#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure White-Box GBDT (Logistic Loss) for COMP5434 Task 2
- No sklearn/lightgbm/xgboost/catboost. Only numpy/pandas/json/pickle.
- Second-order (Newton) boosting:
    F(x) = F0 + sum_{m=1..M} eta * f_m(x)
  where each f_m is a shallow CART regression tree.
- Split gain uses standard 2nd-order formula:
    Gain = G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G^2/(H+λ) - γ
  Leaf value = G/(H+λ).
- With stratified split, macro-F1 threshold tuning, feature importance (by gain),
  and full tree export for audit.

Usage:
  python gbdt_whitebox.py --train ./train.csv
  python gbdt_whitebox.py --train ./train.csv --test ./test.csv --out ./submission.csv
"""

import argparse, json, pickle, os
from dataclasses import dataclass
import numpy as np
import pandas as pd

# -------------------------
# Utilities: metrics, split
# -------------------------
def stratified_split(y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    pos = np.where(y==1)[0]; rng.shuffle(pos)
    neg = np.where(y==0)[0]; rng.shuffle(neg)
    npos_va = int(len(pos)*test_size); nneg_va = int(len(neg)*test_size)
    va = np.concatenate([pos[:npos_va], neg[:nneg_va]])
    tr = np.concatenate([pos[npos_va:], neg[nneg_va:]])
    rng.shuffle(va); rng.shuffle(tr)
    return tr, va

def sigmoid(x):
    # numerically stable
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1 / (1 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1 + expx)
    return out

def confusion_matrix(y_true, y_pred):
    y_true = y_true.astype(int); y_pred = y_pred.astype(int)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    return np.array([[tn, fp],[fn, tp]])

def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    def prf(tp, fp, fn):
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        return p,r,f1
    p1,r1,f1_1 = prf(tp, fp, fn)
    p0,r0,f1_0 = prf(tn, fn, fp)  # treat 0 as positive
    macro_f1 = (f1_0 + f1_1)/2.0
    acc = (tp+tn) / max(1,(tp+tn+fp+fn))
    return {"acc":acc,"macro_f1":macro_f1,"cm":cm.tolist(),
            "class1":{"precision":p1,"recall":r1,"f1":f1_1},
            "class0":{"precision":p0,"recall":r0,"f1":f1_0}}

def tune_threshold(y_true, proba, low=0.05, high=0.95, steps=19):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(low, high, steps):
        y_pred = (proba >= t).astype(int)
        f1 = precision_recall_f1(y_true, y_pred)["macro_f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1

# -------------------------
# Simple One-Hot for object
# -------------------------
class OneHotSimple:
    """Simple one-hot for categorical columns. Keeps all categories (or top-K), adds __OTHER__."""
    def __init__(self, max_categories=None, other="__OTHER__"):
        self.max_categories = max_categories
        self.other = other
        self.cat_cols = []
        self.num_cols = []
        self.col2cats = {}
        self.out_feature_names = []

    def fit(self, df, cat_cols, num_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        feat_names = list(self.num_cols)
        for c in self.cat_cols:
            vc = df[c].astype(str).fillna(self.other).value_counts()
            if self.max_categories and len(vc) > self.max_categories:
                cats = sorted(vc.head(self.max_categories).index.tolist())
                if self.other not in cats: cats.append(self.other)
            else:
                cats = sorted(vc.index.tolist())
                if self.other not in cats: cats.append(self.other)
            self.col2cats[c] = cats
            feat_names += [f"{c}__{v}" for v in cats]
        self.out_feature_names = feat_names
        return self

    def transform(self, df):
        X_num = df[self.num_cols].astype(float).to_numpy() if self.num_cols else np.zeros((len(df),0))
        mats = []
        for c in self.cat_cols:
            cats = self.col2cats[c]; cat2i = {v:i for i,v in enumerate(cats)}
            col = df[c].astype(str).fillna(self.other).to_numpy()
            idx = np.array([cat2i.get(v, cat2i[self.other]) for v in col], dtype=int)
            M = np.zeros((len(df), len(cats)), dtype=float)
            M[np.arange(len(df)), idx] = 1.0
            mats.append(M)
        X_cat = np.concatenate(mats, axis=1) if mats else np.zeros((len(df),0))
        X = np.concatenate([X_num, X_cat], axis=1)
        return X, list(self.out_feature_names)

# -------------------------
# GBDT CART Regression Tree
# -------------------------
@dataclass
class Node:
    is_leaf: bool
    value: float = 0.0         # leaf value (logit increment)
    feature: int = None
    thr: float = None
    left: 'Node' = None
    right: 'Node' = None
    gain: float = 0.0
    n: int = 0

class GBDTTree:
    """A single regression tree trained with 2nd-order gain (Logloss)."""
    def __init__(self, max_depth=3, min_samples_leaf=20, lambda_l2=1.0,
                 gamma=0.0, n_bins=32, random_state=42, feature_subsample=None):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.lambda_l2 = float(lambda_l2)
        self.gamma = float(gamma)          # minimum gain to split
        self.n_bins = int(n_bins)
        self.rng = np.random.default_rng(random_state)
        self.feature_subsample = feature_subsample  # int or float or None
        self.root = None
        self.feature_gain_ = None

    def _cand_thresholds(self, x):
        ux = np.unique(x)
        if len(ux) <= 2:
            return [0.5*(ux.min()+ux.max())]
        if len(ux) > self.n_bins:
            qs = np.linspace(0,1,self.n_bins+2)[1:-1]
            return np.unique(np.quantile(x, qs))
        return (ux[:-1] + ux[1:]) / 2.0

    def _best_split_feature(self, x, g, h):
        # sums on parent
        G = g.sum(); H = h.sum()
        base = (G*G)/(H + self.lambda_l2)

        best_gain, best_thr, best_left_mask = -1.0, None, None
        for t in self._cand_thresholds(x):
            L = x <= t
            nL = int(L.sum()); nR = len(x) - nL
            if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
                continue
            GL, HL = g[L].sum(), h[L].sum()
            GR, HR = G-GL, H-HL
            gain = (GL*GL)/(HL + self.lambda_l2) + (GR*GR)/(HR + self.lambda_l2) - base - self.gamma
            if gain > best_gain:
                best_gain, best_thr, best_left_mask = float(gain), float(t), L
        return best_gain, best_thr, best_left_mask

    def _build(self, X, g, h, depth):
        node = Node(is_leaf=True, value=(g.sum()/(h.sum()+self.lambda_l2)) if len(h)>0 else 0.0,
                    n=int(len(h)))
        if depth >= self.max_depth:
            return node
        if node.n < 2*self.min_samples_leaf:
            return node

        m = X.shape[1]
        if self.feature_subsample is None:
            feat_idx = np.arange(m)
        elif isinstance(self.feature_subsample, int):
            k = min(m, self.feature_subsample)
            feat_idx = self.rng.choice(np.arange(m), size=k, replace=False)
        else:
            # float in (0,1]
            k = max(1, int(m * float(self.feature_subsample)))
            feat_idx = self.rng.choice(np.arange(m), size=k, replace=False)

        best_feat, best_thr, best_gain, best_mask = None, None, -1.0, None
        for f in feat_idx:
            gain, thr, mask = self._best_split_feature(X[:,f], g, h)
            if gain > best_gain:
                best_gain, best_thr, best_feat, best_mask = gain, thr, int(f), mask

        if best_gain <= 1e-12 or best_mask is None:
            return node

        left = self._build(X[best_mask], g[best_mask], h[best_mask], depth+1)
        right = self._build(X[~best_mask], g[~best_mask], h[~best_mask], depth+1)

        parent = Node(is_leaf=False, feature=best_feat, thr=best_thr,
                      left=left, right=right, gain=best_gain, n=int(len(h)))
        return parent

    def fit(self, X, g, h):
        self.root = self._build(X, g, h, depth=0)
        # feature gain aggregation
        gains = {}

        def walk(n: Node):
            if n is None or n.is_leaf: return
            gains[n.feature] = gains.get(n.feature, 0.0) + n.gain * n.n
            walk(n.left); walk(n.right)

        walk(self.root)
        self.feature_gain_ = gains
        return self

    def _pred_one(self, x):
        n = self.root
        while not n.is_leaf:
            n = n.left if x[n.feature] <= n.thr else n.right
        return n.value

    def predict_value(self, X):
        return np.array([self._pred_one(x) for x in X], dtype=float)

    def to_dict(self, n=None):
        n = self.root if n is None else n
        if n.is_leaf:
            return {"leaf": True, "value": float(n.value), "n": n.n}
        return {"leaf": False, "f": n.feature, "thr": float(n.thr), "gain": float(n.gain), "n": n.n,
                "left": self.to_dict(n.left), "right": self.to_dict(n.right)}

# -------------------------
# GBDT model
# -------------------------
class GBDTWhiteBox:
    def __init__(self, n_estimators=200, learning_rate=0.1,
                 max_depth=3, min_samples_leaf=20, lambda_l2=1.0, gamma=0.0,
                 subsample=1.0, colsample=1.0, n_bins=32, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.lambda_l2 = lambda_l2
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
        self.n_bins = n_bins
        self.rng = np.random.default_rng(random_state)

        self.trees_ = []
        self.init_score_ = 0.0
        self.feature_importances_ = None

    def fit(self, X, y):
        n, m = X.shape
        # init with log-odds
        pos_rate = max(1e-6, min(1-1e-6, y.mean()))
        self.init_score_ = np.log(pos_rate/(1-pos_rate))

        F = np.full(n, self.init_score_, dtype=float)
        self.trees_ = []
        gain_sum = {}

        for t in range(self.n_estimators):
            p = sigmoid(F)
            g = y - p                 # negative gradient for logistic
            h = p*(1-p)               # Hessian

            # row subsample
            if self.subsample < 1.0:
                idx = self.rng.choice(np.arange(n), size=int(n*self.subsample), replace=False)
            else:
                idx = np.arange(n)

            # col subsample
            feat_sub = None
            if self.colsample < 1.0:
                k = max(1, int(m*self.colsample))
                feat_sub = self.rng.choice(np.arange(m), size=k, replace=False)
                X_sub = X[np.ix_(idx, feat_sub)]
            else:
                feat_sub = np.arange(m)
                X_sub = X[idx]

            tree = GBDTTree(max_depth=self.max_depth,
                            min_samples_leaf=self.min_samples_leaf,
                            lambda_l2=self.lambda_l2,
                            gamma=self.gamma,
                            n_bins=self.n_bins,
                            random_state=int(self.rng.integers(0, 1<<31)),
                            feature_subsample=None).fit(X_sub, g[idx], h[idx])

            # update predictions on all samples (need full X)
            # tree was trained on subset of features; we must map back:
            def pred_full(Xall):
                if len(feat_sub) == m:
                    return tree.predict_value(Xall)
                # project Xall to feat_sub
                return tree.predict_value(Xall[:, feat_sub])

            increment = self.learning_rate * pred_full(X)
            F += increment

            # collect gains
            for f, gsum in (tree.feature_gain_.items() if tree.feature_gain_ else []):
                real_f = feat_sub[f] if len(feat_sub) != X.shape[1] else f
                gain_sum[real_f] = gain_sum.get(real_f, 0.0) + gsum

            self.trees_.append({"tree": tree, "feat_sub": np.array(feat_sub, dtype=int)})

        # normalize feature importances
        total = sum(gain_sum.values()) if gain_sum else 0.0
        imp = np.zeros(m, dtype=float)
        if total > 0:
            for f,v in gain_sum.items():
                imp[f] = v/total
        self.feature_importances_ = imp
        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.init_score_, dtype=float)
        for item in self.trees_:
            tree = item["tree"]; feat_sub = item["feat_sub"]
            Fx = tree.predict_value(X[:, feat_sub]) if len(feat_sub)!=X.shape[1] else tree.predict_value(X)
            F += self.learning_rate * Fx
        return sigmoid(F)

    def dump(self):
        # export all trees to list of dicts
        return [{"feat_sub": item["feat_sub"].tolist(),
                 "tree": item["tree"].to_dict()} for item in self.trees_]

# -------------------------
# IO & main
# -------------------------
def detect_columns(df, target="label"):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    drop_cols = [c for c in ["id", target] if c in df.columns]
    Xdf = df.drop(columns=drop_cols, errors="ignore")
    y = df[target].astype(int).to_numpy()
    cat_cols = Xdf.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = [c for c in Xdf.columns if c not in cat_cols]
    return Xdf, y, num_cols, cat_cols

def fill_missing(df, num_cols, cat_cols):
    X = df.copy()
    for c in num_cols:
        X[c] = X[c].astype(float)
        med = X[c].median()
        X[c] = X[c].fillna(med)
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("__OTHER__")
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--target", default="label")
    ap.add_argument("--out", default="submission.csv")
    # GBDT hyper-params
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--min_leaf", type=int, default=20)
    ap.add_argument("--lambda_l2", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--colsample", type=float, default=1.0)
    ap.add_argument("--n_bins", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.train)
    Xdf, y, num_cols, cat_cols = detect_columns(df, target=args.target)
    Xdf = fill_missing(Xdf, num_cols, cat_cols)

    # Fit encoder
    enc = OneHotSimple(max_categories=None, other="__OTHER__")
    enc.fit(Xdf, cat_cols=cat_cols, num_cols=num_cols)
    X, feat_names = enc.transform(Xdf)

    # Split
    tr_idx, va_idx = stratified_split(y, test_size=0.2, seed=args.seed)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    # Train GBDT
    gbdt = GBDTWhiteBox(n_estimators=args.n_estimators, learning_rate=args.lr,
                        max_depth=args.max_depth, min_samples_leaf=args.min_leaf,
                        lambda_l2=args.lambda_l2, gamma=args.gamma,
                        subsample=args.subsample, colsample=args.colsample,
                        n_bins=args.n_bins, random_state=args.seed).fit(Xtr, ytr)

    # Validation + threshold tuning
    proba_va = gbdt.predict_proba(Xva)
    best_t, best_f1 = tune_threshold(yva, proba_va)
    ypred_va = (proba_va >= best_t).astype(int)
    rpt = precision_recall_f1(yva, ypred_va)
    print("="*60)
    print(f"[GBDT-WhiteBox] Validation Macro F1: {rpt['macro_f1']:.6f} @ threshold={best_t:.2f}")
    print(f"[GBDT-WhiteBox] Accuracy: {rpt['acc']:.6f}")
    print("Confusion Matrix:", rpt["cm"])
    print("="*60)

    # Exports: feature importance, trees, model
    fi = pd.DataFrame({"feature": feat_names, "importance_gain": gbdt.feature_importances_})
    fi.sort_values("importance_gain", ascending=False).to_csv("feature_importance_gbdt.csv", index=False)
    with open("gbdt_trees.json","w",encoding="utf-8") as f:
        json.dump(gbdt.dump(), f, ensure_ascii=False, indent=2)
    with open("gbdt_model.pkl","wb") as f:
        pickle.dump({
            "encoder": enc,
            "model": gbdt,
            "feature_names": feat_names,
            "best_threshold": best_t,
            "target": args.target
        }, f)
    with open("gbdt_config.json","w",encoding="utf-8") as f:
        json.dump({
            "n_estimators": args.n_estimators,
            "learning_rate": args.lr,
            "max_depth": args.max_depth,
            "min_leaf": args.min_leaf,
            "lambda_l2": args.lambda_l2,
            "gamma": args.gamma,
            "subsample": args.subsample,
            "colsample": args.colsample,
            "n_bins": args.n_bins,
            "seed": args.seed,
            "best_threshold": best_t
        }, f, ensure_ascii=False, indent=2)
    print("✅ 已保存：feature_importance_gbdt.csv, gbdt_trees.json, gbdt_model.pkl, gbdt_config.json")

    # Predict test
    if args.test and os.path.exists(args.test):
        tdf = pd.read_csv(args.test).copy()
        if "id" not in tdf.columns:
            raise ValueError("test.csv 缺少 id 列")
        Xtest_df = tdf.drop(columns=[c for c in ["id", args.target] if c in tdf.columns], errors="ignore")
        Xtest_df = fill_missing(Xtest_df, num_cols, cat_cols)
        Xtest, _ = enc.transform(Xtest_df)
        proba_te = gbdt.predict_proba(Xtest)
        ypred_te = (proba_te >= best_t).astype(int)
        pd.DataFrame({"id": tdf["id"], args.target: ypred_te}).to_csv(args.out, index=False)
        print(f"✅ 提交文件已保存：{args.out}")
    else:
        print("未提供 --test，跳过生成提交文件。")

if __name__ == "__main__":
    main()
