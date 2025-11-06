#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure White-Box Random Forest (CART) for COMP5434 Task 2
- No sklearn/lightgbm/catboost; only numpy/pandas/pickle/json.
- Handles numeric + categorical (own one-hot encoder, unseen -> __OTHER__).
- Stratified split, OOB estimate, Macro-F1, threshold tuning.
- Exports feature importance (impurity decrease) & full tree structures.

Usage:
  python rf_whitebox.py --train ./train.csv
  python rf_whitebox.py --train ./train.csv --test ./test.csv --out ./submission.csv
"""

import argparse, json, pickle, math, os, sys
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


# --------------------------
# Utils: metrics & splitting
# --------------------------
def stratified_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    n_pos_va = int(len(idx_pos) * test_size)
    n_neg_va = int(len(idx_neg) * test_size)
    va_idx = np.concatenate([idx_pos[:n_pos_va], idx_neg[:n_neg_va]])
    tr_idx = np.concatenate([idx_pos[n_pos_va:], idx_neg[n_neg_va:]])
    rng.shuffle(va_idx); rng.shuffle(tr_idx)
    return tr_idx, va_idx

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
        p = tp / (tp + fp) if (tp+fp)>0 else 0.0
        r = tp / (tp + fn) if (tp+fn)>0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        return p, r, f1
    p1,r1,f1_1 = prf(tp, fp, fn)  # class 1
    p0,r0,f1_0 = prf(tn, fn, fp)  # class 0 (swap roles)
    macro_f1 = (f1_0 + f1_1)/2.0
    acc = (tp+tn) / (tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0.0
    return {"acc":acc,"macro_f1":macro_f1,
            "class1":{"precision":p1,"recall":r1,"f1":f1_1},
            "class0":{"precision":p0,"recall":r0,"f1":f1_0},
            "cm":cm.tolist()}

def tune_threshold(y_true, proba, low=0.05, high=0.95, steps=19):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(low, high, steps):
        y_pred = (proba >= t).astype(int)
        f1 = precision_recall_f1(y_true, y_pred)["macro_f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# --------------------------
# Simple One-Hot for objects
# --------------------------
class OneHotSimple:
    def __init__(self, max_categories=None, other_token="__OTHER__"):
        self.max_categories = max_categories
        self.other_token = other_token
        self.col2cats = {}      # col -> [cats...]
        self.out_feature_names = []
        self.cat_cols = []
        self.num_cols = []

    def fit(self, df: pd.DataFrame, cat_cols, num_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        out_names = []

        for c in self.cat_cols:
            vc = df[c].astype(str).fillna(self.other_token).value_counts()
            if self.max_categories and len(vc) > self.max_categories:
                keep = set(vc.head(self.max_categories).index.tolist())
            else:
                keep = set(vc.index.tolist())
            cats = sorted(list(keep))
            if self.other_token not in cats:
                cats.append(self.other_token)
            self.col2cats[c] = cats
            for v in cats:
                out_names.append(f"{c}__{v}")

        # numeric features go as-is
        out_names = self.num_cols + out_names
        self.out_feature_names = out_names
        return self

    def transform(self, df: pd.DataFrame):
        # numeric
        X_num = df[self.num_cols].astype(float).to_numpy() if self.num_cols else np.zeros((len(df),0),dtype=float)
        # categorical
        mats = []
        for c in self.cat_cols:
            cats = self.col2cats[c]
            cat2i = {v:i for i,v in enumerate(cats)}
            col = df[c].astype(str).fillna(self.other_token).to_numpy()
            idx = np.array([cat2i.get(v, cat2i[self.other_token]) for v in col], dtype=int)
            M = np.zeros((len(df), len(cats)), dtype=float)
            M[np.arange(len(df)), idx] = 1.0
            mats.append(M)
        X_cat = np.concatenate(mats, axis=1) if mats else np.zeros((len(df),0),dtype=float)
        X = np.concatenate([X_num, X_cat], axis=1)
        return X, list(self.out_feature_names)


# --------------------------
# CART Decision Tree (binary)
# --------------------------
class TreeNode:
    __slots__ = ("is_leaf","pred","feature","threshold","left","right","impurity","n_samples")
    def __init__(self, is_leaf=True, pred=0.0, feature=None, threshold=None,
                 left=None, right=None, impurity=None, n_samples=0):
        self.is_leaf = is_leaf
        self.pred = float(pred)
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.impurity = impurity
        self.n_samples = int(n_samples)

    def to_dict(self):
        if self.is_leaf:
            return {"leaf":True,"pred":self.pred,"impurity":self.impurity,"n":self.n_samples}
        return {"leaf":False,"f":self.feature,"th":self.threshold,"impurity":self.impurity,"n":self.n_samples,
                "left": self.left.to_dict(),"right": self.right.to_dict()}

def gini_impurity(y):
    if len(y)==0: return 0.0
    p = y.mean()
    return 2.0*p*(1.0-p)

class CARTTree:
    def __init__(self, max_depth=10, min_samples_split=50, min_samples_leaf=25,
                 m_try=None, n_bins=32, random_state=42):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.m_try = m_try            # number of features to consider per split
        self.n_bins = int(n_bins)
        self.rng = np.random.default_rng(random_state)
        self.root = None
        self.n_features_ = None
        self.feature_importances_ = None

    def _best_split_feature(self, x, y):
        N = len(y)
        parent_imp = gini_impurity(y)
        best_gain, best_thr = -1.0, None

        # candidate thresholds
        ux = np.unique(x)
        if len(ux) <= 2:
            # binary or near-binary -> try 0.5 (or mid)
            cand = [0.5*(ux.min()+ux.max())]
        else:
            if len(ux) > self.n_bins:
                qs = np.linspace(0,1,self.n_bins+2)[1:-1]  # exclude 0 and 1
                cand = np.unique(np.quantile(x, qs))
            else:
                cand = (ux[:-1] + ux[1:]) / 2.0

        best_left_idx = None
        for t in cand:
            left = x <= t
            n_l = int(left.sum())
            n_r = N - n_l
            if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                continue
            imp_l = gini_impurity(y[left])
            imp_r = gini_impurity(y[~left])
            gain = parent_imp - (n_l/N)*imp_l - (n_r/N)*imp_r
            if gain > best_gain:
                best_gain = gain
                best_thr = float(t)
                best_left_idx = left
        return best_gain, best_thr, best_left_idx, parent_imp

    def _build(self, X, y, depth, feat_idx_all, importances):
        node = TreeNode(is_leaf=True, pred=float(y.mean()), impurity=gini_impurity(y), n_samples=len(y))
        # stopping
        if (depth >= self.max_depth or len(y) < self.min_samples_split
            or node.impurity == 0.0):
            return node

        n_features = X.shape[1]
        if self.m_try is None:
            k = n_features
        else:
            k = min(self.m_try, n_features)
        feats = self.rng.choice(feat_idx_all, size=k, replace=False)

        best_feat, best_thr, best_gain, best_left = None, None, -1.0, None
        parent_imp = node.impurity

        for f in feats:
            gain, thr, left_mask, _ = self._best_split_feature(X[:,f], y)
            if gain > best_gain:
                best_gain, best_thr, best_feat, best_left = gain, thr, int(f), left_mask

        if best_gain <= 1e-12 or best_left is None:
            return node

        # split & recurse
        left_X, left_y = X[best_left], y[best_left]
        right_X, right_y = X[~best_left], y[~best_left]

        # record importance by impurity decrease
        importances[best_feat] += best_gain * len(y)

        left_child  = self._build(left_X,  left_y,  depth+1, feat_idx_all, importances)
        right_child = self._build(right_X, right_y, depth+1, feat_idx_all, importances)

        node.is_leaf = False
        node.feature = best_feat
        node.threshold = best_thr
        node.left = left_child
        node.right = right_child
        return node

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        importances = np.zeros(self.n_features_, dtype=float)
        feat_idx_all = np.arange(self.n_features_, dtype=int)
        self.root = self._build(X, y, depth=0, feat_idx_all=feat_idx_all, importances=importances)
        imp_sum = importances.sum()
        self.feature_importances_ = importances/imp_sum if imp_sum>0 else importances
        return self

    def _predict_one_proba(self, x, node):
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.pred

    def predict_proba(self, X):
        return np.array([self._predict_one_proba(x, self.root) for x in X], dtype=float)


# --------------------------
# Random Forest (bagging)
# --------------------------
class RandomForestWB:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=50, min_samples_leaf=25,
                 max_features="sqrt", n_bins=32, random_state=42, oob=True):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.n_bins = int(n_bins)
        self.random_state = int(random_state)
        self.oob = bool(oob)

        self.trees_ = []
        self.bootstrap_indices_ = []   # per tree
        self.feature_importances_ = None

    def _m_try(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if isinstance(self.max_features, float):
            return max(1, int(n_features * self.max_features))
        if isinstance(self.max_features, int):
            return min(n_features, self.max_features)
        return n_features

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, m = X.shape
        m_try = self._m_try(m)

        self.trees_ = []
        self.bootstrap_indices_ = []
        all_importances = np.zeros(m, dtype=float)

        for i in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            self.bootstrap_indices_.append(idx)
            Xb, yb = X[idx], y[idx]

            tree = CARTTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf,
                            m_try=m_try, n_bins=self.n_bins,
                            random_state=int(rng.integers(0, 1<<31)))
            tree.fit(Xb, yb)
            all_importances += tree.feature_importances_
            self.trees_.append(tree)

        s = all_importances.sum()
        self.feature_importances_ = all_importances / s if s>0 else all_importances
        return self

    def predict_proba(self, X):
        # mean of tree leaf probabilities
        probs = np.zeros(len(X), dtype=float)
        for t in self.trees_:
            probs += t.predict_proba(X)
        probs /= max(1, len(self.trees_))
        return probs

    def oob_score(self, X, y):
        if not self.oob: return None
        n = len(X)
        sums = np.zeros(n, dtype=float)
        cnts = np.zeros(n, dtype=int)
        for idx, tree in zip(self.bootstrap_indices_, self.trees_):
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[idx] = False
            ix = np.where(oob_mask)[0]
            if len(ix)==0: continue
            p = tree.predict_proba(X[ix])
            sums[ix] += p
            cnts[ix] += 1
        valid = cnts > 0
        proba = np.zeros(n, dtype=float)
        proba[valid] = sums[valid] / cnts[valid]
        mask = valid
        if mask.sum()==0:
            return None
        y_true = y[mask]
        best_t, best_f1 = tune_threshold(y_true, proba[mask])
        y_pred = (proba[mask] >= best_t).astype(int)
        report = precision_recall_f1(y_true, y_pred)
        report["best_threshold"] = best_t
        report["coverage"] = float(mask.mean())
        return report


# --------------------------
# IO & main
# --------------------------
def detect_columns(df, target="label"):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    drop_cols = [c for c in ["id", target] if c in df.columns]
    Xdf = df.drop(columns=drop_cols, errors="ignore")
    y = df[target].astype(int).to_numpy()
    cat_cols = Xdf.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = [c for c in Xdf.columns if c not in cat_cols]
    return Xdf, y, num_cols, cat_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--target", default="label")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--n_estimators", type=int, default=120)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--min_split", type=int, default=80)
    ap.add_argument("--min_leaf", type=int, default=40)
    ap.add_argument("--max_features", default="sqrt")  # "sqrt" or float in (0,1] or int
    ap.add_argument("--n_bins", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load train
    df = pd.read_csv(args.train)
    Xdf, y, num_cols, cat_cols = detect_columns(df, target=args.target)

    # Fit encoder
    enc = OneHotSimple(max_categories=None, other_token="__OTHER__")
    enc.fit(df, cat_cols=cat_cols, num_cols=num_cols)
    X, feat_names = enc.transform(Xdf)

    # Split
    tr_idx, va_idx = stratified_split(X, y, test_size=0.2, seed=args.seed)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    # Train forest
    rf = RandomForestWB(n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        min_samples_split=args.min_split,
                        min_samples_leaf=args.min_leaf,
                        max_features=(args.max_features if args.max_features=="sqrt"
                                      else float(args.max_features) if "." in str(args.max_features)
                                      else int(args.max_features) if str(args.max_features).isdigit()
                                      else "sqrt"),
                        n_bins=args.n_bins,
                        random_state=args.seed,
                        oob=True).fit(Xtr, ytr)

    # OOB report
    oob = rf.oob_score(Xtr, ytr)
    if oob is not None:
        with open("oob_report.json","w",encoding="utf-8") as f:
            json.dump(oob, f, ensure_ascii=False, indent=2)
        print("✅ OOB 评估已保存：oob_report.json")

    # Validation & threshold
    proba_va = rf.predict_proba(Xva)
    best_t, best_f1 = tune_threshold(yva, proba_va)
    ypred_va = (proba_va >= best_t).astype(int)
    rpt = precision_recall_f1(yva, ypred_va)
    print("="*60)
    print(f"[RF-WhiteBox] Validation Macro F1: {rpt['macro_f1']:.6f} @ threshold={best_t:.2f}")
    print(f"[RF-WhiteBox] Accuracy: {rpt['acc']:.6f}")
    print("Confusion Matrix:", rpt["cm"])
    print("="*60)

    # Export feature importances
    fi = pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_})
    fi.sort_values("importance", ascending=False).to_csv("feature_importance.csv", index=False)
    print("✅ 特征重要性已保存：feature_importance.csv")

    # Export tree structures
    with open("forest_trees.jsonl","w",encoding="utf-8") as f:
        for t in rf.trees_:
            f.write(json.dumps(t.root.to_dict(), ensure_ascii=False)+"\n")
    print("✅ 树结构已保存：forest_trees.jsonl")

    # Save model
    with open("model.pkl","wb") as f:
        pickle.dump({
            "encoder": enc,
            "forest": rf,
            "feature_names": feat_names,
            "best_threshold": best_t,
            "target": args.target
        }, f)
    with open("training_config.json","w",encoding="utf-8") as f:
        json.dump({
            "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth,
            "min_samples_split": rf.min_samples_split,
            "min_samples_leaf": rf.min_samples_leaf,
            "max_features": rf.max_features,
            "n_bins": rf.n_bins,
            "seed": args.seed,
            "best_threshold": best_t
        }, f, ensure_ascii=False, indent=2)
    print("✅ 模型与配置已保存：model.pkl, training_config.json")

    # Optional: predict test
    if args.test and os.path.exists(args.test):
        test_df = pd.read_csv(args.test).copy()
        if "id" not in test_df.columns:
            raise ValueError("test.csv 缺少 id 列")
        Xtest_df = test_df.drop(columns=[c for c in ["id", args.target] if c in test_df.columns], errors="ignore")
        Xtest, _ = enc.transform(Xtest_df)
        proba_test = rf.predict_proba(Xtest)
        ypred_test = (proba_test >= best_t).astype(int)
        sub = pd.DataFrame({"id": test_df["id"].values, args.target: ypred_test})
        sub.to_csv(args.out, index=False)
        print(f"✅ 提交文件已保存：{args.out}")
    else:
        print("未提供 --test，跳过生成提交文件。")

if __name__ == "__main__":
    main()
