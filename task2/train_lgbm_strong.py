#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMP5434 - Task 2 购房意向分类（LightGBM 强化版）
功能：
- 自动识别数值/类别列并做预处理（缺失填充 + OneHot）
- 训练 LightGBM，分层划分验证集
- 以 Macro F1 为主指标，并在验证集上搜索最优阈值
- 用全部训练数据重训并保存 model.joblib
- （可选）对 test.csv 生成提交文件（id,label）
用法示例：
    python train_lgbm_strong.py --train .\train.csv
    python train_lgbm_strong.py --train .\train.csv --test .\test.csv --out .\submission.csv
"""

import argparse
import os
import inspect
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


def detect_columns(df: pd.DataFrame, target: str = "label"):
    """自动识别数值/类别列，并移除不参与训练的列（id/target）"""
    if target not in df.columns:
        raise ValueError(f"未找到目标列 '{target}'")
    drop_cols = [c for c in ["id", target] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target].astype(int).values
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, num_cols, cat_cols


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray, grid=None):
    """在验证集上搜索最优概率阈值（最大化 macro F1）"""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        f1m = f1_score(y_true, y_pred, average="macro")
        if f1m > best_f1:
            best_f1, best_t = f1m, t
    return best_t, best_f1


def build_pipeline(num_cols, cat_cols):
    """构建 预处理 + LightGBM 的 Pipeline（兼容新老 sklearn 的 OHE 参数）"""
    num_tf = Pipeline([("imp", SimpleImputer(strategy="median"))])

    # 兼容 scikit-learn>=1.4（使用 sparse_output），老版本退回 sparse
    ohe_params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe = OneHotEncoder(**ohe_params, sparse_output=True)
    else:
        ohe = OneHotEncoder(**ohe_params, sparse=True)

    cat_tf = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    pre = ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols)
    ])

    model = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",   # 类别不均衡更稳
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="训练集 CSV 路径（需包含 label）")
    ap.add_argument("--test", default=None, help="（可选）测试集 CSV 路径（含 id，无 label）")
    ap.add_argument("--target", default="label", help="目标列名（默认 label）")
    ap.add_argument("--out", default="submission.csv", help="提交文件输出路径")
    args = ap.parse_args()

    # 读取训练集并识别列
    df = pd.read_csv(args.train)
    X, y, num_cols, cat_cols = detect_columns(df, target=args.target)

    # 划分训练/验证
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 训练
    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_tr, y_tr)

    # 验证评估 + 阈值调优
    proba_va = pipe.predict_proba(X_va)[:, 1]
    best_t, best_f1 = tune_threshold(y_va, proba_va)
    y_pred_va = (proba_va >= best_t).astype(int)
    acc = accuracy_score(y_va, y_pred_va)

    print("=" * 60)
    print(f"[LGBM] Validation Macro F1: {best_f1:.6f} @ threshold={best_t:.2f}")
    print(f"[LGBM] Accuracy: {acc:.6f}")
    print("Confusion Matrix:\n", confusion_matrix(y_va, y_pred_va))
    print("Classification Report:\n", classification_report(y_va, y_pred_va, digits=4))
    print("=" * 60)

    # 用全部训练数据重训并保存
    pipe.fit(X, y)
    joblib.dump(
        {"pipeline": pipe, "best_threshold": best_t, "target": args.target},
        "model.joblib"
    )
    print("✅ 模型已保存：model.joblib")

    # （可选）生成提交文件
    if args.test and os.path.exists(args.test):
        test_df = pd.read_csv(args.test).copy()
        if "id" not in test_df.columns:
            raise ValueError("test.csv 缺少 id 列")
        X_test = test_df.drop(
            columns=[c for c in ["id", args.target] if c in test_df.columns],
            errors="ignore"
        )
        proba_test = pipe.predict_proba(X_test)[:, 1]
        pred = (proba_test >= best_t).astype(int)

        sub = pd.DataFrame({"id": test_df["id"], args.target: pred})
        sub.to_csv(args.out, index=False)
        print(f"✅ 提交文件已保存：{args.out}")
    else:
        print("未提供 --test，跳过生成提交文件。")


if __name__ == "__main__":
    main()
