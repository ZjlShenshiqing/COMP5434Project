import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from category_encoders import TargetEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# ======================
# 1️⃣ 数据读取与初步清理
# ======================
df = pd.read_csv("train.csv")
original_df = df.copy()  # 保存原始数据用于后续处理
df = df.drop(columns=["id"])
y = df["label"]

# ======================
# 2️⃣ 特征分组
# ======================
normal_cols = ["no_of_adults", "avg_price_per_room"]
longtail_cols = [
    "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
    "lead_time", "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled", "no_of_special_requests"
]
cat_cols = [
    "type_of_meal_plan", "required_car_parking_space",
    "room_type_reserved", "market_segment_type", "repeated_guest"
]
time_cols = ["arrival_year", "arrival_month", "arrival_date"]

# ======================
# 3️⃣ 时间特征周期化
# ======================
for col in ["arrival_month", "arrival_date"]:
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / df[col].max())
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / df[col].max())
df = df.drop(columns=time_cols)

# ======================
# 4️⃣ 长尾分布 log 平滑
# ======================
df[longtail_cols] = df[longtail_cols].apply(lambda x: np.log1p(x))

# ======================
# 5️⃣ 数值特征缩放
# ======================
scaler_normal = StandardScaler()
df[normal_cols] = scaler_normal.fit_transform(df[normal_cols])

scaler_longtail = MinMaxScaler()
df[longtail_cols] = scaler_longtail.fit_transform(df[longtail_cols])

# ======================
# 6️⃣ 分类特征目标编码
# ======================
encoder = TargetEncoder(cols=cat_cols)
df[cat_cols] = encoder.fit_transform(df[cat_cols], y)

# ======================
# 7️⃣ 生成最终特征集并保存
# ======================
X = df.drop(columns=["label"])
X.to_csv("preprocessed_features.csv", index=False)
y.to_csv("labels.csv", index=False)
print("✅ 预处理完成并保存为：preprocessed_features.csv 与 labels.csv")
print("特征维度:", X.shape)

# ======================
# 8️⃣ 训练/验证集划分
# ======================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 9️⃣ 类别权重计算
# ======================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("📊 类别权重:", class_weights)

# ======================
# 🔟 模型构建与训练
# ======================
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ======================
# 11️⃣ 模型评估（Macro F1）
# ======================
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

acc = accuracy_score(y_val, y_pred)
f1_macro = f1_score(y_val, y_pred, average='macro')

print("\n✅ 评估结果:")
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1-score: {f1_macro:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred, digits=4))

# # ======================
# # 12️⃣ 保存模型与预处理器
# # ======================
# model.save("booking_model.h5")
# joblib.dump(scaler_normal, "scaler_normal.pkl")
# joblib.dump(scaler_longtail, "scaler_longtail.pkl")
# joblib.dump(encoder, "target_encoder.pkl")
# print("\n💾 模型与预处理器已保存：booking_model.h5 + 预处理器 .pkl 文件")

# ======================
# 13️⃣ 加载测试集并进行推理
# ======================
print("\n🔍 开始处理测试集...")
test_df = pd.read_csv("test.csv")
test_ids = test_df["id"].copy()  # 保存ID用于输出结果

# 应用相同的数据预处理步骤
# 复制测试数据以避免修改原始数据
test_data = test_df.copy()

# 删除不需要的列
test_data = test_data.drop(columns=["id"])

# 时间特征周期化
for col in ["arrival_month", "arrival_date"]:
    test_data[f"{col}_sin"] = np.sin(2 * np.pi * test_data[col] / original_df[col].max())
    test_data[f"{col}_cos"] = np.cos(2 * np.pi * test_data[col] / original_df[col].max())
test_data = test_data.drop(columns=[col for col in time_cols if col in test_data.columns])

# 长尾分布 log 平滑
test_data[longtail_cols] = test_data[longtail_cols].apply(lambda x: np.log1p(x))

# 数值特征缩放
test_data[normal_cols] = scaler_normal.transform(test_data[normal_cols])
test_data[longtail_cols] = scaler_longtail.transform(test_data[longtail_cols])

# 分类特征目标编码
test_data[cat_cols] = encoder.transform(test_data[cat_cols])

# 进行预测
test_pred_prob = model.predict(test_data)
test_pred = (test_pred_prob > 0.5).astype(int).flatten()

# 创建结果DataFrame并保存
results_df = pd.DataFrame({
    'id': test_ids,
    'label': test_pred
})

results_df.to_csv('test_predictions1.csv', index=False)
print("✅ 测试集预测完成，结果已保存至 test_predictions1.csv")