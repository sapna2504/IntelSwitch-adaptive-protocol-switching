# train_wisetrans_style.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import joblib

# === Load dataset ===
df = pd.read_csv("tcp_quic_paired_with_best.csv")

# === Dataset summary ===
print("=== Dataset Summary ===")
print(f"Shape: {df.shape}")
print("Class distribution (Best column):")
print(df["Best"].value_counts())
print()

# === Feature groups (exclude cpu-pressure & memory-pressure) ===
metrics = [
    "current-bitrate", "last-bitrate", "rebuffering", "throughput",
    "bitrate-variation", "buffer-length", "dropped-frames",
    "latency"
]

tcp_features = [m + "_tcp" for m in metrics]
quic_features = [m + "_quic" for m in metrics]

# === Prepare TCP dataset ===
X_tcp = df[tcp_features]
y_tcp = df["Best"]
X_train_tcp, X_test_tcp, y_train_tcp, y_test_tcp = train_test_split(
    X_tcp, y_tcp, test_size=0.2, random_state=42
)

# === Prepare QUIC dataset ===
X_quic = df[quic_features]
y_quic = df["Best"]
X_train_quic, X_test_quic, y_train_quic, y_test_quic = train_test_split(
    X_quic, y_quic, test_size=0.2, random_state=42
)

# === Define and train TCP model ===
tcp_model = xgb.XGBClassifier(
    learning_rate=0.3,
    n_estimators=150,
    max_depth=7,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0,
    reg_lambda=1,
    n_jobs=-1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

print("Training TCP model...")
tcp_model.fit(X_train_tcp, y_train_tcp)

# Training predictions
y_train_pred_tcp = tcp_model.predict(X_train_tcp)
y_train_proba_tcp = tcp_model.predict_proba(X_train_tcp)
train_acc_tcp = accuracy_score(y_train_tcp, y_train_pred_tcp)
train_loss_tcp = log_loss(y_train_tcp, y_train_proba_tcp)

# Test predictions
y_test_pred_tcp = tcp_model.predict(X_test_tcp)
y_test_proba_tcp = tcp_model.predict_proba(X_test_tcp)
test_acc_tcp = accuracy_score(y_test_tcp, y_test_pred_tcp)
test_loss_tcp = log_loss(y_test_tcp, y_test_proba_tcp)

print(f"✅ TCP Model Training Accuracy: {train_acc_tcp:.4f}, LogLoss: {train_loss_tcp:.4f}")
print(f"✅ TCP Model Test Accuracy: {test_acc_tcp:.4f}, LogLoss: {test_loss_tcp:.4f}\n")

# Save TCP model
joblib.dump(tcp_model, "tcp_xgb_model.pkl")

# === Define and train QUIC model ===
quic_model = xgb.XGBClassifier(
    learning_rate=0.3,
    n_estimators=150,
    max_depth=7,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=0.3,
    reg_alpha=0,
    reg_lambda=1,
    n_jobs=-1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

print("Training QUIC model...")
quic_model.fit(X_train_quic, y_train_quic)

# Training predictions
y_train_pred_quic = quic_model.predict(X_train_quic)
y_train_proba_quic = quic_model.predict_proba(X_train_quic)
train_acc_quic = accuracy_score(y_train_quic, y_train_pred_quic)
train_loss_quic = log_loss(y_train_quic, y_train_proba_quic)

# Test predictions
y_test_pred_quic = quic_model.predict(X_test_quic)
y_test_proba_quic = quic_model.predict_proba(X_test_quic)
test_acc_quic = accuracy_score(y_test_quic, y_test_pred_quic)
test_loss_quic = log_loss(y_test_quic, y_test_proba_quic)

print(f"✅ QUIC Model Training Accuracy: {train_acc_quic:.4f}, LogLoss: {train_loss_quic:.4f}")
print(f"✅ QUIC Model Test Accuracy: {test_acc_quic:.4f}, LogLoss: {test_loss_quic:.4f}\n")

# Save QUIC model
joblib.dump(quic_model, "quic_xgb_model.pkl")

# === Feature importance for interpretability ===
print("\n--- TCP Model Feature Importances ---")
for feat, score in zip(tcp_features, tcp_model.feature_importances_):
    print(f"{feat}: {score:.4f}")

print("\n--- QUIC Model Feature Importances ---")
for feat, score in zip(quic_features, quic_model.feature_importances_):
    print(f"{feat}: {score:.4f}")

print("\n🎯 Training complete. Models saved as tcp_xgb_model.pkl and quic_xgb_model.pkl")