# predict_protocol.py

import pandas as pd
import joblib
import sys

# === Load trained models ===
tcp_model = joblib.load("tcp_xgb_model.pkl")
quic_model = joblib.load("quic_xgb_model.pkl")

# === Define expected features ===
metrics = [
    "current-bitrate", "last-bitrate", "rebuffering", "throughput",
    "bitrate-variation", "buffer-length", "dropped-frames",
    "latency"
]

tcp_features = [m + "_tcp" for m in metrics]
quic_features = [m + "_quic" for m in metrics]

def predict_protocol(input_data: dict):
    """Decide whether to call TCP or QUIC model based on keys in input_data."""

    # Detect protocol
    keys = list(input_data.keys())
    if all(k.endswith("_tcp") for k in keys):
        protocol = "tcp"
        model = tcp_model
        features = tcp_features
    elif all(k.endswith("_quic") for k in keys):
        protocol = "quic"
        model = quic_model
        features = quic_features
    else:
        raise ValueError("Input keys must all end with either '_tcp' or '_quic'")

    # Convert dict to DataFrame with correct column order
    df = pd.DataFrame([input_data], columns=features)

    # Run prediction
    pred_class = model.predict(df)[0]
    pred_proba = model.predict_proba(df)[0]

    # Interpret prediction
    meaning = "TCP is better" if pred_class == 0 else "QUIC is better"

    print(f"Protocol detected: {protocol.upper()}")
    print(f"Prediction: {pred_class} ({meaning})")
    print(f"Probabilities: TCP={pred_proba[0]:.4f}, QUIC={pred_proba[1]:.4f}")

    return pred_class, pred_proba, meaning

# === Example usage ===
if __name__ == "__main__":
    # Example input: replace values with real datapoint
    sample_input_tcp = {
        "current-bitrate_tcp": 2500,
        "last-bitrate_tcp": 2400,
        "rebuffering_tcp": 0.2,
        "throughput_tcp": 3000,
        "bitrate-variation_tcp": 0.05,
        "buffer-length_tcp": 25,
        "dropped-frames_tcp": 2,
        "latency_tcp": 80
    }

    sample_input_quic = {
        "current-bitrate_quic": 2600,
        "last-bitrate_quic": 2500,
        "rebuffering_quic": 0.1,
        "throughput_quic": 3100,
        "bitrate-variation_quic": 0.03,
        "buffer-length_quic": 27,
        "dropped-frames_quic": 1,
        "latency_quic": 70
    }

    print("=== Testing with TCP input ===")
    predict_protocol(sample_input_tcp)
    print("\n=== Testing with QUIC input ===")
    predict_protocol(sample_input_quic)
