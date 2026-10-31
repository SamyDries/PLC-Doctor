
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "fault_dataset.csv"
MODEL_PATH = MODEL_DIR / "plc_fault_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# --------------------------------------------------
# 2. FAULT CATEGORIES
# --------------------------------------------------

FAULT_TYPES = [
    "Motor Overload",
    "Sensor Failure",
    "Power Supply Failure",
    "VFD Fault",
    "PLC Communication Failure",
    "Normal Operation"
]

# --------------------------------------------------
# 3. SYNTHETIC DATA GENERATION
# --------------------------------------------------

def generate_dataset(samples_per_class=100):

    rows = []

    for fault in FAULT_TYPES:

        for _ in range(samples_per_class):

            # Default values: normal machine operation
            motor_current = rng.uniform(5, 12)
            motor_temperature = rng.uniform(35, 65)
            vibration = rng.uniform(0.5, 3.0)
            sensor_status = 1
            voltage = rng.uniform(215, 240)
            vfd_fault = 0
            plc_communication = 1
            machine_running = 1

            if fault == "Motor Overload":
                motor_current = rng.uniform(19, 30)
                motor_temperature = rng.uniform(80, 110)
                vibration = rng.uniform(2, 6)

            elif fault == "Sensor Failure":
                sensor_status = 0
                motor_current = rng.uniform(5, 12)
                motor_temperature = rng.uniform(35, 65)

            elif fault == "Power Supply Failure":
                voltage = rng.uniform(165, 195)
                machine_running = 0
                motor_current = rng.uniform(0, 3)

            elif fault == "VFD Fault":
                vfd_fault = 1
                machine_running = 0
                motor_current = rng.uniform(0, 3)

            elif fault == "PLC Communication Failure":
                plc_communication = 0

            elif fault == "Normal Operation":
                pass

            rows.append({
                "motor_current": round(motor_current, 2),
                "motor_temperature": round(motor_temperature, 2),
                "vibration": round(vibration, 2),
                "sensor_status": sensor_status,
                "voltage": round(voltage, 2),
                "vfd_fault": vfd_fault,
                "plc_communication": plc_communication,
                "machine_running": machine_running,
                "fault_type": fault
            })

    dataset = pd.DataFrame(rows)

    # Shuffle observations
    dataset = dataset.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    return dataset


# --------------------------------------------------
# 4. GENERATE AND SAVE DATASET
# --------------------------------------------------

print("\n" + "=" * 60)
print("PLC DOCTOR - MACHINE LEARNING TRAINING")
print("=" * 60)

df = generate_dataset(samples_per_class=100)

df.to_csv(DATASET_PATH, index=False)

print("\nDataset successfully generated.")
print(f"Number of observations: {len(df)}")
print(f"Number of features: {len(df.columns) - 1}")
print(f"Dataset saved to: {DATASET_PATH}")

print("\nClass distribution:")
print(df["fault_type"].value_counts())


# --------------------------------------------------
# 5. PREPARE FEATURES AND TARGET
# --------------------------------------------------

FEATURES = [
    "motor_current",
    "motor_temperature",
    "vibration",
    "sensor_status",
    "voltage",
    "vfd_fault",
    "plc_communication",
    "machine_running"
]

X = df[FEATURES]
y = df["fault_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))


# --------------------------------------------------
# 6. TRAIN THE RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

print("\nRandom Forest training completed.")


# --------------------------------------------------
# 7. EVALUATE THE MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=FAULT_TYPES,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=FAULT_TYPES
    )
)


# --------------------------------------------------
# 8. SAVE TRAINED MODEL AND METADATA
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

metadata = {
    "model_name": "DecisionTreeClassifier",
    "features": FEATURES,
    "classes": FAULT_TYPES,
    "accuracy": round(float(accuracy), 4),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "dataset_samples": len(df),
    "random_state": RANDOM_STATE,
    "dataset_type": "Synthetic educational dataset"
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\n" + "=" * 60)
print("SAVED FILES")
print("=" * 60)

print(f"Dataset: {DATASET_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Metadata: {METADATA_PATH}")

print("\nPLC DOCTOR MODEL IS READY.")

