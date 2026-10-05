"""
ml_model.py - ML pipeline: dataset creation, training, evaluation, prediction.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
CSV_PATH   = DATA_DIR / "student_marks.csv"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
META_PATH  = MODEL_DIR / "model_metadata.json"

FEATURE_COLS = [
    "attendance",
    "internal_mark",
    "assignment_mark",
    "previous_mark",
    "study_hours",
    "assignment_completion",
]
TARGET_COL = "final_mark"


# ── Sample Dataset ────────────────────────────────────────────────────────────

SAMPLE_STUDENTS = [
    ("ST001","Aarav Sharma","Computer Science",2,3,92,78,85,80,4.5,95,88),
    ("ST002","Priya Nair","Electronics",1,2,85,65,70,68,3.0,80,72),
    ("ST003","Rohan Mehta","Mechanical",3,5,78,55,60,58,2.5,70,63),
    ("ST004","Sneha Pillai","Computer Science",2,4,95,88,90,85,5.0,98,92),
    ("ST005","Kiran Joshi","Civil",4,7,70,48,52,50,2.0,65,55),
    ("ST006","Divya Reddy","Electronics",1,1,88,72,78,74,3.5,85,79),
    ("ST007","Arjun Kumar","Computer Science",3,6,60,38,45,40,1.5,55,44),
    ("ST008","Meena Iyer","Mechanical",2,3,82,68,72,70,3.0,78,73),
    ("ST009","Vijay Patel","Civil",1,2,55,32,40,35,1.0,48,38),
    ("ST010","Anita Singh","Computer Science",4,8,96,92,95,90,6.0,99,95),
    ("ST011","Rahul Gupta","Electronics",3,5,74,58,63,60,2.5,72,65),
    ("ST012","Lakshmi Menon","Mechanical",2,4,89,75,80,76,4.0,88,82),
    ("ST013","Suresh Nair","Computer Science",1,1,67,45,50,47,2.0,60,52),
    ("ST014","Pooja Agarwal","Civil",3,6,91,82,86,83,4.5,92,87),
    ("ST015","Nikhil Sharma","Electronics",4,7,76,62,67,64,3.0,75,68),
    ("ST016","Deepa Krishnan","Computer Science",2,3,84,70,74,71,3.5,82,76),
    ("ST017","Ajay Verma","Mechanical",1,2,58,35,42,38,1.5,52,41),
    ("ST018","Ritu Bajaj","Civil",3,5,93,85,88,86,5.0,94,90),
    ("ST019","Sanjay Tiwari","Computer Science",4,8,71,55,60,57,2.5,68,61),
    ("ST020","Kavya Nambiar","Electronics",2,4,87,74,78,75,4.0,86,80),
    ("ST021","Harsh Malhotra","Mechanical",3,6,64,42,48,44,2.0,58,48),
    ("ST022","Swati Kulkarni","Civil",1,1,90,80,84,81,4.5,91,86),
    ("ST023","Aditya Rao","Computer Science",2,3,79,63,68,65,3.0,77,70),
    ("ST024","Nandita Bose","Electronics",4,7,94,89,92,88,5.5,96,93),
    ("ST025","Manish Chopra","Mechanical",1,2,66,44,50,46,1.5,62,50),
    ("ST026","Shruti Desai","Civil",3,5,83,69,73,70,3.5,81,75),
    ("ST027","Varun Mishra","Computer Science",4,8,61,40,46,42,1.5,57,46),
    ("ST028","Preeti Jain","Electronics",2,4,88,76,80,77,4.0,87,81),
    ("ST029","Gaurav Tiwari","Mechanical",3,6,75,60,65,62,2.5,73,66),
    ("ST030","Amrita Das","Civil",1,2,97,93,96,91,6.0,100,96),
]

SAMPLE_COLUMNS = [
    "roll_number","name","department","year","semester",
    "attendance","internal_mark","assignment_mark","previous_mark",
    "study_hours","assignment_completion","final_mark",
]


def create_sample_dataset():
    """Write sample CSV if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        df = pd.DataFrame(SAMPLE_STUDENTS, columns=SAMPLE_COLUMNS)
        df.to_csv(str(CSV_PATH), index=False)
    return CSV_PATH


def load_dataset() -> pd.DataFrame:
    """Load CSV dataset, creating sample data if missing."""
    create_sample_dataset()
    df = pd.read_csv(str(CSV_PATH))
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame = None) -> dict:
    """
    Train three models, pick best by R², save model + metadata.
    Returns dict with results for all models and best model info.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_dataset()

    # Use only rows with valid feature & target values
    required = FEATURE_COLS + [TARGET_COL]
    df = df[required].dropna()

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression":      LinearRegression(),
        "Decision Tree":          DecisionTreeRegressor(random_state=42),
        "Random Forest":          RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    best_name  = None
    best_r2    = -float("inf")
    best_model = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = float(mean_absolute_error(y_test, y_pred))
        mse  = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2   = float(r2_score(y_test, y_pred))

        results[name] = {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2":  round(r2, 4),
            "model": model,
        }

        if r2 > best_r2:
            best_r2    = r2
            best_name  = name
            best_model = model

    # Save best model
    joblib.dump(best_model, str(MODEL_PATH))

    # Save metadata
    metadata = {
        "best_model_name":    best_name,
        "best_r2":            round(best_r2, 4),
        "feature_cols":       FEATURE_COLS,
        "target_col":         TARGET_COL,
        "training_samples":   len(X_train),
        "test_samples":       len(X_test),
        "model_results": {
            k: {m: v for m, v in v.items() if m != "model"}
            for k, v in results.items()
        },
    }
    with open(str(META_PATH), "w") as f:
        json.dump(metadata, f, indent=2)

    # Return results without model objects (for display)
    display_results = {
        k: {m: v for m, v in v.items() if m != "model"}
        for k, v in results.items()
    }
    return {
        "results":    display_results,
        "best_name":  best_name,
        "best_r2":    round(best_r2, 4),
        "n_train":    len(X_train),
        "n_test":     len(X_test),
    }


# ── Loading & Prediction ──────────────────────────────────────────────────────

def load_model():
    """Load saved model. Trains first if model file missing."""
    if not MODEL_PATH.exists():
        train_models()
    return joblib.load(str(MODEL_PATH))


def get_model_metadata() -> dict:
    """Return metadata dict, or empty dict if not found."""
    if META_PATH.exists():
        with open(str(META_PATH), "r") as f:
            return json.load(f)
    return {}


def predict_mark(student_dict: dict) -> float:
    """
    Predict final mark for a student.
    student_dict must contain all FEATURE_COLS keys.
    Returns float clamped to [0, 100].
    """
    model = load_model()
    features = np.array([[
        float(student_dict.get("attendance",           0)),
        float(student_dict.get("internal_mark",        0)),
        float(student_dict.get("assignment_mark",      0)),
        float(student_dict.get("previous_mark",        0)),
        float(student_dict.get("study_hours",          0)),
        float(student_dict.get("assignment_completion",0)),
    ]])
    prediction = model.predict(features)[0]
    return float(np.clip(prediction, 0.0, 100.0))


def get_best_model_name() -> str:
    """Return name of the best saved model."""
    meta = get_model_metadata()
    return meta.get("best_model_name", "Random Forest")


def ensure_model_exists():
    """Train model if it doesn't exist yet."""
    if not MODEL_PATH.exists():
        train_models()
