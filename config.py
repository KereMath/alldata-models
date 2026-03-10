"""
alldata-models - Konfigürasyon
Tüm Generated Data tiplerinden 2-head multi-class model egitimi.
"""
from pathlib import Path

BASE_DIR           = Path(__file__).parent
GENERATED_DATA_DIR = Path(r"C:\Users\user\Desktop\Generated Data")
COMBINATIONS_DIR   = GENERATED_DATA_DIR / "Combinations"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR         = BASE_DIR / "trained_models"
RESULTS_DIR        = BASE_DIR / "results"

# -------------------------------------------------------------------
# Label tanimlari
# -------------------------------------------------------------------
BASE_LABELS = [
    "stationary",
    "deterministic_trend",
    "stochastic_trend",
    "volatility",
]

ANOMALY_LABELS = [
    "none",
    "collective_anomaly",
    "contextual_anomaly",
    "mean_shift",
    "point_anomaly",
    "trend_shift",
    "variance_shift",
]

# -------------------------------------------------------------------
# Örnekleme parametreleri
# -------------------------------------------------------------------
SAMPLES_PER_CATEGORY = 350   # Her kategoriden alinacak max ornek
TEST_SIZE            = 0.20
VALIDATION_SIZE      = 0.10
RANDOM_STATE         = 42
MIN_SERIES_LENGTH    = 50

# -------------------------------------------------------------------
# Tekli kategoriler:
# (Generated Data altindaki klasor adi, base_label, anomaly_label)
# -------------------------------------------------------------------
SINGLE_CATEGORIES = [
    # Anomalisiz taban tipleri
    ("stationary",          "stationary",         "none"),
    ("deterministic_trend", "deterministic_trend", "none"),
    ("Stochastic Trend",    "stochastic_trend",    "none"),
    ("Volatility",          "volatility",          "none"),
    # Stokastik taban + tek anomali tipleri
    ("collective_anomaly",  "stationary",          "collective_anomaly"),
    ("contextual_anomaly",  "stationary",          "contextual_anomaly"),
    ("mean_shift",          "stationary",          "mean_shift"),
    ("point_anomaly",       "stationary",          "point_anomaly"),
    ("trend_shift",         "stationary",          "trend_shift"),
    ("variance_shift",      "stationary",          "variance_shift"),
]

# -------------------------------------------------------------------
# Kombinasyon yapraklari:
# (path_parts from COMBINATIONS_DIR, base_label, anomaly_label)
# -------------------------------------------------------------------
COMBO_LEAVES = [
    # ---- Cubic Base ----
    (["Cubic Base", "Cubic Base", "Cubic + Mean Shift"],             "deterministic_trend", "mean_shift"),
    (["Cubic Base", "Cubic Base", "Cubic + Point Anomaly"],          "deterministic_trend", "point_anomaly"),
    (["Cubic Base", "Cubic Base", "Cubic + Variance Shift"],         "deterministic_trend", "variance_shift"),
    (["Cubic Base", "Cubic Base", "cubic_collective_anomaly"],       "deterministic_trend", "collective_anomaly"),
    # ---- Damped Base ----
    (["Damped Base", "Damped Base", "Damped + Collective Anomaly"],  "deterministic_trend", "collective_anomaly"),
    (["Damped Base", "Damped Base", "Damped + Mean Shift"],          "deterministic_trend", "mean_shift"),
    (["Damped Base", "Damped Base", "Damped + Point Anomaly"],       "deterministic_trend", "point_anomaly"),
    (["Damped Base", "Damped Base", "Damped + Variance Shift"],      "deterministic_trend", "variance_shift"),
    # ---- Exponential Base ----
    (["Exponential Base", "Exponential Base", "Exponential + Mean Shift"],       "deterministic_trend", "mean_shift"),
    (["Exponential Base", "Exponential Base", "exponential_collective_anomaly"], "deterministic_trend", "collective_anomaly"),
    (["Exponential Base", "Exponential Base", "exponential_point_anomaly"],      "deterministic_trend", "point_anomaly"),
    (["Exponential Base", "Exponential Base", "exponential_variance_shift"],     "deterministic_trend", "variance_shift"),
    # ---- Linear Base ----
    (["Linear Base", "Linear Base", "Linear + Collective Anomaly"],  "deterministic_trend", "collective_anomaly"),
    (["Linear Base", "Linear Base", "Linear + Mean Shift"],          "deterministic_trend", "mean_shift"),
    (["Linear Base", "Linear Base", "Linear + Point Anomaly"],       "deterministic_trend", "point_anomaly"),
    (["Linear Base", "Linear Base", "Linear + Trend Shift"],         "deterministic_trend", "trend_shift"),
    (["Linear Base", "Linear Base", "Linear + Variance Shift"],      "deterministic_trend", "variance_shift"),
    # ---- Quadratic Base ----
    (["Quadratic Base", "Quadratic Base", "Quadratic + Collective anomaly"], "deterministic_trend", "collective_anomaly"),
    (["Quadratic Base", "Quadratic Base", "Quadratic + Mean Shift"],         "deterministic_trend", "mean_shift"),
    (["Quadratic Base", "Quadratic Base", "Quadratic + Point Anomaly"],      "deterministic_trend", "point_anomaly"),
    (["Quadratic Base", "Quadratic Base", "Quadratic + Variance Shift"],     "deterministic_trend", "variance_shift"),
    # ---- Stochastic Trend + anomali ----
    (["Stochastic Trend + Collective Anomaly"], "stochastic_trend", "collective_anomaly"),
    (["Stochastic Trend + Mean Shift"],         "stochastic_trend", "mean_shift"),
    (["Stochastic Trend + Point Anomaly"],      "stochastic_trend", "point_anomaly"),
    (["Stochastic Trend + Variance Shift", "Stochastic Trend + Variance Shift"], "stochastic_trend", "variance_shift"),
    # ---- Volatility + anomali ----
    (["Volatility + Collective Anomaly"], "volatility", "collective_anomaly"),
    (["Volatility + Mean Shift"],         "volatility", "mean_shift"),
    (["Volatility + Point Anomaly"],      "volatility", "point_anomaly"),
    (["Volatility + Variance Shift"],     "volatility", "variance_shift"),
]
