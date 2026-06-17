"""
Central configuration for Niveles3 EEG analysis pipeline.
All parameters in one place — no magic numbers in code.
"""
import os
import multiprocessing

N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# ===== MONTAJE DE ELECTRODOS (corrected) =====
CHANNEL_NAMES = [
    "Fp1", "Fp3", "F3", "Fz", "F4",  # 0-4: frontal
    "Cz",                              # 5: central-motor
    "Pz", "P5", "P3",                  # 6-8: parietal
    "O1", "Oz",                        # 9-10: occipital
    "T8", "P8", "P6", "P4",           # 11-14: temporo-parietal right
    "T7",                              # 15: temporal left
]
N_CHANNELS = 16

IDX_CZ = 5          # Cz — central motor channel (was 7 in Niveles2, which was Pz)
IDX_MOTOR = [3, 5, 2, 4, 6, 8]       # Fz, Cz, F3, F4, Pz, P3
IDX_FRONTAL = [0, 1, 2, 3, 4]        # Fp1, Fp3, F3, Fz, F4
IDX_PARIETAL = [6, 7, 8, 13, 14]     # Pz, P5, P3, P6, P4
IDX_ALL = list(range(N_CHANNELS))

# ===== CHANNEL SUBSETS FOR CLASSIFICATION =====
CHANNEL_SUBSETS = {
    "Cz":               [5],                           # 1 ch: motor central
    "Motor-2":          [3, 5],                        # 2 ch: Fz, Cz
    "Motor-4":          [3, 5, 6, 8],                  # 4 ch: Fz, Cz, Pz, P3
    "Motor-6":          [3, 5, 2, 4, 6, 8],           # 6 ch: Fz, Cz, F3, F4, Pz, P3
    "Frontal-5":        [0, 1, 2, 3, 4],              # 5 ch: Fp1, Fp3, F3, Fz, F4
    "Parietal-5":       [6, 7, 8, 13, 14],            # 5 ch: Pz, P5, P3, P6, P4
    "Motor-Occipital":  [3, 5, 6, 8, 9, 10],          # 6 ch: Fz, Cz, Pz, P3, O1, Oz
    "All-16":           list(range(N_CHANNELS)),        # 16 ch: todos
}

# ===== HRS (HURST WITH PARTITIONS) — Martinez-Peon 2024 =====
HRS_PARTITION_VALUES = [2, 4, 8, 16, 32, 64, 128]
BEST_HRS_PARTITION = 64
HRS_PARTITION_NAMES = [f"HRS_p{p}" for p in HRS_PARTITION_VALUES]  # ["HRS_p2", ..., "HRS_p128"]
MARTINEZ_FEATURE_NAMES = ["HO", "HRS_p64", "HV"]  # The 3 from the paper
MARTINEZ_ALL_P_NAMES = ["HO", "HV"] + [f"HRS_p{p}" for p in HRS_PARTITION_VALUES]

# ===== SPECTRAL FEATURE BANDS =====
SPECTRAL_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "mu": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}
SPECTRAL_BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

# ===== CLASSIFICATION FEATURE SUBSETS =====
CLASS_FEATURE_SUBSETS = {
    "Delta":          ["Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram", "Delta_Average"],
    "Delta-noVar":    ["Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Average"],
    "Delta-Stable":   None,
    "Martinez":       ["Delta_HO", "Delta_HRS_p64", "Delta_HV"],

    "Martinez-All-p": ["Delta_HO", "Delta_HV"] + [f"Delta_HRS_p{p}" for p in HRS_PARTITION_VALUES],
    "Delta+Martinez": [
        "Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram", "Delta_Average",
        "Delta_HO", "Delta_HRS_p64", "Delta_HV",
    ],
    "Delta+Spectral": [
        "Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram", "Delta_Average",
        "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
        "delta_erd", "theta_erd", "alpha_erd", "beta_erd", "gamma_erd",
        "alpha_beta_ratio", "mu_beta_ratio", "spectral_entropy",
    ],
    "Spectral": [
        "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
        "delta_erd", "theta_erd", "alpha_erd", "beta_erd", "gamma_erd",
        "alpha_beta_ratio", "mu_beta_ratio", "spectral_entropy",
    ],
    # Individual fractal algorithms (1 feature each, spatial mean)
    "RS_only":                ["Active_RS"],
    "Higuchi_only":           ["Active_Higuchi"],
    "DFA_only":               ["Active_DFA"],
    "Semivariograma_only":    ["Active_Variogram"],
    "HO_only":                ["Active_HO"],
    "HRS_only":               ["Active_HRS_p64"],
    "HV_only":                ["Active_HV"],
    "All_Fractals":           [
        "Active_RS", "Active_Higuchi", "Active_DFA", "Active_Variogram",
        "Active_HO", "Active_HRS_p64", "Active_HV",
    ],
    "Active+Delta":   [
        "Active_RS", "Active_Higuchi", "Active_DFA", "Active_Variogram", "Active_Average",
        "Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram", "Delta_Average",
    ],
    "All":            None,
}

# ===== CLASSIFICATION MODELS =====
CLASSIFICATION_MODELS = {
    "SVM": {
        "import": "from sklearn.svm import SVC",
        "model": "SVC(random_state=42, probability=True)",
        "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "poly"], "degree": [3]},
    },
    "kNN": {
        "import": "from sklearn.neighbors import KNeighborsClassifier",
        "model": "KNeighborsClassifier(n_jobs=-1)",
        "params": {"n_neighbors": [5, 10], "weights": ["uniform", "distance"],
                   "metric": ["euclidean", "chebyshev"]},
    },
    "MLP": {
        "import": "from sklearn.neural_network import MLPClassifier",
        "model": "MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)",
        "params": {"hidden_layer_sizes": [(7,), (32,)], "learning_rate_init": [0.001, 0.3],
                   "alpha": [0.001, 0.01]},
    },
    "RandomForest": {
        "import": "from sklearn.ensemble import RandomForestClassifier",
        "model": "RandomForestClassifier(random_state=42, n_jobs=-1)",
        "params": {"n_estimators": [100, 300], "max_depth": [10, None]},
    },
    "NaiveBayes": {
        "import": "from sklearn.naive_bayes import GaussianNB",
        "model": "GaussianNB()",
        "params": {"var_smoothing": [1e-9, 1e-7]},
    },
    "BayesNet": {
        "import": "from sklearn.naive_bayes import GaussianNB",
        "model": "GaussianNB()",
        "params": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    },
    "RandomTree": {
        "import": "from sklearn.tree import DecisionTreeClassifier",
        "model": "DecisionTreeClassifier(random_state=42)",
        "params": {"max_features": ["sqrt", "log2"], "max_depth": [None, 10, 20]},
    },
    "LDA": {
        "import": "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
        "model": "LinearDiscriminantAnalysis()",
        "params": {"solver": ["svd", "lsqr", "eigen"]},
    },
    "LogisticRegression": {
        "import": "from sklearn.linear_model import LogisticRegression",
        "model": "LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)",
        "params": {"C": [0.01, 0.1, 1.0, 10.0]},
    },
}

# ===== SAMPLING & FILTERS =====
FS = 250
F_NOTCH = 60.0
F_LOW = 0.5
F_HIGH = 45.0
NOTCH_Q = 30.0
BANDPASS_ORDER = 4

# ===== ICA =====
USE_ICA = True
N_ICA_REMOVE = 3
ICA_MAX_ITER = 500
ICA_RANDOM_STATE = 42

# ===== EVENT DETECTION =====
CHANNEL_ERD = IDX_CZ  # Cz for ERD detection (INDEX 5, not 7)
MU_BAND = (8, 30)     # Hz — mu/beta band for sensorimotor activity
ERD_PERCENTILE_MIN = 10
ERD_PERCENTILE_MAX = 90
ERD_THRESHOLD_FACTOR = 0.2
MIN_EVENT_DISTANCE_SEC = 3.0  # deduplication window
MIN_EVENT_WIDTH_SEC = 0.5     # minimum ERD peak width

# Marker field names — searched in order
MARKER_FIELDS = ["stims", "stim", "mrk", "y", "markers", "events"]

# EEG data field names — searched in order
EEG_FIELDS = ["EEG_all", "B", "data", "eeg", "EEG"]

# Minimum signal duration in seconds (rejects <10s excerpts)
MIN_SIGNAL_DURATION_SEC = 30.0
MIN_SIGNAL_SAMPLES = int(FS * MIN_SIGNAL_DURATION_SEC)

# ===== FRACTAL FEATURES — ADAPTIVE PARAMETERS =====
RS_SCALE_MIN_SEC = 0.128     # 32 samples at 250Hz
RS_SCALE_MAX_RATIO = 0.40    # max scale = N * ratio
RS_MIN_VALID_SCALES = 5       # minimum points for log-log fit
RS_MIN_BLOCKS_PER_SCALE = 5   # minimum windows per scale

HIGUCHI_K_MAX_RATIO = 0.25   # k_max = min(N * ratio, ceil)
HIGUCHI_K_MAX_CEIL = 30      # absolute max kmax
HIGUCHI_K_MAX_FLOOR = 5       # absolute min kmax
HIGUCHI_MIN_VALID_POINTS = 5  # min points for log-log fit

DFA_SCALE_MIN_SEC = 0.064    # 16 samples at 250Hz
DFA_SCALE_MAX_RATIO = 0.33   # max scale = N * ratio (wider than Niveles2's 0.25)
DFA_MIN_VALID_SCALES = 5
DFA_MIN_WINDOWS_PER_SCALE = 3

VARIOGRAM_MAX_LAG_RATIO = 0.20   # max lag as fraction of N (was fixed at 15 in Niveles2)
VARIOGRAM_MIN_LAGS = 10           # at least 10 lag points
VARIOGRAM_MIN_VALID_POINTS = 5

# ===== STABILITY VALIDATION =====
STABILITY_WINDOWS_SEC = [2.0, 3.0, 4.0, 5.0]
CV_STABLE_THRESHOLD = 0.10
CV_MARGINAL_THRESHOLD = 0.20

# ===== BASAL EPOCHS =====
MIN_REST_GAP_SEC = 5.0
PRE_EVENT_MARGIN_SEC = 1.0
BASAL_SEGMENT_DURATION_SEC = 2.0
MAX_BASAL_PER_GAP = 3

# ===== WEKA EXPORT =====
WEKA_FEATURE_SUBSETS = {
    "DELTA": ["Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram", "Delta_Promedio"],
    "ACTIVO": ["Active_RS", "Active_Higuchi", "Active_DFA", "Active_Variogram", "Active_Promedio"],
    "ACTIVO_DELTA": [
        "Active_RS", "Active_Higuchi", "Active_DFA", "Active_Variogram", "Active_Promedio",
        "Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram", "Delta_Promedio",
    ],
    "TODOS": [
        "Basal_RS", "Basal_Higuchi", "Basal_DFA", "Basal_Variogram", "Basal_Promedio",
        "Active_RS", "Active_Higuchi", "Active_DFA", "Active_Variogram", "Active_Promedio",
        "Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram", "Delta_Promedio",
    ],
}
WEKA_DEFAULT_SUBSET = "DELTA"

# ===== ASYNC/SYNC CLASSIFICATION =====
IMAGERY_KEYWORDS = ["img"]

# Folder name patterns for month extraction
MONTH_FOLDER_PATTERNS = {
    r"1": 1, r"month1": 1, r"mes1": 1, r"m1": 1,
    r"3": 3, r"month3": 3, r"mes3": 3, r"m3": 3,
    r"6": 6, r"month6": 6, r"mes6": 6, r"m6": 6,
}

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
WEKA_DIR = os.path.join(OUTPUT_DIR, "weka")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
VALIDATION_DIR = os.path.join(OUTPUT_DIR, "validation")
CLASSIFICATION_DIR = os.path.join(OUTPUT_DIR, "classification")
CONFUSION_DIR = os.path.join(CLASSIFICATION_DIR, "confusion_matrices")
ROC_DATA_DIR = os.path.join(CLASSIFICATION_DIR, "roc_data")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(WEKA_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)
os.makedirs(ROC_DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
