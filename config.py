from __future__ import annotations

APP_TITLE = "Datareduktion Playground"
APP_ICON = "📊"
APP_LAYOUT = "wide"

PCA_DATASET_NAME = "Wine"
UMAP_DATASET_NAME = "Penguins"
CA_DATASET_NAME = "Studieform × favorit AI-verktyg"

PCA_DEFAULT_STANDARDIZE = True
PCA_DEFAULT_N_COMPONENTS = 2
PCA_SHOW_LOADINGS_DEFAULT = True
PCA_SHOW_CORR_DEFAULT = False

UMAP_DEFAULT_STANDARDIZE = True
UMAP_DEFAULT_N_NEIGHBORS = 15
UMAP_DEFAULT_MIN_DIST = 0.10
UMAP_DEFAULT_METRIC = "euclidean"
UMAP_DEFAULT_N_COMPONENTS = 2
UMAP_DEFAULT_RANDOM_STATE = 42
UMAP_ALLOWED_METRICS = ["euclidean", "manhattan", "cosine"]

CA_DEFAULT_VIEW_MODE = "Frekvenser"
CA_SHOW_INERTIA_DEFAULT = True
CA_SHOW_PROFILES_DEFAULT = True
CA_DEFAULT_N_COMPONENTS = 2

WINE_TARGET_NAMES = {
    0: "Wine A",
    1: "Wine B",
    2: "Wine C",
}

PENGUINS_FEATURE_COLUMNS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

PENGUINS_TARGET_COLUMN = "species"

METHOD_DESCRIPTIONS = {
    "PCA": (
        "PCA är en linjär metod för numerisk data. "
        "Den hittar nya axlar som fångar maximal varians i datan."
    ),
    "UMAP": (
        "UMAP är en icke-linjär metod som bygger på lokala grannskap i data. "
        "Den är särskilt användbar för att utforska struktur och kluster."
    ),
    "CA": (
        "Korrespondensanalys används för kategoriska data i kontingenstabeller "
        "och visualiserar samband mellan rader och kolumner."
    ),
}