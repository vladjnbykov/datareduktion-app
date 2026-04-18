# Datareduktion Playground

En pedagogisk Streamlit-app för att hjälpa studenter utforska tre metoder för datareduktion:

- PCA
- UMAP
- Korrespondensanalys (CA)

Appen är byggd för undervisning och exploration före egna projekt.

## Funktioner

### PCA
- Wine-dataset
- Val av features
- Standardisering på/av
- Scree plot
- Kumulativ förklarad varians
- Loadings
- PCA-visualisering i 2D

### UMAP
- Penguins-dataset
- Val av features
- Standardisering på/av
- Justering av:
  - `n_neighbors`
  - `min_dist`
  - `metric`
  - `random_state`
- 2D/3D-visualisering

### Korrespondensanalys
- Syntetisk kontingenstabell
- Biplot för rader och kolumner
- Inertia
- Rad- och kolumnprofiler

### Jämförelse
- PCA vs UMAP på samma dataset

## Projektstruktur

```text
data_reduction_app/
│
├── app.py
├── config.py
│
├── data/
├── methods/
├── components/
├── app_pages/
├── utils/
│
├── requirements.txt
├── Procfile
└── README.md