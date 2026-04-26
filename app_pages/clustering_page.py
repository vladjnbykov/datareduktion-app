from __future__ import annotations

import streamlit as st

from components.controls import render_clustering_controls
from components.dataset_view import (
    render_data_preview,
    render_dataset_summary,
    render_feature_list,
)
from components.intro import render_note_box, render_page_title
from components.plots import (
    plot_cluster_sizes,
    plot_elbow_curve,
    plot_kmeans_pca_scatter,
    plot_silhouette_scores,
    plot_umap_2d,
)
from data.loaders import load_penguins_data, load_wine_data
from methods.clustering_analysis import (
    compute_elbow_and_silhouette,
    run_kmeans,
    run_pca_for_kmeans_visualization,
)
from utils.validators import validate_min_selected_features


def render_clustering_page() -> None:
    render_page_title(
        "Klustring med K-means",
        "Utforska antal kluster, elbow method, silhouette score och PCA-visualisering.",
    )

    st.info(
        """
K-means försöker dela observationer i grupper baserat på likhet.
Här använder vi Penguins-datasetet för att se hur kluster kan uppstå i numerisk data.
        """
    )

    st.info(
    """
K-means får inte veta pingvinarten. Den hittar grupper endast från numeriska mått.
Det innebär att klustren inte nödvändigtvis motsvarar verkliga arter.
"""
    )

    dataset_choice = st.selectbox(
        "Välj dataset för klustring",
        options=["Penguins", "Wine"],
    )

    previous_dataset = st.session_state.get("active_clustering_dataset")

    if previous_dataset is not None and previous_dataset != dataset_choice:
        for key in [
            "kmeans_results",
            "kmeans_elbow_df",
            "kmeans_show_pca",
            "kmeans_selected_features",
            "kmeans_standardize",
            "kmeans_df",
            "kmeans_target_column",
            "kmeans_dataset_choice",
        ]:
            st.session_state.pop(key, None)

    st.session_state["active_clustering_dataset"] = dataset_choice

    if dataset_choice == "Wine":
        st.caption(
            "Wine har fler numeriska variabler än Penguins. Därför kan PCA före K-means påverka klustringen tydligare."
        )
    else:
        st.caption(
            "Penguins har få numeriska variabler, så K-means på originaldata och PCA-komponenter kan ibland bli ganska lika."
        )

    if dataset_choice == "Penguins":
        payload = load_penguins_data()
        df = payload["df"]
        feature_columns = payload["feature_columns"]
        target_column = payload["target_column"]
    else:
        payload = load_wine_data()
        df = payload["df"]
        feature_columns = payload["feature_columns"]
        target_column = payload["target_name_column"]


    render_dataset_summary(df)
    render_feature_list(feature_columns)
    render_data_preview(df[feature_columns + [target_column]])

    left_col, right_col = st.columns([1, 2])

    with left_col:
        controls = render_clustering_controls(feature_columns)

        clustering_space = controls["clustering_space"]
        pca_n_components = controls["pca_n_components"]
        umap_n_neighbors = controls["umap_n_neighbors"]
        umap_min_dist = controls["umap_min_dist"]
        umap_metric = controls["umap_metric"]

        if clustering_space == "Originalvariabler":
            embedding_method = "original"
        elif clustering_space == "PCA-komponenter":
            embedding_method = "pca"
        else:
            embedding_method = "umap"

        if embedding_method == "original":
            st.caption(
                "K-means körs direkt på de valda variablerna. PCA används endast för visualisering."
            )
        elif embedding_method == "pca":
            st.caption(
                "K-means körs efter att datan först reducerats med PCA."
            )
        else:
            st.caption(
                "K-means körs efter att datan först reducerats med UMAP. Detta är explorativt och ska tolkas försiktigt."
            )

        st.caption("K-means körs först när du klickar på knappen.")
        run_clicked = st.button("Kör K-means", type="primary")

    selected_features = controls["selected_features"]
    standardize = controls["standardize"]
    n_clusters = controls["n_clusters"]
    random_state = controls["random_state"]
    show_pca_visualization = controls["show_pca_visualization"]
    clustering_space = controls["clustering_space"]
    pca_n_components = controls["pca_n_components"]

    try:
        validate_min_selected_features(selected_features, min_required=2)
    except ValueError as exc:
        with right_col:
            render_note_box(str(exc), kind="error")
        return

    if run_clicked:
        kmeans_results = run_kmeans(
            df=df,
            feature_cols=selected_features,
            n_clusters=n_clusters,
            standardize=standardize,
            random_state=random_state,
            embedding_method=embedding_method,
            pca_n_components=pca_n_components,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
        )

        elbow_df = compute_elbow_and_silhouette(
            df=df,
            feature_cols=selected_features,
            k_values=list(range(2, 11)),
            standardize=standardize,
            random_state=random_state,
            embedding_method=embedding_method,
            pca_n_components=pca_n_components,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
        )
        st.session_state["kmeans_dataset_choice"] = dataset_choice
        st.session_state["kmeans_df"] = df
        st.session_state["kmeans_target_column"] = target_column
        st.session_state["kmeans_results"] = kmeans_results
        st.session_state["kmeans_elbow_df"] = elbow_df
        st.session_state["kmeans_show_pca"] = show_pca_visualization
        st.session_state["kmeans_selected_features"] = selected_features
        st.session_state["kmeans_standardize"] = standardize

    kmeans_results = st.session_state.get("kmeans_results")
    elbow_df = st.session_state.get("kmeans_elbow_df")

    if kmeans_results is None or elbow_df is None:
        with right_col:
            render_note_box(
                "Välj features och klicka på **Kör K-means** för att skapa klusteranalysen.",
                kind="info",
            )
        return

    with right_col:
        st.markdown("## Resultat")

        sil = kmeans_results["silhouette_score"]
        if sil is not None:
            st.metric("Silhouette score för valt k", f"{sil:.3f}")

        st.metric("Inertia för valt k", f"{kmeans_results['inertia']:.2f}")

        plot_cluster_sizes(kmeans_results["result_df"])

        st.markdown("## Välj antal kluster")
        plot_elbow_curve(elbow_df)
        plot_silhouette_scores(elbow_df)

        if st.session_state.get("kmeans_show_pca", True):
            if kmeans_results.get("used_umap") and kmeans_results.get("umap_scores_df") is not None:
                umap_scores_df = kmeans_results["umap_scores_df"]
                
                plot_umap_2d(umap_scores_df, target_col="cluster")

            elif kmeans_results.get("used_pca") and kmeans_results.get("pca_scores_df") is not None:
                pca_scores_df = kmeans_results["pca_scores_df"]
                plot_kmeans_pca_scatter(pca_scores_df)

            else:
                pca_scores_df = run_pca_for_kmeans_visualization(
                    df=st.session_state["kmeans_df"],
                    feature_cols=st.session_state["kmeans_selected_features"],
                    cluster_labels=kmeans_results["labels"],
                    standardize=st.session_state["kmeans_standardize"],
                )
                plot_kmeans_pca_scatter(pca_scores_df)
    st.markdown("## Tolkning")
    
    if kmeans_results["used_umap"]:
        st.info(
            """
    - **Elbow method** används för att se när inertia slutar minska kraftigt.
    - **Silhouette score** visar hur väl observationer passar i sina kluster.
    - Här kördes **K-means på UMAP-komponenter**, alltså efter icke-linjär dimensionsreduktion.
    - Detta kan ge tydliga visuella grupper, men bör tolkas försiktigt eftersom UMAP främst är ett visualiseringsverktyg.
            """
        ) 
    
    elif kmeans_results["used_pca"]:
        st.info(
            """
    - **Elbow method** används för att se när inertia slutar minska kraftigt.
    - **Silhouette score** visar hur väl observationer passar i sina kluster.
    - Här kördes **K-means på PCA-komponenter**, alltså efter dimensionsreduktion.
    - PCA-plotten visar samma typ av reducerat rum som användes före klustringen.
            """
        )

    else:
        st.info(
            """
    - **Elbow method** används för att se när inertia slutar minska kraftigt.
    - **Silhouette score** visar hur väl observationer passar i sina kluster.
    - Här kördes **K-means på originalvariablerna**.
    - PCA-plotten används endast för att visualisera klustren i två dimensioner.
            """
        )

    st.markdown("## Reflektionsfrågor")
    st.markdown(
        """
1. Vilket k verkar rimligt enligt elbow plot?
2. Vilket k ger högst silhouette score?
3. Ser klustren tydliga ut i visualiseringen?
4. Stämmer klustren ungefär med pingvinarter, eller hittar K-means något annat?
        """
    )