from __future__ import annotations

import streamlit as st

from components.controls import render_hierarchical_controls
from components.dataset_view import (
    render_data_preview,
    render_dataset_summary,
    render_feature_list,
)
from components.intro import render_note_box, render_page_title
from components.plots import (
    plot_cluster_sizes,
    plot_dendrogram,
    plot_kmeans_pca_scatter,
)
from data.loaders import load_penguins_data, load_wine_data
from methods.clustering_analysis import (
    run_hierarchical_clustering,
    run_pca_for_kmeans_visualization,
)
from utils.validators import validate_min_selected_features


def render_hierarchical_page() -> None:
    render_page_title(
        "Hierarkisk klustring",
        "Utforska dendrogram, linkage methods, antal kluster och PCA-visualisering.",
    )

    st.info(
        """
Hierarkisk klustring bygger ett träd av observationer. 
Vi behöver inte välja antal kluster från början, men vi kan “klippa” trädet 
på olika nivåer för att få olika antal grupper.
        """
    )

    st.info(
        """
Dendrogrammet visar hur observationer eller grupper slås ihop steg för steg.
Långa vertikala avstånd kan antyda tydligare separation mellan grupper.
        """
    )

    dataset_choice = st.selectbox(
        "Välj dataset för hierarkisk klustring",
        options=["Penguins", "Wine"],
    )

    previous_dataset = st.session_state.get("active_hierarchical_dataset")

    if previous_dataset is not None and previous_dataset != dataset_choice:
        for key in [
            "hierarchical_results",
            "hierarchical_selected_features",
            "hierarchical_standardize",
            "hierarchical_df",
            "hierarchical_target_column",
            "hierarchical_dataset_choice",
        ]:
            st.session_state.pop(key, None)

    st.session_state["active_hierarchical_dataset"] = dataset_choice

    if dataset_choice == "Penguins":
        payload = load_penguins_data()
        df = payload["df"]
        feature_columns = payload["feature_columns"]
        target_column = payload["target_column"]

        st.caption(
            "Penguins har få numeriska variabler och är bra för en första intuition kring klustring."
        )

    else:
        payload = load_wine_data()
        df = payload["df"]
        feature_columns = payload["feature_columns"]
        target_column = payload["target_name_column"]

        st.caption(
            "Wine har fler numeriska variabler, vilket ofta gör klustringsstrukturen mer intressant att jämföra."
        )

    render_dataset_summary(df)
    render_feature_list(feature_columns)
    render_data_preview(df[feature_columns + [target_column]])

    left_col, right_col = st.columns([1, 2])

    with left_col:
        controls = render_hierarchical_controls(feature_columns)

        st.info(
            "Till skillnad från K-means behöver vi inte välja antal kluster i början. "
            "Vi bygger först ett dendrogram och väljer sedan hur många grupper vi vill dela upp datan i."
        )

        st.caption(
            "Hierarkisk klustring körs först när du klickar på knappen."
        )
        st.caption(
            "Dendrogrammet byggs först. Slidern anger hur många grupper vi vill klippa ut från trädet."
        )
        run_clicked = st.button("Kör hierarkisk klustring", type="primary")

    selected_features = controls["selected_features"]
    standardize = controls["standardize"]
    n_clusters = controls["n_clusters"]
    linkage_method = controls["linkage_method"]

    try:
        validate_min_selected_features(selected_features, min_required=2)
    except ValueError as exc:
        with right_col:
            render_note_box(str(exc), kind="error")
        return

    if run_clicked:
        results = run_hierarchical_clustering(
            df=df,
            feature_cols=selected_features,
            n_clusters=n_clusters,
            standardize=standardize,
            linkage_method=linkage_method,
        )

        st.session_state["hierarchical_results"] = results
        st.session_state["hierarchical_selected_features"] = selected_features
        st.session_state["hierarchical_standardize"] = standardize
        st.session_state["hierarchical_df"] = df
        st.session_state["hierarchical_target_column"] = target_column
        st.session_state["hierarchical_dataset_choice"] = dataset_choice

    results = st.session_state.get("hierarchical_results")

    if results is None:
        with right_col:
            render_note_box(
                "Välj features och klicka på **Kör hierarkisk klustring** för att skapa analysen.",
                kind="info",
            )
        return

    with right_col:
        st.markdown("## Resultat")

        sil = results["silhouette_score"]
        if sil is not None:
            st.metric("Silhouette score", f"{sil:.3f}")

        st.metric("Antal grupper efter klippning", results["n_clusters"])
        st.metric("Linkage method", results["linkage_method"])

        plot_cluster_sizes(results["result_df"])

        st.markdown("## Dendrogram")
        plot_dendrogram(
            results["linkage_matrix"],
            cut_height=results.get("cut_height"),
        )


        st.caption(
            "Dendrogrammet visar trädet. Tabellen nedan visar vilka observationer som hamnade i varje kluster efter klippningen."
        )

        st.markdown("### Observationer per kluster")

        available_clusters = sorted(results["result_df"]["cluster"].unique())

        selected_cluster = st.selectbox(
            "Välj kluster att inspektera",
            options=available_clusters,
            key="hierarchical_selected_cluster",
        )

        cluster_df = results["result_df"][
            results["result_df"]["cluster"] == selected_cluster
        ]

        st.write(
            f"Kluster **{selected_cluster}** innehåller **{len(cluster_df)}** observationer."
        )

        st.dataframe(
            cluster_df,
            width="stretch",
        )

        st.markdown("### Sammanfattning per kluster, means")

        summary_df = results["result_df"].groupby("cluster")[
            st.session_state["hierarchical_selected_features"]
        ].mean()

        st.dataframe(summary_df.round(2), width="stretch")





        st.markdown("## Kluster visualiserade med PCA")
        pca_scores_df = run_pca_for_kmeans_visualization(
            df=st.session_state["hierarchical_df"],
            feature_cols=st.session_state["hierarchical_selected_features"],
            cluster_labels=results["labels"],
            standardize=st.session_state["hierarchical_standardize"],
        )
        plot_kmeans_pca_scatter(pca_scores_df)

    st.markdown("## Tolkning")
    st.info(
        """
- Dendrogrammet visar hur observationer eller grupper slås ihop steg för steg.
- **Linkage method** påverkar hur avstånd mellan grupper beräknas.
- **Ward** försöker minimera variation inom kluster och fungerar bäst med euklidiskt avstånd.
- **Complete** tenderar att skapa kompaktare kluster.
- **Average** använder genomsnittliga avstånd mellan grupper.
- **Single** kan skapa kedjeeffekter där kluster blir utdragna.
- PCA-plotten är en 2D-visualisering av klustren, men själva klustringen sker med den hierarkiska metoden.
        """
    )

    st.markdown("## Reflektionsfrågor")
    st.markdown(
        """
1. Var verkar det vara rimligt att “klippa” dendrogrammet?
2. Hur ändras klustren när du byter linkage method?
3. Verkar silhouette score stödja valt antal kluster?
4. Ser klustren tydliga ut i PCA-visualiseringen?
5. Liknar klustren de kända grupperna i datasetet, eller hittar metoden något annat?
        """
    )