from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from wordcloud import WordCloud

from src import cleaning, preprocessing

st.set_page_config(page_title="AMIAI", layout="wide")

# ── Session state defaults ─────────────────────────────────────────
if "extra_rows" not in st.session_state:
    st.session_state.extra_rows = []
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

EXPECTED_COLUMNS = [
    "Carimbo de data/hora",
    "Qual é o seu perfil de estudante?",
    "O que te motivou a estudar Inteligência Artificial? (até 15 linhas)",
]

# ── Header ─────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>AMIAI</h1>"
    "<p style='text-align:center;color:gray;'>Análise de Motivação Acadêmica em Inteligência Artificial</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────
tab_data, tab_config, tab_results = st.tabs([
    "1. Seleção dos dados",
    "2. Configurações",
    "3. Resultados",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — Seleção dos dados
# ════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Carregar arquivo")
    st.caption("Envie um CSV exportado do formulário com as colunas: "
               "*Carimbo de data/hora*, *Qual é o seu perfil de estudante?* e "
               "*O que te motivou a estudar Inteligência Artificial?*")

    uploaded_file = st.file_uploader(
        "Arquivo CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        missing = [c for c in EXPECTED_COLUMNS if c not in df_raw.columns]
        if missing:
            st.error(f"Colunas ausentes no arquivo: {missing}")
            st.stop()

        df_base = df_raw[EXPECTED_COLUMNS].rename(columns={
            "Carimbo de data/hora": "date_time",
            "Qual é o seu perfil de estudante?": "student_profile",
            "O que te motivou a estudar Inteligência Artificial? (até 15 linhas)": "background",
        })

        st.success(f"Arquivo carregado com **{len(df_base)}** registros.")

        st.divider()
        st.subheader("Adicionar nova resposta")
        st.caption("Opcionalmente, inclua respostas extras que serão anexadas aos dados do CSV.")

        col_form, col_preview = st.columns([1, 1], gap="large")

        with col_form:
            new_profile = st.selectbox(
                "Perfil do estudante",
                ["Graduação", "Pós-graduação"],
            )
            new_background = st.text_area(
                "Motivação para estudar IA (até 15 linhas)",
                height=180,
                placeholder="Descreva aqui a sua motivação...",
            )
            if st.button("Adicionar resposta", use_container_width=True):
                if new_background.strip():
                    st.session_state.extra_rows.append({
                        "date_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        "student_profile": new_profile,
                        "background": new_background.strip(),
                    })
                    st.rerun()
                else:
                    st.warning("Preencha o campo de motivação.")

        with col_preview:
            if st.session_state.extra_rows:
                st.markdown(f"**{len(st.session_state.extra_rows)}** resposta(s) adicionada(s)")
                for i, row in enumerate(st.session_state.extra_rows):
                    with st.expander(f"{row['student_profile']} — {row['date_time']}"):
                        st.write(row["background"])
            else:
                st.info("Nenhuma resposta extra adicionada.")

        # Merge extra rows
        if st.session_state.extra_rows:
            df = pd.concat(
                [df_base, pd.DataFrame(st.session_state.extra_rows)],
                ignore_index=True,
            )
        else:
            df = df_base.copy()

        st.divider()
        st.subheader("Visualização dos dados")
        st.dataframe(
            df.rename(columns={
                "date_time": "Data/Hora",
                "student_profile": "Perfil",
                "background": "Motivação",
            }),
            use_container_width=True,
            height=300,
        )

# ════════════════════════════════════════════════════════════════════
# TAB 2 — Configurações
# ════════════════════════════════════════════════════════════════════
with tab_config:
    if uploaded_file is None:
        st.warning("Carregue um arquivo CSV na aba **Seleção dos dados** para continuar.")
        st.stop()

    st.subheader("Parâmetros do pipeline")
    st.caption("Ajuste os hiperparâmetros de redução de dimensionalidade e clusterização antes de executar a análise.")

    col_pca, col_dbscan = st.columns(2, gap="large")

    with col_pca:
        st.markdown("**PCA — Análise de Componentes Principais**")
        n_components = st.number_input(
            "Número de componentes",
            min_value=2,
            max_value=50,
            value=2,
            step=1,
            help="Número de dimensões após a redução. Use 2 para visualização em dispersão.",
        )

    with col_dbscan:
        st.markdown("**DBSCAN — Clusterização baseada em densidade**")
        eps = st.number_input(
            "Epsilon (raio máximo entre vizinhos)",
            min_value=0.01,
            max_value=10.0,
            value=0.4,
            step=0.05,
            format="%.2f",
            help="Distância máxima entre dois pontos para serem considerados vizinhos.",
        )
        min_samples = st.number_input(
            "Mínimo de amostras por cluster",
            min_value=1,
            max_value=50,
            value=2,
            step=1,
            help="Quantidade mínima de pontos para formar um cluster.",
        )

    st.divider()

    if st.button("Executar pipeline", type="primary", use_container_width=True):
        progress = st.progress(0, text="Iniciando...")

        progress.progress(10, text="Limpando textos...")
        df["background"] = cleaning.clean_text(df["background"])
        df["background"] = df["background"].str.lower()

        progress.progress(30, text="Tokenizando e removendo stopwords...")
        df["tokens"] = df["background"].apply(preprocessing.tokenize)
        df["tokens"] = df["tokens"].apply(preprocessing.remove_stopwords)

        progress.progress(40, text="Aplicando stemming...")
        df["stems"] = df["tokens"].apply(preprocessing.stemming)

        progress.progress(50, text="Lematizando...")
        df["lemmas"] = df["tokens"].apply(preprocessing.lemmatization)

        progress.progress(65, text="Vetorizando documentos...")
        matrix_base = preprocessing.vectorization_docs(df["lemmas"])

        progress.progress(80, text="Reduzindo dimensionalidade (PCA)...")
        pca = PCA(n_components=n_components)
        matrix_reduced = pca.fit_transform(matrix_base)

        progress.progress(90, text="Clusterizando (DBSCAN)...")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = model.fit_predict(matrix_reduced)
        df["cluster"] = clusters
        df["lemmas"] = df["lemmas"].apply(lambda x: np.array(x))

        progress.progress(100, text="Concluído!")

        st.session_state.pipeline_results = {
            "df": df.copy(),
            "matrix_reduced": matrix_reduced,
            "clusters": clusters,
            "n_components": n_components,
            "explained_variance": pca.explained_variance_ratio_,
        }
        st.rerun()

    if st.session_state.pipeline_results is not None:
        st.success("Pipeline executado. Veja os resultados na aba **Resultados**.")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — Resultados
# ════════════════════════════════════════════════════════════════════
with tab_results:
    if st.session_state.pipeline_results is None:
        st.warning("Execute o pipeline na aba **Configurações** para ver os resultados.")
        st.stop()

    res = st.session_state.pipeline_results
    df_res = res["df"]
    matrix_reduced = res["matrix_reduced"]
    clusters = res["clusters"]
    n_components = res["n_components"]
    explained_variance = res["explained_variance"]

    unique_clusters = sorted(df_res["cluster"].unique())
    named_clusters = [c for c in unique_clusters if c != -1]
    n_noise = int((clusters == -1).sum())

    # ── KPI metrics row ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total de respostas", len(df_res))
    m2.metric("Clusters encontrados", len(named_clusters))
    m3.metric("Respostas sem cluster (ruído)", n_noise)
    m4.metric("Variância explicada (PCA)", f"{explained_variance.sum():.1%}")

    st.divider()

    # ── Scatter + Profile distribution side by side ────────────────
    if n_components == 2:
        col_scatter, col_profile = st.columns([3, 2], gap="large")

        with col_scatter:
            st.markdown("##### Dispersão dos clusters")
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
            scatter = ax_scatter.scatter(
                matrix_reduced[:, 0],
                matrix_reduced[:, 1],
                c=clusters,
                cmap="tab10",
                alpha=0.75,
                edgecolors="w",
                linewidths=0.4,
                s=60,
            )
            ax_scatter.set_xlabel("Componente 1")
            ax_scatter.set_ylabel("Componente 2")
            plt.colorbar(scatter, ax=ax_scatter, label="Cluster")
            fig_scatter.tight_layout()
            st.pyplot(fig_scatter)

        with col_profile:
            st.markdown("##### Perfil dos estudantes por cluster")
            profile_data = (
                df_res.groupby(["cluster", "student_profile"])
                .size()
                .unstack(fill_value=0)
            )
            fig_profile, ax_profile = plt.subplots(figsize=(6, 5))
            profile_data.plot.barh(
                stacked=True, ax=ax_profile, color=["#4e79a7", "#e15759"]
            )
            ax_profile.set_xlabel("Quantidade")
            ax_profile.set_ylabel("Cluster")
            ax_profile.legend(title="Perfil", loc="lower right")
            fig_profile.tight_layout()
            st.pyplot(fig_profile)

    # ── Word counts per cluster ────────────────────────────────────
    words_cluster = {}
    for cid in unique_clusters:
        lemmas = np.concatenate(df_res[df_res["cluster"] == cid]["lemmas"].to_numpy())
        wc_dict = {}
        for lemma in lemmas:
            wc_dict[lemma] = wc_dict.get(lemma, 0) + 1
        words_cluster[cid] = wc_dict

    st.divider()
    st.markdown("##### Palavras mais frequentes e nuvens de palavras")

    cluster_tabs = st.tabs(
        [f"Cluster {c}" if c != -1 else "Sem cluster (ruído)" for c in unique_clusters]
    )

    for tab, cid in zip(cluster_tabs, unique_clusters):
        with tab:
            col_bar, col_wc = st.columns(2, gap="large")

            sorted_words = sorted(
                words_cluster[cid].items(), key=lambda x: x[1], reverse=True
            )[:10]
            words, counts = zip(*sorted_words)

            with col_bar:
                fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
                ax_bar.barh(words[::-1], counts[::-1], color="#4e79a7")
                ax_bar.set_xlabel("Frequência")
                ax_bar.set_title(f"Top 10 palavras")
                fig_bar.tight_layout()
                st.pyplot(fig_bar)

            with col_wc:
                lemmas_arr = np.concatenate(
                    df_res[df_res["cluster"] == cid]["lemmas"].to_numpy()
                )
                text = " ".join(lemmas_arr)
                wc_img = WordCloud(
                    background_color="white",
                    width=960,
                    height=540,
                    max_words=80,
                    colormap="viridis",
                ).generate(text)
                fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                ax_wc.imshow(wc_img, interpolation="bilinear")
                ax_wc.axis("off")
                fig_wc.tight_layout()
                st.pyplot(fig_wc)

            # Show responses in this cluster
            df_cluster = df_res[df_res["cluster"] == cid]
            with st.expander(f"Ver {len(df_cluster)} resposta(s) deste grupo"):
                st.dataframe(
                    df_cluster[["date_time", "student_profile", "background"]].rename(
                        columns={
                            "date_time": "Data/Hora",
                            "student_profile": "Perfil",
                            "background": "Motivação",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # ── Full table ─────────────────────────────────────────────────
    st.divider()
    st.markdown("##### Tabela completa com clusters atribuídos")
    st.dataframe(
        df_res[["date_time", "student_profile", "background", "cluster"]].rename(
            columns={
                "date_time": "Data/Hora",
                "student_profile": "Perfil",
                "background": "Motivação",
                "cluster": "Cluster",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
