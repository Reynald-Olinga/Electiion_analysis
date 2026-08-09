# ==============================================================================
# PAGE INDICATEURS DÉCISIONNELS — LES 9 INDICATEURS (script 08)
# ==============================================================================
#
# 6 onglets :
#   1. Vue d'ensemble     : les 9 courbes superposées (comme le tableau de bord du script)
#   2. Zoom sur un indicateur : sélection + graphique individuel
#   3. Composition du score : pondérations du Score de Risque Global
#   4. Corrélations       : les indicateurs sont-ils redondants entre eux ?
#   5. Validation empirique : le score réagit-il vraiment aux vraies crises ?
#   6. Volume quotidien   : contexte — combien de publications chaque jour
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.chargement_donnees import (
    charger_indicateurs_quotidiens, charger_volume_quotidien, charger_correlation_indicateurs,
    charger_validation_villes_mortes, PONDERATIONS_SCORE_RISQUE, NOMS_AFFICHAGE_INDICATEURS,
)

st.set_page_config(page_title="Indicateurs — Veille post-électorale", layout="wide", page_icon="📐")
st.title("📐 Indicateurs décisionnels")
st.caption("ITE, ITR, Activité, Engagement, Viralité, Polarisation, Mobilisation, Désinformation, Score de Risque Global — script 08")

indicateurs = charger_indicateurs_quotidiens()

COULEURS_INDICATEURS = {
    "ITE": "#6A0572", "ITR": "#9D0208", "Polarisation": "#7209B7", "Mobilisation": "#BC6C25",
    "Desinformation": "#495057", "Viralite": "#F72585", "Activite": "#F77F00", "Engagement": "#2A9D8F",
}

onglet1, onglet2, onglet3, onglet4, onglet5, onglet6 = st.tabs([
    "📊 Vue d'ensemble", "🔎 Zoom par indicateur", "🧮 Composition du score",
    "🔗 Corrélations", "✅ Validation empirique", "📉 Volume quotidien",
])

# ==============================================================================
# ONGLET 1 — VUE D'ENSEMBLE (les 9 courbes superposées)
# ==============================================================================
with onglet1:
    dernier_jour_fiable = indicateurs[indicateurs["jour_fiable"]].index.max()
    col1, col2, col3 = st.columns(3)
    col1.metric("Score de Risque Global moyen", f"{indicateurs['Score_Risque_Global'].mean():.1f}/100")
    col2.metric("Score maximal observé", f"{indicateurs['Score_Risque_Global'].max():.1f}/100")
    col3.metric("Jour du pic", indicateurs["Score_Risque_Global"].idxmax().strftime("%d/%m/%Y"))

    fig_ensemble = go.Figure()
    for colonne, couleur in COULEURS_INDICATEURS.items():
        fig_ensemble.add_trace(go.Scatter(
            x=indicateurs.index, y=indicateurs[colonne], mode="lines",
            name=NOMS_AFFICHAGE_INDICATEURS[colonne], line=dict(color=couleur, width=1), opacity=0.7,
        ))
    fig_ensemble.add_trace(go.Scatter(
        x=indicateurs.index, y=indicateurs["Score_Risque_Global"], mode="lines",
        name="Score de Risque Global", line=dict(color="black", width=3),
    ))
    fig_ensemble.add_vline(x="2025-10-12", line_dash="dash", line_color="red")
    fig_ensemble.update_layout(height=520, yaxis_title="Indice (0-100)", hovermode="x unified")
    st.plotly_chart(fig_ensemble, use_container_width=True)
    st.caption(
        "La ligne noire épaisse est le Score de Risque Global (synthèse pondérée). "
        "La ligne pointillée rouge marque le jour du vote (12/10/2025)."
    )

# ==============================================================================
# ONGLET 2 — ZOOM SUR UN INDICATEUR
# ==============================================================================
with onglet2:
    indicateur_choisi = st.selectbox(
        "Choisir un indicateur à examiner en détail",
        options=list(NOMS_AFFICHAGE_INDICATEURS.keys()),
        format_func=lambda c: NOMS_AFFICHAGE_INDICATEURS[c],
        index=8,  # Score_Risque_Global par défaut
    )

    fig_zoom = go.Figure()
    fig_zoom.add_trace(go.Scatter(
        x=indicateurs.index, y=indicateurs[indicateur_choisi], mode="lines",
        line=dict(color=COULEURS_INDICATEURS.get(indicateur_choisi, "black"), width=2),
        fill="tozeroy", fillcolor="rgba(0,0,0,0.05)",
    ))
    # On grise les jours non fiables (moins de 5 publications ce jour-là)
    jours_non_fiables = indicateurs[~indicateurs["jour_fiable"]]
    for jour in jours_non_fiables.index:
        fig_zoom.add_vrect(x0=jour, x1=jour, fillcolor="gray", opacity=0.08, line_width=0)
    fig_zoom.add_vline(x="2025-10-12", line_dash="dash", line_color="red")
    fig_zoom.update_layout(height=450, title=NOMS_AFFICHAGE_INDICATEURS[indicateur_choisi], yaxis_title="Indice (0-100)")
    st.plotly_chart(fig_zoom, use_container_width=True)
    st.caption("Les zones grisées correspondent aux jours avec moins de 5 publications (indicateur statistiquement fragile).")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moyenne", f"{indicateurs[indicateur_choisi].mean():.1f}")
    col2.metric("Minimum", f"{indicateurs[indicateur_choisi].min():.1f}")
    col3.metric("Maximum", f"{indicateurs[indicateur_choisi].max():.1f}")
    col4.metric("Écart-type", f"{indicateurs[indicateur_choisi].std():.1f}")

# ==============================================================================
# ONGLET 3 — COMPOSITION DU SCORE DE RISQUE GLOBAL
# ==============================================================================
with onglet3:
    st.subheader("Pondération des 8 indicateurs dans le Score de Risque Global")
    st.markdown(
        """
        Le Score de Risque Global n'est pas une simple moyenne : chaque indicateur reçoit un poids
        selon son importance conceptuelle pour anticiper une crise (voir script 08) :
        - **ITE + ITR (40%)** : signaux les plus directement liés à la nature de la crise
        - **Polarisation + Mobilisation (30%)** : dynamiques de mobilisation sociale
        - **Désinformation + Viralité (20%)** : signaux amplificateurs
        - **Activité + Engagement (10%)** : contexte de volume, le moins déterminant seul
        """
    )
    df_ponderations = pd.DataFrame({
        "indicateur": [NOMS_AFFICHAGE_INDICATEURS[c] for c in PONDERATIONS_SCORE_RISQUE.keys()],
        "poids": list(PONDERATIONS_SCORE_RISQUE.values()),
    }).sort_values("poids", ascending=False)

    col_gauche, col_droite = st.columns([1, 1])
    with col_gauche:
        fig_ponderations = px.bar(df_ponderations, x="poids", y="indicateur", orientation="h", text="poids")
        fig_ponderations.update_traces(texttemplate="%{text:.0%}", textposition="outside")
        fig_ponderations.update_layout(height=400, xaxis_title="Poids dans le score", yaxis_title="", xaxis_tickformat=".0%")
        st.plotly_chart(fig_ponderations, use_container_width=True)
    with col_droite:
        fig_camembert = px.pie(df_ponderations, names="indicateur", values="poids")
        fig_camembert.update_layout(height=400)
        st.plotly_chart(fig_camembert, use_container_width=True)

    st.warning(
        "⚠️ Ce choix de pondération est transparent mais reste une proposition méthodologique. "
        "Dans un système réellement déployé, ces poids devraient être calibrés avec des experts "
        "du contexte camerounais et/ou par comparaison avec des crises passées documentées."
    )

# ==============================================================================
# ONGLET 4 — CORRÉLATIONS ENTRE INDICATEURS
# ==============================================================================
with onglet4:
    st.subheader("Les indicateurs sont-ils redondants entre eux ?")
    st.caption(
        "Deux indicateurs très fortement corrélés apporteraient une information redondante. "
        "Un peu de corrélation est normal (ex. ITE et ITR sont conceptuellement proches), "
        "mais ça permet de vérifier qu'on n'a pas construit 8 fois le même indicateur sous des noms différents."
    )
    correlation = charger_correlation_indicateurs()
    correlation.index = [NOMS_AFFICHAGE_INDICATEURS.get(c, c) for c in correlation.index]
    correlation.columns = [NOMS_AFFICHAGE_INDICATEURS.get(c, c) for c in correlation.columns]
    fig_correlation = px.imshow(
        correlation.values, x=correlation.columns, y=correlation.index, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto", text_auto=".2f", labels=dict(color="Corrélation"),
    )
    fig_correlation.update_layout(height=500)
    st.plotly_chart(fig_correlation, use_container_width=True)

# ==============================================================================
# ONGLET 5 — VALIDATION EMPIRIQUE
# ==============================================================================
with onglet5:
    st.subheader("Le Score de Risque Global réagit-il vraiment aux vraies crises ?")
    st.caption(
        "Test de validité de façade : le score doit être nettement plus élevé les jours de forte "
        "mention de 'villes mortes' (un événement de crise bien identifié) que les jours calmes."
    )
    validation = charger_validation_villes_mortes()
    couleurs_validation = ["#9D0208", "#2A9D8F"]
    fig_validation = go.Figure(go.Bar(
        x=validation["categorie"], y=validation["score_moyen"], marker_color=couleurs_validation,
        text=validation["score_moyen"].round(1), textposition="outside",
    ))
    fig_validation.update_layout(height=420, yaxis_title="Score de Risque Global moyen (0-100)")
    st.plotly_chart(fig_validation, use_container_width=True)

    if len(validation) == 2:
        score_crise = validation.loc[validation["categorie"].str.contains("villes_mortes", case=False), "score_moyen"]
        score_calme = validation.loc[validation["categorie"].str.contains("autres", case=False), "score_moyen"]
        if len(score_crise) and len(score_calme):
            if score_crise.values[0] > score_calme.values[0]:
                st.success(
                    f"✅ Résultat cohérent : le score est plus élevé les jours de forte mention "
                    f"'villes mortes' ({score_crise.values[0]:.1f}/100) que les autres jours "
                    f"({score_calme.values[0]:.1f}/100) — le système distingue bien les journées de crise."
                )
            else:
                st.error("⚠️ Le score ne distingue pas bien les journées de crise des journées calmes — à investiguer.")

# ==============================================================================
# ONGLET 6 — VOLUME QUOTIDIEN (CONTEXTE)
# ==============================================================================
with onglet6:
    st.subheader("Volume brut de publications par jour")
    volume = charger_volume_quotidien()
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=volume["jour"], y=volume["volume"], marker_color="#457B9D"))
    fig_volume.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Seuil de fiabilité (5 publications)")
    fig_volume.add_vline(x="2025-10-12", line_dash="dash", line_color="black")
    fig_volume.update_layout(height=420, yaxis_title="Nombre de publications")
    st.plotly_chart(fig_volume, use_container_width=True)
    st.caption(
        "Les jours en dessous de la ligne rouge (moins de 5 publications) sont ceux dont les "
        "indicateurs sont marqués comme statistiquement fragiles ailleurs dans le dashboard."
    )
