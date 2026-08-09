# ==============================================================================
# PAGE THÉMATIQUES — TOPIC MODELING LDA (script 06)
# ==============================================================================
#
# 6 onglets :
#   1. Vue d'ensemble (FR)   : répartition des 5 thèmes + top mots-clés
#   2. Évolution & comparaisons : hebdo, avant/après vote, par plateforme, par acteur
#   3. Sentiment & émotions par thème (LDA)
#   4. Qualité du modèle     : similarité entre thèmes, diversité (entropie),
#                              validation croisée avec les catégories de crise
#   5. Anglais / Pidgin      : équivalent des thèmes français, en plus succinct
#   6. Verbatims représentatifs
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.chargement_donnees import (
    charger_repartition_themes_fr, charger_mots_cles_themes_fr, charger_evolution_themes_hebdo,
    charger_themes_avant_apres, charger_themes_par_plateforme, charger_themes_par_acteur,
    charger_similarite_themes, charger_sentiment_par_theme_lda, charger_profil_emotionnel_themes_lda,
    charger_engagement_par_theme_lda, charger_entropie_plateforme, charger_entropie_periode,
    charger_validation_croisee, charger_mots_cles_themes_en, charger_repartition_themes_en,
    charger_engagement_par_theme_en, charger_publications_representatives_themes,
)

st.set_page_config(page_title="Thématiques — Veille post-électorale", layout="wide", page_icon="🗂️")
st.title("🗂️ Extraction des thématiques")
st.caption("Topic modeling LDA, français et anglais/pidgin traités séparément — script 06")

COULEURS_PLATEFORME = {"Facebook": "#1877F2", "TikTok": "#EE1D52", "X": "#6c757d"}

onglet1, onglet2, onglet3, onglet4, onglet5, onglet6 = st.tabs([
    "📊 Vue d'ensemble (FR)", "📈 Évolution & comparaisons", "😊 Sentiment & émotions",
    "🔍 Qualité du modèle", "🌍 Anglais / Pidgin", "📝 Verbatims",
])

# ==============================================================================
# ONGLET 1 — VUE D'ENSEMBLE (FRANÇAIS)
# ==============================================================================
with onglet1:
    repartition = charger_repartition_themes_fr()

    st.subheader("Répartition des publications par thème")
    fig_repartition = px.bar(
        repartition.sort_values("count"), x="count", y="theme_nom", orientation="h",
        color="theme_nom", color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_repartition.update_layout(height=350, showlegend=False, xaxis_title="Nombre de publications", yaxis_title="")
    st.plotly_chart(fig_repartition, use_container_width=True)

    st.subheader("Top mots-clés de chaque thème")
    st.caption("Les mots les plus représentatifs de chaque thème selon le modèle LDA (par ordre décroissant de poids).")
    mots_cles = charger_mots_cles_themes_fr()
    colonnes_theme = st.columns(len(mots_cles.columns))
    for i, nom_theme in enumerate(mots_cles.columns):
        with colonnes_theme[i]:
            st.markdown(f"**{nom_theme}**")
            for mot in mots_cles[nom_theme].dropna():
                st.markdown(f"- {mot}")

# ==============================================================================
# ONGLET 2 — ÉVOLUTION & COMPARAISONS
# ==============================================================================
with onglet2:
    st.subheader("Évolution hebdomadaire (aire empilée)")
    evolution = charger_evolution_themes_hebdo()
    colonnes_themes = [c for c in evolution.columns if c != "semaine"]
    fig_evolution = go.Figure()
    for theme in colonnes_themes:
        fig_evolution.add_trace(go.Scatter(
            x=evolution["semaine"], y=evolution[theme], mode="lines", stackgroup="un", name=theme,
        ))
    fig_evolution.add_vline(x="2025-10-12", line_dash="dash", line_color="black")
    fig_evolution.update_layout(height=420, yaxis_title="Part des publications (%)")
    st.plotly_chart(fig_evolution, use_container_width=True)

    col_gauche, col_droite = st.columns(2)

    with col_gauche:
        st.subheader("Poids de chaque thème, avant/après le vote")
        avant_apres = charger_themes_avant_apres()
        fig_avant_apres = go.Figure()
        fig_avant_apres.add_trace(go.Bar(y=avant_apres["theme_nom"], x=avant_apres["avant le vote"], name="Avant le vote", orientation="h", marker_color="#4C9A2A"))
        fig_avant_apres.add_trace(go.Bar(y=avant_apres["theme_nom"], x=avant_apres["après le vote"], name="Après le vote", orientation="h", marker_color="#B22222"))
        fig_avant_apres.update_layout(barmode="group", height=400, xaxis_title="% des publications de la période")
        st.plotly_chart(fig_avant_apres, use_container_width=True)

    with col_droite:
        st.subheader("Répartition des thèmes par plateforme (%)")
        par_plateforme = charger_themes_par_plateforme()
        fig_plateforme = px.imshow(
            par_plateforme.values, x=par_plateforme.columns, y=par_plateforme.index,
            color_continuous_scale="YlGnBu", aspect="auto", labels=dict(color="%"),
        )
        fig_plateforme.update_layout(height=400)
        st.plotly_chart(fig_plateforme, use_container_width=True)

    st.subheader("Répartition thématique des publications mentionnant chaque acteur (%)")
    par_acteur = charger_themes_par_acteur()
    fig_acteur = px.imshow(
        par_acteur.values, x=par_acteur.columns, y=par_acteur.index,
        color_continuous_scale="PuRd", aspect="auto", labels=dict(color="%"), text_auto=".0f",
    )
    fig_acteur.update_layout(height=380)
    st.plotly_chart(fig_acteur, use_container_width=True)

# ==============================================================================
# ONGLET 3 — SENTIMENT & ÉMOTIONS PAR THÈME (LDA)
# ==============================================================================
with onglet3:
    col_gauche, col_droite = st.columns(2)

    with col_gauche:
        st.subheader("Sentiment moyen par thème")
        sentiment_theme = charger_sentiment_par_theme_lda().sort_values("mean")
        couleurs = ["#E63946" if v < 0 else "#2A9D8F" for v in sentiment_theme["mean"]]
        fig_sentiment = go.Figure(go.Bar(x=sentiment_theme["mean"], y=sentiment_theme["theme_nom"], orientation="h", marker_color=couleurs))
        fig_sentiment.add_vline(x=0, line_color="black")
        fig_sentiment.update_layout(height=400, xaxis_title="Score de sentiment moyen")
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col_droite:
        st.subheader("Engagement moyen par thème")
        engagement_theme = charger_engagement_par_theme_lda().sort_values("mean")
        fig_engagement = px.bar(engagement_theme, x="mean", y="theme_nom", orientation="h", color="theme_nom",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
        fig_engagement.update_layout(height=400, showlegend=False, xaxis_title="Nombre moyen de likes", yaxis_title="")
        st.plotly_chart(fig_engagement, use_container_width=True)

    st.subheader("Profil des 8 émotions pour chaque thème")
    profil_emotionnel = charger_profil_emotionnel_themes_lda()
    fig_profil = px.imshow(
        profil_emotionnel.values, x=profil_emotionnel.columns, y=profil_emotionnel.index,
        color_continuous_scale="magma", aspect="auto", labels=dict(color="Intensité"),
    )
    fig_profil.update_layout(height=380)
    st.plotly_chart(fig_profil, use_container_width=True)

# ==============================================================================
# ONGLET 4 — QUALITÉ DU MODÈLE
# ==============================================================================
with onglet4:
    st.subheader("Similarité entre les 5 thèmes (cosinus)")
    st.caption("Deux thèmes trop similaires (proches de 1) feraient un peu doublon — ici, on vérifie qu'ils sont bien distincts.")
    similarite = charger_similarite_themes()
    fig_similarite = px.imshow(
        similarite.values, x=similarite.columns, y=similarite.index, color_continuous_scale="RdBu_r",
        zmin=0, zmax=1, aspect="auto", text_auto=".2f", labels=dict(color="Similarité"),
    )
    fig_similarite.update_layout(height=420)
    st.plotly_chart(fig_similarite, use_container_width=True)

    col_gauche, col_droite = st.columns(2)
    with col_gauche:
        st.subheader("Diversité thématique par plateforme")
        st.caption("Entropie de Shannon : plus c'est élevé, plus les sujets abordés sont variés et équilibrés.")
        entropie_plateforme = charger_entropie_plateforme()
        fig_entropie_p = px.bar(entropie_plateforme, x="plateforme", y="entropie", color="plateforme",
                                 color_discrete_map=COULEURS_PLATEFORME)
        fig_entropie_p.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_entropie_p, use_container_width=True)

    with col_droite:
        st.subheader("Diversité thématique par période")
        entropie_periode = charger_entropie_periode()
        fig_entropie_per = px.bar(entropie_periode, x="periode", y="entropie", color="periode")
        fig_entropie_per.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_entropie_per, use_container_width=True)

    st.subheader("Validation croisée : catégories de crise (a priori) vs thèmes LDA (automatiques)")
    st.caption("Une catégorie de crise très concentrée sur un seul thème LDA = les deux méthodes se confirment mutuellement.")
    validation = charger_validation_croisee()
    fig_validation = px.imshow(
        validation.values, x=validation.columns, y=validation.index, color_continuous_scale="OrRd",
        aspect="auto", text_auto=".0f", labels=dict(color="% des publications"),
    )
    fig_validation.update_layout(height=420)
    st.plotly_chart(fig_validation, use_container_width=True)

# ==============================================================================
# ONGLET 5 — ANGLAIS / PIDGIN
# ==============================================================================
with onglet5:
    st.info(
        "ℹ️ Le nombre de thèmes anglais/pidgin (k) est choisi automatiquement par "
        "le script selon un score de cohérence, et peut varier d'une exécution à "
        "l'autre. Contrairement au français, les libellés ci-dessous restent donc "
        "génériques (\"Thème 0\", \"Thème 1\"...) plutôt que des noms interprétés, "
        "pour ne pas afficher un nom qui ne correspondrait plus au thème réel."
    )

    col_gauche, col_droite = st.columns([2, 1])

    with col_gauche:
        st.subheader("Top mots-clés par thème")
        mots_cles_en = charger_mots_cles_themes_en()
        colonnes_theme_en = st.columns(len(mots_cles_en.columns))
        for i, nom_theme in enumerate(mots_cles_en.columns):
            with colonnes_theme_en[i]:
                st.markdown(f"**{nom_theme}**")
                for mot in mots_cles_en[nom_theme].dropna():
                    st.markdown(f"- {mot}")

    with col_droite:
        st.subheader("Répartition")
        repartition_en = charger_repartition_themes_en()
        fig_repartition_en = px.pie(repartition_en, names="theme_nom", values="count")
        fig_repartition_en.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_repartition_en, use_container_width=True)

    st.subheader("Engagement moyen par thème (anglais/pidgin)")
    engagement_en = charger_engagement_par_theme_en().sort_values("mean")
    fig_engagement_en = px.bar(engagement_en, x="mean", y="theme_nom", orientation="h")
    fig_engagement_en.update_layout(height=350, xaxis_title="Nombre moyen de likes", yaxis_title="")
    st.plotly_chart(fig_engagement_en, use_container_width=True)

# ==============================================================================
# ONGLET 6 — VERBATIMS REPRÉSENTATIFS
# ==============================================================================
with onglet6:
    st.subheader("Publications les plus représentatives de chaque thème")
    st.caption("Les 3 publications avec la plus forte probabilité d'appartenance à chaque thème (verbatims).")
    verbatims = charger_publications_representatives_themes()
    for theme in verbatims["theme"].unique():
        with st.expander(f"**{theme}**"):
            sous_ensemble = verbatims[verbatims["theme"] == theme]
            for _, ligne in sous_ensemble.iterrows():
                st.markdown(f"— *{ligne['plateforme']}*, probabilité `{ligne['probabilite']:.2f}`, 👍 {int(ligne['nb_likes'])}")
                st.write(ligne["texte"])
                st.divider()
