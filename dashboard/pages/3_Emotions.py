# ==============================================================================
# PAGE ÉMOTIONS — LES 8 ÉMOTIONS DE PLUTCHIK (script 05)
# ==============================================================================
#
# 4 onglets :
#   1. Vue d'ensemble    : distribution des émotions dominantes + radar global
#   2. Évolution         : radar avant/après le vote, courbes hebdo, heatmap temps
#   3. Profils comparés  : par thème, par acteur, par plateforme
#   4. Engagement        : quelle émotion suscite le plus de réactions ?
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.chargement_donnees import (
    charger_repartition_emotions, charger_scores_emotionnels_globaux, charger_emotions_avant_apres,
    charger_evolution_hebdo_emotions, charger_heatmap_emotions_temps, charger_profils_emotionnels_themes,
    charger_profils_emotionnels_acteurs, charger_profils_emotionnels_plateformes, charger_engagement_par_emotion,
    EMOTIONS, COULEURS_EMOTIONS,
)

st.set_page_config(page_title="Émotions — Veille post-électorale", layout="wide", page_icon="🎭")
st.title("🎭 Analyse des émotions")
st.caption("Modèle des 8 émotions de Plutchik — lexique français maison + NRCLex (anglais/pidgin), script 05")

COULEURS_PLATEFORME = {"Facebook": "#1877F2", "TikTok": "#EE1D52", "X": "#6c757d"}


def tracer_radar(dict_series, titre, hauteur=420):
    """Petite fonction utilitaire : trace un radar à partir d'un ou plusieurs
    dictionnaires {emotion: valeur}. dict_series est un dict {nom_serie: {emotion: valeur}}."""
    fig = go.Figure()
    for nom_serie, valeurs_dict in dict_series.items():
        valeurs = [valeurs_dict[e] for e in EMOTIONS]
        valeurs += valeurs[:1]
        fig.add_trace(go.Scatterpolar(r=valeurs, theta=EMOTIONS + [EMOTIONS[0]], fill="toself", name=nom_serie))
    fig.update_layout(height=hauteur, title=titre, polar=dict(radialaxis=dict(visible=True)))
    return fig


onglet1, onglet2, onglet3, onglet4 = st.tabs([
    "📊 Vue d'ensemble", "📈 Évolution", "⚖️ Profils comparés", "❤️ Engagement"
])

# ==============================================================================
# ONGLET 1 — VUE D'ENSEMBLE
# ==============================================================================
with onglet1:
    repartition = charger_repartition_emotions()
    scores_globaux = charger_scores_emotionnels_globaux()

    col_gauche, col_droite = st.columns(2)

    with col_gauche:
        st.subheader("Distribution des émotions dominantes")
        repartition_visible = repartition[repartition["emotion_dominante"] != "aucune détectée"]
        repartition_visible = repartition_visible.set_index("emotion_dominante").reindex(EMOTIONS).dropna().reset_index()
        fig_bar = px.bar(
            repartition_visible, x="emotion_dominante", y="count", color="emotion_dominante",
            color_discrete_map=COULEURS_EMOTIONS,
        )
        fig_bar.update_layout(height=400, showlegend=False, xaxis_title="", yaxis_title="Nombre de publications")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_droite:
        st.subheader("Radar émotionnel global")
        dict_scores = dict(zip(scores_globaux["emotion"], scores_globaux["score"]))
        st.plotly_chart(tracer_radar({"Ensemble du corpus": dict_scores}, ""), use_container_width=True)

# ==============================================================================
# ONGLET 2 — ÉVOLUTION TEMPORELLE
# ==============================================================================
with onglet2:
    col_gauche, col_droite = st.columns(2)

    with col_gauche:
        st.subheader("Radar : avant vs après le vote")
        avant_apres = charger_emotions_avant_apres()
        dict_radar = {
            "Avant le vote": avant_apres["avant le vote"].to_dict(),
            "Après le vote": avant_apres["après le vote"].to_dict(),
        }
        st.plotly_chart(tracer_radar(dict_radar, ""), use_container_width=True)

    with col_droite:
        st.subheader("Heatmap émotionnelle dans le temps")
        heatmap_temps = charger_heatmap_emotions_temps()
        fig_heatmap = px.imshow(
            heatmap_temps.values, x=heatmap_temps.columns, y=heatmap_temps.index,
            color_continuous_scale="inferno", aspect="auto", labels=dict(color="Intensité"),
        )
        fig_heatmap.update_layout(height=420)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Évolution hebdomadaire de chaque émotion")
    evolution = charger_evolution_hebdo_emotions()
    fig_evolution = go.Figure()
    for emotion in EMOTIONS:
        fig_evolution.add_trace(go.Scatter(
            x=evolution["semaine"], y=evolution[emotion], mode="lines",
            name=emotion, line=dict(color=COULEURS_EMOTIONS[emotion], width=2),
        ))
    fig_evolution.add_vline(x="2025-10-12", line_dash="dash", line_color="black")
    fig_evolution.update_layout(height=420, yaxis_title="Intensité moyenne (proportion)")
    st.plotly_chart(fig_evolution, use_container_width=True)

# ==============================================================================
# ONGLET 3 — PROFILS COMPARÉS
# ==============================================================================
with onglet3:
    st.subheader("Signature émotionnelle de chaque thème de crise")
    profils_themes = charger_profils_emotionnels_themes()
    fig_themes = px.imshow(
        profils_themes.values, x=profils_themes.columns, y=profils_themes.index,
        color_continuous_scale="YlOrRd", aspect="auto", labels=dict(color="Intensité"),
    )
    fig_themes.update_layout(height=380)
    st.plotly_chart(fig_themes, use_container_width=True)

    st.subheader("Profil émotionnel de chaque acteur politique (barres empilées, %)")
    profils_acteurs = charger_profils_emotionnels_acteurs()
    proportions_acteurs = profils_acteurs.div(profils_acteurs.sum(axis=1), axis=0) * 100
    fig_acteurs = go.Figure()
    for emotion in EMOTIONS:
        fig_acteurs.add_trace(go.Bar(
            x=proportions_acteurs.index, y=proportions_acteurs[emotion], name=emotion,
            marker_color=COULEURS_EMOTIONS[emotion],
        ))
    fig_acteurs.update_layout(barmode="stack", height=420, yaxis_title="%")
    st.plotly_chart(fig_acteurs, use_container_width=True)

    st.subheader("Profil émotionnel comparé des 3 plateformes (radar)")
    profils_plateformes = charger_profils_emotionnels_plateformes()  # index = émotion, colonnes = plateforme
    fig_radar_plateformes = go.Figure()
    for plateforme in profils_plateformes.columns:
        valeurs = [profils_plateformes.loc[e, plateforme] for e in EMOTIONS]
        valeurs += valeurs[:1]
        fig_radar_plateformes.add_trace(go.Scatterpolar(
            r=valeurs, theta=EMOTIONS + [EMOTIONS[0]], fill="toself", name=plateforme,
            line=dict(color=COULEURS_PLATEFORME.get(plateforme)),
        ))
    fig_radar_plateformes.update_layout(height=420, polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig_radar_plateformes, use_container_width=True)

# ==============================================================================
# ONGLET 4 — ENGAGEMENT PAR ÉMOTION
# ==============================================================================
with onglet4:
    st.subheader("Nombre moyen de likes selon l'émotion dominante")
    engagement = charger_engagement_par_emotion().set_index("emotion_dominante").reindex(EMOTIONS).dropna().reset_index()
    fig_engagement = px.bar(
        engagement, x="emotion_dominante", y="mean", color="emotion_dominante",
        color_discrete_map=COULEURS_EMOTIONS, text="count",
    )
    fig_engagement.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig_engagement.update_layout(height=450, showlegend=False, xaxis_title="", yaxis_title="Nombre moyen de likes")
    st.plotly_chart(fig_engagement, use_container_width=True)
    st.caption("Le nombre au-dessus de chaque barre (n=...) indique le nombre de publications concernées.")
