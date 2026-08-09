# ==============================================================================
# PAGE SENTIMENTS — ANALYSE DES SENTIMENTS (script 04)
# ==============================================================================
#
# Cette page reprend les 14 tableaux produits par 04_analyse_sentiments.py,
# organisés en 4 onglets pour rester lisible :
#   1. Vue d'ensemble    : répartition globale + évolution quotidienne
#   2. Évolutions        : par plateforme, par thème, par origine, hebdo empilé
#   3. Comparaisons       : boxplot plateformes, heatmap jour x heure, radar,
#                           sentiment par thème, acteurs avant/après, engagement
#   4. Verbatims          : publications les plus positives / négatives
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.chargement_donnees import (
    charger_repartition_sentiment, charger_sentiment_quotidien, charger_proportions_hebdo_sentiment,
    charger_sentiment_plateforme_hebdo, charger_sentiment_themes_hebdo, charger_sentiment_origine_hebdo,
    charger_stats_sentiment_plateforme, charger_sentiment_par_theme, charger_sentiment_acteurs_avant_apres,
    charger_heatmap_jour_heure, charger_sentiment_engagement, charger_radar_plateformes,
    charger_verbatims_positifs, charger_verbatims_negatifs,
)

st.set_page_config(page_title="Sentiments — Veille post-électorale", layout="wide", page_icon="💬")
st.title("💬 Analyse des sentiments")
st.caption("Positif / Neutre / Négatif — lexiques français + VADER (anglais/pidgin), script 04")

COULEURS_SENTIMENT = {"Positif": "#2A9D8F", "Neutre": "#E9C46A", "Négatif": "#E63946", "indéterminé": "#B0B0B0"}
COULEURS_PLATEFORME = {"Facebook": "#1877F2", "TikTok": "#EE1D52", "X": "#6c757d"}

onglet1, onglet2, onglet3, onglet4 = st.tabs([
    "📊 Vue d'ensemble", "📈 Évolutions temporelles", "⚖️ Comparaisons", "📝 Verbatims"
])

# ==============================================================================
# ONGLET 1 — VUE D'ENSEMBLE
# ==============================================================================
with onglet1:
    repartition = charger_repartition_sentiment()
    quotidien = charger_sentiment_quotidien()

    col_gauche, col_droite = st.columns([1, 2])

    with col_gauche:
        st.subheader("Répartition globale")
        repartition_visible = repartition[repartition["sentiment"] != "indéterminé"]
        fig_camembert = px.pie(
            repartition_visible, names="sentiment", values="count",
            color="sentiment", color_discrete_map=COULEURS_SENTIMENT,
        )
        st.plotly_chart(fig_camembert, use_container_width=True)
        st.dataframe(repartition, use_container_width=True, hide_index=True)

    with col_droite:
        st.subheader("Évolution quotidienne du sentiment moyen")
        quotidien["moyenne_mobile_7j"] = quotidien["score_sentiment"].rolling(7, min_periods=1).mean()
        fig_quotidien = go.Figure()
        fig_quotidien.add_trace(go.Scatter(
            x=quotidien["jour"], y=quotidien["score_sentiment"], mode="lines",
            name="Score brut (par jour)", line=dict(color="lightsteelblue", width=1),
        ))
        fig_quotidien.add_trace(go.Scatter(
            x=quotidien["jour"], y=quotidien["moyenne_mobile_7j"], mode="lines",
            name="Moyenne mobile (7 jours)", line=dict(color="#1d3557", width=2.5),
        ))
        fig_quotidien.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_quotidien.add_vline(x="2025-10-12", line_dash="dash", line_color="red")
        fig_quotidien.update_layout(height=420, yaxis_title="Score de sentiment moyen", xaxis_title="Date")
        st.plotly_chart(fig_quotidien, use_container_width=True)
        st.caption("La ligne pointillée rouge marque le jour du vote (12/10/2025).")

    st.divider()
    st.subheader("Évolution hebdomadaire de la composition (aire empilée)")
    proportions = charger_proportions_hebdo_sentiment()
    fig_aire = go.Figure()
    for sentiment in ["Positif", "Neutre", "Négatif"]:
        fig_aire.add_trace(go.Scatter(
            x=proportions["semaine"], y=proportions[sentiment], mode="lines", stackgroup="un",
            name=sentiment, line=dict(width=0.5), fillcolor=COULEURS_SENTIMENT[sentiment],
        ))
    fig_aire.add_vline(x="2025-10-12", line_dash="dash", line_color="black")
    fig_aire.update_layout(height=400, yaxis_title="% des publications", xaxis_title="Semaine")
    st.plotly_chart(fig_aire, use_container_width=True)

# ==============================================================================
# ONGLET 2 — ÉVOLUTIONS TEMPORELLES (plateforme / thème / origine)
# ==============================================================================
with onglet2:
    st.subheader("Par plateforme")
    sentiment_plateforme = charger_sentiment_plateforme_hebdo()
    fig_plateforme = go.Figure()
    for plateforme in [c for c in sentiment_plateforme.columns if c != "semaine"]:
        fig_plateforme.add_trace(go.Scatter(
            x=sentiment_plateforme["semaine"], y=sentiment_plateforme[plateforme], mode="lines+markers",
            name=plateforme, line=dict(color=COULEURS_PLATEFORME.get(plateforme), width=2), marker=dict(size=4),
        ))
    fig_plateforme.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_plateforme.add_vline(x="2025-10-12", line_dash="dash", line_color="red")
    fig_plateforme.update_layout(height=380, yaxis_title="Score de sentiment moyen")
    st.plotly_chart(fig_plateforme, use_container_width=True)

    st.subheader("Par thème de crise")
    sentiment_themes = charger_sentiment_themes_hebdo()
    fig_themes = go.Figure()
    for theme in sentiment_themes.columns:
        fig_themes.add_trace(go.Scatter(
            x=sentiment_themes.index, y=sentiment_themes[theme], mode="lines+markers",
            name=theme, marker=dict(size=4),
        ))
    fig_themes.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_themes.add_vline(x="2025-10-12", line_dash="dash", line_color="black")
    fig_themes.update_layout(height=380, yaxis_title="Score de sentiment moyen")
    st.plotly_chart(fig_themes, use_container_width=True)

    st.subheader("Par origine géographique de l'auteur")
    sentiment_origine = charger_sentiment_origine_hebdo()
    fig_origine = go.Figure()
    for origine in [c for c in sentiment_origine.columns if c != "semaine"]:
        fig_origine.add_trace(go.Scatter(
            x=sentiment_origine["semaine"], y=sentiment_origine[origine], mode="lines+markers",
            name=origine, marker=dict(size=4),
        ))
    fig_origine.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_origine.add_vline(x="2025-10-12", line_dash="dash", line_color="red")
    fig_origine.update_layout(height=380, yaxis_title="Score de sentiment moyen")
    st.plotly_chart(fig_origine, use_container_width=True)

# ==============================================================================
# ONGLET 3 — COMPARAISONS
# ==============================================================================
with onglet3:
    col_gauche, col_droite = st.columns(2)

    with col_gauche:
        st.subheader("Distribution par plateforme")
        stats_plateforme = charger_stats_sentiment_plateforme()
        fig_box = go.Figure()
        for _, ligne in stats_plateforme.iterrows():
            fig_box.add_trace(go.Box(
                q1=[ligne["25%"]], median=[ligne["50%"]], q3=[ligne["75%"]],
                lowerfence=[ligne["min"]], upperfence=[ligne["max"]], mean=[ligne["mean"]],
                name=ligne["plateforme"], marker_color=COULEURS_PLATEFORME.get(ligne["plateforme"]),
            ))
        fig_box.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_box.update_layout(height=400, showlegend=False, yaxis_title="Score de sentiment")
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Sentiment par thème (barres divergentes)")
        sentiment_theme = charger_sentiment_par_theme().sort_values("score_sentiment")
        couleurs_barres = ["#E63946" if v < 0 else "#2A9D8F" for v in sentiment_theme["score_sentiment"]]
        fig_theme = go.Figure(go.Bar(
            x=sentiment_theme["score_sentiment"], y=sentiment_theme["theme"], orientation="h", marker_color=couleurs_barres,
        ))
        fig_theme.add_vline(x=0, line_color="black")
        fig_theme.update_layout(height=400, xaxis_title="Score de sentiment moyen")
        st.plotly_chart(fig_theme, use_container_width=True)

    with col_droite:
        st.subheader("Motif temporel (jour x heure)")
        heatmap = charger_heatmap_jour_heure()
        fig_heatmap = px.imshow(
            heatmap.values, x=heatmap.columns, y=heatmap.index, color_continuous_scale="RdYlGn",
            zmin=-0.3, zmax=0.3, aspect="auto", labels=dict(x="Heure", y="", color="Sentiment moyen"),
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.subheader("Comparaison des plateformes (radar)")
        radar = charger_radar_plateformes()
        radar_normalise = (radar - radar.min()) / (radar.max() - radar.min() + 1e-9)
        fig_radar = go.Figure()
        for plateforme in radar_normalise.index:
            valeurs = radar_normalise.loc[plateforme].tolist()
            valeurs += valeurs[:1]
            fig_radar.add_trace(go.Scatterpolar(
                r=valeurs, theta=list(radar_normalise.columns) + [radar_normalise.columns[0]],
                fill="toself", name=plateforme, line=dict(color=COULEURS_PLATEFORME.get(plateforme)),
            ))
        fig_radar.update_layout(height=400, polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    col_gauche2, col_droite2 = st.columns(2)

    with col_gauche2:
        st.subheader("Sentiment par acteur, avant/après le vote")
        acteurs = charger_sentiment_acteurs_avant_apres()
        fig_acteurs = go.Figure()
        fig_acteurs.add_trace(go.Bar(x=acteurs.index, y=acteurs["avant le vote"], name="Avant le vote", marker_color="#4C9A2A"))
        fig_acteurs.add_trace(go.Bar(x=acteurs.index, y=acteurs["après le vote"], name="Après le vote", marker_color="#B22222"))
        fig_acteurs.add_hline(y=0, line_color="black")
        fig_acteurs.update_layout(barmode="group", height=400, yaxis_title="Score de sentiment moyen")
        st.plotly_chart(fig_acteurs, use_container_width=True)

    with col_droite2:
        st.subheader("Sentiment x Engagement")
        bulles = charger_sentiment_engagement()
        couleurs_bulles = ["#E63946" if x < 0 else ("#E9C46A" if x == 0 else "#2A9D8F") for x in bulles["tranche_sentiment"]]
        fig_bulles = go.Figure(go.Scatter(
            x=bulles["tranche_sentiment"], y=bulles["engagement_moyen"], mode="markers",
            marker=dict(size=bulles["nb_publications"] / bulles["nb_publications"].max() * 60 + 5,
                        color=couleurs_bulles, line=dict(color="black", width=0.5)),
        ))
        fig_bulles.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_bulles.update_layout(height=400, xaxis_title="Tranche de sentiment", yaxis_title="Nombre moyen de likes")
        st.plotly_chart(fig_bulles, use_container_width=True)

# ==============================================================================
# ONGLET 4 — VERBATIMS
# ==============================================================================
with onglet4:
    col_gauche, col_droite = st.columns(2)

    with col_gauche:
        st.subheader("🟢 Top 10 — Publications les plus positives")
        positives = charger_verbatims_positifs()
        for _, ligne in positives.iterrows():
            with st.container(border=True):
                st.markdown(f"**{ligne['plateforme']}** — {ligne['date_publication']} — score : `{ligne['score_sentiment']:.2f}` — 👍 {int(ligne['nb_likes'])}")
                st.write(ligne["texte"])

    with col_droite:
        st.subheader("🔴 Top 10 — Publications les plus négatives")
        negatives = charger_verbatims_negatifs()
        for _, ligne in negatives.iterrows():
            with st.container(border=True):
                st.markdown(f"**{ligne['plateforme']}** — {ligne['date_publication']} — score : `{ligne['score_sentiment']:.2f}` — 👍 {int(ligne['nb_likes'])}")
                st.write(ligne["texte"])
