# ==============================================================================
# PAGE ALERTES — HISTORIQUE COMPLET ET SCÉNARIOS COMBINÉS
# ==============================================================================
#
# Cette page détaille tout ce que la page d'accueil ne fait qu'esquisser sur
# un seul jour : l'historique complet des alertes sur toute la fenêtre
# d'analyse, les épisodes (regroupement des jours consécutifs), et surtout
# les deux SCÉNARIOS COMBINÉS du script 09 :
#   - Règle combinée 1 : Risque thématique (ITR) + Tension émotionnelle (ITE)
#   - Règle combinée 2 : Mobilisation + Polarisation
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.chargement_donnees import (
    charger_alertes_completes, charger_episodes_alerte,
    charger_seuils_alerte, charger_journal_alertes,
)

st.set_page_config(page_title="Alertes — Veille post-électorale", layout="wide", page_icon="🚨")
st.title("🚨 Historique des alertes")
st.caption("Vert / Orange / Rouge / Critique — sur toute la fenêtre d'analyse (script 09)")

# ------------------------------------------------------------------------------
# CHARGEMENT
# ------------------------------------------------------------------------------
alertes = charger_alertes_completes()
episodes = charger_episodes_alerte()
seuils = charger_seuils_alerte()
journal = charger_journal_alertes()

COULEURS_ALERTE = {"Vert": "#2A9D8F", "Orange": "#F4A300", "Rouge": "#E63946", "Critique": "#6A0572"}

# ------------------------------------------------------------------------------
# FILTRE DE PÉRIODE (dans la barre latérale)
# ------------------------------------------------------------------------------
st.sidebar.header("Filtrer la période")
date_min, date_max = alertes.index.min().date(), alertes.index.max().date()
periode_choisie = st.sidebar.date_input(
    "Période à afficher", value=(date_min, date_max), min_value=date_min, max_value=date_max,
)

# date_input peut renvoyer une seule date le temps que l'utilisateur choisisse
# la seconde borne : on gère ce cas pour ne pas planter la page.
if isinstance(periode_choisie, tuple) and len(periode_choisie) == 2:
    debut_filtre, fin_filtre = periode_choisie
else:
    debut_filtre, fin_filtre = date_min, date_max

masque_periode = (alertes.index.date >= debut_filtre) & (alertes.index.date <= fin_filtre)
alertes_filtrees = alertes.loc[masque_periode]

# ------------------------------------------------------------------------------
# INDICATEURS CLÉS SUR LA PÉRIODE FILTRÉE
# ------------------------------------------------------------------------------
nb_jours = len(alertes_filtrees)
nb_jours_alerte = alertes_filtrees["est_en_alerte"].sum()
nb_jours_critique = (alertes_filtrees["ALERTE_GLOBALE"] == "Critique").sum()
nb_jours_ite_itr = alertes_filtrees["regle_combinee_1_risque_tension"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jours analysés", nb_jours)
col2.metric(
    "Jours en alerte", int(nb_jours_alerte),
    f"{nb_jours_alerte/nb_jours*100:.0f}% de la période" if nb_jours else "",
    delta_color="off",  # neutre : un taux élevé n'est pas une "bonne nouvelle", pas de flèche verte trompeuse
)
col3.metric("Jours Critique", int(nb_jours_critique))
col4.metric("Jours ITE+ITR élevés", int(nb_jours_ite_itr))

st.divider()

# ------------------------------------------------------------------------------
# CALENDRIER VISUEL INTERACTIF
# ------------------------------------------------------------------------------
st.subheader("Calendrier des alertes")

fig_calendrier = go.Figure()
for niveau, couleur in COULEURS_ALERTE.items():
    sous_ensemble = alertes_filtrees[alertes_filtrees["ALERTE_GLOBALE"] == niveau]
    fig_calendrier.add_trace(go.Bar(
        x=sous_ensemble.index, y=[1] * len(sous_ensemble),
        name=niveau, marker_color=couleur, width=1000 * 60 * 60 * 24 * 0.9,
    ))
fig_calendrier.update_layout(
    barmode="stack", height=180, showlegend=True,
    yaxis=dict(showticklabels=False, showgrid=False),
    xaxis_title="Date", legend_title="Niveau",
)
st.plotly_chart(fig_calendrier, use_container_width=True)

# ------------------------------------------------------------------------------
# LES DEUX SCÉNARIOS COMBINÉS — LE CŒUR DE LA PAGE
# ------------------------------------------------------------------------------
st.divider()
st.subheader("🟣 Scénario 1 : Risque thématique + Tension émotionnelle (ITE + ITR)")
st.markdown(
    "Ce scénario se déclenche quand **l'ITR et l'ITE sont tous les deux en zone "
    "Orange ou Rouge le même jour**. Les deux signaux se renforcent : les "
    "publications parlent beaucoup de sujets à risque (fraude, tribalisme, "
    "contestation) *et* elles sont chargées d'émotions négatives (colère, peur)."
)

jours_scenario_1 = alertes_filtrees[alertes_filtrees["regle_combinee_1_risque_tension"]]
if len(jours_scenario_1) > 0:
    st.dataframe(
        jours_scenario_1[["ITE", "ITR", "Score_Risque_Global", "alerte_ITE", "alerte_ITR"]].round(1),
        use_container_width=True,
    )
    st.caption(f"→ {len(jours_scenario_1)} jour(s) concerné(s) sur la période sélectionnée.")
else:
    st.info("Aucun jour de ce type sur la période sélectionnée.")

st.subheader("🟠 Scénario 2 : Mobilisation + Polarisation")
st.markdown(
    "Ce scénario se déclenche quand des **appels à l'action collective** "
    "(manifestations, villes mortes...) et un **public divisé** (opinions "
    "très dispersées) sont détectés le même jour — un terrain propice à un "
    "débordement."
)

jours_scenario_2 = alertes_filtrees[alertes_filtrees["regle_combinee_2_mobilisation_polarisation"]]
if len(jours_scenario_2) > 0:
    st.dataframe(
        jours_scenario_2[["Mobilisation", "Polarisation", "Score_Risque_Global"]].round(1),
        use_container_width=True,
    )
    st.caption(f"→ {len(jours_scenario_2)} jour(s) concerné(s) sur la période sélectionnée.")
else:
    st.info("Aucun jour de ce type sur la période sélectionnée.")

# ------------------------------------------------------------------------------
# ÉPISODES D'ALERTE (jours consécutifs regroupés)
# ------------------------------------------------------------------------------
st.divider()
st.subheader("Épisodes d'alerte (jours consécutifs regroupés)")

episodes_filtres = episodes[
    (episodes["date_fin"].dt.date >= debut_filtre) & (episodes["date_debut"].dt.date <= fin_filtre)
].sort_values("date_debut", ascending=False)

st.dataframe(
    episodes_filtres.style.map(
        lambda v: f"background-color: {COULEURS_ALERTE.get(v, 'white')}33", subset=["niveau_maximal"]
    ),
    use_container_width=True,
)

if len(episodes_filtres) > 0:
    fig_duree = px.bar(
        episodes_filtres.sort_values("date_debut"), x="date_debut", y="duree_jours",
        color="niveau_maximal", color_discrete_map=COULEURS_ALERTE,
        title="Durée de chaque épisode d'alerte", labels={"date_debut": "Date de début", "duree_jours": "Durée (jours)"},
    )
    st.plotly_chart(fig_duree, use_container_width=True)

# ------------------------------------------------------------------------------
# SEUILS DE CALIBRATION (transparence méthodologique)
# ------------------------------------------------------------------------------
with st.expander("📏 Voir les seuils de calibration utilisés (percentiles 75/90)"):
    st.dataframe(seuils, use_container_width=True)
    st.caption(
        "Rappel méthodologique : ces seuils sont calibrés de façon relative à la "
        "fenêtre de données observée (75e et 90e percentile), pas des seuils "
        "absolus universels — voir la limite déjà documentée dans le mémoire."
    )

# ------------------------------------------------------------------------------
# JOURNAL DÉTAILLÉ DES CAUSES
# ------------------------------------------------------------------------------
st.divider()
st.subheader("Journal détaillé des causes d'alerte")

journal_filtre = journal[(journal.index.date >= debut_filtre) & (journal.index.date <= fin_filtre)]
st.dataframe(journal_filtre.sort_index(ascending=False), use_container_width=True)
