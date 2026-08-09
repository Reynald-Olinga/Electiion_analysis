# ==============================================================================
# PAGE RECHERCHE — FILTRAGE LIBRE DANS LE CORPUS
# ==============================================================================
#
# Cette page donne un accès direct aux 48 138 publications (bd_f_themes.csv),
# avec des filtres combinables : mot-clé/hashtag, plateforme, période,
# sentiment, thème, auteur. Utile pour qu'un décideur ou un analyste vérifie
# concrètement les publications derrière un indicateur ou une alerte.
# ==============================================================================

import streamlit as st
import pandas as pd
from utils.chargement_donnees import charger_corpus_complet

st.set_page_config(page_title="Recherche — Veille post-électorale", layout="wide", page_icon="🔍")
st.title("🔍 Recherche dans le corpus")
st.caption("48 138 publications — filtrage par mot-clé, plateforme, période, sentiment, thème ou auteur")

corpus = charger_corpus_complet()

# ------------------------------------------------------------------------------
# FILTRES (barre latérale)
# ------------------------------------------------------------------------------
st.sidebar.header("Filtres de recherche")

texte_recherche = st.sidebar.text_input(
    "Mot-clé ou hashtag", placeholder="ex : fraude, #villesmortes, kamto...",
    help="Recherche insensible à la casse dans le texte brut des publications.",
)

plateformes_disponibles = sorted(corpus["plateforme"].dropna().unique())
plateformes_choisies = st.sidebar.multiselect("Plateforme", options=plateformes_disponibles, default=plateformes_disponibles)

date_min = corpus["date_publication"].min()
date_max = corpus["date_publication"].max()
periode_choisie = st.sidebar.date_input(
    "Période", value=(date_min.date(), date_max.date()), min_value=date_min.date(), max_value=date_max.date(),
)

sentiments_disponibles = sorted(corpus["sentiment"].dropna().unique()) if "sentiment" in corpus.columns else []
sentiments_choisis = st.sidebar.multiselect("Sentiment", options=sentiments_disponibles, default=sentiments_disponibles)

themes_disponibles = sorted(corpus["theme_nom"].dropna().unique()) if "theme_nom" in corpus.columns else []
themes_choisis = st.sidebar.multiselect("Thème", options=themes_disponibles, default=themes_disponibles)

auteur_recherche = st.sidebar.text_input(
    "Auteur (pseudonyme)", placeholder="ex : un pseudo précis...",
    help="Recherche exacte, insensible à la casse, sur le pseudonyme déjà anonymisé.",
)

st.sidebar.caption(
    "ℹ️ Conformément à l'engagement d'anonymisation du mémoire, les identifiants "
    "affichés ici sont déjà des pseudonymes ; évite de les diffuser tels quels "
    "dans un rapport destiné à un public externe."
)

# ------------------------------------------------------------------------------
# APPLICATION DES FILTRES
# ------------------------------------------------------------------------------
resultats = corpus.copy()

if texte_recherche.strip():
    resultats = resultats[resultats["texte"].str.contains(texte_recherche, case=False, na=False, regex=False)]

if plateformes_choisies:
    resultats = resultats[resultats["plateforme"].isin(plateformes_choisies)]

if isinstance(periode_choisie, tuple) and len(periode_choisie) == 2:
    debut_filtre, fin_filtre = periode_choisie
    resultats = resultats[
        (resultats["date_publication"].dt.date >= debut_filtre) & (resultats["date_publication"].dt.date <= fin_filtre)
    ]

if sentiments_choisis and "sentiment" in resultats.columns:
    resultats = resultats[resultats["sentiment"].isin(sentiments_choisis)]

if themes_choisis and "theme_nom" in resultats.columns:
    resultats = resultats[resultats["theme_nom"].isin(themes_choisis)]

if auteur_recherche.strip():
    resultats = resultats[resultats["auteur_pseudo"].str.contains(auteur_recherche, case=False, na=False, regex=False)]

# ------------------------------------------------------------------------------
# AFFICHAGE DES RÉSULTATS
# ------------------------------------------------------------------------------
st.metric("Publications trouvées", f"{len(resultats):,}".replace(",", " "))

colonnes_a_afficher = [
    c for c in ["date_publication", "plateforme", "auteur_pseudo", "texte", "score_sentiment",
                "sentiment", "emotion_dominante", "theme_nom", "nb_likes", "nb_reponses"]
    if c in resultats.columns
]

st.dataframe(
    resultats[colonnes_a_afficher].sort_values("date_publication", ascending=False),
    use_container_width=True, height=500,
)

# ------------------------------------------------------------------------------
# EXPORT DES RÉSULTATS FILTRÉS
# ------------------------------------------------------------------------------
if len(resultats) > 0:
    csv_export = resultats[colonnes_a_afficher].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Télécharger ces résultats (CSV)", data=csv_export,
        file_name="resultats_recherche.csv", mime="text/csv",
    )
else:
    st.info("Aucune publication ne correspond à ces filtres — essaie d'élargir la période ou de retirer un critère.")
