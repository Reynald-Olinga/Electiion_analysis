# ==============================================================================
# MODULE COMMUN — CHARGEMENT DES DONNÉES
# ==============================================================================
#
# À quoi sert ce fichier ?
# ------------------------
# Toutes les pages du dashboard (app.py + fichiers dans pages/) ont besoin de
# lire les mêmes CSV, déjà produits par les scripts 01 à 09. Plutôt que de
# répéter le même code pd.read_csv(...) dans chaque page, on centralise ces
# fonctions ici, et chaque page fera simplement :
#
#   from utils.chargement_donnees import charger_indicateurs_quotidiens
#
# Le décorateur @st.cache_data est important : il dit à Streamlit "si les
# arguments de la fonction n'ont pas changé, ne relis pas le fichier CSV à
# chaque clic de l'utilisateur, garde le résultat en mémoire". Sans ça,
# chaque interaction (changer un filtre, cliquer un bouton) relirait 48 138
# lignes depuis le disque, ce qui rendrait l'application lente.
# ==============================================================================

import pandas as pd
import streamlit as st

# Chemins relatifs au dossier dashboard/ (d'où Streamlit sera lancé)
CHEMIN_TABLEAUX = "../outputs/tables"
CHEMIN_DATA = "../data"


# ==============================================================================
# CHARGEMENTS GÉNÉRAUX (page d'accueil + indicateurs)
# ==============================================================================

@st.cache_data
def charger_indicateurs_quotidiens():
    """
    Charge le tableau de bord quotidien produit par le script 08 :
    une ligne par jour, avec ITE, ITR, Activité, Engagement, Viralité,
    Polarisation, Mobilisation, Désinformation, Score_Risque_Global.
    """
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/INDICATEURS_tableau_complet_quotidien.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "jour"
    return df


@st.cache_data
def charger_alertes_quotidiennes():
    """
    Charge le tableau détaillé des alertes produit par le script 09 :
    contient, en plus des indicateurs, les colonnes alerte_ITE, alerte_ITR,
    alerte_Score_Risque_Global, les règles combinées, et ALERTE_GLOBALE.
    """
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/C1_regles_combinees.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "jour"
    return df


@st.cache_data
def charger_alertes_completes():
    """
    Charge C1_regles_combinees.csv (script 09, section C) et Y AJOUTE la
    colonne ALERTE_GLOBALE, calculée avec EXACTEMENT la même règle que la
    fonction determiner_alerte_globale() du script 09 :
      - Critique si une règle combinée (1 ou 2) est déclenchée
      - Rouge si un des 3 indicateurs clés est en zone Rouge
      - Orange si un des 3 est en zone Orange
      - Vert sinon
    """
    df = charger_alertes_quotidiennes().copy()

    def determiner_alerte_globale(ligne):
        if ligne["regle_combinee_1_risque_tension"] or ligne["regle_combinee_2_mobilisation_polarisation"]:
            return "Critique"
        niveaux = [ligne["alerte_ITE"], ligne["alerte_ITR"], ligne["alerte_Score_Risque_Global"]]
        if "Rouge" in niveaux:
            return "Rouge"
        elif "Orange" in niveaux:
            return "Orange"
        else:
            return "Vert"

    df["ALERTE_GLOBALE"] = df.apply(determiner_alerte_globale, axis=1)
    df["est_en_alerte"] = df["ALERTE_GLOBALE"] != "Vert"
    return df


@st.cache_data
def charger_journal_alertes():
    """Journal des jours en alerte uniquement (Orange/Rouge/Critique), script 09."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/JOURNAL_alertes_final.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "jour"
    return df


@st.cache_data
def charger_episodes_alerte():
    """Épisodes d'alerte regroupés (jours consécutifs), script 09."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/F1_episodes_alerte.csv")
    df["date_debut"] = pd.to_datetime(df["date_debut"])
    df["date_fin"] = pd.to_datetime(df["date_fin"])
    return df


@st.cache_data
def charger_seuils_alerte():
    """Seuils Orange/Rouge calibrés par percentile (script 09)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/A1_seuils_alerte.csv", index_col=0)


@st.cache_data
def charger_synthese_themes():
    """Tableau de bord récapitulatif par thème (script 07)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/SYNTHESE_tableau_de_bord_themes.csv", index_col=0)


@st.cache_data
def charger_corpus_complet():
    """
    Charge le corpus complet avec sentiment + émotions + thèmes (script 06).
    Fichier volumineux (48 138 lignes) : à n'utiliser que sur les pages qui
    en ont vraiment besoin (ex. page Recherche), pas sur toutes les pages.
    """
    df = pd.read_csv(f"{CHEMIN_DATA}/bd_f_themes.csv", low_memory=False)
    df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
    return df


# ==============================================================================
# CHARGEMENTS POUR LA PAGE SENTIMENTS (script 04)
# ==============================================================================

@st.cache_data
def charger_repartition_sentiment():
    """Répartition Positif/Neutre/Négatif/indéterminé (script 04, section E)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/E1_repartition_sentiment.csv")


@st.cache_data
def charger_sentiment_quotidien():
    """Score de sentiment moyen, un par jour, sur tout le corpus (section F)."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/F1_sentiment_moyen_par_jour.csv")
    df["jour"] = pd.to_datetime(df["jour"])
    return df


@st.cache_data
def charger_proportions_hebdo_sentiment():
    """Proportions hebdomadaires Positif/Neutre/Négatif, pour l'aire empilée (section G)."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/G1_proportions_hebdomadaires_sentiment.csv")
    df["semaine"] = pd.to_datetime(df["semaine"])
    return df


@st.cache_data
def charger_sentiment_plateforme_hebdo():
    """Évolution hebdomadaire du sentiment moyen, par plateforme (section H)."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/H1_evolution_sentiment_plateforme.csv")
    df["semaine"] = pd.to_datetime(df["semaine"])
    return df


@st.cache_data
def charger_sentiment_themes_hebdo():
    """Évolution du sentiment moyen par thème de crise (section I)."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/I1_evolution_sentiment_themes.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def charger_sentiment_origine_hebdo():
    """Évolution du sentiment moyen par origine géographique de l'auteur (section J)."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/J1_evolution_sentiment_origine.csv")
    df["semaine"] = pd.to_datetime(df["semaine"])
    return df


@st.cache_data
def charger_stats_sentiment_plateforme():
    """Statistiques descriptives du sentiment par plateforme (min/quartiles/max), section K."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/K1_stats_sentiment_plateforme.csv")


@st.cache_data
def charger_sentiment_par_theme():
    """Sentiment moyen selon le thème de crise évoqué, barres divergentes (section L)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/L1_sentiment_par_theme.csv", header=0, names=["theme", "score_sentiment"])


@st.cache_data
def charger_sentiment_acteurs_avant_apres():
    """Sentiment envers chaque acteur politique, avant/après le vote (section M)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/M1_sentiment_acteurs_avant_apres.csv", index_col=0)


@st.cache_data
def charger_heatmap_jour_heure():
    """Sentiment moyen selon le jour de la semaine et l'heure (section N)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/N1_heatmap_sentiment_jour_heure.csv", index_col=0)


@st.cache_data
def charger_sentiment_engagement():
    """Engagement moyen par tranche de score de sentiment (section O)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/O1_sentiment_x_engagement.csv")


@st.cache_data
def charger_radar_plateformes():
    """Indicateurs comparés des plateformes pour le radar (section P)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/P1_indicateurs_radar_plateformes.csv", index_col=0)


@st.cache_data
def charger_verbatims_positifs():
    """Top 10 publications les plus positives (section Q)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/Q1_top_10_publications_positives.csv")


@st.cache_data
def charger_verbatims_negatifs():
    """Top 10 publications les plus négatives (section Q)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/Q2_top_10_publications_negatives.csv")


# ==============================================================================
# CHARGEMENTS POUR LA PAGE ÉMOTIONS (script 05)
# ==============================================================================

EMOTIONS = ["Colère", "Peur", "Joie", "Tristesse", "Dégoût", "Surprise", "Confiance", "Anticipation"]
COULEURS_EMOTIONS = {
    "Colère": "#E63946", "Peur": "#6A0572", "Joie": "#F4A300", "Tristesse": "#457B9D",
    "Dégoût": "#588157", "Surprise": "#F72585", "Confiance": "#2A9D8F", "Anticipation": "#B08968",
}


@st.cache_data
def charger_repartition_emotions():
    """Nombre de publications par émotion dominante (section D)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/D1_repartition_emotions.csv")


@st.cache_data
def charger_scores_emotionnels_globaux():
    """Intensité moyenne de chaque émotion sur tout le corpus, pour le radar global (section E)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/E1_scores_emotionnels_globaux.csv", header=0, names=["emotion", "score"])


@st.cache_data
def charger_emotions_avant_apres():
    """Scores émotionnels moyens, avant vs après le vote (section F)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/F1_scores_emotionnels_avant_apres.csv", index_col=0)


@st.cache_data
def charger_evolution_hebdo_emotions():
    """Évolution hebdomadaire de l'intensité de chaque émotion (section G)."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/G1_evolution_hebdomadaire_emotions.csv")
    df["semaine"] = pd.to_datetime(df["semaine"])
    return df


@st.cache_data
def charger_heatmap_emotions_temps():
    """Heatmap émotions x semaines, 25 dernières semaines (section H)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/H1_heatmap_emotions_temps.csv", index_col=0)


@st.cache_data
def charger_profils_emotionnels_themes():
    """Signature émotionnelle de chaque thème de crise (section I)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/I1_profils_emotionnels_themes.csv", index_col=0)


@st.cache_data
def charger_profils_emotionnels_acteurs():
    """Profil émotionnel de chaque acteur politique (section J)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/J1_profils_emotionnels_acteurs.csv", index_col=0)


@st.cache_data
def charger_profils_emotionnels_plateformes():
    """Profil émotionnel comparé des 3 plateformes (section K) — index = émotion, colonnes = plateforme."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/K1_profils_emotionnels_plateformes.csv", index_col=0)


@st.cache_data
def charger_engagement_par_emotion():
    """Engagement moyen (likes) selon l'émotion dominante de la publication (section L)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/L1_engagement_par_emotion.csv")


# ==============================================================================
# CHARGEMENTS POUR LA PAGE THÉMATIQUES (script 06 — topic modeling LDA)
# ==============================================================================

# Mapping fixe des 5 thèmes français, tel qu'interprété manuellement à partir
# des mots-clés (section E du script 06). C'est le même mapping repris dans
# le script 07 (themes_francais). Utilisé uniquement pour donner un nom
# lisible aux colonnes 0/1/2/3/4 du fichier D1 (mots-clés bruts par thème).
LIBELLES_THEMES_FR = {
    "0": "Tribalisme, régime et tensions régionales (Ouest)",
    "1": "Solidarité, foi et encouragements au peuple",
    "2": "Candidats de l'opposition et espoir de changement",
    "3": "Communauté Bamiléké, famille et problèmes du pays",
    "4": "Paul Biya : pouvoir, vie et mort",
}


@st.cache_data
def charger_repartition_themes_fr():
    """Nombre de publications par thème français dominant (section J)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/J1_repartition_themes_francais.csv")


@st.cache_data
def charger_mots_cles_themes_fr():
    """
    Top 12 mots-clés de chaque thème français (section D). Les colonnes du
    fichier brut sont juste "0","1","2","3","4" (l'identifiant numérique du
    thème LDA) : on les renomme avec LIBELLES_THEMES_FR pour la lisibilité.
    """
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/D1_mots_cles_themes_francais.csv")
    df = df.rename(columns=LIBELLES_THEMES_FR)
    return df


@st.cache_data
def charger_evolution_themes_hebdo():
    """Évolution hebdomadaire de la part de chaque thème français (section K)."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/K1_evolution_themes_hebdomadaire.csv")
    df["semaine"] = pd.to_datetime(df["semaine"])
    return df


@st.cache_data
def charger_themes_avant_apres():
    """Poids de chaque thème français, avant vs après le vote (section M)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/M1_themes_avant_apres.csv")


@st.cache_data
def charger_themes_par_plateforme():
    """Répartition des thèmes français par plateforme, % (section L)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/L1_themes_par_plateforme.csv", index_col=0)


@st.cache_data
def charger_themes_par_acteur():
    """Répartition thématique des publications mentionnant chaque acteur (section N)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/N1_themes_par_acteur.csv", index_col=0)


@st.cache_data
def charger_similarite_themes():
    """Similarité cosinus entre les 5 thèmes français (contrôle qualité, section G)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/G1_similarite_themes.csv", index_col=0)


@st.cache_data
def charger_sentiment_par_theme_lda():
    """Sentiment moyen selon le thème LDA dominant (section Q)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/Q1_sentiment_par_theme.csv")


@st.cache_data
def charger_profil_emotionnel_themes_lda():
    """Profil des 8 émotions pour chaque thème LDA (section R)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/R1_profil_emotionnel_themes.csv", index_col=0)


@st.cache_data
def charger_engagement_par_theme_lda():
    """Engagement moyen (likes) selon le thème LDA dominant (section O)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/O1_engagement_par_theme.csv")


@st.cache_data
def charger_entropie_plateforme():
    """Diversité thématique (entropie de Shannon) par plateforme (section P)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/P1_entropie_par_plateforme.csv", header=0, names=["plateforme", "entropie"])


@st.cache_data
def charger_entropie_periode():
    """Diversité thématique (entropie de Shannon) par période (section P)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/P2_entropie_par_periode.csv", header=0, names=["periode", "entropie"])


@st.cache_data
def charger_validation_croisee():
    """Validation croisée : catégories de crise a priori vs thèmes LDA (section S)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/S1_validation_croisee_crise_themes.csv", index_col=0)


@st.cache_data
def charger_mots_cles_themes_en():
    """
    Top 12 mots-clés de chaque thème anglais/pidgin (section T). ATTENTION :
    contrairement au français, le nombre de thèmes anglais (k_en) varie
    d'une exécution à l'autre du script (choisi automatiquement par cohérence),
    donc on ne renomme PAS les colonnes ici (pas de mapping fiable garanti) —
    elles restent "Thème 0", "Thème 1"... La page l'indique explicitement.
    """
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/T1_mots_cles_themes_anglais.csv")
    df.columns = [f"Thème {c}" for c in df.columns]
    return df


@st.cache_data
def charger_repartition_themes_en():
    """Nombre de publications par thème anglais/pidgin dominant (section T)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/T2_repartition_themes_anglais.csv")


@st.cache_data
def charger_engagement_par_theme_en():
    """Engagement moyen par thème anglais/pidgin (section T)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/T3_engagement_par_theme_anglais.csv")


@st.cache_data
def charger_publications_representatives_themes():
    """Les 3 publications les plus représentatives de chaque thème français (section I)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/I1_publications_representatives_themes.csv")


# ==============================================================================
# CHARGEMENTS POUR LA PAGE INDICATEURS DÉCISIONNELS (script 08)
# ==============================================================================

# Pondérations du Score de Risque Global, reprises telles quelles du script 08
# (dictionnaire PONDERATIONS). Rien n'est recalculé : ce sont EXACTEMENT les
# mêmes poids que ceux utilisés pour produire Score_Risque_Global.
PONDERATIONS_SCORE_RISQUE = {
    "ITE": 0.20, "ITR": 0.20, "Polarisation": 0.15, "Mobilisation": 0.15,
    "Desinformation": 0.10, "Viralite": 0.10, "Activite": 0.05, "Engagement": 0.05,
}

# Noms d'affichage (avec accents) pour les colonnes du tableau d'indicateurs,
# qui elles n'ont pas d'accents (contrainte historique du script 08).
NOMS_AFFICHAGE_INDICATEURS = {
    "ITE": "ITE", "ITR": "ITR", "Activite": "Activité", "Engagement": "Engagement",
    "Viralite": "Viralité", "Polarisation": "Polarisation", "Mobilisation": "Mobilisation",
    "Desinformation": "Désinformation", "Score_Risque_Global": "Score de Risque Global",
}


@st.cache_data
def charger_volume_quotidien():
    """Volume brut de publications par jour, sur la fenêtre d'analyse (script 08, étape 2)."""
    df = pd.read_csv(f"{CHEMIN_TABLEAUX}/00_volume_quotidien_fenetre.csv", header=0, names=["jour", "volume"])
    df["jour"] = pd.to_datetime(df["jour"])
    return df


@st.cache_data
def charger_correlation_indicateurs():
    """Matrice de corrélation entre les 8 indicateurs (script 08, VALIDATION1)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/VALIDATION1_correlation_indicateurs.csv", index_col=0)


@st.cache_data
def charger_validation_villes_mortes():
    """Score de Risque Global moyen : jours de forte mention 'villes mortes' vs autres jours (VALIDATION2)."""
    return pd.read_csv(f"{CHEMIN_TABLEAUX}/VALIDATION2_score_vs_villes_mortes.csv", header=0, names=["categorie", "score_moyen"])
