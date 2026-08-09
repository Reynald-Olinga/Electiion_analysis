# ==============================================================================
# PHASE 5 — ANALYSE DES SENTIMENTS (version enrichie)
# ==============================================================================
#
# Cette version va plus loin que la précédente : en plus des analyses de
# base (histogramme, évolution quotidienne), on ajoute de VRAIES évolutions
# temporelles par plateforme, par thème et par origine géographique (comme
# le prévoyait le canevas), et on varie beaucoup les types de graphiques :
# histogramme, camembert, courbes, aires empilées, boxplots, heatmap,
# bulles, diagramme radar, barres divergentes, barres groupées.
#
# Sections :
#   A. Re-lemmatisation en conservant les négations
#   B. Lexique français de sentiment
#   C. Lexique anglais/pidgin (VADER)
#   D. Calcul du score de sentiment
#   E. Classification et distribution générale (histogramme + camembert)
#   F. Évolution quotidienne globale (autour du vote)
#   G. Évolution hebdomadaire EN PROPORTIONS (aire empilée)
#   H. Évolution temporelle PAR PLATEFORME (courbes superposées)
#   I. Évolution temporelle PAR THÈME de crise (courbes superposées)
#   J. Évolution temporelle PAR ORIGINE géographique (courbes superposées)
#   K. Distribution du sentiment par plateforme (boxplot)
#   L. Sentiment par thème de crise (barres divergentes)
#   M. Sentiment par acteur politique, avant/après le vote (barres groupées)
#   N. Motif temporel jour x heure (heatmap)
#   O. Sentiment x engagement (bulles)
#   P. Comparaison des plateformes sur plusieurs indicateurs (radar)
#   Q. Verbatims : publications les plus positives / les plus négatives
#   R. Sauvegarde du corpus enrichi
# ==============================================================================


# ------------------------------------------------------------------------------
# ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("Chargement des modèles de langue...")
nlp_francais = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
nlp_anglais = spacy.load("en_core_web_sm", disable=["parser", "ner"])
print("Modèles chargés.")


# ------------------------------------------------------------------------------
# ÉTAPE 1 : PRÉPARER LES DOSSIERS DE SORTIE
# ------------------------------------------------------------------------------

dossier_data = "../data"
dossier_figures = "../outputs/figures"
dossier_tableaux = "../outputs/tables"
os.makedirs(dossier_figures, exist_ok=True)
os.makedirs(dossier_tableaux, exist_ok=True)


# ------------------------------------------------------------------------------
# ÉTAPE 2 : CHARGER LE CORPUS NETTOYÉ (Phase 3)
# ------------------------------------------------------------------------------

print("\nLecture du corpus nettoyé...")
df = pd.read_csv(f"{dossier_data}/bd_f_nettoye.csv", low_memory=False)
df["texte_nettoye"] = df["texte_nettoye"].fillna("")
df["nb_likes"] = pd.to_numeric(df["nb_likes"], errors="coerce")
print(f"{len(df)} publications chargées.")

df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
date_du_vote = pd.Timestamp("2025-10-12", tz="UTC")
df["periode"] = "après le vote"
df.loc[df["date_publication"] < date_du_vote, "periode"] = "avant le vote"


# ==============================================================================
# SECTION A — RE-LEMMATISATION EN GARDANT LES MOTS DE NÉGATION
# ==============================================================================
# Rappel : la liste standard de mots vides de spaCy enlève "pas", "ne" et
# "sans"... qui sont justement indispensables pour l'analyse de sentiment
# ("content" et "pas content" n'ont pas le même sens !). On refait donc une
# lemmatisation spécifique à cette phase, qui garde ces mots de négation.

print("\n--- A : re-lemmatisation du texte en conservant les négations ---")

from spacy.lang.fr.stop_words import STOP_WORDS as mots_vides_francais_originaux
from spacy.lang.en.stop_words import STOP_WORDS as mots_vides_anglais_originaux

mots_de_negation_fr = {"pas", "ne", "non", "jamais", "aucun", "aucune", "rien", "sans", "ni", "plus"}
mots_de_negation_en = {"no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "without"}

mots_vides_francais_sentiment = mots_vides_francais_originaux - mots_de_negation_fr
mots_vides_anglais_sentiment = mots_vides_anglais_originaux - mots_de_negation_en


def relemmatiser_en_gardant_negations(textes, modele_spacy, mots_vides):
    resultats = []
    for document in modele_spacy.pipe(textes, batch_size=200):
        mots_gardes = [
            token.lemma_.strip().lower() for token in document
            if len(token.lemma_.strip()) > 1 and token.lemma_.strip().lower() not in mots_vides
        ]
        resultats.append(" ".join(mots_gardes))
    return resultats


df["texte_sentiment"] = ""

indices_francais = df.index[df["langue_detectee"] == "fr"]
df.loc[indices_francais, "texte_sentiment"] = relemmatiser_en_gardant_negations(
    df.loc[indices_francais, "texte_nettoye"].tolist(), nlp_francais, mots_vides_francais_sentiment
)
print(f"  -> {len(indices_francais)} textes français re-lemmatisés")

indices_anglais_pidgin = df.index[df["langue_detectee"].isin(["en", "pidgin"])]
df.loc[indices_anglais_pidgin, "texte_sentiment"] = relemmatiser_en_gardant_negations(
    df.loc[indices_anglais_pidgin, "texte_nettoye"].tolist(), nlp_anglais, mots_vides_anglais_sentiment
)
print(f"  -> {len(indices_anglais_pidgin)} textes anglais/pidgin re-lemmatisés")


# ==============================================================================
# SECTION B — LE LEXIQUE FRANÇAIS DE SENTIMENT
# ==============================================================================

print("\n--- B : construction du lexique de sentiment français ---")

lexique_positif_fr = {
    "content": 1, "heureux": 2, "heureuse": 2, "joie": 2, "joyeux": 2, "bonheur": 2,
    "beau": 1, "belle": 1, "bien": 1, "bon": 1, "bonne": 1, "excellent": 2, "génial": 2,
    "super": 1, "magnifique": 2, "formidable": 2, "merveilleux": 2, "aimer": 1, "adorer": 2,
    "merci": 1, "bravo": 2, "félicitation": 2, "féliciter": 2, "réussir": 1, "réussite": 1,
    "gagner": 1, "victoire": 2, "vainqueur": 1, "espoir": 1, "espérer": 1, "confiant": 1,
    "confiance": 1, "fier": 1, "fierté": 1, "sourire": 1, "rire": 1, "encourager": 1,
    "courage": 1, "courageux": 1, "soutenir": 1, "soutien": 1, "aimable": 1, "sympa": 1,
    "paix": 2, "unité": 1, "uni": 1, "union": 1, "démocratie": 1, "liberté": 1, "justice": 1,
    "vérité": 1, "dignité": 1, "respect": 1, "solidarité": 1, "progrès": 1, "développement": 1,
    "changement": 1, "réforme": 1, "réconciliation": 2, "dialogue": 1, "calme": 1, "stabilité": 1,
    "prospérité": 1, "bénédiction": 1, "bénir": 1, "dieu": 1,
}

lexique_negatif_fr = {
    "triste": 1, "tristesse": 1, "malheureux": 2, "colère": 2, "fâché": 1, "haine": 2,
    "détester": 2, "peur": 1, "honte": 2, "honteux": 2, "dommage": 1, "déçu": 1, "déception": 1,
    "mauvais": 1, "mauvaise": 1, "terrible": 2, "horrible": 2, "affreux": 2, "grave": 1,
    "pleurer": 1, "souffrir": 2, "souffrance": 2, "douleur": 2, "inquiet": 1, "inquiétude": 1,
    "échec": 1, "échouer": 1, "perdre": 1, "perdu": 1, "abandonner": 1, "abandon": 1,
    "critiquer": 1, "critique": 1, "insulter": 1, "insulte": 1, "ridicule": 1, "stupide": 1,
    "faible": 1, "lâche": 2, "traître": 2, "traîtrise": 2, "menteur": 1, "mensonge": 2,
    "fraude": 2, "frauder": 2, "tricher": 2, "tricherie": 2, "violence": 2, "violent": 2,
    "manifestation": 1, "manifestant": 1, "répression": 2, "réprimer": 2, "arrestation": 1,
    "arrêter": 1, "tuer": 2, "tué": 2, "mort": 2, "morts": 2, "décès": 2, "blessé": 2,
    "corruption": 2, "corrompu": 2, "dictateur": 2, "dictature": 2, "tyran": 2, "tyrannie": 2,
    "oppression": 2, "opprimer": 2, "injustice": 2, "injuste": 2, "abus": 1, "abuser": 1,
    "chaos": 2, "crise": 1, "danger": 1, "dangereux": 1, "conflit": 1, "guerre": 2,
    "tribalisme": 2, "tribaliste": 2, "ethnie": 1, "diviser": 1, "division": 1, "instable": 1,
    "instabilité": 1, "insécurité": 2, "pauvreté": 1, "misère": 2, "faim": 1,
    "scandale": 1, "scandaleux": 2, "vol": 1, "voler": 1, "voleur": 1, "gaspillage": 1,
}

intensificateurs_fr = {"très": 1.5, "trop": 1.5, "vraiment": 1.4, "extrêmement": 1.8,
                        "tellement": 1.4, "totalement": 1.3, "complètement": 1.3}

print(f"Lexique français : {len(lexique_positif_fr)} mots positifs, {len(lexique_negatif_fr)} mots négatifs")


def calculer_score_francais(texte):
    mots = texte.split()
    score_total = 0
    nb_mots_de_sentiment_trouves = 0

    for position, mot in enumerate(mots):
        poids = None
        if mot in lexique_positif_fr:
            poids = lexique_positif_fr[mot]
        elif mot in lexique_negatif_fr:
            poids = -lexique_negatif_fr[mot]

        if poids is not None:
            fenetre_precedente = mots[max(0, position - 3):position]
            if any(mot_precedent in mots_de_negation_fr for mot_precedent in fenetre_precedente):
                poids = -poids
            if position > 0 and mots[position - 1] in intensificateurs_fr:
                poids = poids * intensificateurs_fr[mots[position - 1]]
            score_total += poids
            nb_mots_de_sentiment_trouves += 1

    if nb_mots_de_sentiment_trouves == 0:
        return 0.0

    score_moyen = score_total / nb_mots_de_sentiment_trouves
    return np.tanh(score_moyen)


# ==============================================================================
# SECTION C — SENTIMENT DE L'ANGLAIS ET DU PIDGIN (VADER)
# ==============================================================================

print("\n--- C : préparation de VADER (anglais + pidgin) ---")

analyseur_vader = SentimentIntensityAnalyzer()

mots_pidgin_supplementaires = {
    "wahala": -2.0, "vex": -1.5, "sabi": 0.0, "waka": 0.0, "suffer": -2.0,
    "enjoy": 2.0, "sweet": 1.5, "fine": 1.5, "wicked": -2.0, "corrupt": -2.5,
}
analyseur_vader.lexicon.update(mots_pidgin_supplementaires)
print(f"Lexique VADER enrichi de {len(mots_pidgin_supplementaires)} mots de pidgin camerounais.")


def calculer_score_anglais(texte):
    if texte.strip() == "":
        return 0.0
    return analyseur_vader.polarity_scores(texte)["compound"]


# ==============================================================================
# SECTION D — APPLICATION DU CALCUL DE SENTIMENT À TOUT LE CORPUS
# ==============================================================================

print("\n--- D : calcul du score de sentiment pour toutes les publications ---")

scores_sentiment = []
for indice, ligne in df.iterrows():
    langue = ligne["langue_detectee"]
    texte = ligne["texte_sentiment"]
    if langue == "fr":
        score = calculer_score_francais(texte)
    elif langue in ("en", "pidgin"):
        score = calculer_score_anglais(texte)
    else:
        score = np.nan
    scores_sentiment.append(score)

df["score_sentiment"] = scores_sentiment
print(f"Score calculé pour {df['score_sentiment'].notna().sum()} publications "
      f"({df['score_sentiment'].isna().sum()} langues non prises en charge).")


# ==============================================================================
# SECTION E — CLASSIFICATION ET DISTRIBUTION GÉNÉRALE
# ==============================================================================

print("\n--- E : classification en Positif / Neutre / Négatif ---")


def classifier_sentiment(score):
    if pd.isna(score):
        return "indéterminé"
    elif score >= 0.05:
        return "Positif"
    elif score <= -0.05:
        return "Négatif"
    else:
        return "Neutre"


df["sentiment"] = df["score_sentiment"].apply(classifier_sentiment)

couleurs_sentiment = {"Positif": "#2A9D8F", "Neutre": "#E9C46A", "Négatif": "#E63946", "indéterminé": "#B0B0B0"}

repartition_sentiment = df["sentiment"].value_counts()
print(repartition_sentiment)
repartition_sentiment.to_csv(f"{dossier_tableaux}/E1_repartition_sentiment.csv")

scores_valides = df["score_sentiment"].dropna()
plt.figure(figsize=(10, 6))
plt.hist(scores_valides, bins=50, color="#6A4C93", edgecolor="white")
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Distribution des scores de sentiment (-1 = très négatif, +1 = très positif)")
plt.xlabel("Score de sentiment")
plt.ylabel("Nombre de publications")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/E1_histogramme_sentiment.png")
plt.close()

repartition_a_afficher = repartition_sentiment.drop("indéterminé", errors="ignore")
plt.figure(figsize=(7, 7))
plt.pie(repartition_a_afficher.values, labels=repartition_a_afficher.index, autopct="%1.1f%%",
        colors=[couleurs_sentiment[s] for s in repartition_a_afficher.index])
plt.title("Répartition Positif / Neutre / Négatif")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/E2_camembert_sentiment.png")
plt.close()

print("Enregistré : E1 (tableau + histogramme) et E2 (camembert).")


# ==============================================================================
# SECTION F — ÉVOLUTION QUOTIDIENNE GLOBALE (autour du vote)
# ==============================================================================

print("\n--- F : évolution quotidienne du sentiment ---")

df["jour"] = df["date_publication"].dt.date
sentiment_moyen_par_jour = df.groupby("jour")["score_sentiment"].mean()
sentiment_moyen_par_jour.to_csv(f"{dossier_tableaux}/F1_sentiment_moyen_par_jour.csv")

fenetre_debut = date_du_vote - pd.Timedelta(days=60)
fenetre_fin = date_du_vote + pd.Timedelta(days=60)
masque_fenetre = (df["date_publication"] >= fenetre_debut) & (df["date_publication"] <= fenetre_fin)
sentiment_zoome = df[masque_fenetre].groupby("jour")["score_sentiment"].mean()
moyenne_mobile_sentiment = sentiment_zoome.rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(14, 6))
plt.plot(sentiment_zoome.index, sentiment_zoome.values, color="lightsteelblue", alpha=0.6, label="Sentiment moyen par jour (brut)")
plt.plot(moyenne_mobile_sentiment.index, moyenne_mobile_sentiment.values, color="#1d3557", linewidth=2.5, label="Moyenne mobile (7 jours)")
plt.axhline(0, color="gray", linestyle=":", linewidth=1)
plt.axvline(date_du_vote.date(), color="red", linestyle="--", label="Jour du vote (12/10/2025)")
plt.title("Évolution du sentiment moyen : 60 jours avant/après le vote")
plt.xlabel("Date")
plt.ylabel("Score de sentiment moyen")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/F1_evolution_sentiment_quotidien.png")
plt.close()

print("Enregistré : F1 (tableau + graphique).")


# ==============================================================================
# SECTION G — ÉVOLUTION HEBDOMADAIRE EN PROPORTIONS (AIRE EMPILÉE)
# ==============================================================================
# Plutôt qu'un simple score moyen, on regarde ici comment les PROPORTIONS de
# publications positives/neutres/négatives évoluent semaine après semaine.
# Une "aire empilée" (stacked area chart) montre bien comment la composition
# du corpus se transforme dans le temps, pas juste sa moyenne.

print("\n--- G : évolution hebdomadaire des proportions (aire empilée) ---")

df["semaine"] = df["date_publication"].dt.to_period("W").dt.start_time

proportions_hebdo = pd.crosstab(df["semaine"], df["sentiment"], normalize="index") * 100
for colonne in ["Positif", "Neutre", "Négatif"]:
    if colonne not in proportions_hebdo.columns:
        proportions_hebdo[colonne] = 0
proportions_hebdo = proportions_hebdo[["Positif", "Neutre", "Négatif"]]
proportions_hebdo.to_csv(f"{dossier_tableaux}/G1_proportions_hebdomadaires_sentiment.csv")

plt.figure(figsize=(14, 6))
plt.stackplot(
    proportions_hebdo.index,
    proportions_hebdo["Positif"], proportions_hebdo["Neutre"], proportions_hebdo["Négatif"],
    labels=["Positif", "Neutre", "Négatif"],
    colors=[couleurs_sentiment["Positif"], couleurs_sentiment["Neutre"], couleurs_sentiment["Négatif"]],
    alpha=0.85,
)
plt.axvline(date_du_vote, color="black", linestyle="--", label="Jour du vote")
plt.title("Évolution hebdomadaire de la composition du sentiment (%)")
plt.xlabel("Semaine")
plt.ylabel("Pourcentage de publications")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/G1_aire_empilee_sentiment_hebdo.png")
plt.close()

print("Enregistré : G1 (tableau + aire empilée).")


# ==============================================================================
# SECTION H — ÉVOLUTION TEMPORELLE PAR PLATEFORME (courbes superposées)
# ==============================================================================
# C'est ici la vraie "évolution par plateforme" prévue par le canevas :
# on regarde, semaine après semaine, si Facebook/TikTok/X évoluent PAREIL
# ou DIFFÉREMMENT dans le temps.

print("\n--- H : évolution temporelle du sentiment par plateforme ---")

sentiment_hebdo_plateforme = df.groupby(["semaine", "plateforme"])["score_sentiment"].mean().unstack()
sentiment_hebdo_plateforme.to_csv(f"{dossier_tableaux}/H1_evolution_sentiment_plateforme.csv")

couleurs_plateformes_lignes = {"Facebook": "#1877F2", "TikTok": "#EE1D52", "X": "#000000"}

plt.figure(figsize=(14, 6))
for plateforme in sentiment_hebdo_plateforme.columns:
    plt.plot(
        sentiment_hebdo_plateforme.index, sentiment_hebdo_plateforme[plateforme],
        label=plateforme, color=couleurs_plateformes_lignes.get(plateforme), linewidth=2, marker="o", markersize=3,
    )
plt.axhline(0, color="gray", linestyle=":", linewidth=1)
plt.axvline(date_du_vote, color="red", linestyle="--", label="Jour du vote")
plt.title("Évolution du sentiment moyen par semaine, par plateforme")
plt.xlabel("Semaine")
plt.ylabel("Score de sentiment moyen")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/H1_evolution_sentiment_plateforme.png")
plt.close()

print("Enregistré : H1 (tableau + graphique).")


# ==============================================================================
# SECTION I — ÉVOLUTION TEMPORELLE PAR THÈME DE CRISE (courbes superposées)
# ==============================================================================

print("\n--- I : évolution temporelle du sentiment par thème de crise ---")

evenements_de_crise = {
    "Villes mortes": ["ville morte", "villes mortes", "ghost town"],
    "Manifestations": ["manifestation", "manifestant"],
    "Violence": ["violence", "violent"],
    "Fraude électorale": ["fraude"],
}  # on limite à 4 thèmes pour garder un graphique lisible

plt.figure(figsize=(14, 6))
couleurs_themes_lignes = cm.Set1(np.linspace(0, 0.8, len(evenements_de_crise)))
tableau_evolution_themes = pd.DataFrame(index=sorted(df["semaine"].unique()))

for (theme, mots_cles), couleur in zip(evenements_de_crise.items(), couleurs_themes_lignes):
    motif = "|".join(mots_cles)
    masque = df["texte_nettoye"].str.contains(motif, regex=True, na=False)
    evolution_theme = df[masque].groupby("semaine")["score_sentiment"].mean()
    tableau_evolution_themes[theme] = evolution_theme
    plt.plot(evolution_theme.index, evolution_theme.values, label=theme, color=couleur, linewidth=2, marker="o", markersize=3)

plt.axhline(0, color="gray", linestyle=":", linewidth=1)
plt.axvline(date_du_vote, color="black", linestyle="--", label="Jour du vote")
plt.title("Évolution du sentiment moyen par semaine, par thème de crise")
plt.xlabel("Semaine")
plt.ylabel("Score de sentiment moyen")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/I1_evolution_sentiment_themes.png")
plt.close()

tableau_evolution_themes.to_csv(f"{dossier_tableaux}/I1_evolution_sentiment_themes.csv")
print("Enregistré : I1 (tableau + graphique).")


# ==============================================================================
# SECTION J — ÉVOLUTION TEMPORELLE PAR ORIGINE GÉOGRAPHIQUE (courbes superposées)
# ==============================================================================

print("\n--- J : évolution temporelle du sentiment par origine géographique ---")

# On regroupe les origines avec peu de publications dans "Autre" pour ne pas
# surcharger le graphique de lignes basées sur très peu de données
origines_principales = ["Cameroun", "indéterminé (anglophone)", "Autre (diaspora)", "France"]
df["origine_simplifiee"] = df["pays_auteur"].where(df["pays_auteur"].isin(origines_principales), "Autre")

sentiment_hebdo_origine = df.groupby(["semaine", "origine_simplifiee"])["score_sentiment"].mean().unstack()
sentiment_hebdo_origine.to_csv(f"{dossier_tableaux}/J1_evolution_sentiment_origine.csv")

couleurs_origine_lignes = cm.Dark2(np.linspace(0, 1, len(sentiment_hebdo_origine.columns)))

plt.figure(figsize=(14, 6))
for origine, couleur in zip(sentiment_hebdo_origine.columns, couleurs_origine_lignes):
    plt.plot(sentiment_hebdo_origine.index, sentiment_hebdo_origine[origine], label=origine, color=couleur, linewidth=2, marker="o", markersize=3)
plt.axhline(0, color="gray", linestyle=":", linewidth=1)
plt.axvline(date_du_vote, color="red", linestyle="--", label="Jour du vote")
plt.title("Évolution du sentiment moyen par semaine, par origine de l'auteur")
plt.xlabel("Semaine")
plt.ylabel("Score de sentiment moyen")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/J1_evolution_sentiment_origine.png")
plt.close()

print("Enregistré : J1 (tableau + graphique).")


# ==============================================================================
# SECTION K — DISTRIBUTION DU SENTIMENT PAR PLATEFORME (BOXPLOT)
# ==============================================================================
# Un score moyen cache beaucoup de choses : deux plateformes peuvent avoir la
# même moyenne mais une dispersion très différente (l'une avec des avis
# plutôt tièdes, l'autre avec des avis très tranchés dans les deux sens). Le
# boxplot montre la médiane, les quartiles, et les valeurs extrêmes.

print("\n--- K : distribution du sentiment par plateforme (boxplot) ---")

plateformes_liste = df["plateforme"].unique().tolist()
donnees_boxplot = [df.loc[df["plateforme"] == p, "score_sentiment"].dropna() for p in plateformes_liste]

plt.figure(figsize=(9, 6))
boites = plt.boxplot(donnees_boxplot, tick_labels=plateformes_liste, patch_artist=True, showmeans=True)
couleurs_boxplot = ["#1877F2", "#EE1D52", "#6c757d"]
for boite, couleur in zip(boites["boxes"], couleurs_boxplot):
    boite.set_facecolor(couleur)
    boite.set_alpha(0.6)
plt.axhline(0, color="black", linestyle=":", linewidth=1)
plt.title("Distribution du score de sentiment par plateforme")
plt.ylabel("Score de sentiment")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/K1_boxplot_sentiment_plateforme.png")
plt.close()

sentiment_par_plateforme_stats = df.groupby("plateforme")["score_sentiment"].describe().round(3)
sentiment_par_plateforme_stats.to_csv(f"{dossier_tableaux}/K1_stats_sentiment_plateforme.csv")

print("Enregistré : K1 (tableau + boxplot).")


# ==============================================================================
# SECTION L — SENTIMENT PAR THÈME DE CRISE (BARRES DIVERGENTES)
# ==============================================================================

print("\n--- L : sentiment moyen par thème (barres divergentes) ---")

evenements_de_crise_complets = {
    "Villes mortes": ["ville morte", "villes mortes", "ghost town"],
    "Manifestations": ["manifestation", "manifestant"],
    "Violence": ["violence", "violent"],
    "Morts / tués": ["tué", "tue ", " mort ", "morts", "décès"],
    "Arrestations": ["arrestation", "arrêté", "arrete"],
    "Répression": ["répression", "repression"],
    "Fraude électorale": ["fraude"],
    "Forces de l'ordre": ["armée", "armee", "police", "gendarme", "militaire"],
}

sentiment_par_theme = {}
for theme, mots_cles in evenements_de_crise_complets.items():
    motif = "|".join(mots_cles)
    masque = df["texte_nettoye"].str.contains(motif, regex=True, na=False)
    sentiment_par_theme[theme] = df.loc[masque, "score_sentiment"].mean()

moyenne_globale = df["score_sentiment"].mean()
sentiment_par_theme["Ensemble du corpus (référence)"] = moyenne_globale

serie_sentiment_theme = pd.Series(sentiment_par_theme).sort_values()
serie_sentiment_theme.to_csv(f"{dossier_tableaux}/L1_sentiment_par_theme.csv")

couleurs_theme = ["#E63946" if v < 0 else "#2A9D8F" for v in serie_sentiment_theme.values]
plt.figure(figsize=(9, 6))
plt.barh(serie_sentiment_theme.index, serie_sentiment_theme.values, color=couleurs_theme)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Sentiment moyen selon le thème évoqué (barres divergentes)")
plt.xlabel("Score de sentiment moyen")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/L1_sentiment_par_theme.png")
plt.close()

print("Enregistré : L1 (tableau + graphique).")


# ==============================================================================
# SECTION M — SENTIMENT PAR ACTEUR POLITIQUE, AVANT/APRÈS LE VOTE (BARRES GROUPÉES)
# ==============================================================================
# On regarde si le sentiment envers chaque acteur a CHANGÉ après le vote —
# ce qui est beaucoup plus parlant qu'un simple score global par acteur.

print("\n--- M : sentiment par acteur politique, avant/après le vote ---")

acteurs_politiques = {
    "Paul Biya": ["biya"],
    "Issa Tchiroma": ["tchiroma"],
    "Cabral Libii": ["cabral", "libii", "libie"],
    "Maurice Kamto": ["kamto"],
    "Bello Bouba Maigari": ["bello bouba", "maigari"],
    "Patricia Ndam Njoya": ["ndam njoya", "ndam"],
}

resultats_acteurs_periode = {}
for acteur, mots_cles in acteurs_politiques.items():
    motif = "|".join(mots_cles)
    masque = df["texte_nettoye"].str.contains(motif, regex=True, na=False)
    score_avant = df.loc[masque & (df["periode"] == "avant le vote"), "score_sentiment"].mean()
    score_apres = df.loc[masque & (df["periode"] == "après le vote"), "score_sentiment"].mean()
    resultats_acteurs_periode[acteur] = {"avant le vote": score_avant, "après le vote": score_apres}

tableau_acteurs_periode = pd.DataFrame(resultats_acteurs_periode).T
tableau_acteurs_periode.to_csv(f"{dossier_tableaux}/M1_sentiment_acteurs_avant_apres.csv")
print(tableau_acteurs_periode)

positions = np.arange(len(tableau_acteurs_periode))
largeur_barre = 0.35

plt.figure(figsize=(10, 6))
plt.bar(positions - largeur_barre / 2, tableau_acteurs_periode["avant le vote"], width=largeur_barre, label="Avant le vote", color="#4C9A2A")
plt.bar(positions + largeur_barre / 2, tableau_acteurs_periode["après le vote"], width=largeur_barre, label="Après le vote", color="#B22222")
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(positions, tableau_acteurs_periode.index, rotation=25, ha="right")
plt.ylabel("Score de sentiment moyen")
plt.title("Sentiment envers chaque acteur politique, avant vs après le vote")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/M1_sentiment_acteurs_avant_apres.png")
plt.close()

print("Enregistré : M1 (tableau + graphique).")


# ==============================================================================
# SECTION N — MOTIF TEMPOREL JOUR x HEURE (HEATMAP)
# ==============================================================================
# On regarde s'il y a un motif régulier : est-ce que les publications sont
# plus négatives à certaines heures de la journée, ou certains jours de la
# semaine (par exemple le week-end, ou tard le soir) ?

print("\n--- N : motif temporel du sentiment (jour x heure) ---")

df["heure"] = df["date_publication"].dt.hour
df["jour_semaine"] = df["date_publication"].dt.day_name()

ordre_jours = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
noms_jours_fr = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

tableau_heatmap = df.groupby(["jour_semaine", "heure"])["score_sentiment"].mean().unstack()
tableau_heatmap = tableau_heatmap.reindex(ordre_jours)
tableau_heatmap.index = noms_jours_fr
tableau_heatmap.to_csv(f"{dossier_tableaux}/N1_heatmap_sentiment_jour_heure.csv")

plt.figure(figsize=(14, 6))
carte = plt.imshow(tableau_heatmap.values, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.3)
plt.colorbar(carte, label="Score de sentiment moyen")
plt.yticks(range(len(tableau_heatmap.index)), tableau_heatmap.index)
plt.xticks(range(24), range(24))
plt.xlabel("Heure de la journée")
plt.title("Sentiment moyen selon le jour de la semaine et l'heure de publication")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/N1_heatmap_sentiment_jour_heure.png")
plt.close()

print("Enregistré : N1 (tableau + heatmap).")


# ==============================================================================
# SECTION O — SENTIMENT x ENGAGEMENT (BULLES)
# ==============================================================================
# Question importante pour un système d'aide à la décision : est-ce que les
# publications les plus NÉGATIVES sont aussi celles qui suscitent le plus de
# réactions (le phénomène bien documenté "l'indignation fait plus réagir") ?
# On regroupe les scores de sentiment par tranche de 0.1, et pour chaque
# tranche on regarde l'engagement moyen et le nombre de publications
# concernées (taille de la bulle).

print("\n--- O : sentiment x engagement (bulles) ---")

df["tranche_sentiment"] = (df["score_sentiment"] / 0.1).round() * 0.1
tableau_bulles_sentiment = df.groupby("tranche_sentiment").agg(
    engagement_moyen=("nb_likes", "mean"),
    nb_publications=("nb_likes", "count"),
).dropna()
tableau_bulles_sentiment.to_csv(f"{dossier_tableaux}/O1_sentiment_x_engagement.csv")

plt.figure(figsize=(11, 7))
couleurs_bulles = ["#E63946" if x < 0 else ("#E9C46A" if x == 0 else "#2A9D8F") for x in tableau_bulles_sentiment.index]
plt.scatter(
    tableau_bulles_sentiment.index, tableau_bulles_sentiment["engagement_moyen"],
    s=tableau_bulles_sentiment["nb_publications"] / 3, color=couleurs_bulles, alpha=0.7, edgecolors="black",
)
plt.axvline(0, color="gray", linestyle=":", linewidth=1)
plt.xlabel("Tranche de score de sentiment")
plt.ylabel("Nombre moyen de likes")
plt.title("Engagement moyen selon la tranche de sentiment (taille = nb de publications)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/O1_bulles_sentiment_engagement.png")
plt.close()

print("Enregistré : O1 (tableau + bulles).")


# ==============================================================================
# SECTION P — COMPARAISON DES PLATEFORMES SUR PLUSIEURS INDICATEURS (RADAR)
# ==============================================================================
# On compare les 3 plateformes sur 4 indicateurs à la fois, sur un même
# graphique "radar" (en toile d'araignée). Chaque indicateur est ramené
# entre 0 et 1 pour être comparable sur les mêmes axes.

print("\n--- P : comparaison des plateformes sur plusieurs indicateurs (radar) ---")

indicateurs_radar = {}
for plateforme in plateformes_liste:
    sous_ensemble = df[df["plateforme"] == plateforme]
    indicateurs_radar[plateforme] = {
        "Score moyen": sous_ensemble["score_sentiment"].mean(),
        "% Positif": (sous_ensemble["sentiment"] == "Positif").mean() * 100,
        "% Négatif": (sous_ensemble["sentiment"] == "Négatif").mean() * 100,
        "Volatilité (écart-type)": sous_ensemble["score_sentiment"].std(),
    }

tableau_radar = pd.DataFrame(indicateurs_radar).T
tableau_radar.to_csv(f"{dossier_tableaux}/P1_indicateurs_radar_plateformes.csv")

# On normalise chaque colonne entre 0 et 1 pour que les indicateurs (qui ont
# des échelles très différentes : un score entre -1/1, un pourcentage entre
# 0/100) soient comparables sur le même graphique
tableau_radar_normalise = (tableau_radar - tableau_radar.min()) / (tableau_radar.max() - tableau_radar.min() + 1e-9)

categories = list(tableau_radar_normalise.columns)
nb_categories = len(categories)
angles = np.linspace(0, 2 * np.pi, nb_categories, endpoint=False).tolist()
angles += angles[:1]  # on referme le cercle

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
couleurs_radar = {"Facebook": "#1877F2", "TikTok": "#EE1D52", "X": "#6c757d"}

for plateforme in tableau_radar_normalise.index:
    valeurs = tableau_radar_normalise.loc[plateforme].tolist()
    valeurs += valeurs[:1]
    ax.plot(angles, valeurs, label=plateforme, color=couleurs_radar.get(plateforme), linewidth=2)
    ax.fill(angles, valeurs, color=couleurs_radar.get(plateforme), alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title("Comparaison des plateformes (indicateurs normalisés)")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(f"{dossier_figures}/P1_radar_plateformes.png")
plt.close()

print("Enregistré : P1 (tableau + radar).")


# ==============================================================================
# SECTION Q — VERBATIMS : PUBLICATIONS LES PLUS POSITIVES / LES PLUS NÉGATIVES
# ==============================================================================
# Ces exemples concrets sont très utiles pour illustrer le chapitre III du
# mémoire avec de vrais verbatims, pas seulement des chiffres.

print("\n--- Q : publications les plus positives et les plus négatives ---")

colonnes_verbatims = ["plateforme", "date_publication", "texte", "score_sentiment", "nb_likes"]

top_10_positives = df.dropna(subset=["score_sentiment"]).sort_values("score_sentiment", ascending=False)[colonnes_verbatims].head(10)
top_10_negatives = df.dropna(subset=["score_sentiment"]).sort_values("score_sentiment", ascending=True)[colonnes_verbatims].head(10)

top_10_positives.to_csv(f"{dossier_tableaux}/Q1_top_10_publications_positives.csv", index=False)
top_10_negatives.to_csv(f"{dossier_tableaux}/Q2_top_10_publications_negatives.csv", index=False)

print("Enregistré : Q1 (top positives) et Q2 (top négatives).")


# ==============================================================================
# SECTION R — SAUVEGARDE DU CORPUS AVEC SENTIMENT
# ==============================================================================

print("\n--- R : sauvegarde du corpus enrichi avec le sentiment ---")

chemin_sortie = f"{dossier_data}/bd_f_sentiment.csv"
df.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")
print(f"Corpus sauvegardé ici : {chemin_sortie}")


# ==============================================================================
# FIN DU SCRIPT
# ==============================================================================

print("\n=== TERMINÉ ! ===")
print(f"Tous les graphiques sont dans : {dossier_figures}")
print(f"Tous les tableaux sont dans   : {dossier_tableaux}")
print("\nRécapitulatif :")
print("  A-D : préparation et calcul du score de sentiment")
print("  E : distribution générale (histogramme + camembert)")
print("  F : évolution quotidienne globale")
print("  G : évolution hebdomadaire en proportions (aire empilée)")
print("  H : évolution temporelle PAR PLATEFORME (courbes)")
print("  I : évolution temporelle PAR THÈME (courbes)")
print("  J : évolution temporelle PAR ORIGINE (courbes)")
print("  K : distribution par plateforme (boxplot)")
print("  L : sentiment par thème (barres divergentes)")
print("  M : sentiment par acteur, avant/après (barres groupées)")
print("  N : motif jour x heure (heatmap)")
print("  O : sentiment x engagement (bulles)")
print("  P : comparaison plateformes (radar)")
print("  Q : verbatims positifs/négatifs")
print(f"  R : corpus final sauvegardé -> {chemin_sortie}")
