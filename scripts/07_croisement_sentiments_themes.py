# ==============================================================================
# PHASE 8 — CROISEMENT SENTIMENTS x THÉMATIQUES
# ==============================================================================
#
# À quoi sert ce script ?
# ------------------------
# C'est la phase la plus directement liée à H2 : "l'analyse combinée des
# sentiments et des thématiques permet une compréhension plus fine du climat
# social". On a déjà vu, phase par phase, le sentiment (Phase 5), les
# émotions (Phase 6) et les thèmes (Phase 7) séparément. Ici, on les CROISE
# vraiment, en répondant aux 5 questions concrètes de ton canevas :
#
#   1. Quels sujets provoquent le plus de colère (et les autres émotions) ?
#   2. Quels sujets deviennent positifs (ou se dégradent) dans le temps ?
#   3. Quels sujets disparaissent (ou émergent) dans le temps ?
#   4. Quels sujets deviennent viraux (engagement croissant) ?
#   5. Quels sujets génèrent le plus de commentaires ?
#
# On termine par un tableau de synthèse qui rassemble tout, et par un
# premier "indice de tension thématique" qui préfigure les indicateurs
# décisionnels de la Phase 9 (ITE, ITR...).
#
# On se concentre sur le sous-corpus FRANÇAIS (5 thèmes propres, sans bruit),
# qui est le plus riche et le plus fiable pour ce genre d'analyse fine. Les
# thèmes anglais/pidgin sont traités plus brièvement en fin de script.
# ==============================================================================


# ------------------------------------------------------------------------------
# ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

dossier_data = "../data"
dossier_figures = "../outputs/figures"
dossier_tableaux = "../outputs/tables"
os.makedirs(dossier_figures, exist_ok=True)
os.makedirs(dossier_tableaux, exist_ok=True)


# ------------------------------------------------------------------------------
# ÉTAPE 1 : CHARGER LE CORPUS (Phase 7, avec sentiment + émotions + thèmes)
# ------------------------------------------------------------------------------

print("Lecture du corpus (avec sentiment + émotions + thèmes déjà calculés)...")
df = pd.read_csv(f"{dossier_data}/bd_f_themes.csv", low_memory=False)
df["nb_likes"] = pd.to_numeric(df["nb_likes"], errors="coerce")
df["nb_reponses"] = pd.to_numeric(df["nb_reponses"], errors="coerce")
print(f"{len(df)} publications chargées.")

df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
date_du_vote = pd.Timestamp("2025-10-12", tz="UTC")
df["periode"] = "après le vote"
df.loc[df["date_publication"] < date_du_vote, "periode"] = "avant le vote"
df["semaine"] = df["date_publication"].dt.to_period("W").dt.start_time

# On se concentre sur les publications françaises qui ont un thème identifié
# (on exclut "non déterminé", qui correspond aux langues non prises en charge)
themes_francais = [
    "Tribalisme, régime et tensions régionales (Ouest)",
    "Solidarité, foi et encouragements au peuple",
    "Candidats de l'opposition et espoir de changement",
    "Communauté Bamiléké, famille et problèmes du pays",
    "Paul Biya : pouvoir, vie et mort",
]
df_themes = df[df["theme_nom"].isin(themes_francais)].copy()
print(f"{len(df_themes)} publications avec un thème français identifié.")

colonnes_emotions = [c for c in df.columns if c.startswith("emotion_") and c != "emotion_dominante"]
noms_emotions = [c.replace("emotion_", "") for c in colonnes_emotions]

# Une couleur FIXE par thème, réutilisée partout dans ce script
couleurs_liste = cm.tab10(np.linspace(0, 1, len(themes_francais)))
COULEURS_THEMES = {theme: couleurs_liste[i] for i, theme in enumerate(themes_francais)}


# ==============================================================================
# QUESTION 1 — QUELS SUJETS PROVOQUENT LE PLUS DE COLÈRE (ET LES AUTRES ÉMOTIONS) ?
# ==============================================================================

print("\n--- Question 1 : quels sujets provoquent le plus quelles émotions ? ---")

profil_emotionnel_themes = df_themes.groupby("theme_nom")[colonnes_emotions].mean()
profil_emotionnel_themes.columns = noms_emotions
profil_emotionnel_themes.to_csv(f"{dossier_tableaux}/Q1_profil_emotionnel_themes.csv")

# Pour CHAQUE émotion, on identifie le thème qui la déclenche le plus
theme_qui_domine_chaque_emotion = profil_emotionnel_themes.idxmax(axis=0)
score_de_cette_domination = profil_emotionnel_themes.max(axis=0)

tableau_dominance = pd.DataFrame({
    "emotion": noms_emotions,
    "theme_dominant": theme_qui_domine_chaque_emotion.values,
    "intensite": score_de_cette_domination.values.round(3),
})
tableau_dominance.to_csv(f"{dossier_tableaux}/Q1_theme_dominant_par_emotion.csv", index=False)
print(tableau_dominance.to_string(index=False))

# Graphique : barres groupées, une couleur par thème, une émotion par groupe
fig, ax = plt.subplots(figsize=(13, 7))
largeur_barre = 0.15
positions_de_base = np.arange(len(noms_emotions))

for i, theme in enumerate(themes_francais):
    valeurs = profil_emotionnel_themes.loc[theme]
    ax.bar(positions_de_base + i * largeur_barre, valeurs, width=largeur_barre,
           label=theme, color=COULEURS_THEMES[theme])

ax.set_xticks(positions_de_base + largeur_barre * (len(themes_francais) - 1) / 2)
ax.set_xticklabels(noms_emotions)
ax.set_ylabel("Intensité moyenne de l'émotion")
ax.set_title("Intensité de chaque émotion selon le thème de la publication")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{dossier_figures}/Q1_emotions_par_theme_barres_groupees.png")
plt.close()

print("Enregistré : Q1 (tableaux + graphique en barres groupées).")


# ==============================================================================
# QUESTION 2 — QUELS SUJETS DEVIENNENT PLUS POSITIFS OU PLUS NÉGATIFS ?
# ==============================================================================
# On regarde l'évolution du sentiment moyen, semaine après semaine, POUR
# CHAQUE THÈME séparément. Puis on résume cette évolution par une simple
# PENTE (régression linéaire) : une pente positive = le thème devient plus
# positif avec le temps ; une pente négative = il se dégrade.

print("\n--- Question 2 : quels sujets deviennent plus positifs ou plus négatifs ? ---")

evolution_sentiment_par_theme = df_themes.groupby(["semaine", "theme_nom"])["score_sentiment"].mean().unstack()
evolution_sentiment_par_theme.to_csv(f"{dossier_tableaux}/Q2_evolution_sentiment_par_theme.csv")

plt.figure(figsize=(15, 7))
for theme in themes_francais:
    if theme in evolution_sentiment_par_theme.columns:
        lissage = evolution_sentiment_par_theme[theme].rolling(window=3, min_periods=1).mean()
        plt.plot(evolution_sentiment_par_theme.index, lissage, label=theme, color=COULEURS_THEMES[theme], linewidth=2)
plt.axhline(0, color="gray", linestyle=":", linewidth=1)
plt.axvline(date_du_vote, color="black", linestyle="--", label="Jour du vote")
plt.title("Évolution du sentiment moyen (lissé sur 3 semaines), par thème")
plt.xlabel("Semaine")
plt.ylabel("Score de sentiment moyen")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{dossier_figures}/Q2_evolution_sentiment_par_theme.png")
plt.close()


def calculer_tendance(serie_dans_le_temps):
    """
    Calcule la pente d'une droite de régression simple sur une série
    temporelle. On utilise numpy.polyfit, qui trouve la droite qui "colle"
    le mieux aux points. Une pente positive = tendance à la hausse dans le
    temps ; une pente négative = tendance à la baisse.
    """
    serie_sans_trous = serie_dans_le_temps.dropna()
    if len(serie_sans_trous) < 3:
        return np.nan
    x = np.arange(len(serie_sans_trous))
    pente, ordonnee_origine = np.polyfit(x, serie_sans_trous.values, deg=1)
    return pente


tendances_sentiment = {}
for theme in themes_francais:
    if theme in evolution_sentiment_par_theme.columns:
        tendances_sentiment[theme] = calculer_tendance(evolution_sentiment_par_theme[theme])

serie_tendances_sentiment = pd.Series(tendances_sentiment).sort_values()
serie_tendances_sentiment.to_csv(f"{dossier_tableaux}/Q2_tendance_sentiment_par_theme.csv")
print("Tendance du sentiment (pente) par thème (négatif = se dégrade, positif = s'améliore) :")
print(serie_tendances_sentiment.round(5))

couleurs_tendance = ["#E63946" if v < 0 else "#2A9D8F" for v in serie_tendances_sentiment.values]
plt.figure(figsize=(10, 6))
plt.barh(serie_tendances_sentiment.index, serie_tendances_sentiment.values, color=couleurs_tendance)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Tendance d'évolution du sentiment par thème (pente sur la période)")
plt.xlabel("Pente (positif = s'améliore, négatif = se dégrade)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/Q2_tendance_sentiment_par_theme.png")
plt.close()

print("Enregistré : Q2 (évolution détaillée + tendances résumées).")


# ==============================================================================
# QUESTION 3 — QUELS SUJETS DISPARAISSENT (OU ÉMERGENT) DANS LE TEMPS ?
# ==============================================================================
# Même principe que la question 2, mais sur la PART (%) de chaque thème dans
# le total des publications de la semaine, plutôt que sur le sentiment.

print("\n--- Question 3 : quels sujets disparaissent ou émergent dans le temps ? ---")

nb_par_semaine_theme = df_themes.groupby(["semaine", "theme_nom"]).size().unstack(fill_value=0)
part_par_semaine_theme = nb_par_semaine_theme.div(nb_par_semaine_theme.sum(axis=1), axis=0) * 100
part_par_semaine_theme.to_csv(f"{dossier_tableaux}/Q3_evolution_part_themes.csv")

tendances_prevalence = {}
for theme in themes_francais:
    if theme in part_par_semaine_theme.columns:
        tendances_prevalence[theme] = calculer_tendance(part_par_semaine_theme[theme])

serie_tendances_prevalence = pd.Series(tendances_prevalence).sort_values()
serie_tendances_prevalence.to_csv(f"{dossier_tableaux}/Q3_tendance_prevalence_par_theme.csv")
print("Tendance de la PART de chaque thème (négatif = perd du terrain, positif = émerge) :")
print(serie_tendances_prevalence.round(4))

couleurs_prevalence = ["#B22222" if v < 0 else "#4C9A2A" for v in serie_tendances_prevalence.values]
plt.figure(figsize=(10, 6))
plt.barh(serie_tendances_prevalence.index, serie_tendances_prevalence.values, color=couleurs_prevalence)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Tendance d'évolution du POIDS de chaque thème dans le débat (pente)")
plt.xlabel("Pente (positif = le thème gagne du terrain, négatif = il en perd)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/Q3_tendance_prevalence_par_theme.png")
plt.close()

print("Enregistré : Q3 (évolution + tendances de la part de chaque thème).")


# ==============================================================================
# QUESTION 4 — QUELS SUJETS DEVIENNENT VIRAUX (ENGAGEMENT CROISSANT) ?
# ==============================================================================
# On regarde l'évolution de l'engagement moyen (likes) par thème, semaine
# après semaine, et on calcule aussi sa tendance.

print("\n--- Question 4 : quels sujets deviennent viraux (engagement croissant) ? ---")

evolution_engagement_par_theme = df_themes.groupby(["semaine", "theme_nom"])["nb_likes"].mean().unstack()
evolution_engagement_par_theme.to_csv(f"{dossier_tableaux}/Q4_evolution_engagement_par_theme.csv")

plt.figure(figsize=(15, 7))
for theme in themes_francais:
    if theme in evolution_engagement_par_theme.columns:
        lissage = evolution_engagement_par_theme[theme].rolling(window=3, min_periods=1).mean()
        plt.plot(evolution_engagement_par_theme.index, lissage, label=theme, color=COULEURS_THEMES[theme], linewidth=2)
plt.axvline(date_du_vote, color="black", linestyle="--", label="Jour du vote")
plt.title("Évolution de l'engagement moyen (likes, lissé sur 3 semaines), par thème")
plt.xlabel("Semaine")
plt.ylabel("Nombre moyen de likes")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{dossier_figures}/Q4_evolution_engagement_par_theme.png")
plt.close()

tendances_engagement = {}
for theme in themes_francais:
    if theme in evolution_engagement_par_theme.columns:
        tendances_engagement[theme] = calculer_tendance(evolution_engagement_par_theme[theme])

serie_tendances_engagement = pd.Series(tendances_engagement).sort_values(ascending=False)
serie_tendances_engagement.to_csv(f"{dossier_tableaux}/Q4_tendance_engagement_par_theme.csv")
print("Tendance de l'engagement (pente) par thème (positif = devient plus viral) :")
print(serie_tendances_engagement.round(3))

# On compare aussi simplement l'engagement moyen avant/après le vote, plus
# facile à lire qu'une pente pour un rapport de mémoire
engagement_avant_apres = df_themes.groupby(["theme_nom", "periode"])["nb_likes"].mean().unstack().round(1)
engagement_avant_apres["variation (%)"] = ((engagement_avant_apres["après le vote"] - engagement_avant_apres["avant le vote"])
                                            / engagement_avant_apres["avant le vote"] * 100).round(1)
engagement_avant_apres = engagement_avant_apres.sort_values("variation (%)", ascending=False)
engagement_avant_apres.to_csv(f"{dossier_tableaux}/Q4_engagement_avant_apres.csv")
print("\nEngagement moyen avant/après le vote, et variation :")
print(engagement_avant_apres)

plt.figure(figsize=(10, 6))
couleurs_variation = ["#2A9D8F" if v > 0 else "#E63946" for v in engagement_avant_apres["variation (%)"]]
plt.barh(engagement_avant_apres.index, engagement_avant_apres["variation (%)"], color=couleurs_variation)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Variation de l'engagement moyen (%) entre avant et après le vote, par thème")
plt.xlabel("Variation en %")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/Q4_variation_engagement_avant_apres.png")
plt.close()

print("Enregistré : Q4 (évolution, tendances et variation avant/après).")


# ==============================================================================
# QUESTION 5 — QUELS SUJETS GÉNÈRENT LE PLUS DE COMMENTAIRES ?
# ==============================================================================
# Les likes mesurent l'approbation ; les commentaires (réponses) mesurent
# plutôt le DÉBAT et la controverse suscités par un sujet. Un thème avec
# peu de likes mais beaucoup de commentaires est un sujet qui divise.

print("\n--- Question 5 : quels sujets génèrent le plus de commentaires ? ---")

commentaires_par_theme = df_themes.groupby("theme_nom")["nb_reponses"].agg(["mean", "median", "sum"]).round(2)
commentaires_par_theme = commentaires_par_theme.sort_values("mean", ascending=False)
commentaires_par_theme.to_csv(f"{dossier_tableaux}/Q5_commentaires_par_theme.csv")
print(commentaires_par_theme)

couleurs_commentaires = [COULEURS_THEMES[theme] for theme in commentaires_par_theme.index]
plt.figure(figsize=(10, 6))
plt.bar(range(len(commentaires_par_theme)), commentaires_par_theme["mean"], color=couleurs_commentaires)
plt.xticks(range(len(commentaires_par_theme)), commentaires_par_theme.index, rotation=25, ha="right")
plt.title("Nombre moyen de commentaires (réponses) selon le thème")
plt.ylabel("Nombre moyen de commentaires")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/Q5_commentaires_par_theme.png")
plt.close()

# Un graphique bulles très parlant : likes (X) vs commentaires (Y) par thème,
# pour voir d'un coup d'œil les sujets "approuvés" vs les sujets "qui divisent"
likes_moyens_par_theme = df_themes.groupby("theme_nom")["nb_likes"].mean()

plt.figure(figsize=(10, 8))
for theme in themes_francais:
    plt.scatter(likes_moyens_par_theme[theme], commentaires_par_theme.loc[theme, "mean"],
                s=300, color=COULEURS_THEMES[theme], edgecolors="black", alpha=0.8, label=theme)
plt.xlabel("Nombre moyen de likes (approbation)")
plt.ylabel("Nombre moyen de commentaires (débat/controverse)")
plt.title("Approbation vs débat suscité, par thème")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{dossier_figures}/Q5_likes_vs_commentaires_par_theme.png")
plt.close()

print("Enregistré : Q5 (tableau + 2 graphiques).")


# ==============================================================================
# SYNTHÈSE — TABLEAU DE BORD RÉCAPITULATIF PAR THÈME
# ==============================================================================
# On rassemble toutes les réponses aux 5 questions dans un seul tableau,
# facile à intégrer directement dans le mémoire.

print("\n--- Synthèse : tableau de bord récapitulatif par thème ---")

tableau_synthese = pd.DataFrame({
    "nb_publications": df_themes["theme_nom"].value_counts(),
    "sentiment_moyen": df_themes.groupby("theme_nom")["score_sentiment"].mean().round(3),
    "emotion_dominante": profil_emotionnel_themes.idxmax(axis=1),
    "tendance_sentiment": serie_tendances_sentiment.round(5),
    "tendance_part_theme": serie_tendances_prevalence.round(4),
    "tendance_engagement": serie_tendances_engagement.round(3),
    "likes_moyens": df_themes.groupby("theme_nom")["nb_likes"].mean().round(1),
    "commentaires_moyens": df_themes.groupby("theme_nom")["nb_reponses"].mean().round(2),
})
tableau_synthese = tableau_synthese.reindex(themes_francais)
tableau_synthese.to_csv(f"{dossier_tableaux}/SYNTHESE_tableau_de_bord_themes.csv")
print(tableau_synthese)

print("Enregistré : SYNTHESE_tableau_de_bord_themes.csv")


# ==============================================================================
# INDICE DE TENSION THÉMATIQUE (préfigure les indicateurs de la Phase 9)
# ==============================================================================
# On construit un premier indice simple qui combine plusieurs signaux de
# tension pour chaque thème : sentiment négatif + colère + peur + engagement
# (un sujet qui est à la fois négatif, générateur de colère/peur, ET viral
# est un sujet à surveiller de près). Chaque composante est d'abord ramenée
# entre 0 et 1 (normalisation min-max), pour pouvoir les additionner
# équitablement malgré leurs échelles très différentes au départ.

print("\n--- Indice de tension thématique (préfiguration de la Phase 9) ---")


def normaliser_0_1(serie):
    return (serie - serie.min()) / (serie.max() - serie.min())


composante_negativite = normaliser_0_1(-tableau_synthese["sentiment_moyen"])
composante_colere = normaliser_0_1(profil_emotionnel_themes["Colère"])
composante_peur = normaliser_0_1(profil_emotionnel_themes["Peur"])
composante_engagement = normaliser_0_1(tableau_synthese["likes_moyens"])

indice_tension = (composante_negativite + composante_colere + composante_peur + composante_engagement) / 4
indice_tension = indice_tension.sort_values(ascending=False)
indice_tension.to_csv(f"{dossier_tableaux}/INDICE_tension_thematique.csv")
print("Indice de tension thématique (0 = calme, 1 = tension maximale relative) :")
print(indice_tension.round(3))

couleurs_indice = cm.Reds(np.linspace(0.4, 0.95, len(indice_tension)))
plt.figure(figsize=(10, 6))
plt.barh(indice_tension.index[::-1], indice_tension.values[::-1], color=couleurs_indice[::-1])
plt.title("Indice de tension thématique (sentiment négatif + colère + peur + engagement)")
plt.xlabel("Indice de tension (0 à 1)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/INDICE_tension_thematique.png")
plt.close()

print("Enregistré : INDICE_tension_thematique (tableau + graphique).")
print("(Cet indice est une PREMIÈRE ébauche, simple et transparente. La Phase 9")
print(" affinera ce principe pour construire les indicateurs officiels du mémoire : ITE, ITR...)")


# ==============================================================================
# COMPLÉMENT — MÊME DÉMARCHE POUR LES THÈMES ANGLAIS/PIDGIN (plus brièvement)
# ==============================================================================

print("\n--- Complément : thèmes anglais/pidgin ---")

themes_anglais = [
    "Vote et actualité internationale", "Villes mortes et pétitions (pidgin)",
    "Discours politique général (honte, ministre)", "Biya, élection et pouvoir",
    "Tchiroma, peur et crise anglophone",
]
df_themes_en = df[df["theme_nom"].isin(themes_anglais)].copy()

synthese_en = pd.DataFrame({
    "nb_publications": df_themes_en["theme_nom"].value_counts(),
    "sentiment_moyen": df_themes_en.groupby("theme_nom")["score_sentiment"].mean().round(3),
    "likes_moyens": df_themes_en.groupby("theme_nom")["nb_likes"].mean().round(1),
    "commentaires_moyens": df_themes_en.groupby("theme_nom")["nb_reponses"].mean().round(2),
})
synthese_en = synthese_en.reindex(themes_anglais)
synthese_en.to_csv(f"{dossier_tableaux}/SYNTHESE_themes_anglais.csv")
print(synthese_en)

print("Enregistré : SYNTHESE_themes_anglais.csv")


# ==============================================================================
# FIN DU SCRIPT
# ==============================================================================

print("\n=== TERMINÉ ! ===")
print(f"Tous les graphiques sont dans : {dossier_figures}")
print(f"Tous les tableaux sont dans   : {dossier_tableaux}")
print("\nRécapitulatif :")
print("  Q1 : quels sujets provoquent le plus quelles émotions")
print("  Q2 : quels sujets deviennent plus positifs/négatifs (évolution + tendance)")
print("  Q3 : quels sujets disparaissent/émergent (évolution + tendance)")
print("  Q4 : quels sujets deviennent viraux (évolution + tendance + avant/après)")
print("  Q5 : quels sujets génèrent le plus de commentaires")
print("  SYNTHÈSE : tableau de bord récapitulatif par thème")
print("  INDICE : indice de tension thématique (préfigure la Phase 9)")
print("  Complément : mêmes analyses pour les thèmes anglais/pidgin")
