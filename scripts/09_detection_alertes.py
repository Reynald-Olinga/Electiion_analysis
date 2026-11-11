# ==============================================================================
# PHASE 10 — DÉTECTION AUTOMATIQUE DES ALERTES
# ==============================================================================
#
# À quoi sert ce script ?
# ------------------------
# La Phase 9 a construit des indicateurs chiffrés (ITE, ITR, Score de Risque
# Global...) jour après jour. Mais un décideur n'a pas le temps de regarder
# une courbe tous les matins : il a besoin d'un SIGNAL CLAIR — "aujourd'hui,
# faut-il s'inquiéter, oui ou non, et à quel point ?". C'est exactement le
# rôle de cette phase : transformer les indicateurs continus en ALERTES
# discrètes et lisibles (Vert / Orange / Rouge / Critique), comme le prévoit
# ton canevas.
#
# On procède en 4 grandes étapes :
#   1. Définir des SEUILS D'ALERTE pour les indicateurs-clés
#   2. Définir des RÈGLES COMBINÉES (plusieurs signaux en même temps, plus
#      fiables qu'un seul indicateur isolé)
#   3. Détecter automatiquement les alertes jour par jour sur toute la
#      période, et les regrouper en "épisodes" (une alerte qui dure 5 jours
#      d'affilée compte comme UN épisode, pas 5 alertes séparées)
#   4. VALIDER le système : est-ce qu'il détecte bien les vraies crises,
#      et est-ce qu'il le fait suffisamment TÔT pour être utile à
#      l'anticipation (c'est tout l'enjeu de H1) ?
#
# NOTE (version actuelle) : le niveau d'alerte global de chaque jour est
# désormais sauvegardé pour TOUS les jours dans ALERTES_quotidiennes_completes.csv
# (colonne ALERTE_GLOBALE), en plus du journal final qui ne liste que les
# jours en alerte.
#
# DÉCISION MÉTHODOLOGIQUE IMPORTANTE SUR LES SEUILS :
# Le canevas donne un exemple illustratif ("ITE > 70 = Alerte Orange").
# Mais nos indicateurs sont normalisés en 0-100 de façon RELATIVE à la
# fenêtre de données observée (voir Phase 9) : un seuil fixe comme "70"
# n'aurait pas le même sens selon l'indicateur (ITE a une moyenne de 20/100,
# ITR a une moyenne de 51/100 !). On calibre donc les seuils de façon
# DATA-DRIVEN, à partir des PERCENTILES réellement observés dans nos
# données : le seuil Orange correspond au 75e percentile (les 25% de
# journées les plus intenses), et le seuil Rouge au 90e percentile (les 10%
# de journées les plus intenses). C'est une méthode standard de calibration
# quand on ne dispose pas encore d'un historique assez long pour fixer des
# seuils absolus universels (voir la limite déjà signalée en Phase 9).
# ==============================================================================


# ------------------------------------------------------------------------------
# ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

dossier_data = "../data"
dossier_figures = "../outputs/figures"
dossier_tableaux = "../outputs/tables"
os.makedirs(dossier_figures, exist_ok=True)
os.makedirs(dossier_tableaux, exist_ok=True)


# ------------------------------------------------------------------------------
# ÉTAPE 1 : CHARGER LES INDICATEURS QUOTIDIENS (Phase 9)
# ------------------------------------------------------------------------------

print("Lecture du tableau des indicateurs quotidiens (Phase 9)...")
indicateurs = pd.read_csv(f"{dossier_tableaux}/INDICATEURS_tableau_complet_quotidien.csv", index_col=0)
indicateurs.index = pd.to_datetime(indicateurs.index).date
print(f"{len(indicateurs)} jours chargés (du {indicateurs.index.min()} au {indicateurs.index.max()}).")

date_du_vote = pd.Timestamp("2025-10-12").date()


# ==============================================================================
# SECTION A — CALIBRATION DES SEUILS D'ALERTE (PAR PERCENTILE)
# ==============================================================================
# Pour chaque indicateur qu'on va surveiller, on calcule 2 seuils :
#   - seuil ORANGE = 75e percentile (au-delà, on est dans le quart de
#     journées les plus intenses observées)
#   - seuil ROUGE  = 90e percentile (au-delà, on est dans les 10% de
#     journées les plus intenses, un niveau clairement exceptionnel)
#
# On calibre ces seuils sur les 3 indicateurs les plus directement liés au
# risque de crise : ITE, ITR et le Score de Risque Global.

print("\n--- Section A : calibration des seuils d'alerte ---")

indicateurs_surveilles = ["ITE", "ITR", "Score_Risque_Global"]
seuils = {}

for nom_indicateur in indicateurs_surveilles:
    valeurs_valides = indicateurs[nom_indicateur].dropna()
    seuil_orange = valeurs_valides.quantile(0.75)
    seuil_rouge = valeurs_valides.quantile(0.90)
    seuils[nom_indicateur] = {"orange": seuil_orange, "rouge": seuil_rouge}
    print(f"  {nom_indicateur:22s} -> seuil Orange = {seuil_orange:.1f}/100 | seuil Rouge = {seuil_rouge:.1f}/100")

pd.DataFrame(seuils).T.to_csv(f"{dossier_tableaux}/A1_seuils_alerte.csv")
print("Enregistré : A1_seuils_alerte.csv")


# ==============================================================================
# SECTION B — ALERTES SIMPLES PAR INDICATEUR
# ==============================================================================
# Pour chaque jour et chaque indicateur surveillé, on détermine un niveau :
# "Vert" (en dessous du seuil orange), "Orange" (entre les 2 seuils), ou
# "Rouge" (au-dessus du seuil rouge).

print("\n--- Section B : détection des alertes simples ---")


def determiner_niveau(valeur, seuil_orange, seuil_rouge):
    if pd.isna(valeur):
        return "Non disponible"
    elif valeur >= seuil_rouge:
        return "Rouge"
    elif valeur >= seuil_orange:
        return "Orange"
    else:
        return "Vert"


for nom_indicateur in indicateurs_surveilles:
    nom_colonne_alerte = f"alerte_{nom_indicateur}"
    indicateurs[nom_colonne_alerte] = indicateurs[nom_indicateur].apply(
        lambda v: determiner_niveau(v, seuils[nom_indicateur]["orange"], seuils[nom_indicateur]["rouge"])
    )
    repartition = indicateurs[nom_colonne_alerte].value_counts()
    print(f"\nRépartition des niveaux d'alerte pour {nom_indicateur} :")
    print(repartition)

indicateurs.to_csv(f"{dossier_tableaux}/B1_alertes_simples_quotidiennes.csv")
print("\nEnregistré : B1_alertes_simples_quotidiennes.csv")


# ==============================================================================
# SECTION C — RÈGLES COMBINÉES (PLUSIEURS SIGNAUX EN MÊME TEMPS)
# ==============================================================================
# Une alerte basée sur UN SEUL indicateur peut être un faux signal (un jour
# isolé où, par exemple, une poignée de publications très négatives suffit à
# faire grimper l'ITE, sans que ça reflète un vrai risque de crise). Une
# alerte qui combine PLUSIEURS signaux différents EN MÊME TEMPS est
# beaucoup plus fiable. On construit ici 2 règles combinées, comme le
# suggère le canevas avec son exemple "Topic Violence + Colère > 80%".
#
# RÈGLE COMBINÉE 1 — "Risque thématique + tension émotionnelle" :
#   ITR en Orange/Rouge  ET  ITE en Orange/Rouge en même temps
#   -> Les publications du jour parlent BEAUCOUP de sujets à risque, ET
#      elles sont chargées d'émotions négatives : les deux signaux se
#      renforcent l'un l'autre.
#
# RÈGLE COMBINÉE 2 — "Mobilisation + Polarisation" :
#   Mobilisation en Orange/Rouge  ET  Polarisation en Orange/Rouge
#   -> Des appels à l'action collective ET un public divisé en même temps :
#      un terrain propice à un débordement, potentiellement conflictuel.

print("\n--- Section C : détection des règles combinées ---")

seuil_mobilisation_orange = indicateurs["Mobilisation"].quantile(0.75)
seuil_mobilisation_rouge = indicateurs["Mobilisation"].quantile(0.90)
seuil_polarisation_orange = indicateurs["Polarisation"].quantile(0.75)
seuil_polarisation_rouge = indicateurs["Polarisation"].quantile(0.90)

indicateurs["alerte_Mobilisation"] = indicateurs["Mobilisation"].apply(
    lambda v: determiner_niveau(v, seuil_mobilisation_orange, seuil_mobilisation_rouge)
)
indicateurs["alerte_Polarisation"] = indicateurs["Polarisation"].apply(
    lambda v: determiner_niveau(v, seuil_polarisation_orange, seuil_polarisation_rouge)
)

est_alerte = lambda niveau: niveau in ("Orange", "Rouge")

indicateurs["regle_combinee_1_risque_tension"] = (
    indicateurs["alerte_ITR"].apply(est_alerte) & indicateurs["alerte_ITE"].apply(est_alerte)
)
indicateurs["regle_combinee_2_mobilisation_polarisation"] = (
    indicateurs["alerte_Mobilisation"].apply(est_alerte) & indicateurs["alerte_Polarisation"].apply(est_alerte)
)

nb_jours_regle1 = indicateurs["regle_combinee_1_risque_tension"].sum()
nb_jours_regle2 = indicateurs["regle_combinee_2_mobilisation_polarisation"].sum()
print(f"Règle combinée 1 (Risque thématique + Tension émotionnelle) déclenchée : {nb_jours_regle1} jours")
print(f"Règle combinée 2 (Mobilisation + Polarisation) déclenchée              : {nb_jours_regle2} jours")

indicateurs.to_csv(f"{dossier_tableaux}/C1_regles_combinees.csv")
print("Enregistré : C1_regles_combinees.csv")


# ==============================================================================
# SECTION D — NIVEAU D'ALERTE GLOBAL DU JOUR (SYNTHÈSE FINALE)
# ==============================================================================
# On résume tout en un seul niveau d'alerte par jour, avec 4 catégories :
#   - VERT      : aucun signal particulier
#   - ORANGE    : au moins un indicateur clé est en zone Orange
#   - ROUGE     : au moins un indicateur clé est en zone Rouge
#   - CRITIQUE  : au moins une des règles combinées est déclenchée (le
#                 niveau le plus grave, car il correspond à plusieurs
#                 signaux qui se confirment mutuellement)

print("\n--- Section D : niveau d'alerte global du jour ---")


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


indicateurs["ALERTE_GLOBALE"] = indicateurs.apply(determiner_alerte_globale, axis=1)

repartition_globale = indicateurs["ALERTE_GLOBALE"].value_counts()
print("Répartition des niveaux d'alerte globale sur toute la période :")
print(repartition_globale)
repartition_globale.to_csv(f"{dossier_tableaux}/D1_repartition_alerte_globale.csv")

print("Enregistré : D1_repartition_alerte_globale.csv")

# Sauvegarde du tableau quotidien COMPLET, avec le niveau d'alerte global de
# CHAQUE jour (Vert / Orange / Rouge / Critique). Jusqu'ici, la colonne
# ALERTE_GLOBALE n'était écrite que dans le journal final, et seulement pour
# les jours en alerte : les fichiers B1 et C1 étaient sauvegardés AVANT son
# calcul. On les réécrit donc ici, et on crée un fichier dédié, pour que tout
# programme qui lit le niveau d'alerte du jour (par exemple la plateforme
# Streamlit) trouve la colonne pour tous les jours de la période.
indicateurs.to_csv(f"{dossier_tableaux}/B1_alertes_simples_quotidiennes.csv")
indicateurs.to_csv(f"{dossier_tableaux}/C1_regles_combinees.csv")
indicateurs.to_csv(f"{dossier_tableaux}/ALERTES_quotidiennes_completes.csv")
print("Enregistré : ALERTES_quotidiennes_completes.csv (+ B1 et C1 réécrits avec ALERTE_GLOBALE)")


# ==============================================================================
# SECTION E — CALENDRIER VISUEL DES ALERTES
# ==============================================================================
# Une frise chronologique en "feu tricolore" : chaque jour est une petite
# barre colorée selon son niveau d'alerte. C'est le type de visuel qu'un
# vrai tableau de bord de surveillance afficherait en continu.

print("\n--- Section E : calendrier visuel des alertes ---")

couleurs_alerte = {"Vert": "#2A9D8F", "Orange": "#F4A300", "Rouge": "#E63946", "Critique": "#6A0572"}
ordre_alerte = {"Vert": 0, "Orange": 1, "Rouge": 2, "Critique": 3}

indicateurs["niveau_numerique"] = indicateurs["ALERTE_GLOBALE"].map(ordre_alerte)

fig, ax = plt.subplots(figsize=(16, 3))
for jour, ligne in indicateurs.iterrows():
    ax.axvspan(jour, jour, color=couleurs_alerte[ligne["ALERTE_GLOBALE"]], linewidth=0)

ax.axvline(date_du_vote, color="black", linestyle="--", linewidth=2, label="Jour du vote")
ax.set_yticks([])
ax.set_title("Calendrier des alertes (Vert / Orange / Rouge / Critique)")
ax.set_xlabel("Date")

# Légende manuelle (des rectangles de couleur, puisqu'on n'a pas de vraies
# courbes à légender ici)
elements_legende = [plt.Rectangle((0, 0), 1, 1, color=couleur) for couleur in couleurs_alerte.values()]
ax.legend(elements_legende, couleurs_alerte.keys(), loc="upper left", bbox_to_anchor=(0, -0.3), ncol=4)

plt.tight_layout()
plt.savefig(f"{dossier_figures}/E1_calendrier_alertes.png")
plt.close()

print("Enregistré : E1_calendrier_alertes.png")


# ==============================================================================
# SECTION F — ÉPISODES D'ALERTE (REGROUPEMENT DES JOURS CONSÉCUTIFS)
# ==============================================================================
# Si un niveau Orange dure 6 jours d'affilée, ce n'est pas "6 alertes", c'est
# UN épisode qui a duré 6 jours. On regroupe donc les jours consécutifs de
# même nature (alerte vs pas alerte) en épisodes, avec leur date de début,
# de fin, leur durée, et leur niveau maximal atteint.

print("\n--- Section F : regroupement en épisodes d'alerte ---")

indicateurs["est_en_alerte"] = indicateurs["ALERTE_GLOBALE"] != "Vert"

# Astuce classique pour regrouper des lignes consécutives qui partagent la
# même valeur : on repère chaque changement d'état, et on numérote les
# groupes avec un cumsum() des changements.
indicateurs["nouveau_groupe"] = (indicateurs["est_en_alerte"] != indicateurs["est_en_alerte"].shift()).cumsum()

episodes = []
for numero_groupe, sous_tableau in indicateurs[indicateurs["est_en_alerte"]].groupby("nouveau_groupe"):
    episodes.append({
        "date_debut": sous_tableau.index.min(),
        "date_fin": sous_tableau.index.max(),
        "duree_jours": len(sous_tableau),
        "niveau_maximal": sous_tableau["ALERTE_GLOBALE"].map(ordre_alerte).idxmax() if False else
                          sous_tableau.loc[sous_tableau["niveau_numerique"].idxmax(), "ALERTE_GLOBALE"],
        "score_risque_moyen": sous_tableau["Score_Risque_Global"].mean().round(1),
    })

tableau_episodes = pd.DataFrame(episodes).sort_values("date_debut")
tableau_episodes.to_csv(f"{dossier_tableaux}/F1_episodes_alerte.csv", index=False)

print(f"Nombre total d'épisodes d'alerte détectés : {len(tableau_episodes)}")
print(f"Durée moyenne d'un épisode : {tableau_episodes['duree_jours'].mean():.1f} jours")
print(f"Épisode le plus long : {tableau_episodes['duree_jours'].max()} jours")
print("\nListe des épisodes :")
print(tableau_episodes.to_string(index=False))

print("\nEnregistré : F1_episodes_alerte.csv")


# ==============================================================================
# SECTION G — DÉLAI D'ANTICIPATION : LES ALERTES PRÉCÈDENT-ELLES LA CRISE ?
# ==============================================================================
# C'est LE test le plus important pour ton mémoire (H1 : "les plateformes
# numériques fournissent des signaux PRÉCURSEURS"). On vérifie si nos
# alertes se déclenchent AVANT les pics réels de mobilisation (mesurés par
# le mot-clé "villes mortes"/"manifestation"), et de COMBIEN DE JOURS à
# l'avance en moyenne. Un système d'alerte qui ne fait que confirmer la
# crise APRÈS qu'elle a éclaté n'a aucune valeur d'anticipation.

print("\n--- Section G : délai d'anticipation des alertes ---")

# On identifie les jours de PIC réel de mobilisation : les jours où
# l'indicateur "Mobilisation" est dans le top 10% (donc des jours où on
# parle vraiment beaucoup, concrètement, d'actions collectives)
seuil_pic_mobilisation = indicateurs["Mobilisation"].quantile(0.90)
jours_pic_mobilisation = indicateurs[indicateurs["Mobilisation"] >= seuil_pic_mobilisation].index.tolist()

print(f"Jours identifiés comme des 'pics de mobilisation réelle' : {len(jours_pic_mobilisation)}")

delais_anticipation = []
for jour_pic in jours_pic_mobilisation:
    # Pour chaque pic réel, on cherche s'il y a eu une alerte (Orange/Rouge/
    # Critique) dans les 10 jours QUI PRÉCÈDENT ce pic
    fenetre_recherche = [jour_pic - pd.Timedelta(days=n) for n in range(1, 11)]
    fenetre_recherche = [j for j in fenetre_recherche if j in indicateurs.index]

    jours_alerte_avant = [j for j in fenetre_recherche if indicateurs.loc[j, "est_en_alerte"]]
    if len(jours_alerte_avant) > 0:
        # Le délai d'anticipation = nombre de jours entre l'alerte la PLUS
        # PRÉCOCE dans cette fenêtre et le jour du pic réel
        alerte_la_plus_precoce = max(jours_alerte_avant, key=lambda j: (jour_pic - j).days)
        delai = (jour_pic - alerte_la_plus_precoce).days
        delais_anticipation.append(delai)
    else:
        delais_anticipation.append(0)  # aucune alerte détectée avant ce pic : délai nul

tableau_anticipation = pd.DataFrame({
    "jour_pic_mobilisation": jours_pic_mobilisation,
    "delai_anticipation_jours": delais_anticipation,
})
tableau_anticipation.to_csv(f"{dossier_tableaux}/G1_delai_anticipation.csv", index=False)

nb_pics_anticipes = (tableau_anticipation["delai_anticipation_jours"] > 0).sum()
taux_anticipation = nb_pics_anticipes / len(tableau_anticipation) * 100 if len(tableau_anticipation) > 0 else 0

print(f"\nPics de mobilisation anticipés par au moins une alerte : {nb_pics_anticipes} / {len(tableau_anticipation)} ({taux_anticipation:.0f}%)")
print(f"Délai d'anticipation moyen (parmi les pics anticipés) : {tableau_anticipation.loc[tableau_anticipation['delai_anticipation_jours']>0, 'delai_anticipation_jours'].mean():.1f} jours")

plt.figure(figsize=(9, 6))
plt.hist(tableau_anticipation["delai_anticipation_jours"], bins=range(0, 12), color="#457B9D", edgecolor="white", align="left")
plt.title("Délai d'anticipation des alertes avant un pic de mobilisation réel")
plt.xlabel("Nombre de jours d'avance (0 = aucune anticipation)")
plt.ylabel("Nombre de pics de mobilisation concernés")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/G1_delai_anticipation.png")
plt.close()

print("Enregistré : G1_delai_anticipation (tableau + histogramme).")


# ==============================================================================
# SECTION H — TAUX DE DÉTECTION ET FAUX POSITIFS
# ==============================================================================
# Deux erreurs sont possibles pour un système d'alerte :
#   - un FAUX NÉGATIF : une vraie crise qui n'a déclenché AUCUNE alerte
#     (dangereux : la crise arrive "par surprise")
#   - un FAUX POSITIF : une alerte déclenchée un jour "calme" (gênant, mais
#     moins grave : ça use la vigilance des décideurs si c'est trop fréquent,
#     mais ça ne rate pas de vraie crise)
# On calcule les deux taux, en utilisant comme référence de "vraie crise"
# les jours de forte mention de "villes mortes" (déjà utilisés en Phase 9).

print("\n--- Section H : taux de détection et faux positifs ---")

# On recharge juste la colonne nécessaire pour identifier les vrais jours de
# crise (même logique que la validation de la Phase 9)
corpus_complet = pd.read_csv(f"{dossier_data}/bd_f_themes.csv", low_memory=False)
corpus_complet["date_publication"] = pd.to_datetime(corpus_complet["date_publication"], utc=True, errors="coerce")
corpus_complet["texte_nettoye"] = corpus_complet["texte_nettoye"].fillna("")
corpus_complet["jour"] = corpus_complet["date_publication"].dt.date
corpus_complet["mention_villes_mortes"] = corpus_complet["texte_nettoye"].str.contains(
    "ville morte|villes mortes|ghost town", regex=True, na=False
)

part_villes_mortes = corpus_complet.groupby("jour")["mention_villes_mortes"].mean().reindex(indicateurs.index, fill_value=0) * 100
seuil_crise_reelle = part_villes_mortes[part_villes_mortes > 0].median() if (part_villes_mortes > 0).any() else 0
indicateurs["jour_de_crise_reelle"] = part_villes_mortes > seuil_crise_reelle

vrais_positifs = ((indicateurs["est_en_alerte"]) & (indicateurs["jour_de_crise_reelle"])).sum()
faux_negatifs = ((~indicateurs["est_en_alerte"]) & (indicateurs["jour_de_crise_reelle"])).sum()
faux_positifs = ((indicateurs["est_en_alerte"]) & (~indicateurs["jour_de_crise_reelle"])).sum()
vrais_negatifs = ((~indicateurs["est_en_alerte"]) & (~indicateurs["jour_de_crise_reelle"])).sum()

taux_detection = vrais_positifs / (vrais_positifs + faux_negatifs) * 100 if (vrais_positifs + faux_negatifs) > 0 else np.nan
taux_faux_positifs = faux_positifs / (faux_positifs + vrais_negatifs) * 100 if (faux_positifs + vrais_negatifs) > 0 else np.nan

print(f"Vrais positifs (alerte ET vraie crise)      : {vrais_positifs} jours")
print(f"Faux négatifs (pas d'alerte MAIS vraie crise) : {faux_negatifs} jours")
print(f"Faux positifs (alerte MAIS jour calme)        : {faux_positifs} jours")
print(f"Vrais négatifs (pas d'alerte ET jour calme)   : {vrais_negatifs} jours")
print(f"\nTaux de détection (sensibilité) : {taux_detection:.1f}%")
print(f"Taux de faux positifs           : {taux_faux_positifs:.1f}%")

tableau_confusion = pd.DataFrame({
    "Vraie crise": [vrais_positifs, faux_negatifs],
    "Jour calme": [faux_positifs, vrais_negatifs],
}, index=["Alerte déclenchée", "Pas d'alerte"])
tableau_confusion.to_csv(f"{dossier_tableaux}/H1_matrice_confusion.csv")
print("\nMatrice de confusion :")
print(tableau_confusion)

plt.figure(figsize=(6, 5))
plt.imshow(tableau_confusion.values, cmap="Blues")
plt.xticks([0, 1], tableau_confusion.columns)
plt.yticks([0, 1], tableau_confusion.index)
for i in range(2):
    for j in range(2):
        plt.text(j, i, tableau_confusion.values[i, j], ha="center", va="center", fontsize=14)
plt.title("Matrice de confusion du système d'alerte")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/H1_matrice_confusion.png")
plt.close()

print("Enregistré : H1_matrice_confusion (tableau + graphique).")


# ==============================================================================
# SECTION I — JOURNAL FINAL DES ALERTES (livrable pour un système réel)
# ==============================================================================
# Le document final : un vrai "journal d'alerte" tel qu'un système déployé
# produirait, avec une ligne par jour, le niveau, et le ou les indicateurs
# responsables du déclenchement.

print("\n--- Section I : journal final des alertes ---")


def lister_causes(ligne):
    causes = []
    for indicateur in indicateurs_surveilles:
        if ligne[f"alerte_{indicateur}"] in ("Orange", "Rouge"):
            causes.append(f"{indicateur}={ligne[f'alerte_{indicateur}']}")
    if ligne["regle_combinee_1_risque_tension"]:
        causes.append("Règle combinée: Risque thématique + Tension émotionnelle")
    if ligne["regle_combinee_2_mobilisation_polarisation"]:
        causes.append("Règle combinée: Mobilisation + Polarisation")
    return " | ".join(causes) if causes else "-"


indicateurs["causes_alerte"] = indicateurs.apply(lister_causes, axis=1)

journal_alertes = indicateurs.loc[indicateurs["ALERTE_GLOBALE"] != "Vert",
                                   ["ALERTE_GLOBALE", "Score_Risque_Global", "causes_alerte"]]
journal_alertes.to_csv(f"{dossier_tableaux}/JOURNAL_alertes_final.csv")

print(f"Journal final : {len(journal_alertes)} jours avec une alerte active sur {len(indicateurs)} jours analysés.")
print("Enregistré : JOURNAL_alertes_final.csv")


# ==============================================================================
# FIN DU SCRIPT
# ==============================================================================

print("\n=== TERMINÉ ! ===")
print(f"Tous les graphiques sont dans : {dossier_figures}")
print(f"Tous les tableaux sont dans   : {dossier_tableaux}")
print("\nRécapitulatif :")
print("  A : calibration des seuils d'alerte (percentiles 75/90)")
print("  B : alertes simples par indicateur (ITE, ITR, Score de Risque Global)")
print("  C : règles combinées (2 règles multi-signaux)")
print("  D : niveau d'alerte global du jour (Vert/Orange/Rouge/Critique)")
print("  E : calendrier visuel des alertes")
print("  F : regroupement en épisodes d'alerte")
print("  G : délai d'anticipation (le test le plus important pour H1)")
print("  H : taux de détection et faux positifs (matrice de confusion)")
print("  I : journal final des alertes (livrable)")




















































































# # ==============================================================================
# # PHASE 10 — DÉTECTION AUTOMATIQUE DES ALERTES
# # ==============================================================================
# #
# # À quoi sert ce script ?
# # ------------------------
# # La Phase 9 a construit des indicateurs chiffrés (ITE, ITR, Score de Risque
# # Global...) jour après jour. Mais un décideur n'a pas le temps de regarder
# # une courbe tous les matins : il a besoin d'un SIGNAL CLAIR — "aujourd'hui,
# # faut-il s'inquiéter, oui ou non, et à quel point ?". C'est exactement le
# # rôle de cette phase : transformer les indicateurs continus en ALERTES
# # discrètes et lisibles (Vert / Orange / Rouge / Critique), comme le prévoit
# # ton canevas.
# #
# # On procède en 4 grandes étapes :
# #   1. Définir des SEUILS D'ALERTE pour les indicateurs-clés
# #   2. Définir des RÈGLES COMBINÉES (plusieurs signaux en même temps, plus
# #      fiables qu'un seul indicateur isolé)
# #   3. Détecter automatiquement les alertes jour par jour sur toute la
# #      période, et les regrouper en "épisodes" (une alerte qui dure 5 jours
# #      d'affilée compte comme UN épisode, pas 5 alertes séparées)
# #   4. VALIDER le système : est-ce qu'il détecte bien les vraies crises,
# #      et est-ce qu'il le fait suffisamment TÔT pour être utile à
# #      l'anticipation (c'est tout l'enjeu de H1)
# #
# # DÉCISION MÉTHODOLOGIQUE IMPORTANTE SUR LES SEUILS :
# # Le canevas donne un exemple illustratif ("ITE > 70 = Alerte Orange").
# # Mais nos indicateurs sont normalisés en 0-100 de façon RELATIVE à la
# # fenêtre de données observée (voir Phase 9) : un seuil fixe comme "70"
# # n'aurait pas le même sens selon l'indicateur (ITE a une moyenne de 20/100,
# # ITR a une moyenne de 51/100 !). On calibre donc les seuils de façon
# # DATA-DRIVEN, à partir des PERCENTILES réellement observés dans nos
# # données : le seuil Orange correspond au 75e percentile (les 25% de
# # journées les plus intenses), et le seuil Rouge au 90e percentile (les 10%
# # de journées les plus intenses). C'est une méthode standard de calibration
# # quand on ne dispose pas encore d'un historique assez long pour fixer des
# # seuils absolus universels (voir la limite déjà signalée en Phase 9).
# #
# # CORRECTION APPORTÉE À CETTE VERSION :
# # Dans la version précédente, le tableau quotidien complet (Vert compris)
# # avec la colonne ALERTE_GLOBALE n'était jamais sauvegardé tel quel dans un
# # fichier CSV — seul B1 (avant les règles combinées), C1 (avant
# # ALERTE_GLOBALE) et JOURNAL_alertes_final.csv (qui EXCLUT les jours Vert)
# # existaient. Un outil externe (comme un tableau de bord) qui a besoin du
# # niveau d'alerte de CHAQUE jour, y compris les jours calmes, devait donc
# # recalculer lui-même ALERTE_GLOBALE à partir des colonnes intermédiaires.
# # On corrige ça en Section D bis : on sauvegarde désormais explicitement
# # le tableau quotidien complet avec ALERTE_GLOBALE dans
# # D2_alerte_globale_quotidienne.csv, qui devient la source de référence
# # unique pour tout ce qui a besoin du niveau d'alerte jour par jour.
# # ==============================================================================


# # ------------------------------------------------------------------------------
# # ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# # ------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import os

# dossier_data = "../data"
# dossier_figures = "../outputs/figures"
# dossier_tableaux = "../outputs/tables"
# os.makedirs(dossier_figures, exist_ok=True)
# os.makedirs(dossier_tableaux, exist_ok=True)


# # ------------------------------------------------------------------------------
# # ÉTAPE 1 : CHARGER LES INDICATEURS QUOTIDIENS (Phase 9)
# # ------------------------------------------------------------------------------

# print("Lecture du tableau des indicateurs quotidiens (Phase 9)...")
# indicateurs = pd.read_csv(f"{dossier_tableaux}/INDICATEURS_tableau_complet_quotidien.csv", index_col=0)
# indicateurs.index = pd.to_datetime(indicateurs.index).date
# print(f"{len(indicateurs)} jours chargés (du {indicateurs.index.min()} au {indicateurs.index.max()}).")

# date_du_vote = pd.Timestamp("2025-10-12").date()


# # ==============================================================================
# # SECTION A — CALIBRATION DES SEUILS D'ALERTE (PAR PERCENTILE)
# # ==============================================================================
# # Pour chaque indicateur qu'on va surveiller, on calcule 2 seuils :
# #   - seuil ORANGE = 75e percentile (au-delà, on est dans le quart de
# #     journées les plus intenses observées)
# #   - seuil ROUGE  = 90e percentile (au-delà, on est dans les 10% de
# #     journées les plus intenses, un niveau clairement exceptionnel)
# #
# # On calibre ces seuils sur les 3 indicateurs les plus directement liés au
# # risque de crise : ITE, ITR et le Score de Risque Global.

# print("\n--- Section A : calibration des seuils d'alerte ---")

# indicateurs_surveilles = ["ITE", "ITR", "Score_Risque_Global"]
# seuils = {}

# for nom_indicateur in indicateurs_surveilles:
#     valeurs_valides = indicateurs[nom_indicateur].dropna()
#     seuil_orange = valeurs_valides.quantile(0.75)
#     seuil_rouge = valeurs_valides.quantile(0.90)
#     seuils[nom_indicateur] = {"orange": seuil_orange, "rouge": seuil_rouge}
#     print(f"  {nom_indicateur:22s} -> seuil Orange = {seuil_orange:.1f}/100 | seuil Rouge = {seuil_rouge:.1f}/100")

# pd.DataFrame(seuils).T.to_csv(f"{dossier_tableaux}/A1_seuils_alerte.csv")
# print("Enregistré : A1_seuils_alerte.csv")


# # ==============================================================================
# # SECTION B — ALERTES SIMPLES PAR INDICATEUR
# # ==============================================================================
# # Pour chaque jour et chaque indicateur surveillé, on détermine un niveau :
# # "Vert" (en dessous du seuil orange), "Orange" (entre les 2 seuils), ou
# # "Rouge" (au-dessus du seuil rouge).

# print("\n--- Section B : détection des alertes simples ---")


# def determiner_niveau(valeur, seuil_orange, seuil_rouge):
#     if pd.isna(valeur):
#         return "Non disponible"
#     elif valeur >= seuil_rouge:
#         return "Rouge"
#     elif valeur >= seuil_orange:
#         return "Orange"
#     else:
#         return "Vert"


# for nom_indicateur in indicateurs_surveilles:
#     nom_colonne_alerte = f"alerte_{nom_indicateur}"
#     indicateurs[nom_colonne_alerte] = indicateurs[nom_indicateur].apply(
#         lambda v: determiner_niveau(v, seuils[nom_indicateur]["orange"], seuils[nom_indicateur]["rouge"])
#     )
#     repartition = indicateurs[nom_colonne_alerte].value_counts()
#     print(f"\nRépartition des niveaux d'alerte pour {nom_indicateur} :")
#     print(repartition)

# indicateurs.to_csv(f"{dossier_tableaux}/B1_alertes_simples_quotidiennes.csv")
# print("\nEnregistré : B1_alertes_simples_quotidiennes.csv")


# # ==============================================================================
# # SECTION C — RÈGLES COMBINÉES (PLUSIEURS SIGNAUX EN MÊME TEMPS)
# # ==============================================================================
# # Une alerte basée sur UN SEUL indicateur peut être un faux signal (un jour
# # isolé où, par exemple, une poignée de publications très négatives suffit à
# # faire grimper l'ITE, sans que ça reflète un vrai risque de crise). Une
# # alerte qui combine PLUSIEURS signaux différents EN MÊME TEMPS est
# # beaucoup plus fiable. On construit ici 2 règles combinées, comme le
# # suggère le canevas avec son exemple "Topic Violence + Colère > 80%".
# #
# # RÈGLE COMBINÉE 1 — "Risque thématique + tension émotionnelle" :
# #   ITR en Orange/Rouge  ET  ITE en Orange/Rouge en même temps
# #   -> Les publications du jour parlent BEAUCOUP de sujets à risque, ET
# #      elles sont chargées d'émotions négatives : les deux signaux se
# #      renforcent l'un l'autre.
# #
# # RÈGLE COMBINÉE 2 — "Mobilisation + Polarisation" :
# #   Mobilisation en Orange/Rouge  ET  Polarisation en Orange/Rouge
# #   -> Des appels à l'action collective ET un public divisé en même temps :
# #      un terrain propice à un débordement, potentiellement conflictuel.

# print("\n--- Section C : détection des règles combinées ---")

# seuil_mobilisation_orange = indicateurs["Mobilisation"].quantile(0.75)
# seuil_mobilisation_rouge = indicateurs["Mobilisation"].quantile(0.90)
# seuil_polarisation_orange = indicateurs["Polarisation"].quantile(0.75)
# seuil_polarisation_rouge = indicateurs["Polarisation"].quantile(0.90)

# indicateurs["alerte_Mobilisation"] = indicateurs["Mobilisation"].apply(
#     lambda v: determiner_niveau(v, seuil_mobilisation_orange, seuil_mobilisation_rouge)
# )
# indicateurs["alerte_Polarisation"] = indicateurs["Polarisation"].apply(
#     lambda v: determiner_niveau(v, seuil_polarisation_orange, seuil_polarisation_rouge)
# )

# est_alerte = lambda niveau: niveau in ("Orange", "Rouge")

# indicateurs["regle_combinee_1_risque_tension"] = (
#     indicateurs["alerte_ITR"].apply(est_alerte) & indicateurs["alerte_ITE"].apply(est_alerte)
# )
# indicateurs["regle_combinee_2_mobilisation_polarisation"] = (
#     indicateurs["alerte_Mobilisation"].apply(est_alerte) & indicateurs["alerte_Polarisation"].apply(est_alerte)
# )

# nb_jours_regle1 = indicateurs["regle_combinee_1_risque_tension"].sum()
# nb_jours_regle2 = indicateurs["regle_combinee_2_mobilisation_polarisation"].sum()
# print(f"Règle combinée 1 (Risque thématique + Tension émotionnelle) déclenchée : {nb_jours_regle1} jours")
# print(f"Règle combinée 2 (Mobilisation + Polarisation) déclenchée              : {nb_jours_regle2} jours")

# indicateurs.to_csv(f"{dossier_tableaux}/C1_regles_combinees.csv")
# print("Enregistré : C1_regles_combinees.csv")


# # ==============================================================================
# # SECTION D — NIVEAU D'ALERTE GLOBAL DU JOUR (SYNTHÈSE FINALE)
# # ==============================================================================
# # On résume tout en un seul niveau d'alerte par jour, avec 4 catégories :
# #   - VERT      : aucun signal particulier
# #   - ORANGE    : au moins un indicateur clé est en zone Orange
# #   - ROUGE     : au moins un indicateur clé est en zone Rouge
# #   - CRITIQUE  : au moins une des règles combinées est déclenchée (le
# #                 niveau le plus grave, car il correspond à plusieurs
# #                 signaux qui se confirment mutuellement)

# print("\n--- Section D : niveau d'alerte global du jour ---")


# def determiner_alerte_globale(ligne):
#     if ligne["regle_combinee_1_risque_tension"] or ligne["regle_combinee_2_mobilisation_polarisation"]:
#         return "Critique"
#     niveaux = [ligne["alerte_ITE"], ligne["alerte_ITR"], ligne["alerte_Score_Risque_Global"]]
#     if "Rouge" in niveaux:
#         return "Rouge"
#     elif "Orange" in niveaux:
#         return "Orange"
#     else:
#         return "Vert"


# indicateurs["ALERTE_GLOBALE"] = indicateurs.apply(determiner_alerte_globale, axis=1)

# repartition_globale = indicateurs["ALERTE_GLOBALE"].value_counts()
# print("Répartition des niveaux d'alerte globale sur toute la période :")
# print(repartition_globale)
# repartition_globale.to_csv(f"{dossier_tableaux}/D1_repartition_alerte_globale.csv")

# print("Enregistré : D1_repartition_alerte_globale.csv")


# # ==============================================================================
# # SECTION D BIS — SAUVEGARDE DU TABLEAU QUOTIDIEN COMPLET (CORRECTION)
# # ==============================================================================
# # C'est la correction demandée : on sauvegarde ICI, explicitement, le
# # tableau contenant TOUS les jours de la fenêtre d'analyse (Vert compris),
# # avec la colonne ALERTE_GLOBALE déjà calculée. Sans ce fichier, un outil
# # externe (comme un tableau de bord) qui a besoin du niveau d'alerte de
# # CHAQUE jour devrait recalculer lui-même ALERTE_GLOBALE à partir des
# # colonnes intermédiaires (alerte_ITE, alerte_ITR, regle_combinee_1...),
# # ce qui duplique inutilement la logique métier à deux endroits différents.
# # Avec ce fichier, il n'y a plus qu'UNE SEULE source de vérité pour le
# # niveau d'alerte quotidien.

# print("\n--- Section D bis : sauvegarde du tableau quotidien complet (avec ALERTE_GLOBALE) ---")

# indicateurs.to_csv(f"{dossier_tableaux}/D2_alerte_globale_quotidienne.csv")
# print(f"Enregistré : D2_alerte_globale_quotidienne.csv ({len(indicateurs)} jours, Vert compris)")


# # ==============================================================================
# # SECTION E — CALENDRIER VISUEL DES ALERTES
# # ==============================================================================
# # Une frise chronologique en "feu tricolore" : chaque jour est une petite
# # barre colorée selon son niveau d'alerte. C'est le type de visuel qu'un
# # vrai tableau de bord de surveillance afficherait en continu.

# print("\n--- Section E : calendrier visuel des alertes ---")

# couleurs_alerte = {"Vert": "#2A9D8F", "Orange": "#F4A300", "Rouge": "#E63946", "Critique": "#6A0572"}
# ordre_alerte = {"Vert": 0, "Orange": 1, "Rouge": 2, "Critique": 3}

# indicateurs["niveau_numerique"] = indicateurs["ALERTE_GLOBALE"].map(ordre_alerte)

# fig, ax = plt.subplots(figsize=(16, 3))
# for jour, ligne in indicateurs.iterrows():
#     ax.axvspan(jour, jour, color=couleurs_alerte[ligne["ALERTE_GLOBALE"]], linewidth=0)

# ax.axvline(date_du_vote, color="black", linestyle="--", linewidth=2, label="Jour du vote")
# ax.set_yticks([])
# ax.set_title("Calendrier des alertes (Vert / Orange / Rouge / Critique)")
# ax.set_xlabel("Date")

# # Légende manuelle (des rectangles de couleur, puisqu'on n'a pas de vraies
# # courbes à légender ici)
# elements_legende = [plt.Rectangle((0, 0), 1, 1, color=couleur) for couleur in couleurs_alerte.values()]
# ax.legend(elements_legende, couleurs_alerte.keys(), loc="upper left", bbox_to_anchor=(0, -0.3), ncol=4)

# plt.tight_layout()
# plt.savefig(f"{dossier_figures}/E1_calendrier_alertes.png")
# plt.close()

# print("Enregistré : E1_calendrier_alertes.png")


# # ==============================================================================
# # SECTION F — ÉPISODES D'ALERTE (REGROUPEMENT DES JOURS CONSÉCUTIFS)
# # ==============================================================================
# # Si un niveau Orange dure 6 jours d'affilée, ce n'est pas "6 alertes", c'est
# # UN épisode qui a duré 6 jours. On regroupe donc les jours consécutifs de
# # même nature (alerte vs pas alerte) en épisodes, avec leur date de début,
# # de fin, leur durée, et leur niveau maximal atteint.

# print("\n--- Section F : regroupement en épisodes d'alerte ---")

# indicateurs["est_en_alerte"] = indicateurs["ALERTE_GLOBALE"] != "Vert"

# # Astuce classique pour regrouper des lignes consécutives qui partagent la
# # même valeur : on repère chaque changement d'état, et on numérote les
# # groupes avec un cumsum() des changements.
# indicateurs["nouveau_groupe"] = (indicateurs["est_en_alerte"] != indicateurs["est_en_alerte"].shift()).cumsum()

# episodes = []
# for numero_groupe, sous_tableau in indicateurs[indicateurs["est_en_alerte"]].groupby("nouveau_groupe"):
#     # CORRECTION : l'ancienne ligne contenait un "if False else" qui ne
#     # servait à rien (la condition était toujours fausse, donc seule la
#     # branche après le "else" s'exécutait). On l'a simplifiée ici sans rien
#     # changer au résultat produit, juste pour la clarté du code.
#     niveau_maximal_episode = sous_tableau.loc[sous_tableau["niveau_numerique"].idxmax(), "ALERTE_GLOBALE"]
#     episodes.append({
#         "date_debut": sous_tableau.index.min(),
#         "date_fin": sous_tableau.index.max(),
#         "duree_jours": len(sous_tableau),
#         "niveau_maximal": niveau_maximal_episode,
#         "score_risque_moyen": sous_tableau["Score_Risque_Global"].mean().round(1),
#     })

# tableau_episodes = pd.DataFrame(episodes).sort_values("date_debut")
# tableau_episodes.to_csv(f"{dossier_tableaux}/F1_episodes_alerte.csv", index=False)

# print(f"Nombre total d'épisodes d'alerte détectés : {len(tableau_episodes)}")
# print(f"Durée moyenne d'un épisode : {tableau_episodes['duree_jours'].mean():.1f} jours")
# print(f"Épisode le plus long : {tableau_episodes['duree_jours'].max()} jours")
# print("\nListe des épisodes :")
# print(tableau_episodes.to_string(index=False))

# print("\nEnregistré : F1_episodes_alerte.csv")


# # ==============================================================================
# # SECTION G — DÉLAI D'ANTICIPATION : LES ALERTES PRÉCÈDENT-ELLES LA CRISE ?
# # ==============================================================================
# # C'est LE test le plus important pour ton mémoire (H1 : "les plateformes
# # numériques fournissent des signaux PRÉCURSEURS"). On vérifie si nos
# # alertes se déclenchent AVANT les pics réels de mobilisation (mesurés par
# # le mot-clé "villes mortes"/"manifestation"), et de COMBIEN DE JOURS à
# # l'avance en moyenne. Un système d'alerte qui ne fait que confirmer la
# # crise APRÈS qu'elle a éclaté n'a aucune valeur d'anticipation.

# print("\n--- Section G : délai d'anticipation des alertes ---")

# # On identifie les jours de PIC réel de mobilisation : les jours où
# # l'indicateur "Mobilisation" est dans le top 10% (donc des jours où on
# # parle vraiment beaucoup, concrètement, d'actions collectives)
# seuil_pic_mobilisation = indicateurs["Mobilisation"].quantile(0.90)
# jours_pic_mobilisation = indicateurs[indicateurs["Mobilisation"] >= seuil_pic_mobilisation].index.tolist()

# print(f"Jours identifiés comme des 'pics de mobilisation réelle' : {len(jours_pic_mobilisation)}")

# delais_anticipation = []
# for jour_pic in jours_pic_mobilisation:
#     # Pour chaque pic réel, on cherche s'il y a eu une alerte (Orange/Rouge/
#     # Critique) dans les 10 jours QUI PRÉCÈDENT ce pic
#     fenetre_recherche = [jour_pic - pd.Timedelta(days=n) for n in range(1, 11)]
#     fenetre_recherche = [j for j in fenetre_recherche if j in indicateurs.index]

#     jours_alerte_avant = [j for j in fenetre_recherche if indicateurs.loc[j, "est_en_alerte"]]
#     if len(jours_alerte_avant) > 0:
#         # Le délai d'anticipation = nombre de jours entre l'alerte la PLUS
#         # PRÉCOCE dans cette fenêtre et le jour du pic réel
#         alerte_la_plus_precoce = max(jours_alerte_avant, key=lambda j: (jour_pic - j).days)
#         delai = (jour_pic - alerte_la_plus_precoce).days
#         delais_anticipation.append(delai)
#     else:
#         delais_anticipation.append(0)  # aucune alerte détectée avant ce pic : délai nul

# tableau_anticipation = pd.DataFrame({
#     "jour_pic_mobilisation": jours_pic_mobilisation,
#     "delai_anticipation_jours": delais_anticipation,
# })
# tableau_anticipation.to_csv(f"{dossier_tableaux}/G1_delai_anticipation.csv", index=False)

# nb_pics_anticipes = (tableau_anticipation["delai_anticipation_jours"] > 0).sum()
# taux_anticipation = nb_pics_anticipes / len(tableau_anticipation) * 100 if len(tableau_anticipation) > 0 else 0

# print(f"\nPics de mobilisation anticipés par au moins une alerte : {nb_pics_anticipes} / {len(tableau_anticipation)} ({taux_anticipation:.0f}%)")
# print(f"Délai d'anticipation moyen (parmi les pics anticipés) : {tableau_anticipation.loc[tableau_anticipation['delai_anticipation_jours']>0, 'delai_anticipation_jours'].mean():.1f} jours")

# plt.figure(figsize=(9, 6))
# plt.hist(tableau_anticipation["delai_anticipation_jours"], bins=range(0, 12), color="#457B9D", edgecolor="white", align="left")
# plt.title("Délai d'anticipation des alertes avant un pic de mobilisation réel")
# plt.xlabel("Nombre de jours d'avance (0 = aucune anticipation)")
# plt.ylabel("Nombre de pics de mobilisation concernés")
# plt.tight_layout()
# plt.savefig(f"{dossier_figures}/G1_delai_anticipation.png")
# plt.close()

# print("Enregistré : G1_delai_anticipation (tableau + histogramme).")


# # ==============================================================================
# # SECTION H — TAUX DE DÉTECTION ET FAUX POSITIFS
# # ==============================================================================
# # Deux erreurs sont possibles pour un système d'alerte :
# #   - un FAUX NÉGATIF : une vraie crise qui n'a déclenché AUCUNE alerte
# #     (dangereux : la crise arrive "par surprise")
# #   - un FAUX POSITIF : une alerte déclenchée un jour "calme" (gênant, mais
# #     moins grave : ça use la vigilance des décideurs si c'est trop fréquent,
# #     mais ça ne rate pas de vraie crise)
# # On calcule les deux taux, en utilisant comme référence de "vraie crise"
# # les jours de forte mention de "villes mortes" (déjà utilisés en Phase 9).

# print("\n--- Section H : taux de détection et faux positifs ---")

# # On recharge juste la colonne nécessaire pour identifier les vrais jours de
# # crise (même logique que la validation de la Phase 9)
# corpus_complet = pd.read_csv(f"{dossier_data}/bd_f_themes.csv", low_memory=False)
# corpus_complet["date_publication"] = pd.to_datetime(corpus_complet["date_publication"], utc=True, errors="coerce")
# corpus_complet["texte_nettoye"] = corpus_complet["texte_nettoye"].fillna("")
# corpus_complet["jour"] = corpus_complet["date_publication"].dt.date
# corpus_complet["mention_villes_mortes"] = corpus_complet["texte_nettoye"].str.contains(
#     "ville morte|villes mortes|ghost town", regex=True, na=False
# )

# part_villes_mortes = corpus_complet.groupby("jour")["mention_villes_mortes"].mean().reindex(indicateurs.index, fill_value=0) * 100
# seuil_crise_reelle = part_villes_mortes[part_villes_mortes > 0].median() if (part_villes_mortes > 0).any() else 0
# indicateurs["jour_de_crise_reelle"] = part_villes_mortes > seuil_crise_reelle

# vrais_positifs = ((indicateurs["est_en_alerte"]) & (indicateurs["jour_de_crise_reelle"])).sum()
# faux_negatifs = ((~indicateurs["est_en_alerte"]) & (indicateurs["jour_de_crise_reelle"])).sum()
# faux_positifs = ((indicateurs["est_en_alerte"]) & (~indicateurs["jour_de_crise_reelle"])).sum()
# vrais_negatifs = ((~indicateurs["est_en_alerte"]) & (~indicateurs["jour_de_crise_reelle"])).sum()

# taux_detection = vrais_positifs / (vrais_positifs + faux_negatifs) * 100 if (vrais_positifs + faux_negatifs) > 0 else np.nan
# taux_faux_positifs = faux_positifs / (faux_positifs + vrais_negatifs) * 100 if (faux_positifs + vrais_negatifs) > 0 else np.nan

# print(f"Vrais positifs (alerte ET vraie crise)      : {vrais_positifs} jours")
# print(f"Faux négatifs (pas d'alerte MAIS vraie crise) : {faux_negatifs} jours")
# print(f"Faux positifs (alerte MAIS jour calme)        : {faux_positifs} jours")
# print(f"Vrais négatifs (pas d'alerte ET jour calme)   : {vrais_negatifs} jours")
# print(f"\nTaux de détection (sensibilité) : {taux_detection:.1f}%")
# print(f"Taux de faux positifs           : {taux_faux_positifs:.1f}%")

# tableau_confusion = pd.DataFrame({
#     "Vraie crise": [vrais_positifs, faux_negatifs],
#     "Jour calme": [faux_positifs, vrais_negatifs],
# }, index=["Alerte déclenchée", "Pas d'alerte"])
# tableau_confusion.to_csv(f"{dossier_tableaux}/H1_matrice_confusion.csv")
# print("\nMatrice de confusion :")
# print(tableau_confusion)

# plt.figure(figsize=(6, 5))
# plt.imshow(tableau_confusion.values, cmap="Blues")
# plt.xticks([0, 1], tableau_confusion.columns)
# plt.yticks([0, 1], tableau_confusion.index)
# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, tableau_confusion.values[i, j], ha="center", va="center", fontsize=14)
# plt.title("Matrice de confusion du système d'alerte")
# plt.tight_layout()
# plt.savefig(f"{dossier_figures}/H1_matrice_confusion.png")
# plt.close()

# print("Enregistré : H1_matrice_confusion (tableau + graphique).")


# # ==============================================================================
# # SECTION I — JOURNAL FINAL DES ALERTES (livrable pour un système réel)
# # ==============================================================================
# # Le document final : un vrai "journal d'alerte" tel qu'un système déployé
# # produirait, avec une ligne par jour, le niveau, et le ou les indicateurs
# # responsables du déclenchement.

# print("\n--- Section I : journal final des alertes ---")


# def lister_causes(ligne):
#     causes = []
#     for indicateur in indicateurs_surveilles:
#         if ligne[f"alerte_{indicateur}"] in ("Orange", "Rouge"):
#             causes.append(f"{indicateur}={ligne[f'alerte_{indicateur}']}")
#     if ligne["regle_combinee_1_risque_tension"]:
#         causes.append("Règle combinée: Risque thématique + Tension émotionnelle")
#     if ligne["regle_combinee_2_mobilisation_polarisation"]:
#         causes.append("Règle combinée: Mobilisation + Polarisation")
#     return " | ".join(causes) if causes else "-"


# indicateurs["causes_alerte"] = indicateurs.apply(lister_causes, axis=1)

# journal_alertes = indicateurs.loc[indicateurs["ALERTE_GLOBALE"] != "Vert",
#                                    ["ALERTE_GLOBALE", "Score_Risque_Global", "causes_alerte"]]
# journal_alertes.to_csv(f"{dossier_tableaux}/JOURNAL_alertes_final.csv")

# print(f"Journal final : {len(journal_alertes)} jours avec une alerte active sur {len(indicateurs)} jours analysés.")
# print("Enregistré : JOURNAL_alertes_final.csv")

# # On resauvegarde aussi le tableau complet (D2) avec les colonnes ajoutées
# # depuis (niveau_numerique, est_en_alerte, causes_alerte, jour_de_crise_reelle)
# # pour que la version finale sur le disque soit la plus complète possible.
# indicateurs.to_csv(f"{dossier_tableaux}/D2_alerte_globale_quotidienne.csv")


# # ==============================================================================
# # FIN DU SCRIPT
# # ==============================================================================

# print("\n=== TERMINÉ ! ===")
# print(f"Tous les graphiques sont dans : {dossier_figures}")
# print(f"Tous les tableaux sont dans   : {dossier_tableaux}")
# print("\nRécapitulatif :")
# print("  A : calibration des seuils d'alerte (percentiles 75/90)")
# print("  B : alertes simples par indicateur (ITE, ITR, Score de Risque Global)")
# print("  C : règles combinées (2 règles multi-signaux)")
# print("  D : niveau d'alerte global du jour (Vert/Orange/Rouge/Critique)")
# print("  D bis : tableau quotidien complet sauvegardé -> D2_alerte_globale_quotidienne.csv")
# print("          (CORRECTION : nouvelle source de référence unique, Vert compris)")
# print("  E : calendrier visuel des alertes")
# print("  F : regroupement en épisodes d'alerte")
# print("  G : délai d'anticipation (le test le plus important pour H1)")
# print("  H : taux de détection et faux positifs (matrice de confusion)")
# print("  I : journal final des alertes (livrable)")
