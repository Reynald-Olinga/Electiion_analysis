# ==============================================================================
# PHASE 9 — CONSTRUCTION DES INDICATEURS DÉCISIONNELS
# ==============================================================================
#
# À quoi sert ce script ?
# ------------------------
# C'est LA phase charnière du mémoire : celle où on arrête de faire "juste"
# de l'analyse académique (sentiments, émotions, thèmes...) pour construire
# un vrai OUTIL D'AIDE À LA DÉCISION. Au lieu de dire "voici ce que les gens
# ont dit", on construit des INDICATEURS chiffrés, calculés jour après jour,
# que des décideurs (autorités, observatoires, ONG...) pourraient surveiller
# en temps réel pour anticiper une crise post-électorale.
#
# On construit ici 9 indicateurs, comme le prévoit ton canevas :
#   1. ITE  — Indice de Tension Émotionnelle
#   2. ITR  — Indice Thématique de Risque
#   3. Indice de Viralité
#   4. Indice de Polarisation
#   5. Indice de Mobilisation
#   6. Indice de Désinformation (proxy)
#   7. Indice d'Activité
#   8. Indice d'Engagement
#   9. Score de Risque Global (combinaison pondérée des 8 précédents)
#
# PRINCIPE GÉNÉRAL DE CONSTRUCTION D'UN INDICATEUR (à bien comprendre avant
# de lire le code) :
#   Étape 1 : on calcule une valeur BRUTE par jour (ex : le score moyen de
#             colère+peur+dégoût+tristesse de toutes les publications du jour)
#   Étape 2 : cette valeur brute a une échelle qui n'a pas de sens en soi
#             (est-ce que 0.15 de "tension" c'est beaucoup ou peu ?). On la
#             ramène donc sur une échelle commune de 0 à 100 par une
#             NORMALISATION MIN-MAX : la valeur la plus basse observée sur la
#             période devient 0, la valeur la plus haute devient 100, et
#             tout le reste est réparti proportionnellement entre les deux.
#             Ça permet de comparer des indicateurs qui, à la base, n'ont
#             rien à voir entre eux (une proportion, un score de -1 à 1, un
#             nombre de likes...) sur une même échelle lisible.
#   Étape 3 : on lisse légèrement (moyenne mobile sur 3 jours) pour ne pas
#             réagir au moindre bruit statistique d'un jour isolé.
#
# LIMITE IMPORTANTE À GARDER EN TÊTE (et à mentionner dans le mémoire) :
# cette normalisation 0-100 est faite par rapport à la fenêtre de données
# qu'on a sous la main (ici : ±90 jours autour du vote). Un "100" veut donc
# dire "le jour le plus extrême QU'ON A OBSERVÉ", pas un seuil absolu et
# universel. Un vrai système déployé en continu aurait besoin de plusieurs
# mois/années d'historique pour calibrer des seuils absolus fiables. C'est
# une limite assumée et normale à ce stade du mémoire.
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
# ÉTAPE 1 : CHARGER LE CORPUS COMPLET (Phase 7, avec sentiment + émotions + thèmes)
# ------------------------------------------------------------------------------

print("Lecture du corpus complet...")
df = pd.read_csv(f"{dossier_data}/bd_f_themes.csv", low_memory=False)
df["nb_likes"] = pd.to_numeric(df["nb_likes"], errors="coerce").fillna(0)
df["nb_reponses"] = pd.to_numeric(df["nb_reponses"], errors="coerce").fillna(0)
df["nb_partages"] = pd.to_numeric(df["nb_partages"], errors="coerce").fillna(0)
df["texte_nettoye"] = df["texte_nettoye"].fillna("")
print(f"{len(df)} publications chargées.")

df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
date_du_vote = pd.Timestamp("2025-10-12", tz="UTC")


# ==============================================================================
# ÉTAPE 2 — DÉFINITION DE LA FENÊTRE D'ANALYSE ET DU SEUIL DE FIABILITÉ
# ==============================================================================
# On limite l'analyse à une fenêtre de ±90 jours autour du vote. Pourquoi ?
# En dehors de cette période, le corpus ne contient que 1 ou 2 publications
# par jour (voir Phase 2) : un indicateur calculé sur si peu de publications
# n'a AUCUNE valeur statistique (une seule publication très en colère
# suffirait à faire croire à une "tension" à 100%, alors que ça ne reflète
# rien de collectif). On préfère donc restreindre l'analyse à une période où
# le volume de données permet des moyennes vraiment représentatives.

print("\n--- Étape 2 : définition de la fenêtre d'analyse ---")

fenetre_debut = date_du_vote - pd.Timedelta(days=90)
fenetre_fin = date_du_vote + pd.Timedelta(days=90)
df_fenetre = df[(df["date_publication"] >= fenetre_debut) & (df["date_publication"] <= fenetre_fin)].copy()
df_fenetre["jour"] = df_fenetre["date_publication"].dt.date

print(f"Fenêtre d'analyse : {fenetre_debut.date()} -> {fenetre_fin.date()} (180 jours)")
print(f"{len(df_fenetre)} publications dans cette fenêtre.")

# On construit une grille avec TOUS les jours de la fenêtre (même ceux sans
# aucune publication), pour ne jamais avoir de "trou" dans les indicateurs
tous_les_jours = pd.date_range(fenetre_debut.date(), fenetre_fin.date(), freq="D").date
volume_quotidien = df_fenetre.groupby("jour").size().reindex(tous_les_jours, fill_value=0)
volume_quotidien.index.name = "jour"

# SEUIL DE FIABILITÉ : en dessous de 5 publications dans la journée, on
# considère que l'indicateur de ce jour-là est statistiquement fragile. On ne
# les supprime pas (on garde l'info), mais on les MARQUE clairement, et les
# graphiques les distingueront visuellement (zone grisée).
SEUIL_FIABILITE = 5
jours_fiables = volume_quotidien >= SEUIL_FIABILITE
print(f"Jours avec au moins {SEUIL_FIABILITE} publications (fiables) : {jours_fiables.sum()} / {len(volume_quotidien)}")

volume_quotidien.to_csv(f"{dossier_tableaux}/00_volume_quotidien_fenetre.csv")


def normaliser_0_100(serie):
    """
    Ramène n'importe quelle série de valeurs sur une échelle de 0 à 100.
    Le minimum observé devient 0, le maximum devient 100, et chaque valeur
    intermédiaire est replacée proportionnellement entre les deux.
    C'est la fonction qu'on va réutiliser pour CHAQUE indicateur de cette
    phase, afin qu'ils soient tous exprimés sur la même échelle et donc
    directement comparables entre eux.
    """
    valeur_min = serie.min()
    valeur_max = serie.max()
    if valeur_max == valeur_min:
        return serie * 0  # évite une division par zéro si toutes les valeurs sont identiques
    return (serie - valeur_min) / (valeur_max - valeur_min) * 100


def lisser(serie, fenetre_jours=3):
    """Moyenne mobile courte (3 jours), pour ne pas réagir au moindre bruit
    d'une journée isolée tout en restant réactif (un système d'anticipation
    de crise ne peut pas se permettre un lissage trop long, qui retarderait
    la détection d'un vrai signal)."""
    return serie.rolling(window=fenetre_jours, min_periods=1, center=False).mean()


def tracer_indicateur(serie_normalisee, titre, chemin_sortie, couleur="#1d3557"):
    """Fonction réutilisée pour tracer chaque indicateur avec la même mise en
    forme : la zone des jours peu fiables est grisée, le jour du vote est
    marqué, et la courbe est lissée."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(serie_normalisee.index, serie_normalisee.values, color=couleur, linewidth=2)

    # On grise en fond les jours en dessous du seuil de fiabilité
    for jour, est_fiable in jours_fiables.items():
        if not est_fiable:
            ax.axvspan(jour, jour, color="gray", alpha=0.08)

    ax.axvline(date_du_vote.date(), color="black", linestyle="--", label="Jour du vote (12/10/2025)")
    ax.set_title(titre)
    ax.set_ylabel("Indice (0-100)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()


# ==============================================================================
# INDICATEUR 1 — ITE : INDICE DE TENSION ÉMOTIONNELLE
# ==============================================================================
# PRINCIPE : parmi les 8 émotions de Plutchik calculées en Phase 6, quatre
# expriment une détresse ou une tension (Colère, Peur, Dégoût, Tristesse),
# alors que les 4 autres (Joie, Surprise, Confiance, Anticipation) sont soit
# neutres soit plutôt positives. L'ITE mesure la part de "charge émotionnelle
# négative" dans l'ensemble des publications d'un jour donné.
#
# FORMULE : pour chaque jour, on fait la moyenne de
#   (emotion_Colère + emotion_Peur + emotion_Dégoût + emotion_Tristesse)
# sur toutes les publications du jour (ces 4 valeurs sont déjà des
# proportions calculées en Phase 6), puis on normalise sur 0-100.

print("\n--- Indicateur 1 : ITE (Indice de Tension Émotionnelle) ---")

df_fenetre["charge_tension_emotionnelle"] = (
    df_fenetre["emotion_Colère"].fillna(0) + df_fenetre["emotion_Peur"].fillna(0)
    + df_fenetre["emotion_Dégoût"].fillna(0) + df_fenetre["emotion_Tristesse"].fillna(0)
)

ite_brut = df_fenetre.groupby("jour")["charge_tension_emotionnelle"].mean().reindex(tous_les_jours)
ite_lisse = lisser(ite_brut)
ITE = normaliser_0_100(ite_lisse)

tracer_indicateur(ITE, "ITE — Indice de Tension Émotionnelle", f"{dossier_figures}/IND1_ITE.png", couleur="#6A0572")
print(f"ITE calculé. Moyenne : {ITE.mean():.1f}/100, jour le plus tendu : {ITE.idxmax()} ({ITE.max():.1f}/100)")


# ==============================================================================
# INDICATEUR 2 — ITR : INDICE THÉMATIQUE DE RISQUE
# ==============================================================================
# PRINCIPE : tous les thèmes ne représentent pas le même niveau de risque
# (voir Phase 8 : "Paul Biya : pouvoir, vie et mort" a un indice de tension
# de 0.638, contre 0.463 pour "Candidats de l'opposition"). L'ITR mesure si
# la conversation du jour se déplace vers des thèmes plus ou moins à risque.
#
# FORMULE : pour chaque jour, on prend la part (%) de publications dans
# chaque thème, et on la multiplie par le "poids de risque" de ce thème
# (calculé en Phase 8), puis on additionne. C'est une MOYENNE PONDÉRÉE : si
# 80% des publications du jour sont dans un thème à risque élevé, l'ITR sera
# élevé, même si le VOLUME total de publications n'a pas changé.

print("\n--- Indicateur 2 : ITR (Indice Thématique de Risque) ---")

# Poids de risque par thème, repris de l'indice de tension thématique de la
# Phase 8 (section "INDICE_tension_thematique")
poids_risque_themes = {
    "Paul Biya : pouvoir, vie et mort": 0.638,
    "Communauté Bamiléké, famille et problèmes du pays": 0.598,
    "Solidarité, foi et encouragements au peuple": 0.596,
    "Tribalisme, régime et tensions régionales (Ouest)": 0.582,
    "Candidats de l'opposition et espoir de changement": 0.463,
}

df_fenetre["poids_risque_theme"] = df_fenetre["theme_nom"].map(poids_risque_themes)

# Pour les publications dont le thème n'est pas dans notre liste (langue non
# française, ou "non déterminé"), on ne peut pas leur donner un poids de
# risque : on les exclut simplement de CE calcul précis (mais elles restent
# comptées dans les autres indicateurs, comme l'activité ou l'engagement).
itr_brut = df_fenetre.groupby("jour")["poids_risque_theme"].mean().reindex(tous_les_jours)
itr_lisse = lisser(itr_brut)
ITR = normaliser_0_100(itr_lisse)

tracer_indicateur(ITR, "ITR — Indice Thématique de Risque", f"{dossier_figures}/IND2_ITR.png", couleur="#9D0208")
print(f"ITR calculé. Moyenne : {ITR.mean():.1f}/100, jour le plus à risque : {ITR.idxmax()} ({ITR.max():.1f}/100)")


# ==============================================================================
# INDICATEUR 3 — INDICE D'ACTIVITÉ
# ==============================================================================
# PRINCIPE : le plus simple des indicateurs. Il mesure juste le VOLUME de
# publications par jour : combien de personnes parlent du sujet aujourd'hui,
# indépendamment de ce qu'elles en disent. Une hausse brutale de l'activité,
# même neutre, est déjà un signal (quelque chose retient l'attention).

print("\n--- Indicateur 3 : Indice d'Activité ---")

activite_lissee = lisser(volume_quotidien.astype(float))
ACTIVITE = normaliser_0_100(activite_lissee)

tracer_indicateur(ACTIVITE, "Indice d'Activité (volume de publications)", f"{dossier_figures}/IND3_ACTIVITE.png", couleur="#F77F00")
print(f"Indice d'Activité calculé. Pic d'activité : {ACTIVITE.idxmax()} ({volume_quotidien[ACTIVITE.idxmax()]} publications ce jour-là)")


# ==============================================================================
# INDICATEUR 4 — INDICE D'ENGAGEMENT
# ==============================================================================
# PRINCIPE : différent de l'activité ! L'activité compte le nombre de
# publications ; l'engagement compte la RÉACTION TOTALE du public à ces
# publications (likes + réponses + partages additionnés). On peut avoir
# beaucoup de publications mais peu de réactions (sujet qui n'intéresse
# personne), ou peu de publications mais énormément de réactions (un post
# qui devient viral). Ici, on additionne le TOTAL de la journée (pas la
# moyenne), car on veut mesurer le volume global de réaction collective.

print("\n--- Indicateur 4 : Indice d'Engagement ---")

df_fenetre["engagement_total_publication"] = (
    df_fenetre["nb_likes"] + df_fenetre["nb_reponses"] + df_fenetre["nb_partages"]
)
engagement_brut = df_fenetre.groupby("jour")["engagement_total_publication"].sum().reindex(tous_les_jours, fill_value=0)
engagement_lisse = lisser(engagement_brut.astype(float))
ENGAGEMENT = normaliser_0_100(engagement_lisse)

tracer_indicateur(ENGAGEMENT, "Indice d'Engagement (réactions totales du public)", f"{dossier_figures}/IND4_ENGAGEMENT.png", couleur="#2A9D8F")
print(f"Indice d'Engagement calculé. Pic : {ENGAGEMENT.idxmax()}")


# ==============================================================================
# INDICATEUR 5 — INDICE DE VIRALITÉ
# ==============================================================================
# PRINCIPE : distinct de l'engagement ! L'engagement mesure le TOTAL de
# réactions (donc il monte automatiquement s'il y a plus de publications).
# La viralité mesure l'INTENSITÉ DE CONTAGION MOYENNE : est-ce qu'une
# publication typique de ce jour-là déclenche beaucoup de réactions PAR
# RAPPORT À SON NOMBRE DE PUBLICATIONS ? On utilise une moyenne pondérée où
# un partage compte plus qu'un commentaire, qui compte plus qu'un like — car
# partager demande plus d'effort et amplifie davantage le message qu'un
# simple like.

print("\n--- Indicateur 5 : Indice de Viralité ---")

df_fenetre["score_viralite_publication"] = (
    df_fenetre["nb_likes"] * 1 + df_fenetre["nb_reponses"] * 2 + df_fenetre["nb_partages"] * 3
)
viralite_brute = df_fenetre.groupby("jour")["score_viralite_publication"].mean().reindex(tous_les_jours, fill_value=0)
viralite_lissee = lisser(viralite_brute.astype(float))
VIRALITE = normaliser_0_100(viralite_lissee)

tracer_indicateur(VIRALITE, "Indice de Viralité (intensité de contagion moyenne par publication)", f"{dossier_figures}/IND5_VIRALITE.png", couleur="#F72585")
print(f"Indice de Viralité calculé. Pic : {VIRALITE.idxmax()}")


# ==============================================================================
# INDICATEUR 6 — INDICE DE POLARISATION
# ==============================================================================
# PRINCIPE : un climat social peut être "négatif en moyenne" (tout le monde
# est mécontent, mais d'accord sur ce mécontentement) OU "polarisé" (une
# partie du public est très positive, une autre très négative, avec peu de
# terrain neutre). Ces deux situations sont TRÈS différentes pour la gestion
# de crise, mais un simple score de sentiment MOYEN ne les distingue pas (les
# deux peuvent donner une moyenne proche de zéro).
#
# FORMULE : on utilise l'ÉCART-TYPE (la dispersion) des scores de sentiment
# individuels du jour, plutôt que leur moyenne. Un écart-type élevé signifie
# que les opinions du jour sont très dispersées (certaines très positives,
# d'autres très négatives) = polarisation. Un écart-type faible signifie que
# tout le monde est à peu près sur la même longueur d'onde (qu'elle soit
# positive, négative ou neutre) = pas de polarisation.

print("\n--- Indicateur 6 : Indice de Polarisation ---")

polarisation_brute = df_fenetre.groupby("jour")["score_sentiment"].std().reindex(tous_les_jours)
polarisation_lissee = lisser(polarisation_brute)
POLARISATION = normaliser_0_100(polarisation_lissee)

tracer_indicateur(POLARISATION, "Indice de Polarisation (dispersion des opinions)", f"{dossier_figures}/IND6_POLARISATION.png", couleur="#7209B7")
print(f"Indice de Polarisation calculé. Pic : {POLARISATION.idxmax()}")


# ==============================================================================
# INDICATEUR 7 — INDICE DE MOBILISATION
# ==============================================================================
# PRINCIPE : mesure si le contenu du jour appelle ou évoque une action
# collective concrète (manifestations, villes mortes, grèves, appels à
# descendre dans la rue...), plutôt qu'un simple commentaire d'opinion. Un
# pic de mobilisation est un signal direct et concret d'un risque de
# débordement dans la rue, donc particulièrement important à surveiller.

print("\n--- Indicateur 7 : Indice de Mobilisation ---")

mots_cles_mobilisation = [
    "manifestation", "manifestant", "ville morte", "villes mortes", "ghost town",
    "grève", "greve", "marche", "rassemblement", "mobilisation", "mobiliser",
    "boycott", "sit-in", "descendre dans la rue", "appel à manifester", "sortir",
]
motif_mobilisation = "|".join(mots_cles_mobilisation)
df_fenetre["contient_mobilisation"] = df_fenetre["texte_nettoye"].str.contains(motif_mobilisation, regex=True, na=False)

mobilisation_brute = df_fenetre.groupby("jour")["contient_mobilisation"].mean().reindex(tous_les_jours, fill_value=0) * 100
mobilisation_lissee = lisser(mobilisation_brute)
MOBILISATION = normaliser_0_100(mobilisation_lissee)

tracer_indicateur(MOBILISATION, "Indice de Mobilisation (appels à l'action collective)", f"{dossier_figures}/IND7_MOBILISATION.png", couleur="#BC6C25")
print(f"Indice de Mobilisation calculé. Pic : {MOBILISATION.idxmax()}")


# ==============================================================================
# INDICATEUR 8 — INDICE DE DÉSINFORMATION (PROXY LEXICAL)
# ==============================================================================
# LIMITE MÉTHODOLOGIQUE IMPORTANTE À ASSUMER CLAIREMENT : on n'a AUCUN moyen,
# avec les données de ce mémoire, de vérifier si une information est vraie ou
# fausse (ça demanderait une base de fact-checking, comme celles tenues par
# des organismes comme Africa Check). Ce qu'on peut mesurer, en revanche,
# c'est la PRÉSENCE d'un VOCABULAIRE ÉVOQUANT le doute sur l'information
# elle-même (rumeur, intox, manipulation...). C'est donc un indicateur
# PROXY (une approximation indirecte), pas une vraie détection de fausses
# informations. Ce point doit être mentionné explicitement dans le mémoire.

print("\n--- Indicateur 8 : Indice de Désinformation (proxy lexical) ---")

mots_cles_desinformation = [
    "rumeur", "intox", "fake", "désinformation", "desinformation", "manipulation",
    "manipuler", "mensonge", "fausse information", "faux", "complot", "propagande",
    "intoxication", "infox",
]
motif_desinformation = "|".join(mots_cles_desinformation)
df_fenetre["contient_desinformation"] = df_fenetre["texte_nettoye"].str.contains(motif_desinformation, regex=True, na=False)

desinformation_brute = df_fenetre.groupby("jour")["contient_desinformation"].mean().reindex(tous_les_jours, fill_value=0) * 100
desinformation_lissee = lisser(desinformation_brute)
DESINFORMATION = normaliser_0_100(desinformation_lissee)

tracer_indicateur(DESINFORMATION, "Indice de Désinformation - PROXY lexical (rumeur/mensonge/complot)", f"{dossier_figures}/IND8_DESINFORMATION.png", couleur="#495057")
print(f"Indice de Désinformation (proxy) calculé. Pic : {DESINFORMATION.idxmax()}")
print("RAPPEL : cet indice mesure la présence d'un VOCABULAIRE du doute, pas la véracité réelle des faits.")


# ==============================================================================
# INDICATEUR 9 — SCORE DE RISQUE GLOBAL
# ==============================================================================
# On combine maintenant les indicateurs précédents en un seul score de
# synthèse. Ce n'est pas une simple moyenne : chaque indicateur reçoit un
# POIDS selon son importance conceptuelle pour anticiper une crise (H1) :
#
#   - ITE et ITR (40% à eux deux) : ce sont les signaux les plus directement
#     liés à la nature même de la crise (tension émotionnelle et thèmes à
#     risque), donc les plus lourds dans le score.
#   - Polarisation et Mobilisation (30% à eux deux) : des signaux de
#     dynamique sociale, moins fréquents mais très significatifs quand ils
#     apparaissent.
#   - Désinformation et Viralité (20% à eux deux) : des signaux
#     "amplificateurs", qui aggravent une situation déjà tendue mais ne la
#     créent pas à eux seuls.
#   - Activité et Engagement (10% à eux deux) : du contexte utile, mais le
#     volume seul ne dit rien sur la nature (bonne ou mauvaise) de ce dont
#     on parle — on leur donne donc le poids le plus faible.
#
# CE CHOIX DE PONDÉRATION EST TRANSPARENT MAIS RESTE UNE PROPOSITION : dans
# un vrai système déployé, ces poids devraient être calibrés et validés avec
# des experts du contexte camerounais et/ou par comparaison avec des crises
# passées documentées. C'est un point à mentionner explicitement dans la
# discussion critique de ton mémoire.

print("\n--- Indicateur 9 : Score de Risque Global ---")

PONDERATIONS = {
    "ITE": 0.20,
    "ITR": 0.20,
    "Polarisation": 0.15,
    "Mobilisation": 0.15,
    "Désinformation": 0.10,
    "Viralité": 0.10,
    "Activité": 0.05,
    "Engagement": 0.05,
}
print("Pondérations utilisées :", PONDERATIONS)
assert abs(sum(PONDERATIONS.values()) - 1.0) < 1e-9, "Les pondérations doivent totaliser 100%"

SCORE_RISQUE_GLOBAL = (
    ITE * PONDERATIONS["ITE"] + ITR * PONDERATIONS["ITR"]
    + POLARISATION * PONDERATIONS["Polarisation"] + MOBILISATION * PONDERATIONS["Mobilisation"]
    + DESINFORMATION * PONDERATIONS["Désinformation"] + VIRALITE * PONDERATIONS["Viralité"]
    + ACTIVITE * PONDERATIONS["Activité"] + ENGAGEMENT * PONDERATIONS["Engagement"]
)

tracer_indicateur(SCORE_RISQUE_GLOBAL, "SCORE DE RISQUE GLOBAL (synthèse pondérée des 8 indicateurs)",
                   f"{dossier_figures}/IND9_SCORE_RISQUE_GLOBAL.png", couleur="#780000")
print(f"Score de Risque Global calculé. Pic : {SCORE_RISQUE_GLOBAL.idxmax()} ({SCORE_RISQUE_GLOBAL.max():.1f}/100)")


# ==============================================================================
# TABLEAU DE BORD COMBINÉ — TOUS LES INDICATEURS SUR UN MÊME GRAPHIQUE
# ==============================================================================

print("\n--- Tableau de bord combiné ---")

tableau_indicateurs = pd.DataFrame({
    "volume_publications": volume_quotidien,
    "jour_fiable": jours_fiables,
    "ITE": ITE, "ITR": ITR, "Activite": ACTIVITE, "Engagement": ENGAGEMENT,
    "Viralite": VIRALITE, "Polarisation": POLARISATION, "Mobilisation": MOBILISATION,
    "Desinformation": DESINFORMATION, "Score_Risque_Global": SCORE_RISQUE_GLOBAL,
})
tableau_indicateurs.to_csv(f"{dossier_tableaux}/INDICATEURS_tableau_complet_quotidien.csv")

fig, ax = plt.subplots(figsize=(16, 8))
couleurs_indicateurs = {
    "ITE": "#6A0572", "ITR": "#9D0208", "Polarisation": "#7209B7", "Mobilisation": "#BC6C25",
    "Desinformation": "#495057", "Viralite": "#F72585", "Activite": "#F77F00", "Engagement": "#2A9D8F",
}
for nom_colonne, couleur in couleurs_indicateurs.items():
    ax.plot(tableau_indicateurs.index, tableau_indicateurs[nom_colonne], label=nom_colonne, color=couleur, linewidth=1, alpha=0.7)

ax.plot(tableau_indicateurs.index, tableau_indicateurs["Score_Risque_Global"], label="SCORE DE RISQUE GLOBAL",
        color="black", linewidth=3)
ax.axvline(date_du_vote.date(), color="red", linestyle="--", linewidth=2, label="Jour du vote")
ax.set_title("Tableau de bord des indicateurs décisionnels (fenêtre ±90 jours autour du vote)")
ax.set_ylabel("Indice (0-100)")
ax.set_xlabel("Date")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{dossier_figures}/TABLEAU_DE_BORD_tous_indicateurs.png")
plt.close()

print("Enregistré : INDICATEURS_tableau_complet_quotidien.csv et TABLEAU_DE_BORD_tous_indicateurs.png")


# ==============================================================================
# VALIDATION 1 — CORRÉLATIONS ENTRE INDICATEURS
# ==============================================================================
# Deux indicateurs très fortement corrélés entre eux apportent une
# information redondante (ils "disent la même chose"). C'est normal d'en
# avoir un peu (ITE et ITR, par exemple, sont conceptuellement proches), mais
# ça permet de vérifier qu'on n'a pas construit 8 fois le même indicateur
# sous des noms différents.

print("\n--- Validation 1 : corrélations entre indicateurs ---")

colonnes_a_correler = ["ITE", "ITR", "Activite", "Engagement", "Viralite", "Polarisation", "Mobilisation", "Desinformation"]
matrice_correlation = tableau_indicateurs[colonnes_a_correler].corr()
matrice_correlation.to_csv(f"{dossier_tableaux}/VALIDATION1_correlation_indicateurs.csv")
print(matrice_correlation.round(2))

plt.figure(figsize=(9, 8))
carte = plt.imshow(matrice_correlation.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(carte, label="Corrélation")
plt.xticks(range(len(colonnes_a_correler)), colonnes_a_correler, rotation=45, ha="right")
plt.yticks(range(len(colonnes_a_correler)), colonnes_a_correler)
for i in range(len(colonnes_a_correler)):
    for j in range(len(colonnes_a_correler)):
        plt.text(j, i, f"{matrice_correlation.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Corrélation entre les indicateurs décisionnels")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/VALIDATION1_correlation_indicateurs.png")
plt.close()

print("Enregistré : VALIDATION1 (tableau + heatmap).")


# ==============================================================================
# VALIDATION 2 — LE SCORE DE RISQUE GLOBAL RÉAGIT-IL AUX ÉVÉNEMENTS CONNUS ?
# ==============================================================================
# Test de VALIDITÉ DE FAÇADE ("face validity") : si notre score de risque
# est pertinent, il devrait être nettement plus élevé lors des journées où
# les publications parlent le plus de "villes mortes" (un évènement de crise
# bien identifié dès la Phase 2) que lors des journées "calmes". C'est une
# vérification simple mais importante avant de proposer cet outil comme
# système d'aide à la décision.

print("\n--- Validation 2 : le score réagit-il aux journées de forte mention 'villes mortes' ? ---")

df_fenetre["mention_villes_mortes"] = df_fenetre["texte_nettoye"].str.contains(
    "ville morte|villes mortes|ghost town", regex=True, na=False
)
part_villes_mortes_quotidien = df_fenetre.groupby("jour")["mention_villes_mortes"].mean().reindex(tous_les_jours, fill_value=0) * 100

# On sépare les jours en 2 groupes : ceux où on parle BEAUCOUP de villes
# mortes (au-dessus de la médiane des jours qui en parlent au moins un peu)
# et les autres, puis on compare le Score de Risque Global moyen des deux groupes
seuil_villes_mortes = part_villes_mortes_quotidien[part_villes_mortes_quotidien > 0].median() if (part_villes_mortes_quotidien > 0).any() else 0
jours_forte_mention = part_villes_mortes_quotidien > seuil_villes_mortes

score_moyen_jours_crise = SCORE_RISQUE_GLOBAL[jours_forte_mention].mean()
score_moyen_jours_calmes = SCORE_RISQUE_GLOBAL[~jours_forte_mention].mean()

print(f"Score de Risque Global moyen les jours de forte mention 'villes mortes' : {score_moyen_jours_crise:.1f}/100")
print(f"Score de Risque Global moyen les autres jours                          : {score_moyen_jours_calmes:.1f}/100")

resultat_validation = pd.Series({
    "score_moyen_jours_forte_mention_villes_mortes": score_moyen_jours_crise,
    "score_moyen_autres_jours": score_moyen_jours_calmes,
})
resultat_validation.to_csv(f"{dossier_tableaux}/VALIDATION2_score_vs_villes_mortes.csv")

plt.figure(figsize=(7, 6))
couleurs_validation = ["#9D0208", "#2A9D8F"]
plt.bar(["Jours de forte mention\n'villes mortes'", "Autres jours"],
        [score_moyen_jours_crise, score_moyen_jours_calmes], color=couleurs_validation)
plt.title("Le Score de Risque Global est-il plus élevé lors des pics de 'villes mortes' ?")
plt.ylabel("Score de Risque Global moyen (0-100)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/VALIDATION2_score_vs_villes_mortes.png")
plt.close()

print("Enregistré : VALIDATION2 (tableau + graphique).")
if score_moyen_jours_crise > score_moyen_jours_calmes:
    print("-> Résultat cohérent : le score réagit bien plus fort lors des journées de crise identifiée.")
else:
    print("-> ATTENTION : le score ne distingue pas bien les journées de crise. À investiguer/ajuster les pondérations.")


# ==============================================================================
# FIN DU SCRIPT
# ==============================================================================

print("\n=== TERMINÉ ! ===")
print(f"Tous les graphiques sont dans : {dossier_figures}")
print(f"Tous les tableaux sont dans   : {dossier_tableaux}")
print("\nRécapitulatif des 9 indicateurs construits :")
print("  1. ITE            - Indice de Tension Émotionnelle")
print("  2. ITR             - Indice Thématique de Risque")
print("  3. Activité        - volume quotidien de publications")
print("  4. Engagement      - total des réactions (likes+réponses+partages)")
print("  5. Viralité        - intensité de contagion moyenne par publication")
print("  6. Polarisation    - dispersion (écart-type) des sentiments")
print("  7. Mobilisation    - présence de vocabulaire d'action collective")
print("  8. Désinformation  - PROXY lexical (rumeur/mensonge/complot)")
print("  9. Score de Risque Global - synthèse pondérée des indicateurs 1 à 8")
print("\nFichier prêt pour la Phase 10 (détection automatique des alertes) :")
print(f"  {dossier_tableaux}/INDICATEURS_tableau_complet_quotidien.csv")
