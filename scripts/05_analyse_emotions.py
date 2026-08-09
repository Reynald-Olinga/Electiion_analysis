# ==============================================================================
# PHASE 6 — ANALYSE DES ÉMOTIONS
# ==============================================================================
#
# À quoi sert ce script ?
# ------------------------
# La Phase 5 a mesuré le sentiment en une seule dimension : positif <-> négatif.
# Mais deux publications "négatives" peuvent exprimer des choses très
# différentes : de la PEUR (on craint pour l'avenir) n'appelle pas la même
# réponse institutionnelle que de la COLÈRE (on veut agir contre quelque
# chose) ou du DÉGOÛT (on rejette moralement une situation). C'est tout
# l'intérêt d'aller plus loin que le sentiment simple.
#
# On utilise le modèle des 8 émotions de base de Plutchik (largement utilisé
# en recherche) : Colère, Peur, Joie, Tristesse, Dégoût, Surprise, Confiance,
# Anticipation.
#
# DÉCISION MÉTHODOLOGIQUE (même logique qu'en Phase 5) :
#   - Pour l'ANGLAIS/PIDGIN : on utilise NRCLex, une librairie basée sur le
#     lexique NRC (National Research Council du Canada), validé
#     scientifiquement et déjà annoté selon les 8 émotions de Plutchik.
#     Comme pour VADER, tout est intégré dans la librairie : pas besoin de
#     connexion internet pour la faire fonctionner.
#   - Pour le FRANÇAIS : NRCLex ne couvre que l'anglais, donc on construit
#     notre propre lexique français des 8 émotions, à la main, avec à la
#     fois du vocabulaire général ET du vocabulaire propre au contexte
#     électoral/crise du mémoire.
#
# On repart du fichier produit par la Phase 5 (bd_f_sentiment.csv), qui
# contient déjà "texte_sentiment" : le texte lemmatisé en conservant les
# négations. Pas besoin de refaire ce travail ici.
# ==============================================================================


# ------------------------------------------------------------------------------
# ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from nrclex import NRCLex

# La liste des 8 émotions de Plutchik qu'on va utiliser partout dans ce script
EMOTIONS = ["Colère", "Peur", "Joie", "Tristesse", "Dégoût", "Surprise", "Confiance", "Anticipation"]

# Une couleur bien distincte pour chaque émotion, réutilisée dans TOUS les
# graphiques du script pour que la même émotion soit toujours reconnaissable
# à sa couleur d'un graphique à l'autre.
COULEURS_EMOTIONS = {
    "Colère": "#E63946",       # rouge
    "Peur": "#6A0572",         # violet foncé
    "Joie": "#F4A300",         # jaune/orange
    "Tristesse": "#457B9D",    # bleu
    "Dégoût": "#588157",       # vert olive
    "Surprise": "#F72585",     # rose vif
    "Confiance": "#2A9D8F",    # turquoise
    "Anticipation": "#B08968",  # marron/beige
}


# ------------------------------------------------------------------------------
# ÉTAPE 1 : PRÉPARER LES DOSSIERS DE SORTIE
# ------------------------------------------------------------------------------

dossier_data = "../data"
dossier_figures = "../outputs/figures"
dossier_tableaux = "../outputs/tables"
os.makedirs(dossier_figures, exist_ok=True)
os.makedirs(dossier_tableaux, exist_ok=True)


# ------------------------------------------------------------------------------
# ÉTAPE 2 : CHARGER LE CORPUS AVEC SENTIMENT (Phase 5)
# ------------------------------------------------------------------------------

print("Lecture du corpus (avec sentiment déjà calculé)...")
df = pd.read_csv(f"{dossier_data}/bd_f_sentiment.csv", low_memory=False)
df["texte_sentiment"] = df["texte_sentiment"].fillna("")
df["texte_nettoye"] = df["texte_nettoye"].fillna("")
df["nb_likes"] = pd.to_numeric(df["nb_likes"], errors="coerce")
print(f"{len(df)} publications chargées.")

df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
date_du_vote = pd.Timestamp("2025-10-12", tz="UTC")
df["periode"] = "après le vote"
df.loc[df["date_publication"] < date_du_vote, "periode"] = "avant le vote"


# ==============================================================================
# SECTION A — LE LEXIQUE FRANÇAIS DES 8 ÉMOTIONS
# ==============================================================================
# Comme pour le lexique de sentiment (Phase 5), on construit ici, à la main,
# une liste de mots français associés à chacune des 8 émotions de Plutchik.
# On mélange du vocabulaire général et du vocabulaire propre au contexte
# électoral/crise camerounais (répression -> peur, fraude -> dégoût/colère...).

print("\n--- A : construction du lexique français des 8 émotions ---")

lexique_emotions_fr = {
    "Colère": [
        "colère", "furieux", "furieuse", "rage", "indigner", "indigné", "révolter", "révolté",
        "scandaliser", "scandalisé", "haine", "haïr", "vengeance", "exaspérer", "exaspéré",
        "hostile", "agacer", "agacé", "insulter", "insulte", "traître", "trahir", "trahison",
        "injustice", "injuste", "abus", "abuser", "tyran", "dictateur", "oppresseur", "provocation",
        "révolte", "protester", "indignation", "outrage", "outragé", "hurler", "cri", "crier",
    ],
    "Peur": [
        "peur", "effrayer", "effrayé", "terreur", "angoisse", "angoissé", "inquiet", "inquiétude",
        "danger", "dangereux", "menace", "menacer", "risque", "risquer", "crainte", "craindre",
        "panique", "paniquer", "terrifiant", "insécurité", "effrayant", "alarmant", "alarme",
        "épouvante", "épouvantable", "trembler", "affoler", "peureux", "redouter", "phobie",
    ],
    "Joie": [
        "joie", "heureux", "heureuse", "bonheur", "content", "ravi", "réjouir", "sourire", "rire",
        "fête", "féter", "célébrer", "victoire", "réussite", "triompher", "triomphe", "allégresse",
        "enthousiasme", "enthousiaste", "gai", "gaieté", "plaisir", "satisfaction", "satisfait",
        "jubiler", "exulter", "épanoui", "vivre", "gagner",
    ],
    "Tristesse": [
        "triste", "tristesse", "chagrin", "douleur", "souffrance", "souffrir", "pleurer", "deuil",
        "désespoir", "désespérer", "malheur", "malheureux", "mélancolie", "regret", "regretter",
        "larme", "peine", "abattu", "découragé", "décourager", "affliger", "affligé", "morose",
        "solitude", "seul", "perdre", "perte", "manquer",
    ],
    "Dégoût": [
        "dégoût", "dégoûtant", "écœurant", "écœurer", "honte", "honteux", "répugnant", "répugner",
        "scandaleux", "corruption", "corrompu", "pourri", "ignoble", "abject", "mépris", "mépriser",
        "méprisable", "sale", "immonde", "vomir", "détestable", "abominable", "vil", "infâme",
    ],
    "Surprise": [
        "surprise", "surprendre", "surpris", "étonnant", "étonner", "étonné", "choc", "choquant",
        "choquer", "stupéfait", "stupéfaction", "incroyable", "inattendu", "soudain", "soudainement",
        "bouleversement", "bouleverser", "sidérant", "sidéré", "ahurissant", "ahuri", "abasourdi",
        "inimaginable", "hallucinant",
    ],
    "Confiance": [
        "confiance", "confier", "fiable", "sûr", "sécurité", "croire", "croyance", "espoir",
        "espérer", "foi", "loyal", "loyauté", "honnête", "honnêteté", "intègre", "intégrité",
        "solidarité", "soutien", "soutenir", "fidèle", "fidélité", "rassurer", "rassurant",
        "certitude", "certain", "garantir", "garantie", "sincère",
    ],
    "Anticipation": [
        "attendre", "attente", "espérer", "prévoir", "prévision", "imminent", "bientôt", "avenir",
        "futur", "préparer", "préparation", "anticiper", "prochain", "imminence", "veille",
        "alerte", "vigilance", "vigilant", "prévoir", "annoncer", "annonce", "prochainement",
        "urgence", "urgent", "attendu", "prochaine",
    ],
}

# On affiche un petit résumé de la taille de chaque lexique
for emotion, mots in lexique_emotions_fr.items():
    print(f"  {emotion} : {len(mots)} mots")

# On transforme chaque liste en "set" (ensemble), ce qui rend la recherche
# d'un mot beaucoup plus rapide qu'avec une liste classique
lexique_emotions_fr = {emotion: set(mots) for emotion, mots in lexique_emotions_fr.items()}

mots_de_negation_fr = {"pas", "ne", "non", "jamais", "aucun", "aucune", "rien", "sans", "ni"}


def calculer_emotions_francais(texte):
    """
    Compte, pour un texte français, combien de mots correspondent à chacune
    des 8 émotions. Si un mot d'émotion est précédé d'une négation ("pas de
    joie"), on ne le compte PAS (plutôt que d'essayer de deviner vers quelle
    autre émotion il faudrait le basculer, ce qui serait trop incertain).
    Retourne un dictionnaire {émotion: nombre_de_mots_trouvés}.
    """
    mots = texte.split()
    compteur_par_emotion = {emotion: 0 for emotion in EMOTIONS}

    for position, mot in enumerate(mots):
        for emotion, lexique in lexique_emotions_fr.items():
            if mot in lexique:
                fenetre_precedente = mots[max(0, position - 2):position]
                est_nie = any(m in mots_de_negation_fr for m in fenetre_precedente)
                if not est_nie:
                    compteur_par_emotion[emotion] += 1

    return compteur_par_emotion


# ==============================================================================
# SECTION B — ÉMOTIONS EN ANGLAIS / PIDGIN (NRCLex)
# ==============================================================================
# NRCLex utilise 10 catégories (les 8 émotions de Plutchik + positif/négatif
# global). On ne garde que les 8 émotions de Plutchik ici, car le
# positif/négatif a déjà été traité en Phase 5.

print("\n--- B : préparation de NRCLex (anglais + pidgin) ---")

# On fait correspondre les noms anglais de NRCLex à nos noms français
correspondance_nrclex = {
    "anger": "Colère", "fear": "Peur", "joy": "Joie", "sadness": "Tristesse",
    "disgust": "Dégoût", "surprise": "Surprise", "trust": "Confiance", "anticipation": "Anticipation",
}


def calculer_emotions_anglais(texte):
    """Utilise NRCLex sur une liste de tokens déjà nettoyés (pas de texte brut,
    pour éviter toute dépendance à des téléchargements externes)."""
    tokens = texte.split()
    if len(tokens) == 0:
        return {emotion: 0 for emotion in EMOTIONS}

    analyseur = NRCLex()
    analyseur.load_token_list(tokens)

    compteur_par_emotion = {emotion: 0 for emotion in EMOTIONS}
    for mot_nrclex, nom_francais in correspondance_nrclex.items():
        compteur_par_emotion[nom_francais] = analyseur.raw_emotion_scores.get(mot_nrclex, 0)

    return compteur_par_emotion


# ==============================================================================
# SECTION C — APPLICATION À TOUT LE CORPUS
# ==============================================================================

print("\n--- C : calcul des scores d'émotion pour toutes les publications ---")
print("(Cette étape peut prendre 2-3 minutes)")

resultats_emotions = []

for indice, ligne in df.iterrows():
    langue = ligne["langue_detectee"]
    texte = ligne["texte_sentiment"]

    if langue == "fr":
        compteur = calculer_emotions_francais(texte)
    elif langue in ("en", "pidgin"):
        compteur = calculer_emotions_anglais(texte)
    else:
        compteur = {emotion: np.nan for emotion in EMOTIONS}

    resultats_emotions.append(compteur)

tableau_emotions_brutes = pd.DataFrame(resultats_emotions, index=df.index)

# On transforme les comptages bruts en PROPORTIONS : pour chaque publication,
# quelle part de ses mots "à charge émotionnelle" va à chaque émotion. Ça
# permet de comparer des posts courts et des posts longs sur la même échelle.
total_mots_emotion = tableau_emotions_brutes[EMOTIONS].sum(axis=1)

tableau_emotions_proportions = tableau_emotions_brutes[EMOTIONS].div(total_mots_emotion.replace(0, np.nan), axis=0)

for emotion in EMOTIONS:
    df[f"emotion_{emotion}"] = tableau_emotions_proportions[emotion]

# L'émotion dominante : celle qui a la plus grande proportion pour cette
# publication (si au moins un mot d'émotion a été trouvé). On calcule
# idxmax() UNIQUEMENT sur les lignes qui ont au moins une valeur valide,
# sinon pandas plante en cherchant un maximum parmi des cases toutes vides.
masque_au_moins_un_mot = (total_mots_emotion.notna()) & (total_mots_emotion > 0)

df["emotion_dominante"] = "aucune détectée"
df.loc[masque_au_moins_un_mot, "emotion_dominante"] = (
    tableau_emotions_proportions.loc[masque_au_moins_un_mot, EMOTIONS].idxmax(axis=1)
)

print("Calcul terminé.")
print(df["emotion_dominante"].value_counts())


# ==============================================================================
# SECTION D — DISTRIBUTION GÉNÉRALE DES ÉMOTIONS DOMINANTES
# ==============================================================================

print("\n--- D : distribution des émotions dominantes ---")

repartition_emotions = df["emotion_dominante"].value_counts()
repartition_emotions.to_csv(f"{dossier_tableaux}/D1_repartition_emotions.csv")

# On enlève "aucune détectée" pour le graphique (pas très informatif à afficher)
repartition_a_afficher = repartition_emotions.drop("aucune détectée", errors="ignore")
repartition_a_afficher = repartition_a_afficher.reindex(EMOTIONS).dropna()

couleurs_ordonnees = [COULEURS_EMOTIONS[e] for e in repartition_a_afficher.index]

fig, (graphique_gauche, graphique_droit) = plt.subplots(1, 2, figsize=(15, 6))

graphique_gauche.bar(repartition_a_afficher.index, repartition_a_afficher.values, color=couleurs_ordonnees)
graphique_gauche.set_title("Nombre de publications par émotion dominante")
graphique_gauche.set_ylabel("Nombre de publications")
graphique_gauche.tick_params(axis="x", rotation=30)

graphique_droit.pie(repartition_a_afficher.values, labels=repartition_a_afficher.index, autopct="%1.1f%%", colors=couleurs_ordonnees)
graphique_droit.set_title("Répartition des émotions dominantes (%)")

plt.tight_layout()
plt.savefig(f"{dossier_figures}/D1_distribution_emotions.png")
plt.close()

print("Enregistré : D1 (tableau + graphiques).")


# ==============================================================================
# SECTION E — RADAR ÉMOTIONNEL GLOBAL
# ==============================================================================
# Le "radar émotionnel" prévu par le canevas : un graphique en toile
# d'araignée qui montre, d'un coup d'œil, l'intensité moyenne de chacune des
# 8 émotions sur l'ensemble du corpus.

print("\n--- E : radar émotionnel global ---")


def dessiner_radar_emotionnel(scores_par_emotion, titre, chemin_sortie, couleur_ligne="#1d3557", scores_secondaires=None, label_principal="Score", label_secondaire=None):
    """
    Dessine un radar (toile d'araignée) à partir d'un dictionnaire
    {émotion: score}. Si scores_secondaires est fourni, on superpose une
    deuxième forme sur le même graphique (utile pour comparer avant/après,
    ou plusieurs plateformes).
    """
    categories = EMOTIONS
    nb_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, nb_categories, endpoint=False).tolist()
    angles += angles[:1]

    valeurs = [scores_par_emotion[e] for e in categories]
    valeurs += valeurs[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.plot(angles, valeurs, color=couleur_ligne, linewidth=2, label=label_principal)
    ax.fill(angles, valeurs, color=couleur_ligne, alpha=0.2)

    if scores_secondaires is not None:
        valeurs_secondaires = [scores_secondaires[e] for e in categories]
        valeurs_secondaires += valeurs_secondaires[:1]
        ax.plot(angles, valeurs_secondaires, color="#E63946", linewidth=2, label=label_secondaire)
        ax.fill(angles, valeurs_secondaires, color="#E63946", alpha=0.15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(titre)
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()


scores_globaux = {emotion: df[f"emotion_{emotion}"].mean() for emotion in EMOTIONS}
pd.Series(scores_globaux).to_csv(f"{dossier_tableaux}/E1_scores_emotionnels_globaux.csv")

dessiner_radar_emotionnel(
    scores_globaux,
    "Radar émotionnel - ensemble du corpus",
    f"{dossier_figures}/E1_radar_emotionnel_global.png",
)

print("Enregistré : E1 (tableau + radar).")


# ==============================================================================
# SECTION F — RADAR ÉMOTIONNEL AVANT / APRÈS LE VOTE
# ==============================================================================

print("\n--- F : radar émotionnel avant/après le vote ---")

scores_avant = {emotion: df.loc[df["periode"] == "avant le vote", f"emotion_{emotion}"].mean() for emotion in EMOTIONS}
scores_apres = {emotion: df.loc[df["periode"] == "après le vote", f"emotion_{emotion}"].mean() for emotion in EMOTIONS}

tableau_avant_apres = pd.DataFrame({"avant le vote": scores_avant, "après le vote": scores_apres})
tableau_avant_apres.to_csv(f"{dossier_tableaux}/F1_scores_emotionnels_avant_apres.csv")
print(tableau_avant_apres)

dessiner_radar_emotionnel(
    scores_avant,
    "Radar émotionnel : avant vs après le vote",
    f"{dossier_figures}/F1_radar_emotionnel_avant_apres.png",
    couleur_ligne="#4C9A2A",
    scores_secondaires=scores_apres,
    label_principal="Avant le vote",
    label_secondaire="Après le vote",
)

print("Enregistré : F1 (tableau + radar comparatif).")


# ==============================================================================
# SECTION G — ÉVOLUTION TEMPORELLE DE CHAQUE ÉMOTION
# ==============================================================================
# On regarde, semaine après semaine, comment chacune des 8 émotions évolue.
# C'est la "courbe des émotions" et l'"évolution des émotions" prévues par
# le canevas.

print("\n--- G : évolution temporelle des émotions ---")

df["semaine"] = df["date_publication"].dt.to_period("W").dt.start_time

evolution_hebdo_emotions = df.groupby("semaine")[[f"emotion_{e}" for e in EMOTIONS]].mean()
evolution_hebdo_emotions.columns = EMOTIONS
evolution_hebdo_emotions.to_csv(f"{dossier_tableaux}/G1_evolution_hebdomadaire_emotions.csv")

plt.figure(figsize=(15, 7))
for emotion in EMOTIONS:
    plt.plot(evolution_hebdo_emotions.index, evolution_hebdo_emotions[emotion],
              label=emotion, color=COULEURS_EMOTIONS[emotion], linewidth=2)
plt.axvline(date_du_vote, color="black", linestyle="--", label="Jour du vote")
plt.title("Évolution hebdomadaire de l'intensité de chaque émotion")
plt.xlabel("Semaine")
plt.ylabel("Intensité moyenne (proportion)")
plt.legend(loc="upper left", ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(f"{dossier_figures}/G1_evolution_emotions_toutes.png")
plt.close()

# On zoome sur les 60 jours autour du vote (comme pour le sentiment), c'est
# la fenêtre la plus riche et la plus lisible
fenetre_debut = date_du_vote - pd.Timedelta(days=60)
fenetre_fin = date_du_vote + pd.Timedelta(days=60)
masque_fenetre = (df["date_publication"] >= fenetre_debut) & (df["date_publication"] <= fenetre_fin)

df["jour"] = df["date_publication"].dt.date
evolution_quotidienne_zoomee = df[masque_fenetre].groupby("jour")[[f"emotion_{e}" for e in EMOTIONS]].mean()
evolution_quotidienne_zoomee.columns = EMOTIONS

plt.figure(figsize=(15, 7))
for emotion in EMOTIONS:
    lissage = evolution_quotidienne_zoomee[emotion].rolling(window=5, min_periods=1).mean()
    plt.plot(evolution_quotidienne_zoomee.index, lissage, label=emotion, color=COULEURS_EMOTIONS[emotion], linewidth=2)
plt.axvline(date_du_vote.date(), color="black", linestyle="--", label="Jour du vote")
plt.title("Évolution quotidienne (lissée sur 5 jours) : 60 jours autour du vote")
plt.xlabel("Date")
plt.ylabel("Intensité moyenne (proportion)")
plt.legend(loc="upper left", ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(f"{dossier_figures}/G2_evolution_emotions_zoome_vote.png")
plt.close()

print("Enregistré : G1 (vue globale) et G2 (zoom autour du vote).")


# ==============================================================================
# SECTION H — HEATMAP ÉMOTIONNELLE (émotions x semaines)
# ==============================================================================
# Une autre façon de visualiser l'évolution : une grille de couleurs où
# chaque ligne est une émotion, chaque colonne une semaine, et la couleur
# l'intensité. Ça permet de voir d'un coup d'œil les pics de chaque émotion.

print("\n--- H : heatmap émotionnelle dans le temps ---")

# On limite aux 25 dernières semaines avec des données pour rester lisible
tableau_heatmap_temps = evolution_hebdo_emotions.tail(25).T  # .T = émotions en ligne, semaines en colonne

plt.figure(figsize=(16, 6))
carte = plt.imshow(tableau_heatmap_temps.values, cmap="inferno", aspect="auto")
plt.colorbar(carte, label="Intensité moyenne (proportion)")
plt.yticks(range(len(EMOTIONS)), EMOTIONS)
plt.xticks(range(len(tableau_heatmap_temps.columns)),
           [d.strftime("%d/%m/%y") for d in tableau_heatmap_temps.columns], rotation=90, fontsize=7)
plt.title("Heatmap émotionnelle : intensité de chaque émotion, semaine par semaine (25 dernières semaines)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/H1_heatmap_emotions_temps.png")
plt.close()

tableau_heatmap_temps.to_csv(f"{dossier_tableaux}/H1_heatmap_emotions_temps.csv")
print("Enregistré : H1 (tableau + heatmap).")


# ==============================================================================
# SECTION I — ÉMOTION DOMINANTE PAR THÈME DE CRISE
# ==============================================================================
# Pour chaque thème (violence, villes mortes, fraude...), quelle est la
# "signature émotionnelle" ? On s'attend par exemple à ce que "Violence"
# soit plutôt associé à la Peur et la Colère, et "Fraude" au Dégoût.

print("\n--- I : profil émotionnel par thème de crise ---")

evenements_de_crise = {
    "Villes mortes": ["ville morte", "villes mortes", "ghost town"],
    "Manifestations": ["manifestation", "manifestant"],
    "Violence": ["violence", "violent"],
    "Morts / tués": ["tué", "tue ", " mort ", "morts", "décès"],
    "Arrestations": ["arrestation", "arrêté", "arrete"],
    "Répression": ["répression", "repression"],
    "Fraude électorale": ["fraude"],
    "Forces de l'ordre": ["armée", "armee", "police", "gendarme", "militaire"],
}

profils_emotionnels_themes = {}
for theme, mots_cles in evenements_de_crise.items():
    motif = "|".join(mots_cles)
    masque = df["texte_nettoye"].str.contains(motif, regex=True, na=False)
    profils_emotionnels_themes[theme] = {emotion: df.loc[masque, f"emotion_{emotion}"].mean() for emotion in EMOTIONS}

tableau_profils_themes = pd.DataFrame(profils_emotionnels_themes).T
tableau_profils_themes.to_csv(f"{dossier_tableaux}/I1_profils_emotionnels_themes.csv")
print(tableau_profils_themes.round(3))

# Heatmap thèmes x émotions (très parlant pour voir la "signature" de chaque thème)
plt.figure(figsize=(10, 7))
carte = plt.imshow(tableau_profils_themes.values, cmap="YlOrRd", aspect="auto")
plt.colorbar(carte, label="Intensité moyenne")
plt.xticks(range(len(EMOTIONS)), EMOTIONS, rotation=30, ha="right")
plt.yticks(range(len(tableau_profils_themes.index)), tableau_profils_themes.index)
plt.title("Signature émotionnelle de chaque thème de crise")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/I1_heatmap_emotions_themes.png")
plt.close()

print("Enregistré : I1 (tableau + heatmap).")


# ==============================================================================
# SECTION J — ÉMOTION DOMINANTE PAR ACTEUR POLITIQUE
# ==============================================================================

print("\n--- J : profil émotionnel par acteur politique ---")

acteurs_politiques = {
    "Paul Biya": ["biya"],
    "Issa Tchiroma": ["tchiroma"],
    "Cabral Libii": ["cabral", "libii", "libie"],
    "Maurice Kamto": ["kamto"],
    "Bello Bouba Maigari": ["bello bouba", "maigari"],
    "Patricia Ndam Njoya": ["ndam njoya", "ndam"],
}

profils_emotionnels_acteurs = {}
for acteur, mots_cles in acteurs_politiques.items():
    motif = "|".join(mots_cles)
    masque = df["texte_nettoye"].str.contains(motif, regex=True, na=False)
    profils_emotionnels_acteurs[acteur] = {emotion: df.loc[masque, f"emotion_{emotion}"].mean() for emotion in EMOTIONS}

tableau_profils_acteurs = pd.DataFrame(profils_emotionnels_acteurs).T
tableau_profils_acteurs.to_csv(f"{dossier_tableaux}/J1_profils_emotionnels_acteurs.csv")
print(tableau_profils_acteurs.round(3))

# Barres empilées : pour chaque acteur, la répartition des 8 émotions (en %)
proportions_acteurs = tableau_profils_acteurs.div(tableau_profils_acteurs.sum(axis=1), axis=0) * 100

plt.figure(figsize=(11, 7))
positions_du_bas = np.zeros(len(proportions_acteurs))
for emotion in EMOTIONS:
    plt.bar(proportions_acteurs.index, proportions_acteurs[emotion], bottom=positions_du_bas,
            label=emotion, color=COULEURS_EMOTIONS[emotion])
    positions_du_bas += proportions_acteurs[emotion].values

plt.title("Profil émotionnel de chaque acteur politique (barres empilées, %)")
plt.ylabel("Pourcentage")
plt.xticks(rotation=20, ha="right")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/J1_profils_emotionnels_acteurs.png")
plt.close()

print("Enregistré : J1 (tableau + barres empilées).")


# ==============================================================================
# SECTION K — PROFIL ÉMOTIONNEL PAR PLATEFORME
# ==============================================================================

print("\n--- K : profil émotionnel par plateforme ---")

profils_emotionnels_plateformes = {}
for plateforme in df["plateforme"].unique():
    sous_ensemble = df[df["plateforme"] == plateforme]
    profils_emotionnels_plateformes[plateforme] = {emotion: sous_ensemble[f"emotion_{emotion}"].mean() for emotion in EMOTIONS}

tableau_profils_plateformes = pd.DataFrame(profils_emotionnels_plateformes)
tableau_profils_plateformes.to_csv(f"{dossier_tableaux}/K1_profils_emotionnels_plateformes.csv")

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
angles = np.linspace(0, 2 * np.pi, len(EMOTIONS), endpoint=False).tolist()
angles += angles[:1]
couleurs_plateformes_radar = {"Facebook": "#1877F2", "TikTok": "#EE1D52", "X": "#6c757d"}

for plateforme in tableau_profils_plateformes.columns:
    valeurs = tableau_profils_plateformes[plateforme].tolist()
    valeurs += valeurs[:1]
    ax.plot(angles, valeurs, label=plateforme, color=couleurs_plateformes_radar.get(plateforme), linewidth=2)
    ax.fill(angles, valeurs, color=couleurs_plateformes_radar.get(plateforme), alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(EMOTIONS)
ax.set_title("Profil émotionnel comparé des 3 plateformes")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(f"{dossier_figures}/K1_radar_emotions_plateformes.png")
plt.close()

print("Enregistré : K1 (tableau + radar comparatif).")


# ==============================================================================
# SECTION L — ÉMOTIONS ET ENGAGEMENT (quelle émotion suscite le plus de likes ?)
# ==============================================================================
# Question utile pour le futur système d'aide à la décision (Phase 9) :
# est-ce que certaines émotions rendent une publication plus "virale" ?

print("\n--- L : engagement moyen selon l'émotion dominante ---")

engagement_par_emotion = df[df["emotion_dominante"] != "aucune détectée"].groupby("emotion_dominante")["nb_likes"].agg(["mean", "count"]).round(1)
engagement_par_emotion = engagement_par_emotion.reindex(EMOTIONS).dropna()
engagement_par_emotion.to_csv(f"{dossier_tableaux}/L1_engagement_par_emotion.csv")
print(engagement_par_emotion)

couleurs_barres_engagement = [COULEURS_EMOTIONS[e] for e in engagement_par_emotion.index]
plt.figure(figsize=(9, 6))
plt.bar(engagement_par_emotion.index, engagement_par_emotion["mean"], color=couleurs_barres_engagement)
plt.title("Nombre moyen de likes selon l'émotion dominante de la publication")
plt.ylabel("Nombre moyen de likes")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/L1_engagement_par_emotion.png")
plt.close()

print("Enregistré : L1 (tableau + graphique).")


# ==============================================================================
# SECTION M — SAUVEGARDE DU CORPUS AVEC LES ÉMOTIONS
# ==============================================================================

print("\n--- M : sauvegarde du corpus enrichi avec les émotions ---")

chemin_sortie = f"{dossier_data}/bd_f_emotions.csv"
df.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")
print(f"Corpus sauvegardé ici : {chemin_sortie}")


# ==============================================================================
# FIN DU SCRIPT
# ==============================================================================

print("\n=== TERMINÉ ! ===")
print(f"Tous les graphiques sont dans : {dossier_figures}")
print(f"Tous les tableaux sont dans   : {dossier_tableaux}")
print("\nRécapitulatif :")
print("  A/B/C : lexiques et calcul des scores d'émotion (fr + en/pidgin)")
print("  D : distribution générale des émotions dominantes")
print("  E : radar émotionnel global")
print("  F : radar émotionnel avant/après le vote")
print("  G : évolution temporelle de chaque émotion (globale + zoom vote)")
print("  H : heatmap émotionnelle dans le temps")
print("  I : profil émotionnel par thème de crise (heatmap)")
print("  J : profil émotionnel par acteur politique (barres empilées)")
print("  K : profil émotionnel par plateforme (radar)")
print("  L : engagement moyen selon l'émotion dominante")
print(f"  M : corpus final sauvegardé -> {chemin_sortie}")
