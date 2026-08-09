# ==============================================================================
# PHASE 2 — EXPLORATION DES DONNÉES (EDA)
# Version adaptée à la problématique du mémoire :
#
# "Dans quelle mesure l'intégration d'un système d'analyse en temps réel des
# sentiments et des thématiques issus des plateformes numériques peut-elle
# améliorer l'anticipation et la gestion des crises post-électorales au
# Cameroun ?"
#
# Ce script explore le corpus non pas de façon générique, mais en gardant en
# tête les hypothèses du mémoire :
#   H1 : les plateformes numériques donnent des signaux précurseurs de tension
#   H2 : croiser sentiments ET thématiques donne une vision plus fine
#   H4 : le système doit être adapté aux réalités camerounaises (langues,
#        diaspora, fracture francophone/anglophone...)
#
# Chaque grande section du script précise à quelle(s) hypothèse(s) elle sert.
# ==============================================================================


# ------------------------------------------------------------------------------
# ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# ------------------------------------------------------------------------------

import pandas as pd               # pour lire et manipuler le tableau de données
import matplotlib.pyplot as plt   # pour dessiner des graphiques
import re                         # pour chercher des motifs dans du texte
import os                         # pour créer des dossiers
from collections import Counter   # pour compter des occurrences facilement


# ------------------------------------------------------------------------------
# ÉTAPE 1 : PRÉPARER LES DOSSIERS DE SORTIE
# ------------------------------------------------------------------------------

dossier_figures = "../outputs/figures"
dossier_tableaux = "../outputs/tables"
os.makedirs(dossier_figures, exist_ok=True)
os.makedirs(dossier_tableaux, exist_ok=True)
print("Dossiers de sortie prêts.")


# ------------------------------------------------------------------------------
# ÉTAPE 2 : CHARGER ET NETTOYER LES DONNÉES
# ------------------------------------------------------------------------------

print("Lecture du fichier CSV...")
df = pd.read_csv("../data/bd_f.csv", low_memory=False)
print(f"On a chargé {len(df)} lignes au départ.")

df = df.dropna(subset=["texte"])
print(f"Après avoir enlevé les lignes sans texte, il reste {len(df)} publications.")

# On transforme la date en vraie date (et pas juste du texte)
df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
df["jour"] = df["date_publication"].dt.date
df["heure"] = df["date_publication"].dt.hour

# Longueur des textes
df["nb_caracteres"] = df["texte"].str.len()
df["nb_mots"] = df["texte"].str.split().str.len()

# On s'assure que les colonnes d'engagement sont bien des nombres
df["nb_likes"] = pd.to_numeric(df["nb_likes"], errors="coerce")
df["nb_reponses"] = pd.to_numeric(df["nb_reponses"], errors="coerce")
df["nb_partages"] = pd.to_numeric(df["nb_partages"], errors="coerce")

# On garde une version du texte en minuscules, sans les accents bizarres,
# qu'on va réutiliser dans plusieurs analyses ci-dessous pour chercher des mots-clés.
df["texte_minuscule"] = df["texte"].astype(str).str.lower()


# ------------------------------------------------------------------------------
# ÉTAPE 3 : DÉFINIR LA DATE DU VOTE ET LES DEUX PÉRIODES (AVANT / APRÈS)
# ------------------------------------------------------------------------------
# Pourquoi c'est important : ton mémoire porte sur la CRISE POST-ÉLECTORALE.
# Tout notre découpage temporel va donc tourner autour de cette date pivot,
# le 12 octobre 2025 (jour du vote). Ça nous permettra de voir si les
# publications avant le vote annoncent déjà la tension qui suit
# (c'est exactement ce que teste H1 : les signaux précurseurs).

date_du_vote = pd.Timestamp("2025-10-12", tz="UTC")

df["periode"] = "après le vote"
df.loc[df["date_publication"] < date_du_vote, "periode"] = "avant le vote"

nb_avant = (df["periode"] == "avant le vote").sum()
nb_apres = (df["periode"] == "après le vote").sum()
print(f"\nPublications avant le vote (< 12 oct. 2025) : {nb_avant}")
print(f"Publications après le vote (>= 12 oct. 2025) : {nb_apres}")


# ------------------------------------------------------------------------------
# ÉTAPE 4 : DÉFINIR LES ACTEURS POLITIQUES ET LES MOTS-CLÉS DE CRISE À SUIVRE
# ------------------------------------------------------------------------------
# Ces dictionnaires listent, pour chaque acteur ou chaque type d'événement,
# les mots qu'on va chercher dans les textes. On les utilisera dans plusieurs
# analyses ci-dessous, donc on les définit une seule fois ici, en haut du script.

acteurs_politiques = {
    "Issa Tchiroma": ["tchiroma"],
    "Cabral Libii": ["cabral", "libii", "libie"],
    "Paul Biya": ["biya"],
    "Maurice Kamto": ["kamto"],
    "Samuel Hiram Iyodi": ["hiram", "iyodi"],
    "Patricia Ndam Njoya": ["ndam njoya", "ndam"],
    "Bello Bouba Maigari": ["bello bouba", "maigari"],
    "Joshua Osih": ["osih"],
}

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


# ==============================================================================
# SECTION A — VOLUMES GÉNÉRAUX (base de toute analyse, sert de point de repère)
# ==============================================================================

print("\n--- A1 : nombre de publications par plateforme ---")

nb_par_plateforme = df["plateforme"].value_counts()
print(nb_par_plateforme)
nb_par_plateforme.to_csv(f"{dossier_tableaux}/A1_nb_publications_par_plateforme.csv")

plt.figure(figsize=(7, 5))
nb_par_plateforme.plot(kind="bar", color=["#1877F2", "#000000", "#555555"])
plt.title("Nombre de publications par plateforme")
plt.xlabel("Plateforme")
plt.ylabel("Nombre de publications")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/A1_nb_publications_par_plateforme.png")
plt.close()


# ==============================================================================
# SECTION B — ÉVOLUTION TEMPORELLE AUTOUR DU VOTE (sert directement H1)
# ==============================================================================
# H1 dit : les plateformes numériques donnent des signaux précurseurs de
# tension. Pour le vérifier, il faut D'ABORD regarder si le volume de
# publications augmente à l'approche du vote, et s'il explose après.
# On trace donc l'évolution jour par jour, avec une ligne verticale qui
# marque le jour du vote.

print("\n--- B1 : évolution du nombre de publications par jour, autour du vote ---")

nb_par_jour = df.groupby("jour").size()
nb_par_jour.to_csv(f"{dossier_tableaux}/B1_nb_publications_par_jour.csv")

plt.figure(figsize=(14, 5))
nb_par_jour.plot(kind="line", color="#1877F2")
plt.axvline(date_du_vote.date(), color="red", linestyle="--", label="Jour du vote (12/10/2025)")
plt.title("Évolution du nombre de publications par jour")
plt.xlabel("Date")
plt.ylabel("Nombre de publications")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/B1_evolution_publications_par_jour.png")
plt.close()

# On refait le même graphique mais zoomé sur les 60 jours avant et après le
# vote, car sur toute la période (2014-2026) le pic autour de l'élection est
# écrasé par l'échelle. C'est cette version zoomée qui sera la plus parlante
# dans le mémoire pour discuter de H1.
fenetre_debut = date_du_vote - pd.Timedelta(days=60)
fenetre_fin = date_du_vote + pd.Timedelta(days=60)
masque_fenetre = (df["date_publication"] >= fenetre_debut) & (df["date_publication"] <= fenetre_fin)
nb_par_jour_zoome = df[masque_fenetre].groupby("jour").size()
nb_par_jour_zoome.to_csv(f"{dossier_tableaux}/B2_nb_publications_par_jour_zoome_vote.csv")

plt.figure(figsize=(14, 5))
nb_par_jour_zoome.plot(kind="line", color="#1877F2", marker="o", markersize=3)
plt.axvline(date_du_vote.date(), color="red", linestyle="--", label="Jour du vote (12/10/2025)")
plt.title("Évolution du nombre de publications : 60 jours avant / après le vote")
plt.xlabel("Date")
plt.ylabel("Nombre de publications")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/B2_evolution_publications_zoome_vote.png")
plt.close()

print("Graphiques enregistrés : B1 (vue globale) et B2 (zoom autour du vote).")


# ==============================================================================
# SECTION C — SIGNAUX DE CRISE DANS LE TEMPS (cœur de la vérification de H1)
# ==============================================================================
# Ici on cherche, jour par jour, le nombre de publications qui contiennent au
# moins un mot lié à la crise (violence, manifestation, arrestation...).
# Si ces mentions augmentent AVANT le 12 octobre 2025, cela va dans le sens
# de H1 : le numérique capte des signaux de tension avant qu'elle n'éclate
# vraiment. On calcule aussi une moyenne mobile sur 7 jours, qui lisse les
# variations jour par jour et permet de mieux voir la tendance de fond.

print("\n--- C1 : évolution des mentions de mots-clés de crise, jour par jour ---")

# On construit une seule grande expression de recherche qui regroupe TOUS les
# mots-clés de crise (toutes catégories confondues), séparés par "|" qui veut
# dire "ou" en langage regex.
tous_les_mots_crise = []
for liste_mots in evenements_de_crise.values():
    tous_les_mots_crise.extend(liste_mots)
motif_crise = "|".join(tous_les_mots_crise)

# str.contains() renvoie True/False pour chaque ligne, selon si le texte
# contient un des mots-clés recherchés.
df["mention_crise"] = df["texte_minuscule"].str.contains(motif_crise, regex=True, na=False)

nb_posts_crise = df["mention_crise"].sum()
print(f"Nombre de publications contenant au moins un mot-clé de crise : {nb_posts_crise}")

# Évolution jour par jour du nombre de publications "de crise"
evolution_crise = df[df["mention_crise"]].groupby("jour").size()
# On calcule la moyenne mobile sur 7 jours pour lisser la courbe
moyenne_mobile_crise = evolution_crise.rolling(window=7, min_periods=1).mean()

tableau_evolution_crise = pd.DataFrame({
    "nb_publications_crise": evolution_crise,
    "moyenne_mobile_7_jours": moyenne_mobile_crise,
})
tableau_evolution_crise.to_csv(f"{dossier_tableaux}/C1_evolution_mentions_crise.csv")

plt.figure(figsize=(14, 5))
plt.plot(evolution_crise.index, evolution_crise.values, color="lightcoral", alpha=0.5, label="Par jour (brut)")
plt.plot(moyenne_mobile_crise.index, moyenne_mobile_crise.values, color="darkred", linewidth=2,
         label="Moyenne mobile (7 jours)")
plt.axvline(date_du_vote.date(), color="black", linestyle="--", label="Jour du vote (12/10/2025)")
plt.title("Évolution des publications mentionnant des mots-clés de crise")
plt.xlabel("Date")
plt.ylabel("Nombre de publications")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/C1_evolution_mentions_crise.png")
plt.close()

print("Graphique enregistré : C1_evolution_mentions_crise.png")

# On regarde maintenant chaque type d'événement séparément (villes mortes,
# manifestations, violence...), pour savoir LEQUEL domine et QUAND.
# Ça sert H2 : ce n'est pas juste "il y a de la tension", mais "quel type de
# tension, et à quel moment" — un premier pas vers le croisement thématique.

print("\n--- C2 : répartition des mentions de crise par catégorie ---")

compte_par_categorie = {}
for categorie, mots_cles in evenements_de_crise.items():
    motif = "|".join(mots_cles)
    nb_mentions = df["texte_minuscule"].str.contains(motif, regex=True, na=False).sum()
    compte_par_categorie[categorie] = nb_mentions

tableau_categories_crise = pd.Series(compte_par_categorie).sort_values(ascending=False)
print(tableau_categories_crise)
tableau_categories_crise.to_csv(f"{dossier_tableaux}/C2_mentions_par_categorie_crise.csv")

plt.figure(figsize=(8, 5))
tableau_categories_crise.sort_values().plot(kind="barh", color="#B22222")
plt.title("Nombre de publications par type d'événement de crise")
plt.xlabel("Nombre de publications")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/C2_mentions_par_categorie_crise.png")
plt.close()

# Et surtout : est-ce que ces catégories sont plus présentes AVANT ou APRÈS
# le vote ? C'est la question centrale de H1.
print("\n--- C3 : mentions de crise, avant vs après le vote ---")

resultat_avant_apres = {}
for categorie, mots_cles in evenements_de_crise.items():
    motif = "|".join(mots_cles)
    contient_mot_cle = df["texte_minuscule"].str.contains(motif, regex=True, na=False)
    nb_avant_cat = (contient_mot_cle & (df["periode"] == "avant le vote")).sum()
    nb_apres_cat = (contient_mot_cle & (df["periode"] == "après le vote")).sum()
    resultat_avant_apres[categorie] = {"avant le vote": nb_avant_cat, "après le vote": nb_apres_cat}

tableau_avant_apres_crise = pd.DataFrame(resultat_avant_apres).T  # .T retourne le tableau (lignes <-> colonnes)
print(tableau_avant_apres_crise)
tableau_avant_apres_crise.to_csv(f"{dossier_tableaux}/C3_mentions_crise_avant_apres.csv")

plt.figure(figsize=(9, 6))
tableau_avant_apres_crise.plot(kind="barh", ax=plt.gca(), color=["#4C9A2A", "#B22222"])
plt.title("Mentions de mots-clés de crise : avant vs après le vote")
plt.xlabel("Nombre de publications")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/C3_mentions_crise_avant_apres.png")
plt.close()

print("Résultats enregistrés : C2 (répartition par type) et C3 (avant/après le vote).")


# ==============================================================================
# SECTION D — LES ACTEURS POLITIQUES : QUI EST MENTIONNÉ, QUAND, ET AVEC QUEL
# ENGAGEMENT ? (sert H1 et H2 : les acteurs sont au cœur des thématiques)
# ==============================================================================

print("\n--- D1 : nombre total de mentions par acteur politique ---")

# Pour chaque acteur, on compte combien de publications le mentionnent
nb_mentions_par_acteur = {}
for acteur, mots_cles in acteurs_politiques.items():
    motif = "|".join(mots_cles)
    nb_mentions_par_acteur[acteur] = df["texte_minuscule"].str.contains(motif, regex=True, na=False).sum()

tableau_acteurs = pd.Series(nb_mentions_par_acteur).sort_values(ascending=False)
print(tableau_acteurs)
tableau_acteurs.to_csv(f"{dossier_tableaux}/D1_nb_mentions_par_acteur.csv")

plt.figure(figsize=(8, 5))
tableau_acteurs.sort_values().plot(kind="barh", color="#4169E1")
plt.title("Nombre de publications mentionnant chaque acteur politique")
plt.xlabel("Nombre de publications")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/D1_nb_mentions_par_acteur.png")
plt.close()

# --- Est-ce que les acteurs sont mentionnés différemment avant/après le vote ? ---
# Ça peut révéler un changement de figure centrale du débat (ex : on parle
# surtout de Biya avant, et de Tchiroma après une contestation des résultats).

print("\n--- D2 : mentions par acteur, avant vs après le vote ---")

resultat_acteurs_periode = {}
for acteur, mots_cles in acteurs_politiques.items():
    motif = "|".join(mots_cles)
    contient_acteur = df["texte_minuscule"].str.contains(motif, regex=True, na=False)
    nb_avant_acteur = (contient_acteur & (df["periode"] == "avant le vote")).sum()
    nb_apres_acteur = (contient_acteur & (df["periode"] == "après le vote")).sum()
    resultat_acteurs_periode[acteur] = {"avant le vote": nb_avant_acteur, "après le vote": nb_apres_acteur}

tableau_acteurs_periode = pd.DataFrame(resultat_acteurs_periode).T
print(tableau_acteurs_periode)
tableau_acteurs_periode.to_csv(f"{dossier_tableaux}/D2_mentions_acteurs_avant_apres.csv")

plt.figure(figsize=(9, 6))
tableau_acteurs_periode.plot(kind="barh", ax=plt.gca(), color=["#4C9A2A", "#B22222"])
plt.title("Mentions de chaque acteur : avant vs après le vote")
plt.xlabel("Nombre de publications")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/D2_mentions_acteurs_avant_apres.png")
plt.close()

# --- Quel acteur génère le plus d'engagement (likes) quand il est mentionné ? ---
# Un acteur peu mentionné mais très "engageant" peut quand même être un point
# de tension important : le nombre de mentions seul ne suffit pas.

print("\n--- D3 : engagement moyen (likes) des publications, par acteur mentionné ---")

resultat_engagement_acteur = {}
for acteur, mots_cles in acteurs_politiques.items():
    motif = "|".join(mots_cles)
    contient_acteur = df["texte_minuscule"].str.contains(motif, regex=True, na=False)
    likes_moyen = df.loc[contient_acteur, "nb_likes"].mean()
    reponses_moyenne = df.loc[contient_acteur, "nb_reponses"].mean()
    resultat_engagement_acteur[acteur] = {
        "likes_moyens": round(likes_moyen, 1),
        "reponses_moyennes": round(reponses_moyenne, 1),
    }

tableau_engagement_acteur = pd.DataFrame(resultat_engagement_acteur).T
tableau_engagement_acteur = tableau_engagement_acteur.sort_values("likes_moyens", ascending=False)
print(tableau_engagement_acteur)
tableau_engagement_acteur.to_csv(f"{dossier_tableaux}/D3_engagement_moyen_par_acteur.csv")

print("Résultats enregistrés : D1, D2 et D3 (mentions et engagement par acteur).")


# ==============================================================================
# SECTION E — FRANCOPHONE / ANGLOPHONE / DIASPORA (sert directement H4 :
# adapter le système aux réalités camerounaises)
# ==============================================================================
# Le Cameroun a une fracture linguistique et politique connue entre les
# régions francophones et anglophones. La colonne "pays_auteur" distingue
# déjà "Cameroun", "indéterminé (anglophone)" et "Autre (diaspora)" : on
# regarde donc comment ces groupes se répartissent, notamment par plateforme,
# ce qui doit guider comment le futur système d'analyse doit être calibré.

print("\n--- E1 : répartition des auteurs (Cameroun / anglophone / diaspora / étranger) ---")

nb_par_origine = df["pays_auteur"].value_counts()
print(nb_par_origine)
nb_par_origine.to_csv(f"{dossier_tableaux}/E1_repartition_origine_auteurs.csv")

plt.figure(figsize=(9, 6))
nb_par_origine.sort_values().plot(kind="barh", color="#DAA520")
plt.title("Répartition des publications par origine déclarée de l'auteur")
plt.xlabel("Nombre de publications")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/E1_repartition_origine_auteurs.png")
plt.close()

# Croisement origine x plateforme : est-ce que la diaspora ou les anglophones
# utilisent une plateforme en particulier ? (utile pour savoir où chercher
# quel type de signal)
print("\n--- E2 : origine des auteurs, par plateforme ---")

origine_par_plateforme = df.groupby(["plateforme", "pays_auteur"]).size().unstack(fill_value=0)
print(origine_par_plateforme)
origine_par_plateforme.to_csv(f"{dossier_tableaux}/E2_origine_par_plateforme.csv")

print("Résultats enregistrés : E1 et E2 (origine des auteurs).")


# ==============================================================================
# SECTION F — RÉPARTITION DES LANGUES (sert aussi H4)
# ==============================================================================
# Rappel : cette colonne est très souvent vide (environ 65% des cas). On
# l'affiche quand même à titre indicatif ; la vraie détection automatique de
# la langue (pour combler les trous) sera faite en Phase 3.

print("\n--- F1 : répartition des langues déclarées ---")

langues = df["langue"].fillna("non renseignée")
nb_par_langue = langues.value_counts()
print(nb_par_langue.head(10))
nb_par_langue.to_csv(f"{dossier_tableaux}/F1_repartition_langues.csv")

top_10_langues = nb_par_langue.head(10).copy()
reste = nb_par_langue.iloc[10:].sum()
if reste > 0:
    top_10_langues["autres"] = reste

plt.figure(figsize=(8, 8))
plt.pie(top_10_langues.values, labels=top_10_langues.index, autopct="%1.1f%%")
plt.title("Répartition des publications par langue déclarée")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/F1_repartition_langues.png")
plt.close()


# ==============================================================================
# SECTION G — MOTS ET HASHTAGS LES PLUS FRÉQUENTS, AVANT VS APRÈS LE VOTE
# ==============================================================================
# On ne regarde plus seulement "les mots les plus fréquents" en général : on
# compare le vocabulaire AVANT et APRÈS le vote. Si le vocabulaire change
# nettement (par exemple : mots plus neutres avant, mots liés à la violence
# après), c'est un argument fort pour H1 et H2.

mots_a_ignorer = set([
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "au", "aux",
    "en", "pour", "par", "sur", "avec", "est", "sont", "ce", "cette", "ces",
    "qui", "que", "quoi", "dont", "où", "il", "elle", "ils", "elles", "nous",
    "vous", "je", "tu", "on", "se", "sa", "son", "ses", "leur", "leurs",
    "mais", "ou", "donc", "car", "ni", "comme", "plus", "moins", "très",
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "with",
    "and", "or", "that", "this", "these", "those", "it", "its",
])


def compter_mots_frequents(sous_dataframe, top_n=20):
    """
    Cette petite fonction fait ce qu'on faisait déjà avant : elle parcourt
    tous les textes d'un groupe de publications (par exemple "avant le vote"),
    et retourne les mots les plus fréquents, en ignorant les mots vides.
    On la met dans une fonction ici, car on va l'utiliser deux fois de suite
    (une fois pour "avant", une fois pour "après") : ça évite de copier-coller
    exactement le même bloc de code deux fois.
    """
    liste_de_mots = []
    for texte in sous_dataframe["texte_minuscule"]:
        texte_sans_liens = re.sub(r"http\S+", "", str(texte))
        mots_trouves = re.findall(r"\b[a-zàâäéèêëîïôöùûüç]{3,}\b", texte_sans_liens)
        for mot in mots_trouves:
            if mot not in mots_a_ignorer:
                liste_de_mots.append(mot)
    return Counter(liste_de_mots).most_common(top_n)


print("\n--- G1 : mots les plus fréquents, avant le vote ---")
mots_avant = compter_mots_frequents(df[df["periode"] == "avant le vote"])
for mot, nombre in mots_avant:
    print(f"  {mot} : {nombre} fois")
pd.DataFrame(mots_avant, columns=["mot", "nombre"]).to_csv(f"{dossier_tableaux}/G1_top_mots_avant_vote.csv", index=False)

print("\n--- G2 : mots les plus fréquents, après le vote ---")
mots_apres = compter_mots_frequents(df[df["periode"] == "après le vote"])
for mot, nombre in mots_apres:
    print(f"  {mot} : {nombre} fois")
pd.DataFrame(mots_apres, columns=["mot", "nombre"]).to_csv(f"{dossier_tableaux}/G2_top_mots_apres_vote.csv", index=False)

# Un graphique côte à côte pour comparer visuellement les deux vocabulaires
fig, (graphique_gauche, graphique_droit) = plt.subplots(1, 2, figsize=(14, 8))

mots_avant_df = pd.DataFrame(mots_avant, columns=["mot", "nombre"]).sort_values("nombre")
graphique_gauche.barh(mots_avant_df["mot"], mots_avant_df["nombre"], color="#4C9A2A")
graphique_gauche.set_title("Top mots AVANT le vote")

mots_apres_df = pd.DataFrame(mots_apres, columns=["mot", "nombre"]).sort_values("nombre")
graphique_droit.barh(mots_apres_df["mot"], mots_apres_df["nombre"], color="#B22222")
graphique_droit.set_title("Top mots APRÈS le vote")

plt.tight_layout()
plt.savefig(f"{dossier_figures}/G3_comparaison_mots_avant_apres.png")
plt.close()

print("Résultats enregistrés : G1, G2 (tableaux) et G3 (comparaison visuelle).")

# --- Même logique pour les hashtags ---


def compter_hashtags_frequents(sous_dataframe, top_n=15):
    liste_hashtags = []
    for texte in sous_dataframe["texte_minuscule"]:
        liste_hashtags.extend(re.findall(r"#(\w+)", str(texte)))
    return Counter(liste_hashtags).most_common(top_n)


print("\n--- G4 : hashtags les plus utilisés, avant vs après le vote ---")
hashtags_avant = compter_hashtags_frequents(df[df["periode"] == "avant le vote"])
hashtags_apres = compter_hashtags_frequents(df[df["periode"] == "après le vote"])

print("Avant le vote :", hashtags_avant[:10])
print("Après le vote :", hashtags_apres[:10])

pd.DataFrame(hashtags_avant, columns=["hashtag", "nombre"]).to_csv(f"{dossier_tableaux}/G4_hashtags_avant_vote.csv", index=False)
pd.DataFrame(hashtags_apres, columns=["hashtag", "nombre"]).to_csv(f"{dossier_tableaux}/G5_hashtags_apres_vote.csv", index=False)

print("Résultats enregistrés : G4 et G5 (hashtags avant/après).")


# ==============================================================================
# SECTION H — ENGAGEMENT GLOBAL ET PUBLICATIONS LES PLUS VIRALES LIÉES À LA CRISE
# ==============================================================================
# On ne regarde pas juste "les publications les plus populaires en général",
# mais spécifiquement celles qui parlent de la crise : ce sont probablement
# les publications les plus utiles à examiner en détail dans le mémoire
# (études de cas, verbatims...).

print("\n--- H1 : engagement moyen par plateforme ---")

engagement_moyen = df.groupby("plateforme")[["nb_likes", "nb_reponses", "nb_partages"]].mean().round(1)
print(engagement_moyen)
engagement_moyen.to_csv(f"{dossier_tableaux}/H1_engagement_moyen_par_plateforme.csv")

print("\n--- H2 : publications les plus populaires, parmi celles liées à la crise ---")

colonnes_a_garder = ["plateforme", "date_publication", "texte", "nb_likes", "nb_reponses", "nb_partages"]

top_populaires_crise = (
    df[df["mention_crise"]]
    .sort_values("nb_likes", ascending=False)[colonnes_a_garder]
    .head(20)
)
top_populaires_crise.to_csv(f"{dossier_tableaux}/H2_top_20_publications_crise_populaires.csv", index=False)

top_commentees_crise = (
    df[df["mention_crise"]]
    .sort_values("nb_reponses", ascending=False)[colonnes_a_garder]
    .head(20)
)
top_commentees_crise.to_csv(f"{dossier_tableaux}/H3_top_20_publications_crise_commentees.csv", index=False)

# Et, à titre de comparaison, les publications les plus populaires TOUTES
# CATÉGORIES confondues (pour voir si ce sont les mêmes ou pas)
top_populaires_general = df.sort_values("nb_likes", ascending=False)[colonnes_a_garder].head(20)
top_populaires_general.to_csv(f"{dossier_tableaux}/H4_top_20_publications_populaires_general.csv", index=False)

print("Résultats enregistrés : H1 à H4 (engagement et publications les plus virales).")


# ==============================================================================
# SECTION I — LONGUEUR DES PUBLICATIONS PAR PLATEFORME (contexte de base)
# ==============================================================================

print("\n--- I1 : longueur moyenne des publications par plateforme ---")

longueur_moyenne = df.groupby("plateforme")["nb_mots"].mean().round(1)
print(longueur_moyenne)
longueur_moyenne.to_csv(f"{dossier_tableaux}/I1_longueur_moyenne_par_plateforme.csv")

plt.figure(figsize=(7, 5))
longueur_moyenne.plot(kind="bar", color="#4C9A2A")
plt.title("Longueur moyenne des publications (nombre de mots) par plateforme")
plt.xlabel("Plateforme")
plt.ylabel("Nombre moyen de mots")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/I1_longueur_moyenne_par_plateforme.png")
plt.close()


# ==============================================================================
# FIN DU SCRIPT
# ==============================================================================

print("\n=== TERMINÉ ! ===")
print(f"Tous les graphiques sont dans : {dossier_figures}")
print(f"Tous les tableaux sont dans   : {dossier_tableaux}")
print("\nRappel des sections et de leur lien avec le mémoire :")
print("  A : volumes généraux (contexte de base)")
print("  B : évolution temporelle autour du vote (H1)")
print("  C : signaux de crise dans le temps, par catégorie, avant/après (H1, H2)")
print("  D : mentions et engagement par acteur politique (H1, H2)")
print("  E : origine des auteurs - Cameroun/anglophone/diaspora (H4)")
print("  F : répartition des langues (H4)")
print("  G : vocabulaire et hashtags avant/après le vote (H1, H2)")
print("  H : publications les plus virales liées à la crise (matière pour le chapitre III)")
print("  I : longueur des publications par plateforme (contexte de base)")
