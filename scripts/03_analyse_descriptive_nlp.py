# ==============================================================================
# PHASE 4 — ANALYSE DESCRIPTIVE NLP (version enrichie)
# ==============================================================================
#
# Ce script va plus loin qu'une exploration NLP générique. Chaque analyse a
# été choisie soit parce qu'elle complète une statistique descriptive
# classique qui manquait (loi de Zipf, richesse lexicale), soit parce
# qu'elle éclaire directement la problématique du mémoire (mots-compagnons
# des acteurs, mots distinctifs par plateforme, bulles fréquence x
# engagement, réseau de cooccurrence).
#
# Sections :
#   A. Top 100 des mots les plus fréquents
#   B. Bigrammes et trigrammes les plus fréquents
#   C. Loi de Zipf (rang-fréquence) — vérifie la "normalité" du corpus
#   D. Richesse lexicale (TTR) par plateforme et par période
#   E. Nuages de mots (global + avant/après le vote)
#   F. Nuage des hashtags
#   G. Treemaps des acteurs / villes / partis mentionnés
#   H. Mots-compagnons de chaque acteur politique (contexte lexical)
#   I. Mots distinctifs par plateforme (TF-IDF Facebook / TikTok / X)
#   J. Mots typiques de chaque période (TF-IDF avant / après le vote)
#   K. Bulles fréquence x engagement
#   L. Réseau de cooccurrence des mots (graphe)
# ==============================================================================


# ------------------------------------------------------------------------------
# ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import os
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import squarify           # pour dessiner des treemaps (rectangles proportionnels)
import networkx as nx      # pour dessiner un réseau de mots (graphe nœuds-liens)


# ------------------------------------------------------------------------------
# ÉTAPE 1 : PRÉPARER LES DOSSIERS DE SORTIE
# ------------------------------------------------------------------------------

dossier_figures = "../outputs/figures"
dossier_tableaux = "../outputs/tables"
os.makedirs(dossier_figures, exist_ok=True)
os.makedirs(dossier_tableaux, exist_ok=True)


# ------------------------------------------------------------------------------
# ÉTAPE 2 : CHARGER LE CORPUS NETTOYÉ (produit par la Phase 3)
# ------------------------------------------------------------------------------

print("Lecture du corpus nettoyé...")
df = pd.read_csv("../data/bd_f_nettoye.csv", low_memory=False)
print(f"{len(df)} publications chargées.")

df["texte_lemmatise"] = df["texte_lemmatise"].fillna("")
df["texte_nettoye"] = df["texte_nettoye"].fillna("")
df["nb_likes"] = pd.to_numeric(df["nb_likes"], errors="coerce")

df_avec_texte = df[df["texte_lemmatise"].str.strip() != ""].copy()
print(f"{len(df_avec_texte)} publications ont un texte lemmatisé exploitable.")

# On recrée la colonne "période" (avant/après le vote), comme en Phase 2
df_avec_texte["date_publication"] = pd.to_datetime(df_avec_texte["date_publication"], utc=True, errors="coerce")
date_du_vote = pd.Timestamp("2025-10-12", tz="UTC")
df_avec_texte["periode"] = "après le vote"
df_avec_texte.loc[df_avec_texte["date_publication"] < date_du_vote, "periode"] = "avant le vote"


# ==============================================================================
# SECTION A — TOP 100 DES MOTS LES PLUS FRÉQUENTS
# ==============================================================================

print("\n--- A : top 100 des mots les plus fréquents ---")

tous_les_mots = []
for texte in df_avec_texte["texte_lemmatise"]:
    tous_les_mots.extend(texte.split())

compteur_mots = Counter(tous_les_mots)
top_100_mots = compteur_mots.most_common(100)

tableau_top_100 = pd.DataFrame(top_100_mots, columns=["mot", "nombre_occurrences"])
tableau_top_100.to_csv(f"{dossier_tableaux}/A_top_100_mots.csv", index=False)

# On colore chaque barre selon son rang, avec un dégradé de couleurs (colormap
# "viridis"), plutôt qu'une seule couleur unie pour toutes les barres.
top_30 = tableau_top_100.head(30).sort_values("nombre_occurrences")
couleurs_degradees = cm.viridis(np.linspace(0.15, 0.95, len(top_30)))

plt.figure(figsize=(8, 10))
plt.barh(top_30["mot"], top_30["nombre_occurrences"], color=couleurs_degradees)
plt.title("Top 30 des mots les plus fréquents (corpus lemmatisé)")
plt.xlabel("Nombre d'occurrences")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/A_top_30_mots.png")
plt.close()

print("Enregistré : A_top_100_mots.csv et A_top_30_mots.png")


# ==============================================================================
# SECTION B — BIGRAMMES ET TRIGRAMMES LES PLUS FRÉQUENTS
# ==============================================================================

print("\n--- B : bigrammes et trigrammes les plus fréquents ---")

tous_les_bigrammes = []
tous_les_trigrammes = []

for texte in df_avec_texte["texte_lemmatise"]:
    mots = texte.split()
    for i in range(len(mots) - 1):
        tous_les_bigrammes.append(mots[i] + " " + mots[i + 1])
    for i in range(len(mots) - 2):
        tous_les_trigrammes.append(mots[i] + " " + mots[i + 1] + " " + mots[i + 2])

top_30_bigrammes = Counter(tous_les_bigrammes).most_common(30)
top_30_trigrammes = Counter(tous_les_trigrammes).most_common(30)

pd.DataFrame(top_30_bigrammes, columns=["bigramme", "nombre"]).to_csv(f"{dossier_tableaux}/B1_top_bigrammes.csv", index=False)
pd.DataFrame(top_30_trigrammes, columns=["trigramme", "nombre"]).to_csv(f"{dossier_tableaux}/B2_top_trigrammes.csv", index=False)

bigrammes_graphique = pd.DataFrame(top_30_bigrammes[:20], columns=["expression", "nombre"]).sort_values("nombre")
couleurs_bigrammes = cm.plasma(np.linspace(0.1, 0.9, len(bigrammes_graphique)))

plt.figure(figsize=(8, 8))
plt.barh(bigrammes_graphique["expression"], bigrammes_graphique["nombre"], color=couleurs_bigrammes)
plt.title("Top 20 des bigrammes (paires de mots) les plus fréquents")
plt.xlabel("Nombre d'occurrences")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/B1_top_bigrammes.png")
plt.close()

print("Enregistré : B1 et B2 (bigrammes/trigrammes).")


# ==============================================================================
# SECTION C — LOI DE ZIPF (RANG-FRÉQUENCE)
# ==============================================================================
# La loi de Zipf est une régularité observée dans presque tous les textes
# humains : si on classe les mots du plus fréquent au moins fréquent, la
# fréquence du mot de rang 2 est environ la moitié de celle du mot de rang 1,
# celle du rang 3 environ le tiers, etc. Autrement dit, fréquence x rang
# est à peu près constant. Vérifier que notre corpus suit cette loi est une
# façon simple de contrôler que le texte collecté est bien du "vrai" langage
# naturel (et pas, par exemple, du spam répétitif ou des données corrompues).
# On trace la courbe réelle et on la compare à la courbe théorique de Zipf.

print("\n--- C : loi de Zipf (rang-fréquence) ---")

frequences_triees = sorted(compteur_mots.values(), reverse=True)
rangs = np.arange(1, len(frequences_triees) + 1)

# La courbe théorique de Zipf : fréquence_théorique(rang) = fréquence_du_mot_1 / rang
frequence_theorique = frequences_triees[0] / rangs

plt.figure(figsize=(8, 6))
plt.loglog(rangs, frequences_triees, color="#1f77b4", label="Corpus réel", linewidth=2)
plt.loglog(rangs, frequence_theorique, color="#d62728", linestyle="--", label="Loi de Zipf théorique (1/rang)")
plt.title("Loi de Zipf : fréquence des mots en fonction de leur rang")
plt.xlabel("Rang du mot (échelle log)")
plt.ylabel("Fréquence du mot (échelle log)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/C1_loi_de_zipf.png")
plt.close()

print("Enregistré : C1_loi_de_zipf.png")
print("(Si la courbe bleue suit à peu près la ligne rouge en pointillés, le corpus")
print(" se comporte comme un texte naturel classique.)")


# ==============================================================================
# SECTION D — RICHESSE LEXICALE (TYPE-TOKEN RATIO) PAR PLATEFORME ET PÉRIODE
# ==============================================================================
# La "richesse lexicale" mesure si un ensemble de textes utilise un
# vocabulaire varié ou au contraire répétitif. On la calcule ainsi :
#
#   richesse = (nombre de mots DIFFÉRENTS) / (nombre de mots TOTAL)
#
# Une richesse proche de 1 = vocabulaire très varié (peu de répétitions).
# Une richesse proche de 0 = on répète beaucoup les mêmes mots.
# On compare cette richesse par plateforme (Facebook/TikTok/X ont-elles un
# vocabulaire différent ?) et par période (le vocabulaire se réduit-il,
# se polarise-t-il, après le vote ?).

print("\n--- D : richesse lexicale (Type-Token Ratio) ---")


def calculer_richesse_lexicale(liste_de_textes):
    """Calcule le ratio mots-uniques / mots-total pour un ensemble de textes."""
    tous_les_mots_du_groupe = []
    for texte in liste_de_textes:
        tous_les_mots_du_groupe.extend(texte.split())
    if len(tous_les_mots_du_groupe) == 0:
        return 0
    nb_mots_uniques = len(set(tous_les_mots_du_groupe))
    nb_mots_total = len(tous_les_mots_du_groupe)
    return nb_mots_uniques / nb_mots_total


# --- Par plateforme ---
richesse_par_plateforme = {}
for plateforme in df_avec_texte["plateforme"].unique():
    textes_plateforme = df_avec_texte.loc[df_avec_texte["plateforme"] == plateforme, "texte_lemmatise"]
    richesse_par_plateforme[plateforme] = calculer_richesse_lexicale(textes_plateforme)

serie_richesse_plateforme = pd.Series(richesse_par_plateforme).sort_values(ascending=False)
print("Richesse lexicale par plateforme :")
print(serie_richesse_plateforme)
serie_richesse_plateforme.to_csv(f"{dossier_tableaux}/D1_richesse_lexicale_plateforme.csv")

# --- Par période ---
richesse_par_periode = {}
for periode in df_avec_texte["periode"].unique():
    textes_periode = df_avec_texte.loc[df_avec_texte["periode"] == periode, "texte_lemmatise"]
    richesse_par_periode[periode] = calculer_richesse_lexicale(textes_periode)

serie_richesse_periode = pd.Series(richesse_par_periode)
print("\nRichesse lexicale par période :")
print(serie_richesse_periode)
serie_richesse_periode.to_csv(f"{dossier_tableaux}/D2_richesse_lexicale_periode.csv")

# Un graphique avec les deux comparaisons côte à côte, chacune avec sa propre
# palette de couleurs qualitatives (des couleurs bien distinctes les unes des
# autres, pas un dégradé).
fig, (graphique_gauche, graphique_droit) = plt.subplots(1, 2, figsize=(12, 5))

couleurs_plateforme = cm.Set2(np.linspace(0, 1, len(serie_richesse_plateforme)))
graphique_gauche.bar(serie_richesse_plateforme.index, serie_richesse_plateforme.values, color=couleurs_plateforme)
graphique_gauche.set_title("Richesse lexicale par plateforme")
graphique_gauche.set_ylabel("Ratio mots uniques / mots total")

couleurs_periode = cm.Set1(np.linspace(0, 0.5, len(serie_richesse_periode)))
graphique_droit.bar(serie_richesse_periode.index, serie_richesse_periode.values, color=couleurs_periode)
graphique_droit.set_title("Richesse lexicale par période")

plt.tight_layout()
plt.savefig(f"{dossier_figures}/D3_richesse_lexicale.png")
plt.close()

print("\nEnregistré : D1, D2 (tableaux) et D3 (graphique).")


# ==============================================================================
# SECTION E — NUAGES DE MOTS (GLOBAL + AVANT/APRÈS LE VOTE)
# ==============================================================================

print("\n--- E : nuages de mots ---")

nuage_global = WordCloud(width=1200, height=800, background_color="white", colormap="turbo").generate_from_frequencies(dict(compteur_mots))
plt.figure(figsize=(12, 8))
plt.imshow(nuage_global, interpolation="bilinear")
plt.axis("off")
plt.title("Nuage de mots - ensemble du corpus")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/E1_nuage_mots_global.png")
plt.close()

mots_avant = []
for texte in df_avec_texte.loc[df_avec_texte["periode"] == "avant le vote", "texte_lemmatise"]:
    mots_avant.extend(texte.split())
nuage_avant = WordCloud(width=900, height=700, background_color="white", colormap="winter").generate_from_frequencies(dict(Counter(mots_avant)))

mots_apres = []
for texte in df_avec_texte.loc[df_avec_texte["periode"] == "après le vote", "texte_lemmatise"]:
    mots_apres.extend(texte.split())
nuage_apres = WordCloud(width=900, height=700, background_color="white", colormap="Wistia").generate_from_frequencies(dict(Counter(mots_apres)))

fig, (graphique_gauche, graphique_droit) = plt.subplots(1, 2, figsize=(16, 8))
graphique_gauche.imshow(nuage_avant, interpolation="bilinear")
graphique_gauche.axis("off")
graphique_gauche.set_title("AVANT le vote")
graphique_droit.imshow(nuage_apres, interpolation="bilinear")
graphique_droit.axis("off")
graphique_droit.set_title("APRÈS le vote")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/E2_nuage_mots_avant_apres.png")
plt.close()

print("Enregistré : E1 (nuage global) et E2 (comparaison avant/après).")


# ==============================================================================
# SECTION F — NUAGE DES HASHTAGS
# ==============================================================================

print("\n--- F : nuage des hashtags ---")

tous_les_hashtags = []
for texte in df["texte"]:
    tous_les_hashtags.extend(re.findall(r"#(\w+)", str(texte).lower()))

compteur_hashtags = Counter(tous_les_hashtags)

if len(compteur_hashtags) > 0:
    nuage_hashtags = WordCloud(width=1200, height=800, background_color="white", colormap="rainbow").generate_from_frequencies(dict(compteur_hashtags))
    plt.figure(figsize=(12, 8))
    plt.imshow(nuage_hashtags, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nuage des hashtags les plus utilisés")
    plt.tight_layout()
    plt.savefig(f"{dossier_figures}/F1_nuage_hashtags.png")
    plt.close()
    pd.Series(dict(compteur_hashtags.most_common(50))).to_csv(f"{dossier_tableaux}/F1_top_50_hashtags.csv")
    print("Enregistré : F1_nuage_hashtags.png et F1_top_50_hashtags.csv")


# ==============================================================================
# SECTION G — TREEMAPS DES ACTEURS / VILLES / PARTIS MENTIONNÉS
# ==============================================================================
# Un treemap représente chaque élément par un rectangle dont la SURFACE est
# proportionnelle à sa valeur. Contrairement à un nuage de mots (où la taille
# du texte donne une impression de grandeur, mais difficile à comparer
# précisément), un treemap permet de comparer les proportions d'un coup
# d'œil, un peu comme des parts de gâteau mais en rectangles.

print("\n--- G : treemaps des acteurs, villes et partis ---")

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

villes_camerounaises = {
    "Yaoundé": ["yaounde", "yaoundé"], "Douala": ["douala"], "Garoua": ["garoua"],
    "Bamenda": ["bamenda"], "Bafoussam": ["bafoussam"], "Maroua": ["maroua"],
    "Ngaoundéré": ["ngaoundere", "ngaoundéré"], "Bertoua": ["bertoua"], "Ebolowa": ["ebolowa"],
    "Buea": ["buea"], "Limbe": ["limbe", "limbé"], "Kribi": ["kribi"], "Foumban": ["foumban"],
    "Dschang": ["dschang"], "Nkongsamba": ["nkongsamba"], "Bafang": ["bafang"], "Mbouda": ["mbouda"],
}

partis_politiques = {
    "RDPC": ["rdpc"], "MRC": ["mrc"], "PCRN": ["pcrn"], "SDF": ["sdf"],
    "UDC": ["udc"], "UPC": ["upc"], "FSNC": ["fsnc"], "UNDP": ["undp"],
}


def compter_mentions(dictionnaire_mots_cles):
    """Compte, pour chaque entrée d'un dictionnaire {nom: [mots-clés]}, le
    nombre de publications qui contiennent au moins un des mots-clés."""
    resultat = {}
    for nom, mots_cles in dictionnaire_mots_cles.items():
        motif = "|".join(mots_cles)
        nb = df["texte_nettoye"].str.contains(motif, regex=True, na=False).sum()
        if nb > 0:
            resultat[nom] = nb
    return resultat


frequence_acteurs = compter_mentions(acteurs_politiques)
frequence_villes = compter_mentions(villes_camerounaises)
frequence_partis = compter_mentions(partis_politiques)

pd.Series(frequence_acteurs).sort_values(ascending=False).to_csv(f"{dossier_tableaux}/G1_frequence_acteurs.csv")
pd.Series(frequence_villes).sort_values(ascending=False).to_csv(f"{dossier_tableaux}/G2_frequence_villes.csv")
pd.Series(frequence_partis).sort_values(ascending=False).to_csv(f"{dossier_tableaux}/G3_frequence_partis.csv")


def dessiner_treemap(dictionnaire_frequences, titre, palette, chemin_sortie):
    """Dessine un treemap à partir d'un dictionnaire {nom: valeur}."""
    noms = list(dictionnaire_frequences.keys())
    valeurs = list(dictionnaire_frequences.values())
    etiquettes = [f"{nom}\n({valeur})" for nom, valeur in zip(noms, valeurs)]
    couleurs = palette(np.linspace(0, 1, len(noms)))

    plt.figure(figsize=(11, 7))
    squarify.plot(sizes=valeurs, label=etiquettes, color=couleurs, text_kwargs={"fontsize": 10})
    plt.title(titre)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.close()


dessiner_treemap(frequence_acteurs, "Treemap des acteurs politiques mentionnés", cm.Set2, f"{dossier_figures}/G1_treemap_acteurs.png")
dessiner_treemap(frequence_villes, "Treemap des villes camerounaises mentionnées", cm.tab20, f"{dossier_figures}/G2_treemap_villes.png")
dessiner_treemap(frequence_partis, "Treemap des partis politiques mentionnés", cm.Pastel1, f"{dossier_figures}/G3_treemap_partis.png")

print("Enregistré : G1, G2, G3 (treemaps acteurs / villes / partis).")


# ==============================================================================
# SECTION H — MOTS-COMPAGNONS DE CHAQUE ACTEUR POLITIQUE (CONTEXTE LEXICAL)
# ==============================================================================
# Pour les 4 acteurs les plus mentionnés, on regarde : "quand on parle de cet
# acteur, quels AUTRES mots apparaissent le plus souvent dans le même post ?"
# Ça permet de voir si un acteur est plutôt associé à des mots de contestation,
# de soutien, de violence, etc. — un premier pas vers l'analyse de sentiment
# et de thématiques (Phases 5 à 8), mais rien qu'avec des fréquences de mots.
#
# On enlève de ces "mots-compagnons" : le nom de l'acteur lui-même (pas très
# informatif de savoir que "biya" apparaît près de "biya"), et les mots les
# plus fréquents du corpus entier (comme "aller", "faire", "cameroun"), qui
# apparaissent de toute façon partout et ne sont donc pas distinctifs.

print("\n--- H : mots-compagnons de chaque acteur politique ---")

# Les 4 acteurs les plus mentionnés (voir Phase 2/G) : Biya, Tchiroma, Cabral
# Libii, Kamto. On ne garde que ceux-là car les autres ont trop peu de
# mentions pour une analyse de contexte lexical fiable.
acteurs_principaux = {
    "Paul Biya": ["biya"],
    "Issa Tchiroma": ["tchiroma"],
    "Cabral Libii": ["cabral", "libii", "libie"],
    "Maurice Kamto": ["kamto"],
}

# Mots à exclure car pas informatifs : les 10 mots les plus fréquents du
# corpus + le nom de l'acteur lui-même (sous ses différentes formes lemmatisées)
mots_generiques_a_exclure = set([mot for mot, nombre in top_100_mots[:10]])
mots_noms_acteurs = set()
for mots_cles in acteurs_principaux.values():
    for mot in mots_cles:
        mots_noms_acteurs.add(mot.split()[0])  # ex: "ndam njoya" -> "ndam"
mots_noms_acteurs.update(["paul", "issa", "maurice"])  # prénoms des acteurs, pas informatifs non plus

fig, grille_graphiques = plt.subplots(2, 2, figsize=(14, 11))
grille_graphiques = grille_graphiques.flatten()  # pour pouvoir boucler dessus facilement
palettes_par_acteur = [cm.Blues, cm.Oranges, cm.Greens, cm.RdPu]

resultats_contexte_acteurs = {}

for indice, (acteur, mots_cles) in enumerate(acteurs_principaux.items()):
    motif = "|".join(mots_cles)
    # On sélectionne les publications qui mentionnent cet acteur
    masque = df_avec_texte["texte_nettoye"].str.contains(motif, regex=True, na=False)
    textes_de_l_acteur = df_avec_texte.loc[masque, "texte_lemmatise"]

    mots_contexte = []
    for texte in textes_de_l_acteur:
        for mot in texte.split():
            if mot not in mots_generiques_a_exclure and mot not in mots_noms_acteurs and len(mot) > 2:
                mots_contexte.append(mot)

    top_15_contexte = Counter(mots_contexte).most_common(15)
    resultats_contexte_acteurs[acteur] = top_15_contexte

    if len(top_15_contexte) > 0:
        tableau_contexte = pd.DataFrame(top_15_contexte, columns=["mot", "nombre"]).sort_values("nombre")
        couleurs = palettes_par_acteur[indice](np.linspace(0.4, 0.9, len(tableau_contexte)))
        grille_graphiques[indice].barh(tableau_contexte["mot"], tableau_contexte["nombre"], color=couleurs)
        grille_graphiques[indice].set_title(f"Mots-compagnons de {acteur}")

plt.tight_layout()
plt.savefig(f"{dossier_figures}/H1_mots_compagnons_acteurs.png")
plt.close()

# On sauvegarde aussi les résultats en tableau
lignes_export = []
for acteur, liste_mots in resultats_contexte_acteurs.items():
    for mot, nombre in liste_mots:
        lignes_export.append({"acteur": acteur, "mot_compagnon": mot, "nombre": nombre})
pd.DataFrame(lignes_export).to_csv(f"{dossier_tableaux}/H1_mots_compagnons_acteurs.csv", index=False)

print("Enregistré : H1_mots_compagnons_acteurs.png et .csv")


# ==============================================================================
# SECTION I — MOTS DISTINCTIFS PAR PLATEFORME (TF-IDF Facebook / TikTok / X)
# ==============================================================================
# On a vu en Phase 2 que Facebook, TikTok et X n'ont pas le même public
# (locale/anglophone/diaspora). On regarde ici si elles n'ont pas non plus le
# même VOCABULAIRE. Comme pour la comparaison avant/après (section J), on
# regroupe tous les textes de chaque plateforme en un seul document, et on
# regarde quels mots sont spécifiques à chacune (score TF-IDF nettement plus
# élevé sur cette plateforme que sur les 2 autres en moyenne).

print("\n--- I : mots distinctifs par plateforme ---")

plateformes = df_avec_texte["plateforme"].unique().tolist()
textes_par_plateforme = []
for plateforme in plateformes:
    texte_complet = " ".join(df_avec_texte.loc[df_avec_texte["plateforme"] == plateforme, "texte_lemmatise"])
    textes_par_plateforme.append(texte_complet)

calculateur_tfidf_plateforme = TfidfVectorizer(max_features=2000, min_df=1, sublinear_tf=True)
matrice_plateforme = calculateur_tfidf_plateforme.fit_transform(textes_par_plateforme).toarray()
mots_plateforme = calculateur_tfidf_plateforme.get_feature_names_out()

tableau_tfidf_plateforme = pd.DataFrame(matrice_plateforme.T, columns=plateformes, index=mots_plateforme)

fig, axes = plt.subplots(1, len(plateformes), figsize=(6 * len(plateformes), 7))
couleurs_par_plateforme = {"Facebook": cm.Blues, "TikTok": cm.Greys, "X": cm.Purples}

resultats_export_plateforme = []
for indice, plateforme in enumerate(plateformes):
    autres_plateformes = [p for p in plateformes if p != plateforme]
    moyenne_autres = tableau_tfidf_plateforme[autres_plateformes].mean(axis=1)
    specificite = tableau_tfidf_plateforme[plateforme] - moyenne_autres
    top_15 = specificite.sort_values(ascending=False).head(15)

    for mot, score in top_15.items():
        resultats_export_plateforme.append({"plateforme": plateforme, "mot": mot, "score_specificite": score})

    palette = couleurs_par_plateforme.get(plateforme, cm.viridis)
    couleurs = palette(np.linspace(0.4, 0.9, len(top_15)))
    axes[indice].barh(top_15.index[::-1], top_15.values[::-1], color=couleurs[::-1])
    axes[indice].set_title(f"Mots typiques de {plateforme}")

plt.tight_layout()
plt.savefig(f"{dossier_figures}/I1_mots_distinctifs_plateformes.png")
plt.close()

pd.DataFrame(resultats_export_plateforme).to_csv(f"{dossier_tableaux}/I1_mots_distinctifs_plateformes.csv", index=False)
print("Enregistré : I1_mots_distinctifs_plateformes.png et .csv")


# ==============================================================================
# SECTION J — MOTS TYPIQUES DE CHAQUE PÉRIODE (TF-IDF AVANT / APRÈS LE VOTE)
# ==============================================================================

print("\n--- J : mots typiques de chaque période (avant/après le vote) ---")

texte_avant_complet = " ".join(df_avec_texte.loc[df_avec_texte["periode"] == "avant le vote", "texte_lemmatise"])
texte_apres_complet = " ".join(df_avec_texte.loc[df_avec_texte["periode"] == "après le vote", "texte_lemmatise"])

calculateur_tfidf = TfidfVectorizer(max_features=2000, min_df=1, sublinear_tf=True)
matrice_tfidf = calculateur_tfidf.fit_transform([texte_avant_complet, texte_apres_complet])
liste_des_mots = calculateur_tfidf.get_feature_names_out()

tableau_tfidf = pd.DataFrame({
    "mot": liste_des_mots,
    "score_tfidf_avant": matrice_tfidf[0].toarray()[0],
    "score_tfidf_apres": matrice_tfidf[1].toarray()[0],
})
tableau_tfidf["ecart_avant_moins_apres"] = tableau_tfidf["score_tfidf_avant"] - tableau_tfidf["score_tfidf_apres"]

top_20_typiques_avant = tableau_tfidf.sort_values("ecart_avant_moins_apres", ascending=False).head(20)
top_20_typiques_apres = tableau_tfidf.sort_values("ecart_avant_moins_apres", ascending=True).head(20)

top_20_typiques_avant.to_csv(f"{dossier_tableaux}/J1_tfidf_typique_avant_vote.csv", index=False)
top_20_typiques_apres.to_csv(f"{dossier_tableaux}/J2_tfidf_typique_apres_vote.csv", index=False)

fig, (graphique_gauche, graphique_droit) = plt.subplots(1, 2, figsize=(14, 8))
couleurs_avant = cm.YlGn(np.linspace(0.4, 0.95, 20))
couleurs_apres = cm.OrRd(np.linspace(0.4, 0.95, 20))

graphique_gauche.barh(top_20_typiques_avant["mot"][::-1], top_20_typiques_avant["score_tfidf_avant"][::-1], color=couleurs_avant)
graphique_gauche.set_title("Mots typiques AVANT le vote (TF-IDF)")
graphique_droit.barh(top_20_typiques_apres["mot"][::-1], top_20_typiques_apres["score_tfidf_apres"][::-1], color=couleurs_apres)
graphique_droit.set_title("Mots typiques APRÈS le vote (TF-IDF)")

plt.tight_layout()
plt.savefig(f"{dossier_figures}/J3_comparaison_tfidf_avant_apres.png")
plt.close()

print("Enregistré : J1, J2 (tableaux) et J3 (graphique).")


# ==============================================================================
# SECTION K — BULLES FRÉQUENCE x ENGAGEMENT
# ==============================================================================
# Un mot très fréquent n'est pas forcément un mot qui "fait réagir". Ici, on
# croise deux informations pour chacun des 40 mots les plus fréquents :
#   - en X : dans combien de publications il apparaît (fréquence)
#   - en Y : le nombre moyen de likes des publications qui le contiennent
#            (engagement)
#   - la TAILLE de la bulle : le nombre de publications concernées
#   - la COULEUR : la catégorie du mot (acteur politique, lieu, mot de crise,
#     ou autre), pour voir si un TYPE de mot se détache des autres
#
# C'est une première ébauche de ce qui devient, en Phase 9, un "indice de
# viralité thématique" : quels sujets, une fois évoqués, suscitent le plus
# de réactions du public ?

print("\n--- K : bulles fréquence x engagement ---")

mots_a_categoriser = [mot for mot, nombre in top_100_mots[:40]]

mots_crise = {"violence", "mort", "manifestation", "arrestation", "répression", "fraude",
              "armée", "police", "gendarme", "militaire", "tuer", "ghost", "mourir", "peur"}
mots_acteurs_lemma = {"biya", "kamto", "tchiroma", "cabral", "libii", "libie"}
mots_lieux_lemma = {"cameroun", "yaoundé", "yaounde", "douala", "garoua", "bamenda", "pays"}


def categoriser_mot(mot):
    if mot in mots_acteurs_lemma:
        return "Acteur politique"
    elif mot in mots_lieux_lemma:
        return "Lieu"
    elif mot in mots_crise:
        return "Mot de crise"
    else:
        return "Autre"


resultats_bulles = []
for mot in mots_a_categoriser:
    masque_contient_mot = df_avec_texte["texte_lemmatise"].str.contains(rf"\b{mot}\b", regex=True, na=False)
    sous_ensemble = df_avec_texte[masque_contient_mot]
    nb_publications = len(sous_ensemble)
    engagement_moyen = sous_ensemble["nb_likes"].mean()
    resultats_bulles.append({
        "mot": mot,
        "nb_publications": nb_publications,
        "engagement_moyen": engagement_moyen if pd.notna(engagement_moyen) else 0,
        "categorie": categoriser_mot(mot),
    })

tableau_bulles = pd.DataFrame(resultats_bulles)
tableau_bulles.to_csv(f"{dossier_tableaux}/K1_bulles_frequence_engagement.csv", index=False)

# On associe une couleur bien distincte à chaque catégorie
couleurs_categories = {
    "Acteur politique": "#E63946",
    "Lieu": "#457B9D",
    "Mot de crise": "#2A9D8F",
    "Autre": "#B0B0B0",
}

plt.figure(figsize=(11, 8))
for categorie, couleur in couleurs_categories.items():
    sous_tableau = tableau_bulles[tableau_bulles["categorie"] == categorie]
    plt.scatter(
        sous_tableau["nb_publications"],
        sous_tableau["engagement_moyen"],
        s=sous_tableau["nb_publications"] / 2,   # taille de bulle proportionnelle à la fréquence
        color=couleur,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        label=categorie,
    )

# On ajoute le nom du mot à côté de chaque bulle
for _, ligne in tableau_bulles.iterrows():
    plt.annotate(ligne["mot"], (ligne["nb_publications"], ligne["engagement_moyen"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

plt.xlabel("Nombre de publications contenant le mot (fréquence)")
plt.ylabel("Nombre moyen de likes des publications contenant le mot")
plt.title("Fréquence des mots vs engagement moyen suscité")
plt.legend(title="Catégorie du mot")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/K1_bulles_frequence_engagement.png")
plt.close()

print("Enregistré : K1_bulles_frequence_engagement.png et .csv")


# ==============================================================================
# SECTION L — RÉSEAU DE COOCCURRENCE DES MOTS (GRAPHE)
# ==============================================================================
# Plutôt qu'une simple grille de couleurs (heatmap), on représente les
# associations entre mots sous forme de RÉSEAU : chaque mot est un point
# (nœud), et deux mots sont reliés par un trait si on les trouve souvent
# ensemble dans les mêmes publications. Plus le trait est épais, plus
# l'association est forte. Plus un point est gros, plus le mot est fréquent.
# Ce type de visualisation fait ressortir des "familles" de mots qui vont
# ensemble, un peu comme une carte mentale.

print("\n--- L : réseau de cooccurrence des mots ---")

mots_pour_reseau = [mot for mot, nombre in top_100_mots[:25]]

listes_de_mots_par_post = [set(texte.split()) for texte in df_avec_texte["texte_lemmatise"]]

# On construit le graphe : un nœud par mot
graphe = nx.Graph()
for mot in mots_pour_reseau:
    graphe.add_node(mot, frequence=compteur_mots[mot])

# Pour chaque publication, pour chaque paire de mots (parmi nos 25 mots)
# présents ensemble, on renforce le lien entre ces deux mots.
compteur_paires = Counter()
for mots_du_post in listes_de_mots_par_post:
    mots_presents = [mot for mot in mots_pour_reseau if mot in mots_du_post]
    for i in range(len(mots_presents)):
        for j in range(i + 1, len(mots_presents)):
            paire = tuple(sorted([mots_presents[i], mots_presents[j]]))
            compteur_paires[paire] += 1

# On ne garde que les liens suffisamment forts, sinon le graphe est illisible
# (on garde les 50 associations les plus fréquentes)
liens_les_plus_forts = compteur_paires.most_common(50)
for (mot1, mot2), poids in liens_les_plus_forts:
    graphe.add_edge(mot1, mot2, weight=poids)

# On calcule une disposition automatique des nœuds (les mots liés se
# regroupent naturellement les uns près des autres)
disposition = nx.spring_layout(graphe, k=0.6, seed=42)

tailles_noeuds = [graphe.nodes[mot]["frequence"] / 3 for mot in graphe.nodes]
epaisseurs_liens = [graphe[mot1][mot2]["weight"] / 15 for mot1, mot2 in graphe.edges]
couleurs_noeuds = cm.tab20(np.linspace(0, 1, len(graphe.nodes)))

plt.figure(figsize=(12, 10))
nx.draw_networkx_edges(graphe, disposition, width=epaisseurs_liens, edge_color="lightgray")
nx.draw_networkx_nodes(graphe, disposition, node_size=tailles_noeuds, node_color=couleurs_noeuds)
nx.draw_networkx_labels(graphe, disposition, font_size=9)
plt.title("Réseau de cooccurrence des 25 mots les plus fréquents")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/L1_reseau_cooccurrence.png")
plt.close()

# On sauvegarde aussi les paires de mots sous forme de tableau
pd.DataFrame(
    [(mot1, mot2, poids) for (mot1, mot2), poids in liens_les_plus_forts],
    columns=["mot_1", "mot_2", "nombre_cooccurrences"]
).to_csv(f"{dossier_tableaux}/L1_paires_mots_frequentes.csv", index=False)

print("Enregistré : L1_reseau_cooccurrence.png et L1_paires_mots_frequentes.csv")


# ==============================================================================
# FIN DU SCRIPT
# ==============================================================================

print("\n=== TERMINÉ ! ===")
print(f"Tous les graphiques sont dans : {dossier_figures}")
print(f"Tous les tableaux sont dans   : {dossier_tableaux}")
print("\nRécapitulatif des sections :")
print("  A : top 100 mots       B : bigrammes/trigrammes    C : loi de Zipf")
print("  D : richesse lexicale  E : nuages de mots          F : nuage hashtags")
print("  G : treemaps           H : mots-compagnons acteurs I : mots distinctifs plateformes")
print("  J : TF-IDF avant/après K : bulles fréquence x engagement")
print("  L : réseau de cooccurrence")
