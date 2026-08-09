# ==============================================================================
# PHASE 7 — EXTRACTION DES THÈMES (TOPIC MODELING) — VERSION ENRICHIE
# ==============================================================================
#
# À quoi sert ce script ?
# ------------------------
# Jusqu'ici, on savait SI une publication était positive/négative (Phase 5)
# et QUELLE émotion elle exprimait (Phase 6). Il manque encore : DE QUOI elle
# parle vraiment. Le topic modeling découvre AUTOMATIQUEMENT des groupes de
# mots qui reviennent souvent ensemble, qu'on peut interpréter comme des
# "sujets" (ex : "biya/pouvoir/élection/vote" évoque "Paul Biya et le vote").
#
# MÉTHODE : LDA (Latent Dirichlet Allocation), comme le prévoit le canevas.
# 100% hors-ligne, aucun modèle à télécharger (contrairement à BERTopic).
#
# Français et anglais/pidgin sont traités SÉPARÉMENT (mélanger les langues
# produirait des thèmes incohérents, sans lien de sens).
#
# Cette version ne se contente pas de découvrir les thèmes : elle va plus
# loin en vérifiant la QUALITÉ du modèle (similarité entre thèmes, confiance
# des attributions), en creusant le lien entre thèmes et contexte du mémoire
# (acteurs, engagement, plateformes), et en croisant les thèmes découverts
# automatiquement avec les catégories de crise définies manuellement dès la
# Phase 2, pour voir si les deux approches se recoupent.
# ==============================================================================


# ------------------------------------------------------------------------------
# ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

dossier_data = "../data"
dossier_figures = "../outputs/figures"
dossier_tableaux = "../outputs/tables"
os.makedirs(dossier_figures, exist_ok=True)
os.makedirs(dossier_tableaux, exist_ok=True)


# ------------------------------------------------------------------------------
# ÉTAPE 1 : CHARGER LE CORPUS (Phase 6, avec sentiment + émotions)
# ------------------------------------------------------------------------------

print("Lecture du corpus (avec sentiment + émotions déjà calculés)...")
df = pd.read_csv(f"{dossier_data}/bd_f_emotions.csv", low_memory=False)
df["texte_lemmatise"] = df["texte_lemmatise"].fillna("")
df["texte_nettoye"] = df["texte_nettoye"].fillna("")
df["nb_likes"] = pd.to_numeric(df["nb_likes"], errors="coerce")
print(f"{len(df)} publications chargées.")

df["date_publication"] = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
date_du_vote = pd.Timestamp("2025-10-12", tz="UTC")
df["periode"] = "après le vote"
df.loc[df["date_publication"] < date_du_vote, "periode"] = "avant le vote"
df["semaine"] = df["date_publication"].dt.to_period("W").dt.start_time

df["theme_id"] = -1
df["theme_nom"] = "non déterminé"


# ------------------------------------------------------------------------------
# ÉTAPE 2 : SÉPARER LES SOUS-CORPUS FRANÇAIS ET ANGLAIS/PIDGIN
# ------------------------------------------------------------------------------

masque_francais = (df["langue_detectee"] == "fr") & (df["texte_lemmatise"].str.strip() != "")
masque_anglais_pidgin = (df["langue_detectee"].isin(["en", "pidgin"])) & (df["texte_lemmatise"].str.strip() != "")

print(f"Sous-corpus français : {masque_francais.sum()} publications")
print(f"Sous-corpus anglais/pidgin : {masque_anglais_pidgin.sum()} publications")


# ==============================================================================
# SECTION A — VECTORISATION
# ==============================================================================

print("\n--- A : vectorisation des textes ---")

mots_generiques_fr = [
    "faire", "aller", "pouvoir", "vouloir", "falloir", "savoir", "devoir", "venir",
    "dire", "voir", "avoir", "prendre", "mettre", "donner", "rester", "luire", "taire",
    "bien", "vraiment", "vrai", "trop", "hein", "oui", "non", "rien", "jamais", "ici",
    "comprendre", "connaître", "regarder",
]
mots_generiques_en = ["go", "make", "know", "want", "think", "say", "come", "get", "look", "tell", "use", "try", "need", "let"]

vectoriseur_fr = CountVectorizer(max_features=3000, min_df=5, max_df=0.5, stop_words=mots_generiques_fr)
matrice_fr = vectoriseur_fr.fit_transform(df.loc[masque_francais, "texte_lemmatise"])
mots_fr = vectoriseur_fr.get_feature_names_out()
print(f"Français : matrice de {matrice_fr.shape[0]} publications x {matrice_fr.shape[1]} mots")

vectoriseur_en = CountVectorizer(max_features=2000, min_df=3, max_df=0.5, stop_words=mots_generiques_en)
matrice_en = vectoriseur_en.fit_transform(df.loc[masque_anglais_pidgin, "texte_lemmatise"])
mots_en = vectoriseur_en.get_feature_names_out()
print(f"Anglais/pidgin : matrice de {matrice_en.shape[0]} publications x {matrice_en.shape[1]} mots")


# ==============================================================================
# SECTION B — CHOIX DU NOMBRE OPTIMAL DE THÈMES (SCORE DE COHÉRENCE)
# ==============================================================================

print("\n--- B : recherche du nombre optimal de thèmes (cohérence) ---")


def calculer_coherence_umass(indices_top_mots, matrice_binaire, frequence_par_mot):
    score_total = 0
    nb_paires = 0
    for i in range(1, len(indices_top_mots)):
        for j in range(i):
            mot_i, mot_j = indices_top_mots[i], indices_top_mots[j]
            cooccurrence = matrice_binaire[:, mot_i].multiply(matrice_binaire[:, mot_j]).sum()
            score_total += np.log((cooccurrence + 1) / frequence_par_mot[mot_j])
            nb_paires += 1
    return score_total / nb_paires


def tester_plusieurs_k(matrice, valeurs_de_k, nb_mots_par_theme=10):
    matrice_binaire = (matrice > 0).astype(int)
    frequence_par_mot = np.array(matrice_binaire.sum(axis=0)).flatten()
    resultats = {}
    for k in valeurs_de_k:
        modele = LatentDirichletAllocation(n_components=k, random_state=42, max_iter=25, learning_method="batch")
        modele.fit(matrice)
        coherences_des_themes = []
        for theme in modele.components_:
            indices_top_mots = theme.argsort()[::-1][:nb_mots_par_theme]
            coherences_des_themes.append(calculer_coherence_umass(indices_top_mots, matrice_binaire, frequence_par_mot))
        resultats[k] = np.mean(coherences_des_themes)
        print(f"  K={k} -> cohérence moyenne = {resultats[k]:.3f}")
    return resultats


print("Test pour le français :")
resultats_k_fr = tester_plusieurs_k(matrice_fr, [5, 6, 7, 8, 9, 10, 12])
meilleur_k_fr = max(resultats_k_fr, key=resultats_k_fr.get)

print("\nTest pour l'anglais/pidgin :")
resultats_k_en = tester_plusieurs_k(matrice_en, [4, 5, 6, 7])
meilleur_k_en = max(resultats_k_en, key=resultats_k_en.get)

print(f"\nNombre de thèmes retenu : {meilleur_k_fr} (français), {meilleur_k_en} (anglais/pidgin)")

fig, (graphique_gauche, graphique_droit) = plt.subplots(1, 2, figsize=(13, 5))
graphique_gauche.plot(list(resultats_k_fr.keys()), list(resultats_k_fr.values()), marker="o", color="#1d3557")
graphique_gauche.axvline(meilleur_k_fr, color="red", linestyle="--", label=f"K retenu = {meilleur_k_fr}")
graphique_gauche.set_title("Cohérence des thèmes (français)")
graphique_gauche.set_xlabel("Nombre de thèmes (K)")
graphique_gauche.set_ylabel("Score de cohérence (plus proche de 0 = mieux)")
graphique_gauche.legend()
graphique_droit.plot(list(resultats_k_en.keys()), list(resultats_k_en.values()), marker="o", color="#E76F51")
graphique_droit.axvline(meilleur_k_en, color="red", linestyle="--", label=f"K retenu = {meilleur_k_en}")
graphique_droit.set_title("Cohérence des thèmes (anglais/pidgin)")
graphique_droit.set_xlabel("Nombre de thèmes (K)")
graphique_droit.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/B1_choix_nombre_de_themes.png")
plt.close()

print("Enregistré : B1_choix_nombre_de_themes.png")


# ==============================================================================
# SECTION C — ENTRAÎNEMENT DES MODÈLES LDA FINAUX
# ==============================================================================

print("\n--- C : entraînement des modèles LDA finaux ---")

lda_fr = LatentDirichletAllocation(n_components=meilleur_k_fr, random_state=42, max_iter=25, learning_method="batch")
distribution_themes_fr = lda_fr.fit_transform(matrice_fr)

lda_en = LatentDirichletAllocation(n_components=meilleur_k_en, random_state=42, max_iter=25, learning_method="batch")
distribution_themes_en = lda_en.fit_transform(matrice_en)

print("Modèles entraînés.")


# ==============================================================================
# SECTION D — MOTS-CLÉS DE CHAQUE THÈME (FRANÇAIS)
# ==============================================================================

print("\n--- D : mots-clés des thèmes français ---")

nb_mots_affiches = 12
couleurs_themes_fr = cm.tab10(np.linspace(0, 1, meilleur_k_fr))

mots_cles_par_theme_fr = {}
fig, grille = plt.subplots(int(np.ceil(meilleur_k_fr / 3)), 3, figsize=(16, 4 * int(np.ceil(meilleur_k_fr / 3))))
grille = grille.flatten()

for i, theme in enumerate(lda_fr.components_):
    indices_top_mots = theme.argsort()[::-1][:nb_mots_affiches]
    mots_du_theme = [mots_fr[idx] for idx in indices_top_mots]
    poids_du_theme = [theme[idx] for idx in indices_top_mots]
    mots_cles_par_theme_fr[i] = mots_du_theme
    print(f"Thème {i} : {', '.join(mots_du_theme)}")
    grille[i].barh(mots_du_theme[::-1], poids_du_theme[::-1], color=couleurs_themes_fr[i])
    grille[i].set_title(f"Thème {i}")

for j in range(meilleur_k_fr, len(grille)):
    grille[j].axis("off")

plt.tight_layout()
plt.savefig(f"{dossier_figures}/D1_mots_cles_themes_francais.png")
plt.close()

pd.DataFrame(mots_cles_par_theme_fr).to_csv(f"{dossier_tableaux}/D1_mots_cles_themes_francais.csv", index=False)
print("Enregistré : D1 (tableau + graphique).")


# ==============================================================================
# SECTION E — LIBELLÉS INTERPRÉTATIFS DES THÈMES
# ==============================================================================
# Ces libellés sont une PROPOSITION issue de la lecture des mots-clés
# (section D). Comme toujours en topic modeling, c'est à l'analyste de
# nommer les thèmes : affine ces noms selon ta propre lecture si besoin.

print("\n--- E : libellés interprétatifs proposés pour les thèmes ---")

libelles_themes_fr = {
    0: "Tribalisme, régime et tensions régionales (Ouest)",
    1: "Solidarité, foi et encouragements au peuple",
    2: "Candidats de l'opposition et espoir de changement",
    3: "Communauté Bamiléké, famille et problèmes du pays",
    4: "Paul Biya : pouvoir, vie et mort",
}
for i in range(meilleur_k_fr):
    if i not in libelles_themes_fr:
        libelles_themes_fr[i] = f"Thème {i} (à nommer manuellement, voir mots-clés)"

for i, nom in libelles_themes_fr.items():
    if i < meilleur_k_fr:
        print(f"  Thème {i} -> {nom}")

libelles_themes_en = {
    0: "Vote et actualité internationale",
    1: "Villes mortes et pétitions (pidgin)",
    2: "Discours politique général (honte, ministre)",
    3: "Biya, élection et pouvoir",
    4: "Bruit résiduel (textes mal détectés comme anglais)",
    5: "Bruit résiduel supplémentaire (mélange linguistique)",
    6: "Tchiroma, peur et crise anglophone",
}
for i in range(meilleur_k_en):
    if i not in libelles_themes_en:
        libelles_themes_en[i] = f"Thème {i} (à nommer manuellement, voir mots-clés)"

# Une couleur FIXE par thème français, réutilisée dans TOUS les graphiques
# suivants du script, pour que le même thème soit toujours reconnaissable à
# la même couleur (répartition, évolution, radar, engagement...).
COULEURS_THEMES_FR = {libelles_themes_fr[i]: couleurs_themes_fr[i] for i in range(meilleur_k_fr)}


# ==============================================================================
# SECTION F — NUAGES DE MOTS PAR THÈME (FRANÇAIS)
# ==============================================================================

print("\n--- F : nuages de mots par thème ---")

fig, grille = plt.subplots(int(np.ceil(meilleur_k_fr / 3)), 3, figsize=(15, 4 * int(np.ceil(meilleur_k_fr / 3))))
grille = grille.flatten()
palettes_wordcloud = ["Blues", "Oranges", "Greens", "Purples", "Reds", "YlOrBr", "PuBuGn", "RdPu", "BuGn", "OrRd"]

for i, theme in enumerate(lda_fr.components_):
    frequences_mots = {mots_fr[idx]: theme[idx] for idx in theme.argsort()[::-1][:40]}
    nuage = WordCloud(width=500, height=350, background_color="white",
                       colormap=palettes_wordcloud[i % len(palettes_wordcloud)]).generate_from_frequencies(frequences_mots)
    grille[i].imshow(nuage, interpolation="bilinear")
    grille[i].axis("off")
    grille[i].set_title(libelles_themes_fr[i], fontsize=10)

for j in range(meilleur_k_fr, len(grille)):
    grille[j].axis("off")

plt.tight_layout()
plt.savefig(f"{dossier_figures}/F1_nuages_mots_themes_francais.png")
plt.close()

print("Enregistré : F1_nuages_mots_themes_francais.png")


# ==============================================================================
# SECTION G — CARTE DE SIMILARITÉ ENTRE LES THÈMES (NOUVEAU)
# ==============================================================================
# Deux analyses complémentaires pour vérifier la QUALITÉ du modèle :
#   1. Une heatmap de similarité entre chaque paire de thèmes (est-ce que
#      deux thèmes se ressemblent trop, au point de faire un peu doublon ?)
#   2. Une "carte" en 2 dimensions où chaque thème est un point : plus deux
#      points sont proches, plus les thèmes se ressemblent. La taille du
#      point représente le nombre de publications concernées.

print("\n--- G : carte de similarité entre les thèmes ---")

similarite_themes = cosine_similarity(lda_fr.components_)

plt.figure(figsize=(8, 7))
carte = plt.imshow(similarite_themes, cmap="coolwarm", vmin=0, vmax=1)
plt.colorbar(carte, label="Similarité (0 = très différents, 1 = identiques)")
noms_courts = [f"T{i}" for i in range(meilleur_k_fr)]
plt.xticks(range(meilleur_k_fr), noms_courts)
plt.yticks(range(meilleur_k_fr), [libelles_themes_fr[i] for i in range(meilleur_k_fr)])
for i in range(meilleur_k_fr):
    for j in range(meilleur_k_fr):
        plt.text(j, i, f"{similarite_themes[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Similarité entre les thèmes français (cosinus)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/G1_similarite_themes.png")
plt.close()

# Carte 2D (PCA) des thèmes, avec une taille de point proportionnelle à leur poids
tailles_themes = distribution_themes_fr.sum(axis=0)
projection = PCA(n_components=2, random_state=42).fit_transform(lda_fr.components_)

plt.figure(figsize=(9, 8))
for i in range(meilleur_k_fr):
    plt.scatter(projection[i, 0], projection[i, 1], s=tailles_themes[i] / 3,
                color=couleurs_themes_fr[i], alpha=0.7, edgecolors="black")
    plt.annotate(f"T{i}: {libelles_themes_fr[i][:30]}", (projection[i, 0], projection[i, 1]),
                 fontsize=8, xytext=(8, 8), textcoords="offset points")
plt.title("Carte des thèmes (proximité = ressemblance, taille = poids dans le corpus)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/G2_carte_2d_themes.png")
plt.close()

pd.DataFrame(similarite_themes, index=[libelles_themes_fr[i] for i in range(meilleur_k_fr)],
             columns=[libelles_themes_fr[i] for i in range(meilleur_k_fr)]).to_csv(f"{dossier_tableaux}/G1_similarite_themes.csv")

print("Enregistré : G1 (heatmap similarité) et G2 (carte 2D).")


# ==============================================================================
# SECTION H — CONFIANCE D'ATTRIBUTION DES THÈMES (NOUVEAU)
# ==============================================================================
# Pour chaque publication, la LDA donne une probabilité par thème (pas une
# catégorie stricte). Certaines publications sont "clairement" d'un thème
# (une probabilité proche de 1 sur un seul thème), d'autres sont un mélange
# flou de plusieurs thèmes à la fois (probabilités proches les unes des
# autres). On regarde la distribution de cette "confiance" pour savoir si
# nos thèmes sont globalement bien tranchés ou pas.

print("\n--- H : confiance d'attribution des thèmes ---")

confiance_fr = distribution_themes_fr.max(axis=1)
confiance_en = distribution_themes_en.max(axis=1)

fig, (graphique_gauche, graphique_droit) = plt.subplots(1, 2, figsize=(13, 5))
graphique_gauche.hist(confiance_fr, bins=40, color="#457B9D", edgecolor="white")
graphique_gauche.axvline(confiance_fr.mean(), color="red", linestyle="--", label=f"Moyenne = {confiance_fr.mean():.2f}")
graphique_gauche.set_title("Confiance d'attribution (français)")
graphique_gauche.set_xlabel("Probabilité du thème dominant")
graphique_gauche.set_ylabel("Nombre de publications")
graphique_gauche.legend()

graphique_droit.hist(confiance_en, bins=40, color="#E76F51", edgecolor="white")
graphique_droit.axvline(confiance_en.mean(), color="red", linestyle="--", label=f"Moyenne = {confiance_en.mean():.2f}")
graphique_droit.set_title("Confiance d'attribution (anglais/pidgin)")
graphique_droit.set_xlabel("Probabilité du thème dominant")
graphique_droit.legend()

plt.tight_layout()
plt.savefig(f"{dossier_figures}/H1_confiance_attribution_themes.png")
plt.close()

print(f"Confiance moyenne : {confiance_fr.mean():.3f} (français), {confiance_en.mean():.3f} (anglais/pidgin)")
print("Enregistré : H1_confiance_attribution_themes.png")


# ==============================================================================
# SECTION I — PUBLICATIONS LES PLUS REPRÉSENTATIVES DE CHAQUE THÈME (NOUVEAU)
# ==============================================================================
# Pour illustrer concrètement chaque thème dans le mémoire (verbatims), on
# retrouve les publications qui appartiennent le PLUS clairement à chaque
# thème (probabilité la plus haute), plutôt que de se contenter des mots-clés.

print("\n--- I : publications les plus représentatives de chaque thème ---")

index_francais = df.index[masque_francais]
lignes_representatives = []
for i in range(meilleur_k_fr):
    probabilites_theme_i = distribution_themes_fr[:, i]
    meilleurs_indices_locaux = probabilites_theme_i.argsort()[::-1][:3]
    for indice_local in meilleurs_indices_locaux:
        indice_reel = index_francais[indice_local]
        lignes_representatives.append({
            "theme": libelles_themes_fr[i],
            "probabilite": round(probabilites_theme_i[indice_local], 3),
            "texte": df.loc[indice_reel, "texte"],
            "plateforme": df.loc[indice_reel, "plateforme"],
            "nb_likes": df.loc[indice_reel, "nb_likes"],
        })

tableau_representatifs = pd.DataFrame(lignes_representatives)
tableau_representatifs.to_csv(f"{dossier_tableaux}/I1_publications_representatives_themes.csv", index=False)
print(f"Enregistré : I1_publications_representatives_themes.csv ({len(tableau_representatifs)} verbatims)")


# ==============================================================================
# SECTION J — ATTRIBUTION DU THÈME DOMINANT ET RÉPARTITION GLOBALE
# ==============================================================================

print("\n--- J : attribution du thème dominant et répartition ---")

themes_dominants_fr = distribution_themes_fr.argmax(axis=1)
df.loc[masque_francais, "theme_id"] = themes_dominants_fr
df.loc[masque_francais, "theme_nom"] = [libelles_themes_fr[t] for t in themes_dominants_fr]

themes_dominants_en = distribution_themes_en.argmax(axis=1)
df.loc[masque_anglais_pidgin, "theme_id"] = themes_dominants_en
df.loc[masque_anglais_pidgin, "theme_nom"] = [libelles_themes_en[t] for t in themes_dominants_en]

df_francais = df[masque_francais].copy()
df_francais["theme_nom"] = df_francais["theme_id"].map(libelles_themes_fr)

repartition_themes_fr = df_francais["theme_nom"].value_counts()
repartition_themes_fr.to_csv(f"{dossier_tableaux}/J1_repartition_themes_francais.csv")
print(repartition_themes_fr)

plt.figure(figsize=(10, 7))
couleurs_ordre = [COULEURS_THEMES_FR[nom] for nom in repartition_themes_fr.index]
plt.barh(repartition_themes_fr.index[::-1], repartition_themes_fr.values[::-1], color=couleurs_ordre[::-1])
plt.title("Nombre de publications par thème dominant (français)")
plt.xlabel("Nombre de publications")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/J1_repartition_themes_francais.png")
plt.close()

print("Enregistré : J1 (tableau + graphique).")


# ==============================================================================
# SECTION K — ÉVOLUTION TEMPORELLE DES THÈMES
# ==============================================================================

print("\n--- K : évolution temporelle des thèmes ---")

evolution_themes_hebdo = df_francais.groupby(["semaine", "theme_nom"]).size().unstack(fill_value=0)
evolution_themes_pourcentage = evolution_themes_hebdo.div(evolution_themes_hebdo.sum(axis=1), axis=0) * 100
evolution_themes_pourcentage.to_csv(f"{dossier_tableaux}/K1_evolution_themes_hebdomadaire.csv")

plt.figure(figsize=(15, 7))
couleurs_stack = [COULEURS_THEMES_FR[colonne] for colonne in evolution_themes_pourcentage.columns]
plt.stackplot(
    evolution_themes_pourcentage.index,
    [evolution_themes_pourcentage[colonne] for colonne in evolution_themes_pourcentage.columns],
    labels=evolution_themes_pourcentage.columns,
    colors=couleurs_stack,
)
plt.axvline(date_du_vote, color="black", linestyle="--", linewidth=2, label="Jour du vote")
plt.title("Évolution hebdomadaire de la part de chaque thème (français)")
plt.xlabel("Semaine")
plt.ylabel("Part des publications (%)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{dossier_figures}/K1_evolution_themes_hebdomadaire.png")
plt.close()

print("Enregistré : K1 (tableau + aire empilée).")


# ==============================================================================
# SECTION L — THÈMES PAR PLATEFORME (heatmap + radar)
# ==============================================================================

print("\n--- L : répartition des thèmes par plateforme ---")

repartition_theme_plateforme = pd.crosstab(df_francais["plateforme"], df_francais["theme_nom"], normalize="index") * 100
repartition_theme_plateforme.to_csv(f"{dossier_tableaux}/L1_themes_par_plateforme.csv")

plt.figure(figsize=(11, 6))
carte = plt.imshow(repartition_theme_plateforme.values, cmap="YlGnBu", aspect="auto")
plt.colorbar(carte, label="% des publications de la plateforme")
plt.yticks(range(len(repartition_theme_plateforme.index)), repartition_theme_plateforme.index)
plt.xticks(range(len(repartition_theme_plateforme.columns)), repartition_theme_plateforme.columns, rotation=45, ha="right")
plt.title("Répartition des thèmes par plateforme (%)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/L1_themes_par_plateforme.png")
plt.close()

# Radar pour visualiser le même résultat autrement (profil comparé des 3 plateformes)
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
angles = np.linspace(0, 2 * np.pi, meilleur_k_fr, endpoint=False).tolist()
angles += angles[:1]
couleurs_plateformes_radar = {"Facebook": "#1877F2", "TikTok": "#EE1D52", "X": "#6c757d"}

noms_themes_courts = [nom[:22] + "..." if len(nom) > 22 else nom for nom in repartition_theme_plateforme.columns]

for plateforme in repartition_theme_plateforme.index:
    valeurs = repartition_theme_plateforme.loc[plateforme].tolist()
    valeurs += valeurs[:1]
    ax.plot(angles, valeurs, label=plateforme, color=couleurs_plateformes_radar.get(plateforme), linewidth=2)
    ax.fill(angles, valeurs, color=couleurs_plateformes_radar.get(plateforme), alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(noms_themes_courts, fontsize=8)
ax.set_title("Profil thématique comparé des 3 plateformes")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(f"{dossier_figures}/L2_radar_themes_plateformes.png")
plt.close()

print("Enregistré : L1 (heatmap) et L2 (radar).")


# ==============================================================================
# SECTION M — THÈMES AVANT / APRÈS LE VOTE
# ==============================================================================

print("\n--- M : répartition des thèmes avant/après le vote ---")

repartition_theme_periode = pd.crosstab(df_francais["theme_nom"], df_francais["periode"], normalize="columns") * 100
repartition_theme_periode.to_csv(f"{dossier_tableaux}/M1_themes_avant_apres.csv")
print(repartition_theme_periode.round(1))

positions = np.arange(len(repartition_theme_periode))
largeur = 0.35
plt.figure(figsize=(11, 7))
plt.barh(positions - largeur / 2, repartition_theme_periode["avant le vote"], height=largeur, label="Avant le vote", color="#4C9A2A")
plt.barh(positions + largeur / 2, repartition_theme_periode["après le vote"], height=largeur, label="Après le vote", color="#B22222")
plt.yticks(positions, repartition_theme_periode.index)
plt.xlabel("Part des publications de la période (%)")
plt.title("Poids de chaque thème, avant vs après le vote")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dossier_figures}/M1_themes_avant_apres.png")
plt.close()

print("Enregistré : M1 (tableau + graphique).")


# ==============================================================================
# SECTION N — THÈMES PAR ACTEUR POLITIQUE (NOUVEAU)
# ==============================================================================
# Pour chaque acteur, dans quel(s) thème(s) tombent les publications qui le
# mentionnent ? Ça permet de voir si un acteur est surtout associé à un sujet
# précis (ex : Biya associé au thème "pouvoir/vie/mort") ou au contraire
# dispersé sur plusieurs thèmes.

print("\n--- N : répartition des thèmes par acteur politique ---")

acteurs_politiques = {
    "Paul Biya": ["biya"],
    "Issa Tchiroma": ["tchiroma"],
    "Cabral Libii": ["cabral", "libii", "libie"],
    "Maurice Kamto": ["kamto"],
    "Bello Bouba Maigari": ["bello bouba", "maigari"],
    "Patricia Ndam Njoya": ["ndam njoya", "ndam"],
}

lignes_acteurs_themes = {}
for acteur, mots_cles in acteurs_politiques.items():
    motif = "|".join(mots_cles)
    masque_acteur = df_francais["texte_nettoye"].str.contains(motif, regex=True, na=False)
    if masque_acteur.sum() > 0:
        lignes_acteurs_themes[acteur] = df_francais.loc[masque_acteur, "theme_nom"].value_counts(normalize=True) * 100

tableau_acteurs_themes = pd.DataFrame(lignes_acteurs_themes).T.fillna(0)
tableau_acteurs_themes = tableau_acteurs_themes[[c for c in libelles_themes_fr.values() if c in tableau_acteurs_themes.columns]]
tableau_acteurs_themes.to_csv(f"{dossier_tableaux}/N1_themes_par_acteur.csv")
print(tableau_acteurs_themes.round(1))

plt.figure(figsize=(11, 7))
carte = plt.imshow(tableau_acteurs_themes.values, cmap="PuRd", aspect="auto")
plt.colorbar(carte, label="% des publications de l'acteur")
plt.xticks(range(len(tableau_acteurs_themes.columns)), tableau_acteurs_themes.columns, rotation=30, ha="right")
plt.yticks(range(len(tableau_acteurs_themes.index)), tableau_acteurs_themes.index)
for i in range(len(tableau_acteurs_themes.index)):
    for j in range(len(tableau_acteurs_themes.columns)):
        plt.text(j, i, f"{tableau_acteurs_themes.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
plt.title("Répartition thématique des publications mentionnant chaque acteur (%)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/N1_themes_par_acteur.png")
plt.close()

print("Enregistré : N1 (tableau + heatmap).")


# ==============================================================================
# SECTION O — ENGAGEMENT MOYEN PAR THÈME (NOUVEAU)
# ==============================================================================
# Quel thème suscite le plus de réactions (likes) ? Une première ébauche de
# ce qui deviendra un "indice de viralité thématique" en Phase 9.

print("\n--- O : engagement moyen par thème ---")

engagement_par_theme = df_francais.groupby("theme_nom")["nb_likes"].agg(["mean", "median", "count"]).round(1)
engagement_par_theme = engagement_par_theme.sort_values("mean", ascending=False)
engagement_par_theme.to_csv(f"{dossier_tableaux}/O1_engagement_par_theme.csv")
print(engagement_par_theme)

couleurs_engagement = [COULEURS_THEMES_FR[nom] for nom in engagement_par_theme.index]
plt.figure(figsize=(10, 6))
plt.bar(range(len(engagement_par_theme)), engagement_par_theme["mean"], color=couleurs_engagement)
plt.xticks(range(len(engagement_par_theme)), engagement_par_theme.index, rotation=25, ha="right")
plt.title("Nombre moyen de likes selon le thème dominant de la publication")
plt.ylabel("Nombre moyen de likes")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/O1_engagement_par_theme.png")
plt.close()

print("Enregistré : O1 (tableau + graphique).")


# ==============================================================================
# SECTION P — DIVERSITÉ THÉMATIQUE (ENTROPIE) PAR PLATEFORME ET PÉRIODE (NOUVEAU)
# ==============================================================================
# La "diversité thématique" mesure si un ensemble de publications parle de
# BEAUCOUP de sujets différents, de façon équilibrée, ou si elle se
# concentre sur un ou deux sujets dominants. On utilise l'entropie de
# Shannon (le même principe mathématique qu'en théorie de l'information) :
# plus l'entropie est élevée, plus les sujets sont variés et équilibrés.

print("\n--- P : diversité thématique (entropie de Shannon) ---")


def calculer_entropie(proportions):
    proportions = proportions[proportions > 0]
    return -np.sum(proportions * np.log2(proportions))


entropie_max_possible = np.log2(meilleur_k_fr)  # l'entropie si tous les thèmes étaient parfaitement équilibrés

entropie_par_plateforme = {}
for plateforme in df_francais["plateforme"].unique():
    proportions = df_francais.loc[df_francais["plateforme"] == plateforme, "theme_nom"].value_counts(normalize=True)
    entropie_par_plateforme[plateforme] = calculer_entropie(proportions.values)

entropie_par_periode = {}
for periode in df_francais["periode"].unique():
    proportions = df_francais.loc[df_francais["periode"] == periode, "theme_nom"].value_counts(normalize=True)
    entropie_par_periode[periode] = calculer_entropie(proportions.values)

print(f"Entropie maximale possible (avec {meilleur_k_fr} thèmes) : {entropie_max_possible:.3f}")
print("Par plateforme :", entropie_par_plateforme)
print("Par période :", entropie_par_periode)

pd.Series(entropie_par_plateforme).to_csv(f"{dossier_tableaux}/P1_entropie_par_plateforme.csv")
pd.Series(entropie_par_periode).to_csv(f"{dossier_tableaux}/P2_entropie_par_periode.csv")

fig, (graphique_gauche, graphique_droit) = plt.subplots(1, 2, figsize=(12, 5))

couleurs_plateforme_entropie = cm.Set2(np.linspace(0, 1, len(entropie_par_plateforme)))
graphique_gauche.bar(entropie_par_plateforme.keys(), entropie_par_plateforme.values(), color=couleurs_plateforme_entropie)
graphique_gauche.axhline(entropie_max_possible, color="black", linestyle="--", label="Diversité maximale possible")
graphique_gauche.set_title("Diversité thématique par plateforme")
graphique_gauche.set_ylabel("Entropie de Shannon")
graphique_gauche.legend()

couleurs_periode_entropie = cm.Set1(np.linspace(0, 0.5, len(entropie_par_periode)))
graphique_droit.bar(entropie_par_periode.keys(), entropie_par_periode.values(), color=couleurs_periode_entropie)
graphique_droit.axhline(entropie_max_possible, color="black", linestyle="--", label="Diversité maximale possible")
graphique_droit.set_title("Diversité thématique par période")
graphique_droit.legend()

plt.tight_layout()
plt.savefig(f"{dossier_figures}/P1_diversite_thematique.png")
plt.close()

print("Enregistré : P1/P2 (tableaux) et P1 (graphique).")


# ==============================================================================
# SECTION Q — SENTIMENT MOYEN PAR THÈME (contribue à H2)
# ==============================================================================

print("\n--- Q : sentiment moyen par thème ---")

sentiment_par_theme = df_francais.groupby("theme_nom")["score_sentiment"].agg(["mean", "count"]).round(3)
sentiment_par_theme = sentiment_par_theme.sort_values("mean")
sentiment_par_theme.to_csv(f"{dossier_tableaux}/Q1_sentiment_par_theme.csv")
print(sentiment_par_theme)

couleurs_sentiment_theme = ["#E63946" if v < 0 else "#2A9D8F" for v in sentiment_par_theme["mean"]]
plt.figure(figsize=(10, 7))
plt.barh(sentiment_par_theme.index, sentiment_par_theme["mean"], color=couleurs_sentiment_theme)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Score de sentiment moyen selon le thème dominant de la publication")
plt.xlabel("Score de sentiment moyen")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/Q1_sentiment_par_theme.png")
plt.close()

print("Enregistré : Q1 (tableau + graphique).")


# ==============================================================================
# SECTION R — PROFIL ÉMOTIONNEL PAR THÈME (contribue à H2)
# ==============================================================================

print("\n--- R : profil émotionnel de chaque thème ---")

colonnes_emotions = [c for c in df.columns if c.startswith("emotion_") and c != "emotion_dominante"]
noms_emotions = [c.replace("emotion_", "") for c in colonnes_emotions]

profil_emotionnel_themes = df_francais.groupby("theme_nom")[colonnes_emotions].mean()
profil_emotionnel_themes.columns = noms_emotions
profil_emotionnel_themes.to_csv(f"{dossier_tableaux}/R1_profil_emotionnel_themes.csv")

plt.figure(figsize=(10, 8))
carte = plt.imshow(profil_emotionnel_themes.values, cmap="magma", aspect="auto")
plt.colorbar(carte, label="Intensité moyenne")
plt.xticks(range(len(noms_emotions)), noms_emotions, rotation=30, ha="right")
plt.yticks(range(len(profil_emotionnel_themes.index)), profil_emotionnel_themes.index)
plt.title("Profil émotionnel de chaque thème")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/R1_profil_emotionnel_themes.png")
plt.close()

print("Enregistré : R1 (tableau + heatmap).")


# ==============================================================================
# SECTION S — VALIDATION CROISÉE : THÈMES LDA vs CATÉGORIES DE CRISE A PRIORI (NOUVEAU)
# ==============================================================================
# Depuis la Phase 2, on utilise des catégories de crise définies À LA MAIN
# (villes mortes, violence, fraude...). Ici, on vérifie si les thèmes
# DÉCOUVERTS AUTOMATIQUEMENT par la LDA recoupent ces catégories. Si un mot-clé
# de crise tombe presque toujours dans le même thème LDA, c'est un signe que
# les deux approches (manuelle et automatique) se confirment l'une l'autre :
# c'est un argument de robustesse important pour le mémoire.

print("\n--- S : validation croisée thèmes LDA vs catégories de crise définies a priori ---")

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

lignes_validation = {}
for evenement, mots_cles in evenements_de_crise.items():
    motif = "|".join(mots_cles)
    masque_evenement = df_francais["texte_nettoye"].str.contains(motif, regex=True, na=False)
    if masque_evenement.sum() > 0:
        lignes_validation[evenement] = df_francais.loc[masque_evenement, "theme_nom"].value_counts(normalize=True) * 100

tableau_validation = pd.DataFrame(lignes_validation).T.fillna(0)
tableau_validation = tableau_validation[[c for c in libelles_themes_fr.values() if c in tableau_validation.columns]]
tableau_validation.to_csv(f"{dossier_tableaux}/S1_validation_croisee_crise_themes.csv")
print(tableau_validation.round(1))

plt.figure(figsize=(11, 7))
carte = plt.imshow(tableau_validation.values, cmap="OrRd", aspect="auto")
plt.colorbar(carte, label="% des publications contenant ce mot-clé")
plt.xticks(range(len(tableau_validation.columns)), tableau_validation.columns, rotation=30, ha="right")
plt.yticks(range(len(tableau_validation.index)), tableau_validation.index)
for i in range(len(tableau_validation.index)):
    for j in range(len(tableau_validation.columns)):
        plt.text(j, i, f"{tableau_validation.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
plt.title("Validation croisée : catégories de crise (a priori) vs thèmes LDA (automatiques)")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/S1_validation_croisee_crise_themes.png")
plt.close()

print("Enregistré : S1 (tableau + heatmap).")
print("(Une catégorie de crise très concentrée sur un seul thème LDA = les deux")
print(" méthodes se confirment. Une catégorie dispersée sur plusieurs thèmes = ")
print(" le mot-clé apparaît dans des contextes plus variés que prévu.)")


# ==============================================================================
# SECTION T — MOTS-CLÉS, RÉPARTITION ET ENGAGEMENT (ANGLAIS/PIDGIN)
# ==============================================================================

print("\n--- T : analyses complémentaires pour l'anglais/pidgin ---")

couleurs_themes_en = cm.Set2(np.linspace(0, 1, meilleur_k_en))

fig, grille = plt.subplots(1, meilleur_k_en, figsize=(5 * meilleur_k_en, 5))
if meilleur_k_en == 1:
    grille = [grille]

mots_cles_par_theme_en = {}
for i, theme in enumerate(lda_en.components_):
    indices_top_mots = theme.argsort()[::-1][:12]
    mots_du_theme = [mots_en[idx] for idx in indices_top_mots]
    poids_du_theme = [theme[idx] for idx in indices_top_mots]
    mots_cles_par_theme_en[i] = mots_du_theme
    print(f"Thème {i} ({libelles_themes_en[i]}) : {', '.join(mots_du_theme)}")
    grille[i].barh(mots_du_theme[::-1], poids_du_theme[::-1], color=couleurs_themes_en[i])
    grille[i].set_title(libelles_themes_en[i], fontsize=9)

plt.tight_layout()
plt.savefig(f"{dossier_figures}/T1_mots_cles_themes_anglais.png")
plt.close()
pd.DataFrame(mots_cles_par_theme_en).to_csv(f"{dossier_tableaux}/T1_mots_cles_themes_anglais.csv", index=False)

df_anglais = df[masque_anglais_pidgin].copy()
df_anglais["theme_nom"] = df_anglais["theme_id"].map(libelles_themes_en)
repartition_themes_en = df_anglais["theme_nom"].value_counts()
repartition_themes_en.to_csv(f"{dossier_tableaux}/T2_repartition_themes_anglais.csv")
print("\nRépartition des thèmes (anglais/pidgin) :")
print(repartition_themes_en)

plt.figure(figsize=(9, 6))
plt.barh(repartition_themes_en.index[::-1], repartition_themes_en.values[::-1],
         color=couleurs_themes_en[:len(repartition_themes_en)][::-1])
plt.title("Nombre de publications par thème dominant (anglais/pidgin)")
plt.xlabel("Nombre de publications")
plt.tight_layout()
plt.savefig(f"{dossier_figures}/T2_repartition_themes_anglais.png")
plt.close()

# Engagement moyen par thème (anglais/pidgin), pour compléter le pendant français (section O)
engagement_par_theme_en = df_anglais.groupby("theme_nom")["nb_likes"].agg(["mean", "count"]).round(1)
engagement_par_theme_en = engagement_par_theme_en.sort_values("mean", ascending=False)
engagement_par_theme_en.to_csv(f"{dossier_tableaux}/T3_engagement_par_theme_anglais.csv")
print("\nEngagement moyen par thème (anglais/pidgin) :")
print(engagement_par_theme_en)

print("\nEnregistré : T1 (mots-clés), T2 (répartition) et T3 (engagement).")


# ==============================================================================
# SECTION U — SAUVEGARDE DU CORPUS AVEC LES THÈMES
# ==============================================================================

print("\n--- U : sauvegarde du corpus enrichi avec les thèmes ---")

chemin_sortie = f"{dossier_data}/bd_f_themes.csv"
df.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")
print(f"Corpus sauvegardé ici : {chemin_sortie}")


# ==============================================================================
# FIN DU SCRIPT
# ==============================================================================

print("\n=== TERMINÉ ! ===")
print(f"Tous les graphiques sont dans : {dossier_figures}")
print(f"Tous les tableaux sont dans   : {dossier_tableaux}")
print("\nRécapitulatif :")
print("  A/B/C : vectorisation, choix de K (cohérence), entraînement LDA")
print("  D/F   : mots-clés (barres + nuages de mots)")
print("  E     : libellés interprétatifs (À AFFINER PAR TOI)")
print("  G     : carte de similarité entre thèmes (heatmap + projection 2D)")
print("  H     : confiance d'attribution des thèmes")
print("  I     : publications représentatives de chaque thème (verbatims)")
print("  J/K   : répartition et évolution temporelle des thèmes")
print("  L     : thèmes par plateforme (heatmap + radar)")
print("  M     : thèmes avant/après le vote")
print("  N     : thèmes par acteur politique (heatmap)")
print("  O     : engagement moyen par thème")
print("  P     : diversité thématique (entropie) par plateforme/période")
print("  Q/R   : sentiment et profil émotionnel par thème (H2)")
print("  S     : validation croisée thèmes LDA vs catégories de crise a priori")
print("  T     : analyses complémentaires (anglais/pidgin)")
print(f"  U     : corpus final sauvegardé -> {chemin_sortie}")
