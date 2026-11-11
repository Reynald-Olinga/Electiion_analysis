# ==============================================================================
# PHASE 3 — PRÉTRAITEMENT NLP
# ==============================================================================
#
# À quoi sert ce script ?
# ------------------------
# Avant de pouvoir faire de l'analyse de sentiment, d'émotion ou de thèmes
# (Phases 5 à 7), il faut d'abord "nettoyer" et préparer le texte brut des
# publications. C'est ce qu'on appelle le prétraitement NLP (NLP = Natural
# Language Processing = traitement automatique du langage).
#
# Un texte brut de réseau social contient plein de choses qui gênent l'analyse :
# des liens internet, des emojis, des fautes de frappe, des mots qui ne veulent
# rien dire ("le", "la", "de"...). On va donc, étape par étape :
#   0. Constituer le corpus : enlever les textes manquants, puis ne garder que
#      la PÉRIODE D'ÉTUDE (15 juillet 2025 -> 15 janvier 2026)
#   1. Enlever les publications en double
#   2. Nettoyer le texte (liens, emojis, hashtags inutiles, caractères bizarres)
#   3. Détecter automatiquement la langue de chaque publication
#   4. Découper le texte en mots (tokenisation) et les ramener à leur forme de
#      base (lemmatisation), puis enlever les mots vides (stopwords)
#
# À la fin, on sauvegarde un nouveau fichier CSV, complété avec toutes ces
# nouvelles colonnes, qui servira de base à toutes les phases suivantes.
#
# IMPORTANT — décision méthodologique :
# La colonne "langue" d'origine est manquante dans environ 65% des publications.
# On ne peut donc pas s'appuyer dessus. On utilise à la place une détection
# automatique de la langue (avec la librairie langdetect), complétée par une
# détection "maison" du pidgin camerounais (aucune librairie ne le fait, car
# c'est une langue très peu outillée en NLP).
#
# MODIFICATIONS DE CETTE VERSION (réexécution cohérente du mémoire) :
#   - la graine de langdetect est FIXÉE (DetectorFactory.seed = 0) : deux
#     exécutions successives donnent désormais exactement les mêmes langues
#     (sans cela, la détection varie légèrement d'une exécution à l'autre) ;
#   - le filtre de PÉRIODE (15/07/2025 -> 15/01/2026) est appliqué ici, AVANT
#     la déduplication : tout le reste de la chaîne (sentiments, émotions,
#     thèmes, indicateurs) travaille donc sur la seule période d'étude ;
#   - chaque étape de la "cascade" de constitution du corpus est comptée,
#     affichée et sauvegardée (cascade_constitution_corpus.csv) : c'est la
#     source des chiffres du tableau de constitution du corpus du mémoire.
# ==============================================================================


# ------------------------------------------------------------------------------
# ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# ------------------------------------------------------------------------------

import pandas as pd                    # pour manipuler le tableau de données
import re                              # pour chercher/remplacer des motifs dans le texte
import os                              # pour créer des dossiers
import unicodedata                     # pour normaliser les caractères un peu bizarres
import emoji                           # pour repérer et enlever les emojis
from langdetect import detect, DetectorFactory   # pour détecter automatiquement la langue d'un texte
import spacy                           # pour tokeniser et lemmatiser (français / anglais)

# Graine fixe : rend la détection de langue REPRODUCTIBLE (même résultat à
# chaque exécution). À faire AVANT le premier appel à detect().
DetectorFactory.seed = 0

# PÉRIODE D'ÉTUDE UNIQUE du mémoire (bornes incluses, heure UTC) :
# du 15 juillet 2025 au 15 janvier 2026, soit 185 jours.
PERIODE_DEBUT = pd.Timestamp("2025-07-15 00:00:00", tz="UTC")
PERIODE_FIN = pd.Timestamp("2026-01-15 23:59:59", tz="UTC")

print("Chargement des modèles de langue (ça peut prendre quelques secondes)...")

# On charge les deux modèles spaCy, un pour le français et un pour l'anglais.
# disable=["parser", "ner"] : on désactive les parties du modèle dont on n'a
# pas besoin ici (l'analyse grammaticale complète et la détection de noms
# propres), ce qui rend le traitement beaucoup plus rapide. On garde
# uniquement ce qu'il faut pour la lemmatisation.
nlp_francais = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
nlp_anglais = spacy.load("en_core_web_sm", disable=["parser", "ner"])

print("Modèles chargés.")


# ------------------------------------------------------------------------------
# ÉTAPE 1 : PRÉPARER LES DOSSIERS DE SORTIE
# ------------------------------------------------------------------------------

dossier_data = "../data"
dossier_tableaux = "../outputs/tables"
os.makedirs(dossier_tableaux, exist_ok=True)


# ------------------------------------------------------------------------------
# ÉTAPE 2 : CHARGER LES DONNÉES
# ------------------------------------------------------------------------------

print("\nLecture du fichier CSV...")
df = pd.read_csv(f"{dossier_data}/bd_f.csv", low_memory=False)
nb_brut = len(df)
print(f"[CASCADE] 1. Publications brutes chargées : {nb_brut}")

# ------------------------------------------------------------------------------
# ÉTAPE 2 bis : CASCADE DE CONSTITUTION DU CORPUS (textes manquants, période)
# ------------------------------------------------------------------------------
# a) Textes manquants : on enlève les lignes sans texte (vides, ou composées
#    uniquement d'espaces) : elles ne contiennent rien d'analysable.
texte_present = df["texte"].notna() & (df["texte"].astype(str).str.strip() != "")
nb_textes_manquants = int((~texte_present).sum())
df = df[texte_present].copy()
nb_avec_texte = len(df)
print(f"[CASCADE] 2. Textes manquants supprimés : {nb_textes_manquants} -> publications avec texte : {nb_avec_texte}")

# b) Période d'étude : on ne garde que les publications datées du 15 juillet
#    2025 au 15 janvier 2026. On convertit les dates dans une série TEMPORAIRE
#    (la colonne d'origine "date_publication" reste inchangée dans le fichier).
dates_utc = pd.to_datetime(df["date_publication"], utc=True, errors="coerce")
nb_date_invalide = int(dates_utc.isna().sum())
nb_avant_periode = int((dates_utc < PERIODE_DEBUT).sum())
nb_apres_periode = int((dates_utc > PERIODE_FIN).sum())
masque_periode = (dates_utc >= PERIODE_DEBUT) & (dates_utc <= PERIODE_FIN)
df = df[masque_periode].copy().reset_index(drop=True)
nb_dans_periode = len(df)
print(f"[CASCADE] 3. Hors période : {nb_avant_periode} avant le {PERIODE_DEBUT.date()}, "
      f"{nb_apres_periode} après le {PERIODE_FIN.date()}, {nb_date_invalide} dates illisibles")
print(f"[CASCADE]    -> publications dans la période d'étude : {nb_dans_periode}")


# ==============================================================================
# SECTION A — SUPPRESSION DES DOUBLONS
# ==============================================================================
# Un "doublon", c'est une publication que le robot de collecte (Apify) a
# aspirée deux fois par erreur : c'est exactement le MÊME message, du MÊME
# auteur, publié à la MÊME date et à la MÊME heure. C'est un doublon de
# collecte, pas un vrai contenu à part.
#
# ATTENTION, piège à éviter : on ne peut PAS se contenter de repérer les
# textes identiques tout seuls. Sur les réseaux sociaux, énormément de gens
# différents écrivent exactement la même chose sans se concerter (par
# exemple "😂😂😂" ou "🥰"). Si on supprimait tous les textes identiques,
# on supprimerait par erreur des dizaines de milliers de publications bien
# réelles et distinctes, juste parce qu'elles se ressemblent par hasard !
# (on l'a vérifié : rien qu'avec "🥰🥰🥰", il y a 448 auteurs DIFFÉRENTS).
#
# On considère donc qu'il y a un vrai doublon seulement quand TROIS choses
# sont identiques à la fois : le texte, l'auteur, ET la date de publication.
# Dans ce cas précis, c'est forcément la même publication capturée deux fois.

print("\n--- A : suppression des doublons ---")

nb_avant_doublons = len(df)
df = df.drop_duplicates(subset=["texte", "auteur_pseudo", "date_publication"], keep="first")
nb_apres_doublons = len(df)

print(f"Nombre de publications avant suppression des doublons : {nb_avant_doublons}")
print(f"Nombre de publications après suppression des doublons  : {nb_apres_doublons}")
print(f"Nombre de doublons supprimés : {nb_avant_doublons - nb_apres_doublons}")
print(f"[CASCADE] 4. Doublons supprimés : {nb_avant_doublons - nb_apres_doublons} -> publications après déduplication : {nb_apres_doublons}")


# ==============================================================================
# SECTION B — NETTOYAGE DE BASE DU TEXTE
# ==============================================================================
# Ici on nettoie chaque texte, un par un, en suivant toujours le même ordre
# d'étapes. On regroupe ces étapes dans une fonction (une sorte de "recette"
# qu'on peut réutiliser), pour ne pas avoir à copier-coller le même bloc de
# code 47 000 fois. La fonction fait, DANS L'ORDRE :
#
#   1. Suppression des URLs (les liens http://... ou https://...)
#   2. Suppression des emojis
#   3. Suppression des hashtags "inutiles" (des hashtags génériques qui ne
#      veulent rien dire sur le fond, comme #fyp ou #viral), et pour les
#      autres hashtags, on enlève juste le symbole # pour garder le mot
#      (ex: "#Cameroun" devient "Cameroun", utilisable comme mot normal)
#   4. Suppression des mentions (@untel), qui désignent des comptes et non
#      des mots de vocabulaire
#   5. Normalisation Unicode (pour corriger des caractères mal encodés)
#   6. Suppression des caractères spéciaux (ponctuation, symboles...) en ne
#      gardant que les lettres, les chiffres et les espaces
#   7. Réduction des lettres répétées (ex: "trooooop" devient "troop"), une
#      première approche simple pour limiter l'effet des fautes de frappe
#      et des exagérations typiques des réseaux sociaux
#   8. Nettoyage des espaces multiples
#   9. Mise en minuscules

print("\n--- B : nettoyage du texte (URLs, emojis, hashtags, caractères spéciaux...) ---")

# Une petite liste de hashtags "génériques" qu'on juge inutiles pour l'analyse,
# car ils ne parlent pas du fond du message (ils servent juste à être vu).
hashtags_a_supprimer = [
    "fyp", "foryou", "foryoupage", "viral", "trending", "follow",
    "like4like", "share", "video", "live", "tiktok", "reels", "explore",
]


def nettoyer_un_texte(texte):
    """
    Prend un texte brut en entrée, et retourne une version nettoyée.
    On applique les étapes une par une, dans l'ordre expliqué au-dessus.
    """
    texte = str(texte)

    # 1. Suppression des URLs
    texte = re.sub(r"http\S+|www\.\S+", " ", texte)

    # 2. Suppression des emojis (la librairie "emoji" sait les reconnaître)
    texte = emoji.replace_emoji(texte, replace=" ")

    # 3. Suppression des hashtags inutiles, et nettoyage des autres hashtags
    for mot_generique in hashtags_a_supprimer:
        texte = re.sub(rf"#{mot_generique}\b", " ", texte, flags=re.IGNORECASE)
    # Pour les hashtags restants, on enlève juste le "#" et on garde le mot
    texte = re.sub(r"#(\w+)", r"\1", texte)

    # 4. Suppression des mentions (@quelqu'un)
    texte = re.sub(r"@\w+", " ", texte)

    # 5. Normalisation Unicode : corrige certains caractères mal encodés
    # (par exemple des accents composés bizarrement)
    texte = unicodedata.normalize("NFKC", texte)

    # 6. Suppression des caractères spéciaux : on garde seulement les lettres
    # (avec accents), les chiffres et les espaces
    texte = re.sub(r"[^a-zA-Zà-ÿÀ-Ÿ0-9\s]", " ", texte)

    # 7. Réduction des lettres répétées : si une lettre est répétée 3 fois ou
    # plus d'affilée (ex: "trooooop"), on la ramène à 2 répétitions ("troop").
    # Ce n'est pas une vraie correction orthographique (ça, c'est complexe et
    # peu fiable pour du français/anglais/pidgin mélangés), mais ça limite
    # déjà beaucoup les déformations volontaires ("mdrrrrrr", "nooonnn"...).
    texte = re.sub(r"(.)\1{2,}", r"\1\1", texte)

    # 8. Nettoyage des espaces multiples
    texte = re.sub(r"\s+", " ", texte).strip()

    # 9. Minuscules
    texte = texte.lower()

    return texte


# On applique la fonction à chaque texte, un par un, avec une boucle for.
textes_nettoyes = []
for texte_original in df["texte"]:
    textes_nettoyes.append(nettoyer_un_texte(texte_original))

df["texte_nettoye"] = textes_nettoyes

# On enlève les publications qui, une fois nettoyées, ne contiennent plus rien
# (par exemple un post qui n'était qu'un lien ou qu'un emoji).
nb_avant_vide = len(df)
df = df[df["texte_nettoye"].str.strip() != ""]
nb_apres_vide = len(df)
print(f"Publications devenues vides après nettoyage (supprimées) : {nb_avant_vide - nb_apres_vide}")
print(f"[CASCADE] 5. Publications vides après nettoyage : {nb_avant_vide - nb_apres_vide} -> corpus analytique : {nb_apres_vide}")

# Un petit exemple pour voir la différence avant/après
print("\nExemple de nettoyage :")
print("AVANT :", df["texte"].iloc[0])
print("APRÈS :", df["texte_nettoye"].iloc[0])


# ==============================================================================
# SECTION C — DÉTECTION DE LA LANGUE (Y COMPRIS LE PIDGIN)
# ==============================================================================
# On détecte la langue de chaque publication automatiquement, car la colonne
# d'origine "langue" est trop souvent vide. On procède en deux temps :
#
#   1. D'abord, on regarde si le texte contient des mots typiques du pidgin
#      camerounais (comme "wetin", "abeg", "sabi", "waka"...). Aucune
#      librairie standard ne sait détecter le pidgin camerounais, donc on
#      fait nous-mêmes une détection simple basée sur une liste de mots.
#   2. Si ce n'est pas du pidgin, on utilise la librairie langdetect, qui sait
#      reconnaître une centaine de langues (dont le français et l'anglais)
#      à partir du texte lui-même.

print("\n--- C : détection de la langue (français / anglais / pidgin / autre) ---")

# Liste de mots caractéristiques du pidgin camerounais / camfranglais.
# Ce n'est pas une liste parfaite, mais elle permet de repérer les cas les
# plus clairs.
mots_pidgin = [
    "wetin", "abeg", "sabi", "waka", "wahala", "sef", "dem", "dey",
    "na so", "no fit", "how far", "chop", "small small", "massa",
    "kie", "ashawo", "wuna", "una", "wey", "dis", "dat one", "vex",
]


def detecter_pidgin(texte_minuscule):
    """Retourne True si le texte contient au moins 2 mots typiques du pidgin."""
    nb_marqueurs_trouves = 0
    for mot in mots_pidgin:
        if mot in texte_minuscule:
            nb_marqueurs_trouves += 1
    return nb_marqueurs_trouves >= 2


def detecter_langue(texte_nettoye):
    """
    Détecte la langue d'un texte nettoyé.
    Retourne un code : 'pidgin', 'fr', 'en', ou le code de langdetect,
    ou 'indetermine' si le texte est trop court ou illisible.
    """
    # IMPORTANT : langdetect n'est fiable que sur des textes d'une certaine
    # longueur. On l'a testé nous-mêmes : sur un texte très court comme
    # "bravo a tous" (12 caractères), langdetect répond "slovaque" avec
    # 99.99% de confiance... alors que ce n'est évidemment pas du slovaque !
    # Le score de confiance ne suffit donc pas à filtrer les erreurs : c'est
    # la LONGUEUR du texte qui pose problème. On préfère donc, par prudence,
    # ne pas essayer de deviner la langue des textes trop courts (moins de
    # 15 caractères), et les marquer clairement comme "indéterminé" plutôt
    # que de leur donner une langue fausse avec une fausse assurance.
    if len(texte_nettoye) < 15:
        return "indetermine"

    # Étape 1 : est-ce du pidgin ?
    if detecter_pidgin(texte_nettoye):
        return "pidgin"

    # Étape 2 : sinon, on demande à langdetect
    try:
        return detect(texte_nettoye)
    except Exception:
        # langdetect peut échouer sur des textes très courts ou bizarres
        return "indetermine"


langues_detectees = []
for texte in df["texte_nettoye"]:
    langues_detectees.append(detecter_langue(texte))

df["langue_detectee"] = langues_detectees

print("Répartition des langues détectées automatiquement :")
print(df["langue_detectee"].value_counts().head(15))


# ==============================================================================
# SECTION D — TOKENISATION, LEMMATISATION ET SUPPRESSION DES STOPWORDS
# ==============================================================================
# - La "tokenisation" consiste à découper un texte en une liste de mots
#   (des "tokens").
# - La "lemmatisation" consiste à ramener chaque mot à sa forme de base
#   (par exemple "manifestions" -> "manifester", "sont" -> "être"). Ça permet
#   de compter "vote", "votes" et "voter" comme un seul et même mot dans les
#   analyses suivantes, plutôt que trois mots différents.
# - Les "stopwords" (mots vides) sont des mots très fréquents mais peu
#   porteurs de sens ("le", "la", "et", "the", "is"...). On les enlève pour
#   que les analyses se concentrent sur les mots qui comptent vraiment.
#
# On a un modèle spaCy pour le français et un pour l'anglais. Comme le
# pidgin camerounais est très proche de l'anglais, on utilise le modèle
# anglais pour le traiter aussi (c'est une approximation raisonnable, à
# mentionner comme limite dans le mémoire). Pour les autres langues (rares
# dans le corpus), on se contente d'une tokenisation simple, sans
# lemmatisation, faute de modèle disponible.

print("\n--- D : tokenisation, lemmatisation et suppression des stopwords ---")

# On récupère les listes de mots vides fournies par spaCy pour le français et l'anglais
from spacy.lang.fr.stop_words import STOP_WORDS as mots_vides_francais
from spacy.lang.en.stop_words import STOP_WORDS as mots_vides_anglais

# On y ajoute quelques mots vides "maison", typiques des réseaux sociaux et
# qui ne veulent rien dire de particulier
mots_vides_supplementaires = {"cest", "quil", "qu", "j", "n", "d", "l", "s", "amp"}
mots_vides_francais = mots_vides_francais | mots_vides_supplementaires
mots_vides_anglais = mots_vides_anglais | mots_vides_supplementaires


def lemmatiser_et_filtrer(texte_nettoye, langue):
    """
    Tokenise et lemmatise un texte avec le bon modèle spaCy selon sa langue,
    puis enlève les stopwords. Retourne le texte final sous forme de chaîne
    de mots séparés par des espaces (utilisable directement dans les phases
    suivantes, par exemple pour le TF-IDF ou le topic modeling).
    """
    if langue == "fr":
        document = nlp_francais(texte_nettoye)
        mots_vides = mots_vides_francais
    elif langue in ("en", "pidgin"):
        # on traite le pidgin avec le modèle anglais, faute de mieux
        document = nlp_anglais(texte_nettoye)
        mots_vides = mots_vides_anglais
    else:
        # Langue non prise en charge : on tokenise simplement, sans lemmatiser
        mots = texte_nettoye.split()
        mots_gardes = [mot for mot in mots if len(mot) > 2]
        return " ".join(mots_gardes)

    mots_lemmatises = []
    for token in document:
        lemme = token.lemma_.strip()
        # On garde le mot seulement si : il n'est pas vide, il fait plus de
        # 2 lettres, et ce n'est pas un mot vide (stopword)
        if len(lemme) > 2 and lemme not in mots_vides:
            mots_lemmatises.append(lemme)

    return " ".join(mots_lemmatises)


# On applique la fonction à chaque publication. Comme spaCy est plus rapide
# quand on lui donne beaucoup de textes d'un coup, on sépare d'abord les
# textes français des textes anglais/pidgin, puis on utilise nlp.pipe()
# (qui traite une liste de textes efficacement) pour chaque groupe.

print("Lemmatisation en cours (peut prendre 2 à 5 minutes pour tout le corpus)...")

# On crée une colonne vide qu'on va remplir petit à petit
df["texte_lemmatise"] = ""

# --- Textes en français ---
indices_francais = df.index[df["langue_detectee"] == "fr"]
textes_francais = df.loc[indices_francais, "texte_nettoye"].tolist()

resultats_francais = []
for document in nlp_francais.pipe(textes_francais, batch_size=200):
    mots_lemmatises = [
        token.lemma_.strip() for token in document
        if len(token.lemma_.strip()) > 2 and token.lemma_.strip() not in mots_vides_francais
    ]
    resultats_francais.append(" ".join(mots_lemmatises))

df.loc[indices_francais, "texte_lemmatise"] = resultats_francais
print(f"  -> {len(textes_francais)} textes français lemmatisés")

# --- Textes en anglais ou en pidgin (on utilise le même modèle anglais) ---
indices_anglais = df.index[df["langue_detectee"].isin(["en", "pidgin"])]
textes_anglais = df.loc[indices_anglais, "texte_nettoye"].tolist()

resultats_anglais = []
for document in nlp_anglais.pipe(textes_anglais, batch_size=200):
    mots_lemmatises = [
        token.lemma_.strip() for token in document
        if len(token.lemma_.strip()) > 2 and token.lemma_.strip() not in mots_vides_anglais
    ]
    resultats_anglais.append(" ".join(mots_lemmatises))

df.loc[indices_anglais, "texte_lemmatise"] = resultats_anglais
print(f"  -> {len(textes_anglais)} textes anglais/pidgin lemmatisés")

# --- Toutes les autres langues : tokenisation simple, sans lemmatisation ---
indices_autres = df.index[~df["langue_detectee"].isin(["fr", "en", "pidgin"])]
for indice in indices_autres:
    texte = df.at[indice, "texte_nettoye"]
    mots = [mot for mot in texte.split() if len(mot) > 2]
    df.at[indice, "texte_lemmatise"] = " ".join(mots)

print(f"  -> {len(indices_autres)} textes dans d'autres langues (tokenisation simple, sans lemmatisation)")

print("\nExemple de lemmatisation :")
exemple_francais = df[df["langue_detectee"] == "fr"].iloc[0]
print("Texte nettoyé   :", exemple_francais["texte_nettoye"])
print("Texte lemmatisé :", exemple_francais["texte_lemmatise"])


# ==============================================================================
# SECTION E — SAUVEGARDE DU CORPUS NETTOYÉ
# ==============================================================================
# On sauvegarde un nouveau fichier CSV qui contient TOUTES les colonnes
# d'origine, PLUS les nouvelles colonnes qu'on vient de créer :
#   - texte_nettoye     : le texte nettoyé (avant lemmatisation)
#   - langue_detectee   : la langue détectée automatiquement
#   - texte_lemmatise   : le texte final, prêt pour les analyses NLP
#     (sentiment, thématiques...) des phases suivantes
#
# C'est ce fichier qu'on utilisera comme point de départ des Phases 4 à 12.

print("\n--- E : sauvegarde du corpus nettoyé ---")

chemin_sortie = f"{dossier_data}/bd_f_nettoye.csv"
df.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")

print(f"Corpus nettoyé sauvegardé ici : {chemin_sortie}")
print(f"Nombre final de publications : {len(df)}")

# On sauvegarde aussi un petit résumé des langues détectées, utile pour le mémoire
resume_langues = df["langue_detectee"].value_counts()
resume_langues.to_csv(f"{dossier_tableaux}/langues_detectees_phase3.csv")

# Couverture du pipeline : part du corpus dont la langue est prise en charge
# pour les analyses de sentiment, d'émotion et de thèmes (français, anglais, pidgin)
nb_couvert = int(df["langue_detectee"].isin(["fr", "en", "pidgin"]).sum())
print(f"Publications traitées (fr + en + pidgin) : {nb_couvert} / {len(df)} ({nb_couvert / len(df) * 100:.1f} %)")
print(f"Publications hors analyse : {len(df) - nb_couvert} ({(len(df) - nb_couvert) / len(df) * 100:.1f} %)")

# Cascade de constitution du corpus, sauvegardée pour le tableau du mémoire
cascade = pd.DataFrame([
    ("Publications brutes", nb_brut, None),
    ("Textes manquants", -nb_textes_manquants, "supprimés"),
    ("Publications avec texte", nb_avec_texte, None),
    (f"Avant le {PERIODE_DEBUT.date()}", -nb_avant_periode, "hors période"),
    (f"Après le {PERIODE_FIN.date()}", -nb_apres_periode, "hors période"),
    ("Dates illisibles", -nb_date_invalide, "hors période"),
    ("Publications dans la période d'étude", nb_dans_periode, None),
    ("Doublons (texte + auteur + date)", -(nb_avant_doublons - nb_apres_doublons), "supprimés"),
    ("Publications après déduplication", nb_apres_doublons, None),
    ("Vides après nettoyage", -(nb_avant_vide - nb_apres_vide), "supprimés"),
    ("Corpus analytique final", len(df), None),
], columns=["etape", "effectif", "nature"])
cascade.to_csv(f"{dossier_tableaux}/cascade_constitution_corpus.csv", index=False, encoding="utf-8-sig")
print("\nCascade de constitution du corpus :")
print(cascade.to_string(index=False))


# ==============================================================================
# FIN DU SCRIPT
# ==============================================================================

print("\n=== TERMINÉ ! ===")
print("Récapitulatif :")
print(f"  - Période d'étude : {PERIODE_DEBUT.date()} -> {PERIODE_FIN.date()}")
print(f"  - Publications brutes : {nb_brut} ; avec texte : {nb_avec_texte} ; dans la période : {nb_dans_periode}")
print(f"  - Doublons supprimés : {nb_avant_doublons - nb_apres_doublons}")
print(f"  - Publications vides supprimées après nettoyage : {nb_avant_vide - nb_apres_vide}")
print(f"  - Publications finales : {len(df)}")
print(f"  - Fichier prêt pour la Phase 4 : {chemin_sortie}")





























































































































































# # ==============================================================================
# # PHASE 3 — PRÉTRAITEMENT NLP
# # ==============================================================================
# #
# # À quoi sert ce script ?
# # ------------------------
# # Avant de pouvoir faire de l'analyse de sentiment, d'émotion ou de thèmes
# # (Phases 5 à 7), il faut d'abord "nettoyer" et préparer le texte brut des
# # publications. C'est ce qu'on appelle le prétraitement NLP (NLP = Natural
# # Language Processing = traitement automatique du langage).
# #
# # Un texte brut de réseau social contient plein de choses qui gênent l'analyse :
# # des liens internet, des emojis, des fautes de frappe, des mots qui ne veulent
# # rien dire ("le", "la", "de"...). On va donc, étape par étape :
# #   1. Enlever les publications en double
# #   2. Nettoyer le texte (liens, emojis, hashtags inutiles, caractères bizarres)
# #   3. Détecter automatiquement la langue de chaque publication
# #   4. Découper le texte en mots (tokenisation) et les ramener à leur forme de
# #      base (lemmatisation), puis enlever les mots vides (stopwords)
# #
# # À la fin, on sauvegarde un nouveau fichier CSV, complété avec toutes ces
# # nouvelles colonnes, qui servira de base à toutes les phases suivantes.
# #
# # IMPORTANT — décision méthodologique :
# # La colonne "langue" d'origine est manquante dans environ 65% des publications.
# # On ne peut donc pas s'appuyer dessus. On utilise à la place une détection
# # automatique de la langue (avec la librairie langdetect), complétée par une
# # détection "maison" du pidgin camerounais (aucune librairie ne le fait, car
# # c'est une langue très peu outillée en NLP).
# # ==============================================================================


# # ------------------------------------------------------------------------------
# # ÉTAPE 0 : IMPORTER LES OUTILS DONT ON A BESOIN
# # ------------------------------------------------------------------------------

# import pandas as pd                    # pour manipuler le tableau de données
# import re                              # pour chercher/remplacer des motifs dans le texte
# import os                              # pour créer des dossiers
# import unicodedata                     # pour normaliser les caractères un peu bizarres
# import emoji                           # pour repérer et enlever les emojis
# from langdetect import detect          # pour détecter automatiquement la langue d'un texte
# import spacy                           # pour tokeniser et lemmatiser (français / anglais)

# print("Chargement des modèles de langue (ça peut prendre quelques secondes)...")

# # On charge les deux modèles spaCy, un pour le français et un pour l'anglais.
# # disable=["parser", "ner"] : on désactive les parties du modèle dont on n'a
# # pas besoin ici (l'analyse grammaticale complète et la détection de noms
# # propres), ce qui rend le traitement beaucoup plus rapide. On garde
# # uniquement ce qu'il faut pour la lemmatisation.
# nlp_francais = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
# nlp_anglais = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# print("Modèles chargés.")


# # ------------------------------------------------------------------------------
# # ÉTAPE 1 : PRÉPARER LES DOSSIERS DE SORTIE
# # ------------------------------------------------------------------------------

# dossier_data = "../data"
# dossier_tableaux = "../outputs/tables"
# os.makedirs(dossier_tableaux, exist_ok=True)


# # ------------------------------------------------------------------------------
# # ÉTAPE 2 : CHARGER LES DONNÉES
# # ------------------------------------------------------------------------------

# print("\nLecture du fichier CSV...")
# df = pd.read_csv(f"{dossier_data}/bd_f.csv", low_memory=False)
# print(f"On a chargé {len(df)} lignes au départ.")

# # On enlève les lignes sans texte, comme dans la Phase 2 : elles ne servent à rien ici.
# df = df.dropna(subset=["texte"])
# print(f"Après avoir enlevé les lignes sans texte, il reste {len(df)} publications.")


# # ==============================================================================
# # SECTION A — SUPPRESSION DES DOUBLONS
# # ==============================================================================
# # Un "doublon", c'est une publication que le robot de collecte (Apify) a
# # aspirée deux fois par erreur : c'est exactement le MÊME message, du MÊME
# # auteur, publié à la MÊME date et à la MÊME heure. C'est un doublon de
# # collecte, pas un vrai contenu à part.
# #
# # ATTENTION, piège à éviter : on ne peut PAS se contenter de repérer les
# # textes identiques tout seuls. Sur les réseaux sociaux, énormément de gens
# # différents écrivent exactement la même chose sans se concerter (par
# # exemple "😂😂😂" ou "🥰"). Si on supprimait tous les textes identiques,
# # on supprimerait par erreur des dizaines de milliers de publications bien
# # réelles et distinctes, juste parce qu'elles se ressemblent par hasard !
# # (on l'a vérifié : rien qu'avec "🥰🥰🥰", il y a 448 auteurs DIFFÉRENTS).
# #
# # On considère donc qu'il y a un vrai doublon seulement quand TROIS choses
# # sont identiques à la fois : le texte, l'auteur, ET la date de publication.
# # Dans ce cas précis, c'est forcément la même publication capturée deux fois.

# print("\n--- A : suppression des doublons ---")

# nb_avant_doublons = len(df)
# df = df.drop_duplicates(subset=["texte", "auteur_pseudo", "date_publication"], keep="first")
# nb_apres_doublons = len(df)

# print(f"Nombre de publications avant suppression des doublons : {nb_avant_doublons}")
# print(f"Nombre de publications après suppression des doublons  : {nb_apres_doublons}")
# print(f"Nombre de doublons supprimés : {nb_avant_doublons - nb_apres_doublons}")


# # ==============================================================================
# # SECTION B — NETTOYAGE DE BASE DU TEXTE
# # ==============================================================================
# # Ici on nettoie chaque texte, un par un, en suivant toujours le même ordre
# # d'étapes. On regroupe ces étapes dans une fonction (une sorte de "recette"
# # qu'on peut réutiliser), pour ne pas avoir à copier-coller le même bloc de
# # code 47 000 fois. La fonction fait, DANS L'ORDRE :
# #
# #   1. Suppression des URLs (les liens http://... ou https://...)
# #   2. Suppression des emojis
# #   3. Suppression des hashtags "inutiles" (des hashtags génériques qui ne
# #      veulent rien dire sur le fond, comme #fyp ou #viral), et pour les
# #      autres hashtags, on enlève juste le symbole # pour garder le mot
# #      (ex: "#Cameroun" devient "Cameroun", utilisable comme mot normal)
# #   4. Suppression des mentions (@untel), qui désignent des comptes et non
# #      des mots de vocabulaire
# #   5. Normalisation Unicode (pour corriger des caractères mal encodés)
# #   6. Suppression des caractères spéciaux (ponctuation, symboles...) en ne
# #      gardant que les lettres, les chiffres et les espaces
# #   7. Réduction des lettres répétées (ex: "trooooop" devient "troop"), une
# #      première approche simple pour limiter l'effet des fautes de frappe
# #      et des exagérations typiques des réseaux sociaux
# #   8. Nettoyage des espaces multiples
# #   9. Mise en minuscules

# print("\n--- B : nettoyage du texte (URLs, emojis, hashtags, caractères spéciaux...) ---")

# # Une petite liste de hashtags "génériques" qu'on juge inutiles pour l'analyse,
# # car ils ne parlent pas du fond du message (ils servent juste à être vu).
# hashtags_a_supprimer = [
#     "fyp", "foryou", "foryoupage", "viral", "trending", "follow",
#     "like4like", "share", "video", "live", "tiktok", "reels", "explore",
# ]


# def nettoyer_un_texte(texte):
#     """
#     Prend un texte brut en entrée, et retourne une version nettoyée.
#     On applique les étapes une par une, dans l'ordre expliqué au-dessus.
#     """
#     texte = str(texte)

#     # 1. Suppression des URLs
#     texte = re.sub(r"http\S+|www\.\S+", " ", texte)

#     # 2. Suppression des emojis (la librairie "emoji" sait les reconnaître)
#     texte = emoji.replace_emoji(texte, replace=" ")

#     # 3. Suppression des hashtags inutiles, et nettoyage des autres hashtags
#     for mot_generique in hashtags_a_supprimer:
#         texte = re.sub(rf"#{mot_generique}\b", " ", texte, flags=re.IGNORECASE)
#     # Pour les hashtags restants, on enlève juste le "#" et on garde le mot
#     texte = re.sub(r"#(\w+)", r"\1", texte)

#     # 4. Suppression des mentions (@quelqu'un)
#     texte = re.sub(r"@\w+", " ", texte)

#     # 5. Normalisation Unicode : corrige certains caractères mal encodés
#     # (par exemple des accents composés bizarrement)
#     texte = unicodedata.normalize("NFKC", texte)

#     # 6. Suppression des caractères spéciaux : on garde seulement les lettres
#     # (avec accents), les chiffres et les espaces
#     texte = re.sub(r"[^a-zA-Zà-ÿÀ-Ÿ0-9\s]", " ", texte)

#     # 7. Réduction des lettres répétées : si une lettre est répétée 3 fois ou
#     # plus d'affilée (ex: "trooooop"), on la ramène à 2 répétitions ("troop").
#     # Ce n'est pas une vraie correction orthographique (ça, c'est complexe et
#     # peu fiable pour du français/anglais/pidgin mélangés), mais ça limite
#     # déjà beaucoup les déformations volontaires ("mdrrrrrr", "nooonnn"...).
#     texte = re.sub(r"(.)\1{2,}", r"\1\1", texte)

#     # 8. Nettoyage des espaces multiples
#     texte = re.sub(r"\s+", " ", texte).strip()

#     # 9. Minuscules
#     texte = texte.lower()

#     return texte


# # On applique la fonction à chaque texte, un par un, avec une boucle for.
# textes_nettoyes = []
# for texte_original in df["texte"]:
#     textes_nettoyes.append(nettoyer_un_texte(texte_original))

# df["texte_nettoye"] = textes_nettoyes

# # On enlève les publications qui, une fois nettoyées, ne contiennent plus rien
# # (par exemple un post qui n'était qu'un lien ou qu'un emoji).
# nb_avant_vide = len(df)
# df = df[df["texte_nettoye"].str.strip() != ""]
# nb_apres_vide = len(df)
# print(f"Publications devenues vides après nettoyage (supprimées) : {nb_avant_vide - nb_apres_vide}")

# # Un petit exemple pour voir la différence avant/après
# print("\nExemple de nettoyage :")
# print("AVANT :", df["texte"].iloc[0])
# print("APRÈS :", df["texte_nettoye"].iloc[0])


# # ==============================================================================
# # SECTION C — DÉTECTION DE LA LANGUE (Y COMPRIS LE PIDGIN)
# # ==============================================================================
# # On détecte la langue de chaque publication automatiquement, car la colonne
# # d'origine "langue" est trop souvent vide. On procède en deux temps :
# #
# #   1. D'abord, on regarde si le texte contient des mots typiques du pidgin
# #      camerounais (comme "wetin", "abeg", "sabi", "waka"...). Aucune
# #      librairie standard ne sait détecter le pidgin camerounais, donc on
# #      fait nous-mêmes une détection simple basée sur une liste de mots.
# #   2. Si ce n'est pas du pidgin, on utilise la librairie langdetect, qui sait
# #      reconnaître une centaine de langues (dont le français et l'anglais)
# #      à partir du texte lui-même.

# print("\n--- C : détection de la langue (français / anglais / pidgin / autre) ---")

# # Liste de mots caractéristiques du pidgin camerounais / camfranglais.
# # Ce n'est pas une liste parfaite, mais elle permet de repérer les cas les
# # plus clairs.
# mots_pidgin = [
#     "wetin", "abeg", "sabi", "waka", "wahala", "sef", "dem", "dey",
#     "na so", "no fit", "how far", "chop", "small small", "massa",
#     "kie", "ashawo", "wuna", "una", "wey", "dis", "dat one", "vex",
# ]


# def detecter_pidgin(texte_minuscule):
#     """Retourne True si le texte contient au moins 2 mots typiques du pidgin."""
#     nb_marqueurs_trouves = 0
#     for mot in mots_pidgin:
#         if mot in texte_minuscule:
#             nb_marqueurs_trouves += 1
#     return nb_marqueurs_trouves >= 2


# def detecter_langue(texte_nettoye):
#     """
#     Détecte la langue d'un texte nettoyé.
#     Retourne un code : 'pidgin', 'fr', 'en', ou le code de langdetect,
#     ou 'indetermine' si le texte est trop court ou illisible.
#     """
#     # IMPORTANT : langdetect n'est fiable que sur des textes d'une certaine
#     # longueur. On l'a testé nous-mêmes : sur un texte très court comme
#     # "bravo a tous" (12 caractères), langdetect répond "slovaque" avec
#     # 99.99% de confiance... alors que ce n'est évidemment pas du slovaque !
#     # Le score de confiance ne suffit donc pas à filtrer les erreurs : c'est
#     # la LONGUEUR du texte qui pose problème. On préfère donc, par prudence,
#     # ne pas essayer de deviner la langue des textes trop courts (moins de
#     # 15 caractères), et les marquer clairement comme "indéterminé" plutôt
#     # que de leur donner une langue fausse avec une fausse assurance.
#     if len(texte_nettoye) < 15:
#         return "indetermine"

#     # Étape 1 : est-ce du pidgin ?
#     if detecter_pidgin(texte_nettoye):
#         return "pidgin"

#     # Étape 2 : sinon, on demande à langdetect
#     try:
#         return detect(texte_nettoye)
#     except Exception:
#         # langdetect peut échouer sur des textes très courts ou bizarres
#         return "indetermine"


# langues_detectees = []
# for texte in df["texte_nettoye"]:
#     langues_detectees.append(detecter_langue(texte))

# df["langue_detectee"] = langues_detectees

# print("Répartition des langues détectées automatiquement :")
# print(df["langue_detectee"].value_counts().head(15))


# # ==============================================================================
# # SECTION D — TOKENISATION, LEMMATISATION ET SUPPRESSION DES STOPWORDS
# # ==============================================================================
# # - La "tokenisation" consiste à découper un texte en une liste de mots
# #   (des "tokens").
# # - La "lemmatisation" consiste à ramener chaque mot à sa forme de base
# #   (par exemple "manifestions" -> "manifester", "sont" -> "être"). Ça permet
# #   de compter "vote", "votes" et "voter" comme un seul et même mot dans les
# #   analyses suivantes, plutôt que trois mots différents.
# # - Les "stopwords" (mots vides) sont des mots très fréquents mais peu
# #   porteurs de sens ("le", "la", "et", "the", "is"...). On les enlève pour
# #   que les analyses se concentrent sur les mots qui comptent vraiment.
# #
# # On a un modèle spaCy pour le français et un pour l'anglais. Comme le
# # pidgin camerounais est très proche de l'anglais, on utilise le modèle
# # anglais pour le traiter aussi (c'est une approximation raisonnable, à
# # mentionner comme limite dans le mémoire). Pour les autres langues (rares
# # dans le corpus), on se contente d'une tokenisation simple, sans
# # lemmatisation, faute de modèle disponible.

# print("\n--- D : tokenisation, lemmatisation et suppression des stopwords ---")

# # On récupère les listes de mots vides fournies par spaCy pour le français et l'anglais
# from spacy.lang.fr.stop_words import STOP_WORDS as mots_vides_francais
# from spacy.lang.en.stop_words import STOP_WORDS as mots_vides_anglais

# # On y ajoute quelques mots vides "maison", typiques des réseaux sociaux et
# # qui ne veulent rien dire de particulier
# mots_vides_supplementaires = {"cest", "quil", "qu", "j", "n", "d", "l", "s", "amp"}
# mots_vides_francais = mots_vides_francais | mots_vides_supplementaires
# mots_vides_anglais = mots_vides_anglais | mots_vides_supplementaires


# def lemmatiser_et_filtrer(texte_nettoye, langue):
#     """
#     Tokenise et lemmatise un texte avec le bon modèle spaCy selon sa langue,
#     puis enlève les stopwords. Retourne le texte final sous forme de chaîne
#     de mots séparés par des espaces (utilisable directement dans les phases
#     suivantes, par exemple pour le TF-IDF ou le topic modeling).
#     """
#     if langue == "fr":
#         document = nlp_francais(texte_nettoye)
#         mots_vides = mots_vides_francais
#     elif langue in ("en", "pidgin"):
#         # on traite le pidgin avec le modèle anglais, faute de mieux
#         document = nlp_anglais(texte_nettoye)
#         mots_vides = mots_vides_anglais
#     else:
#         # Langue non prise en charge : on tokenise simplement, sans lemmatiser
#         mots = texte_nettoye.split()
#         mots_gardes = [mot for mot in mots if len(mot) > 2]
#         return " ".join(mots_gardes)

#     mots_lemmatises = []
#     for token in document:
#         lemme = token.lemma_.strip()
#         # On garde le mot seulement si : il n'est pas vide, il fait plus de
#         # 2 lettres, et ce n'est pas un mot vide (stopword)
#         if len(lemme) > 2 and lemme not in mots_vides:
#             mots_lemmatises.append(lemme)

#     return " ".join(mots_lemmatises)


# # On applique la fonction à chaque publication. Comme spaCy est plus rapide
# # quand on lui donne beaucoup de textes d'un coup, on sépare d'abord les
# # textes français des textes anglais/pidgin, puis on utilise nlp.pipe()
# # (qui traite une liste de textes efficacement) pour chaque groupe.

# print("Lemmatisation en cours (peut prendre 2 à 5 minutes pour tout le corpus)...")

# # On crée une colonne vide qu'on va remplir petit à petit
# df["texte_lemmatise"] = ""

# # --- Textes en français ---
# indices_francais = df.index[df["langue_detectee"] == "fr"]
# textes_francais = df.loc[indices_francais, "texte_nettoye"].tolist()

# resultats_francais = []
# for document in nlp_francais.pipe(textes_francais, batch_size=200):
#     mots_lemmatises = [
#         token.lemma_.strip() for token in document
#         if len(token.lemma_.strip()) > 2 and token.lemma_.strip() not in mots_vides_francais
#     ]
#     resultats_francais.append(" ".join(mots_lemmatises))

# df.loc[indices_francais, "texte_lemmatise"] = resultats_francais
# print(f"  -> {len(textes_francais)} textes français lemmatisés")

# # --- Textes en anglais ou en pidgin (on utilise le même modèle anglais) ---
# indices_anglais = df.index[df["langue_detectee"].isin(["en", "pidgin"])]
# textes_anglais = df.loc[indices_anglais, "texte_nettoye"].tolist()

# resultats_anglais = []
# for document in nlp_anglais.pipe(textes_anglais, batch_size=200):
#     mots_lemmatises = [
#         token.lemma_.strip() for token in document
#         if len(token.lemma_.strip()) > 2 and token.lemma_.strip() not in mots_vides_anglais
#     ]
#     resultats_anglais.append(" ".join(mots_lemmatises))

# df.loc[indices_anglais, "texte_lemmatise"] = resultats_anglais
# print(f"  -> {len(textes_anglais)} textes anglais/pidgin lemmatisés")

# # --- Toutes les autres langues : tokenisation simple, sans lemmatisation ---
# indices_autres = df.index[~df["langue_detectee"].isin(["fr", "en", "pidgin"])]
# for indice in indices_autres:
#     texte = df.at[indice, "texte_nettoye"]
#     mots = [mot for mot in texte.split() if len(mot) > 2]
#     df.at[indice, "texte_lemmatise"] = " ".join(mots)

# print(f"  -> {len(indices_autres)} textes dans d'autres langues (tokenisation simple, sans lemmatisation)")

# print("\nExemple de lemmatisation :")
# exemple_francais = df[df["langue_detectee"] == "fr"].iloc[0]
# print("Texte nettoyé   :", exemple_francais["texte_nettoye"])
# print("Texte lemmatisé :", exemple_francais["texte_lemmatise"])


# # ==============================================================================
# # SECTION E — SAUVEGARDE DU CORPUS NETTOYÉ
# # ==============================================================================
# # On sauvegarde un nouveau fichier CSV qui contient TOUTES les colonnes
# # d'origine, PLUS les nouvelles colonnes qu'on vient de créer :
# #   - texte_nettoye     : le texte nettoyé (avant lemmatisation)
# #   - langue_detectee   : la langue détectée automatiquement
# #   - texte_lemmatise   : le texte final, prêt pour les analyses NLP
# #     (sentiment, thématiques...) des phases suivantes
# #
# # C'est ce fichier qu'on utilisera comme point de départ des Phases 4 à 12.

# print("\n--- E : sauvegarde du corpus nettoyé ---")

# chemin_sortie = f"{dossier_data}/bd_f_nettoye.csv"
# df.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")

# print(f"Corpus nettoyé sauvegardé ici : {chemin_sortie}")
# print(f"Nombre final de publications : {len(df)}")

# # On sauvegarde aussi un petit résumé des langues détectées, utile pour le mémoire
# resume_langues = df["langue_detectee"].value_counts()
# resume_langues.to_csv(f"{dossier_tableaux}/langues_detectees_phase3.csv")


# # ==============================================================================
# # FIN DU SCRIPT
# # ==============================================================================

# print("\n=== TERMINÉ ! ===")
# print("Récapitulatif :")
# print(f"  - Doublons supprimés : {nb_avant_doublons - nb_apres_doublons}")
# print(f"  - Publications vides supprimées après nettoyage : {nb_avant_vide - nb_apres_vide}")
# print(f"  - Publications finales : {len(df)}")
# print(f"  - Fichier prêt pour la Phase 4 : {chemin_sortie}")
