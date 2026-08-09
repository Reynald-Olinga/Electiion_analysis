# Plateforme d'aide à la décision — Analyse des sentiments et des thématiques pour l'anticipation des crises post-électorales au Cameroun

Application web interactive développée dans le cadre du mémoire de Master 2 :
**« Aide à la décision et analyse des sentiments pour les crises post-électorales au Cameroun »**

Rédigé par **ASSA ALA OLINGA Vincent Reynald**, sous la supervision du **Dr NYAWA Serge** — Année académique 2025-2026.

---

## 1. Présentation du projet

Cette plateforme restitue, sous forme de tableau de bord interactif, les résultats de l'analyse d'un corpus de **48 138 publications** collectées sur Facebook, TikTok et X (Twitter) entre le 1er juillet et le 30 octobre 2025, autour de l'élection présidentielle camerounaise d'octobre 2025.

Elle traduit opérationnellement les deux indicateurs synthétiques construits dans le mémoire :

- **ITE — Indice de Tension Émotionnelle** : mesure la charge émotionnelle négative (colère, peur, dégoût, tristesse) du corpus, jour après jour.
- **ITR — Indice Thématique de Risque** : mesure le poids des thématiques à risque (tribalisme, contestation institutionnelle, discours autour du pouvoir) dans les publications quotidiennes.

Ces deux indices, ainsi que 6 indicateurs complémentaires (Activité, Engagement, Viralité, Polarisation, Mobilisation, Désinformation-proxy), sont agrégés en un **Score de Risque Global** qui alimente un système d'alerte à quatre niveaux (Vert / Orange / Rouge / Critique), destiné à des acteurs institutionnels tels que les préfectures, le MINAT, ELECAM ou des observatoires de la société civile.

> ⚠️ **Avertissement méthodologique** : cette plateforme est un prototype académique de recherche. Le Score de Risque Global et les seuils d'alerte sont calibrés de façon relative sur la fenêtre de données disponible (voir Partie 3 du mémoire, section Limites). Elle ne doit pas être utilisée comme unique fondement à une décision opérationnelle réelle sans validation et recalibrage par des experts du terrain.

---

## 2. Fonctionnalités — les 8 pages de l'application

| Page | Contenu |
|---|---|
| **Accueil** | Synthèse générale du corpus, chiffres-clés, navigation |
| **Alertes** | Niveau d'alerte du jour, calendrier des alertes, épisodes détectés, causes du déclenchement |
| **Sentiments** | Distribution positif/neutre/négatif, évolution temporelle, comparaison par plateforme |
| **Émotions** | Profil émotionnel (8 émotions de Plutchik), radar comparatif, évolution hebdomadaire |
| **Thématiques** | Répartition des thèmes issus du topic modeling (LDA), évolution, croisement avec les acteurs |
| **Indicateurs décisionnels** | ITE, ITR, Score de Risque Global et les 6 indicateurs complémentaires, sous forme de courbes et de jauges |
| **Recherche** | Recherche libre dans le corpus (mots-clés, acteurs, période, plateforme) |
| **Rapports** | Génération et export de rapports au format PDF (fpdf2) et Excel (openpyxl) |

---

## 3. Architecture technique

```
├── app.py                        # Point d'entrée principal (page Accueil)
├── pages/                        # Les 7 autres pages (détectées automatiquement par Streamlit)
│   ├── 1_Alertes.py
│   ├── 2_Sentiments.py
│   ├── 3_Emotions.py
│   ├── 4_Thematiques.py
│   ├── 5_Indicateurs_decisionnels.py
│   ├── 6_Recherche.py
│   └── 7_Rapports.py
├── chargement_donnees.py         # Module central d'accès aux données (mise en cache @st.cache_data)
├── data/                         # Fichiers CSV pré-calculés (résultats des scripts 01 à 09)
│   ├── INDICATEURS_tableau_complet_quotidien.csv
│   ├── B1_alertes_simples_quotidiennes.csv
│   ├── bd_f_themes.csv
│   └── ...
├── requirements.txt              # Dépendances Python de la couche application
└── README.md
```

### Principe de séparation pipeline / application

La plateforme **ne recalcule rien** : elle lit exclusivement des fichiers CSV pré-calculés, produits localement par les scripts du pipeline NLP (`01_exploration_eda.py` à `09_detection_alertes.py`). Cette séparation stricte est délibérée :

- Le **pipeline** (scripts 01-09) tourne uniquement en local ou sur une machine de traitement, avec des dépendances lourdes (spaCy, NLTK, scikit-learn, gensim/LDA, VADER, NRCLex).
- L'**application déployée** ne dépend que de `pandas`, `plotly`, `streamlit`, `fpdf2` et `openpyxl` — ce qui garde le déploiement léger, rapide et gratuit sur Streamlit Community Cloud.

Toute mise à jour du corpus nécessite donc de :
1. Relancer le pipeline en local (`python 01_exploration_eda.py`, puis `02_pretraitement_nlp.py`, etc., dans l'ordre).
2. Copier les CSV de résultats mis à jour dans le dossier `data/` du dépôt.
3. Pousser (`git push`) les nouveaux fichiers — Streamlit Cloud redéploie automatiquement.

---

## 4. Installation en local

### Prérequis
- Python 3.10 ou supérieur
- pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-utilisateur>/<nom-du-depot>.git
cd <nom-du-depot>

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # sous Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse `http://localhost:8501`.

---

## 5. Déploiement sur Streamlit Community Cloud

Voir la section dédiée plus bas dans ce document (§ 7) pour la procédure complète, ou directement [share.streamlit.io](https://share.streamlit.io).

En résumé :
1. Le dépôt doit être public (ou accessible via votre compte GitHub connecté).
2. `requirements.txt` doit être à la racine du dépôt.
3. Le fichier d'entrée (`app.py`) doit être précisé lors de la création de l'app sur Streamlit Cloud.
4. Chaque `git push` sur la branche déployée déclenche un redéploiement automatique.

---

## 6. Données et confidentialité

- Les données collectées sont exclusivement des **contenus publiquement accessibles** au moment de la collecte (commentaires et publications publiques), conformément à la loi camerounaise n°2010/012 du 21 décembre 2010 relative à la cybercriminalité.
- Les identifiants d'auteurs affichés dans les extraits (verbatims) sont **anonymisés ou pseudonymisés** avant intégration dans les fichiers `data/`.
- Aucune donnée à caractère privé (messages privés, informations de contact) n'est collectée ni exposée par la plateforme.

---

## 7. Étapes complètes de déploiement sur Streamlit Community Cloud

1. **Préparer le dépôt GitHub** : structure `app.py` + `pages/` + `data/` + `requirements.txt`, tous poussés sur la branche `main`.
2. **Vérifier les chemins de fichiers** dans `chargement_donnees.py` : utiliser des chemins relatifs à la racine du dépôt (ex. `data/bd_f_themes.csv`), jamais de chemins absolus type `C:\Users\...` ou `../data/...` hérités du poste local.
3. **Vérifier la taille des fichiers** : GitHub refuse au-delà de 100 Mo par fichier ; au-delà de 50 Mo, utiliser Git LFS.
4. Se rendre sur **[share.streamlit.io](https://share.streamlit.io)** et se connecter avec le compte GitHub.
5. Cliquer sur **« New app »**, sélectionner le dépôt, la branche `main`, et indiquer `app.py` comme fichier principal.
6. Cliquer sur **« Deploy »** — l'installation des dépendances et le premier lancement prennent 2 à 5 minutes.
7. Une fois déployée, l'application est accessible à une URL du type `https://<nom-app>.streamlit.app`. Vérifier chaque page (chargement des CSV, graphiques, exports PDF/Excel).
8. En cas d'erreur, consulter les logs via le bouton **« Manage app »** dans le tableau de bord Streamlit Cloud.
9. *(Optionnel)* Créer une page GitHub Pages statique (`index.html`) qui redirige ou pointe vers l'URL de l'app déployée, pour disposer d'une vitrine de présentation du projet.

---

## 8. Auteur et encadrement

- **Auteur** : ASSA ALA OLINGA Vincent Reynald
- **Superviseur** : Dr NYAWA Serge
- **Établissement / niveau** : Master 2 (mémoire de recherche)
- **Année académique** : 2025-2026

## 9. Licence

Projet académique — usage à des fins de recherche et d'évaluation universitaire. Toute réutilisation des données ou du code à des fins autres qu'académiques doit faire l'objet d'une autorisation préalable de l'auteur.
#   E l e c t i i o n _ a n a l y s i s  
 #   E l e c t i i o n _ a n a l y s i s  
 