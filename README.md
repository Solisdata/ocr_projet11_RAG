# OCR - Projet 11 : Concevez et déployez un système RAG

*Novembre 2026*

## Contexte du projet

### Entreprise

L'entreprise **Puls-Events** propose une plateforme web qui permet aux utilisateurs de découvrir et de suivre des événements culturels en temps réel.

### Projet

| Avant | Maintenant |
| --- | --- |
| L'utilisateur trouve les événements à l'aide de filtres (lieu, période). | L'utilisateur reçoit des recommandations personnalisées via un **chatbot**, soutenu par un **RAG** (*Retrieval-Augmented Generation*). |

## Objectifs

En tant que **Data Engineer**, réaliser un **POC** (preuve de concept) qui exploite des documents pour répondre à des questions en langage naturel.

**Comment faire chercher une IA dans des documents internes ?**
Un LLM seul ne connaît pas les événements de Puls-Events et peut inventer des réponses. Le RAG procède en deux temps :

1. **Retrieval (recherche)** : on retrouve les événements les plus proches de la question dans une base de données vectorielle.
2. **Generation (génération)** : on transmet ces événements au LLM comme contexte, et il rédige une réponse fondée uniquement sur eux.

**Base de données vectorielle** : une base qui stocke chaque document sous la forme d'un vecteur (une liste de nombres qui représente son sens). Deux textes au sens proche ont des vecteurs proches. On peut donc rechercher par **similarité sémantique** plutôt que par mots-clés : « un spectacle pour enfants ce week-end » retrouve un événement « Marionnettes en famille, samedi », même sans mot commun.

Le POC couvre les événements de **Nantes**, récupérés depuis **OpenAgenda** via l'API OpenDataSoft.

## Structure du projet

```text
projet_11/
├── .github/
│   └── workflows/
│       └── ci.yml                  # Intégration continue : lance pytest à chaque push / PR
├── data/
│   ├── evenements_nantes.csv       # Événements bruts récupérés depuis l'API
│   ├── evenements_nantes_clean.csv # Événements nettoyés (entrée de l'indexation)
│   └── faiss_index/                # Index vectoriel utilisé par le chatbot
│       ├── index.faiss             #   vecteurs
│       └── index.pkl               #   textes et métadonnées des documents
├── docs/
│   ├── architecture_RAG.png        # Schéma d'architecture : briques et outils du RAG
│   ├── logigramme_RAG.png          # Logigramme : enchaînement des étapes et des scripts
│   ├── ocr_projet11_RAG_v6.pptx    # Présentation du POC
│   └── Rapport_technique_du_POC_RAG_MLS.pdf  # Rapport technique du POC
├── script/
│   ├── api_opendatasoft.py         # 1. Collecte paginée des événements
│   ├── preprocessing.py            # 2. Nettoyage, normalisation, filtrage, dédoublonnage
│   ├── embeddings.py               # 3. Création des documents, vectorisation, index FAISS
│   ├── run_pipeline.py             # Enchaîne les étapes 1 → 2 → 3
│   ├── chatbot.py                  # Chaîne RAG : recherche FAISS + génération Ollama (terminal)
│   ├── periode.py                  # Détection de la période (« ce week-end »...) et filtre par date
│   ├── evaluation.py               # Évaluation du RAG sur le jeu de test annoté
│   └── read_index.py               # Inspection de l'index (nombre de vecteurs, dimension, exemples)
├── evaluation/
│   ├── jeu_test.json               # Jeu de test annoté : questions, événements et réponses attendus
│   └── resultats/                  # Résultats de chaque évaluation (un CSV par lancement)
├── tests/
│   ├── test_data.py                # Qualité du CSV nettoyé
│   ├── test_preprocessing.py       # Règles de prétraitement
│   ├── test_embeddings.py          # Création des documents et de l'index FAISS
│   ├── test_periode.py             # Détection des périodes et filtre par date
│   └── test_evaluation.py          # Cohérence du jeu de test et calcul des métriques
├── app.py                          # Interface Streamlit du chatbot
├── config.py                       # Chemins, ville, période, paramètres API et modèles
├── check_env.py                    # Vérifie que les dépendances s'importent correctement
├── requirements.txt                # Dépendances Python
├── .env                            # Variables locales (non versionné)
└── README.md
```

## Architecture du RAG

Les briques du système et l'outil qui porte chacune : Streamlit pour l'interface, LangChain pour la chaîne, Ollama pour les embeddings et le LLM, FAISS pour l'index vectoriel.

![Architecture du RAG](docs/architecture_RAG.png)

## Logigramme

L'enchaînement des étapes, du script de collecte jusqu'à l'interface, avec les tests associés à chaque étape.

![Logigramme du RAG](docs/logigramme_RAG.png)



## Choix technologiques

| Besoin | Choix | Pourquoi |
| --- | --- | --- |
| Source de données | **API OpenDataSoft** (jeu `evenements-publics-openagenda`) | Données publiques, paginées, filtrables par ville et par date. |
| Format intermédiaire | **CSV** | Simple à inspecter et à réutiliser entre les étapes. Le brut reste séparé du nettoyé. |
| Transformations | **Pandas** | Validation, normalisation, filtrage et dédoublonnage de données tabulaires. |
| Découpage | **1 événement = 1 document** | Les informations d'un événement restent ensemble. Les métadonnées rattachent chaque résultat à sa source. |
| Embeddings | **Ollama + `embeddinggemma`** (dimension 768) | Tourne en local : aucune donnée envoyée à l'extérieur, aucun coût au token. |
| LLM de génération | **Ollama + `qwen3:1.7b`** (par défaut) | Petit modèle local, rapide sur CPU, bon en français. Configurable (`qwen3:4b`, `qwen3:8b`…). |
| Base vectorielle | **FAISS** (`faiss-cpu`) | Bibliothèque locale, sans service à déployer. Suffisante pour environ 10 000 événements. |
| Orchestration | **LangChain** (`langchain-community`, `langchain-ollama`) | Chargement de l'index, retriever, prompt, chaîne `prompt | llm | parser`. |
| Interface | **Streamlit** | Chat web en quelques lignes, avec réglage du modèle et affichage des temps de réponse. |
| Qualité | **pytest** + **GitHub Actions** | Tests à chaque push et pull request sur `main`. |

### Prompt du chatbot

Le prompt système impose au modèle de :

- répondre en français à partir **uniquement** des événements fournis ;
- interpréter les dates relatives (« aujourd'hui », « ce week-end ») à partir de la **date du jour** (fuseau Europe/Paris), injectée à chaque question ;
- ne pas recommander d'événement passé ou incompatible avec la période demandée ;
- dire qu'il ne sait pas plutôt que d'inventer.

La température est fixée à 0,2 pour limiter les réponses fantaisistes.

## Tests

```powershell
python -m pytest tests/ -v
```

| Fichier | Ce qui est vérifié |
| --- | --- |
| `tests/test_data.py` | Le CSV nettoyé contient les colonnes requises, des titres, des dates valides, des événements dans la période attendue et dans la bonne ville, et des `uid` présents et uniques. |
| `tests/test_preprocessing.py` | Le nettoyage du texte, la journalisation des événements écartés et le rejet d'un fichier sans les colonnes requises. |
| `tests/test_embeddings.py` | Un document est créé par ligne du CSV. L'index FAISS est créé avec un faux modèle d'embedding, donc sans Ollama. |
| `tests/test_embeddings.py` (intégration) | Ollama renvoie des vecteurs de dimension constante. Ces tests sont ignorés par défaut. |
| `tests/test_periode.py` | Les expressions « ce week-end », « demain », « week-end prochain » donnent les bons intervalles. Un événement n'est gardé que s'il a une séance dans la période. |
| `tests/test_evaluation.py` | Le jeu de test d'évaluation est complet et cohérent avec les données : chaque `uid` attendu existe dans le CSV et a une séance dans la période de la question, la période détectée correspond à l'annotation. Les notes (Recall, MRR, couverture, similarité, refus) sont vérifiées. |

Pour lancer les tests d'intégration avec Ollama démarré et `embeddinggemma` téléchargé :

```powershell
$env:RUN_OLLAMA_INTEGRATION = "1"
python -m pytest tests/test_embeddings.py -v
```

Les tests tournent aussi automatiquement sur GitHub Actions (Ubuntu, Python 3.13) à chaque push et pull request vers `main`. Un test qui échoue bloque la fusion.

## Indicateurs de performances

### Données et index

| Indicateur | Valeur |
| --- | --- |
| Événements récupérés depuis l'API | 10 000 (plafond configuré dans `config.py`) |
| Événements conservés après prétraitement | 9 998 |
| Doublons et dates invalides restants | 0 |
| Vecteurs dans l'index FAISS Ollama | 9 998 |
| Dimension des vecteurs | 768 (`embeddinggemma`) |

Pour vérifier ces valeurs sur votre index :

```powershell
python -m script.read_index
```

### Qualité des réponses : évaluation sur un jeu de test annoté

#### Les indicateurs

| Étape | Indicateur | Ce qu'il mesure |
| --- | --- | --- |
| Recherche | **Recall@k** | Part des événements attendus retrouvés parmi les *k* résultats. |
| Recherche | **MRR** | 1 / rang du premier bon événement (1 = trouvé en tête). |
| Réponse | **Couverture des éléments clés** | Part des éléments clés présents dans la réponse (insensible aux accents, « 14h00 » = « 14h », « 19 € » = « 19 euros »). |
| Réponse | **Similarité sémantique** | Cosinus entre les embeddings (`embeddinggemma`) de la réponse et de la référence. |
| Réponse | **Refus correct** | Le chatbot dit « aucun événement » quand c'est attendu, et seulement dans ce cas. |

#### Lancer l'évaluation

```powershell
python -m script.evaluation                    # recherche + génération (environ 1 à 2 min sur CPU)
python -m script.evaluation --sans-generation  # recherche seule (quelques secondes)
```

Les résultats sont enregistrés dans `evaluation/resultats/` : un CSV par lancement, avec pour chaque question la réponse générée, les événements trouvés et les notes. Les notes et leur moyenne s'affichent aussi dans le terminal.

#### Résultats (13/11/2026, `qwen3:1.7b`, k = 5)

| Question | Recall@5 | MRR | Couverture | Similarité | Refus correct |
| --- | --- | --- | --- | --- | --- |
| q01 enfant de 4 ans | 1,00 | 1,00 | 0,80 | 0,86 | oui |
| q02 Nuit sans fin | 1,00 | 1,00 | 1,00 | 0,85 | oui |
| q03 impro ce soir | **0,25** | 0,25 | 0,67 | 0,77 | oui |
| q04 concert de Noël | 1,00 | 1,00 | **0,25** | 0,68 | oui |
| q05 Lyon | – | – | – | 0,53 | **non** |
| **Global** | **0,81** | **0,81** | **0,68** | **0,74** | **4 / 5** |

Temps moyens : 2,2 s pour la recherche, 14,1 s pour la génération.

**Lecture**

- **La recherche fonctionne quand le sens suffit** (q01, q02, q04 : tous les événements attendus sont retrouvés, le premier est le bon). Elle échoue quand la question repose sur un mot précis : pour « improvisation » (q03), elle ne retrouve qu'un spectacle sur quatre et ramène des spectacles en général.
- **Le LLM peut mélanger les événements** (q04) : il invente un « Concert de Noël » au Warehouse, en accolant le titre d'un événement au lieu et au prix d'un autre, avec un jour erroné (« samedi 20/11 »). La couverture (0,25) signale cette erreur, alors que la recherche avait trouvé le bon concert.
- **Le chatbot ne sait pas dire « je n'ai rien »** (q05) : il reçoit toujours 5 événements et propose des événements nantais pour une question sur Lyon.
- **Limite de la mesure** : 5 questions donnent une première tendance, pas une mesure stable. La couverture est une mesure lexicale : elle ne détecte pas toutes les hallucinations, d'où l'intérêt d'un LLM juge (voir les limites).

## Piste d'amélioration
Récupérer les données supplémentaires sans avoir à relancer le modèle d’embedding pour l’ensemble des données
Rendre le POC scalable en adaptant les outils utilisés
Ajuster le modèle pour maximisés la qualité des réponses
Évaluer les réponses avec RAGAS sur des questions de reference



## Configuration de l'environnement

### 1. Environnement Python

Python 3.13 recommandé (version utilisée par la CI).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_env.py
```

`check_env.py` vérifie que toutes les bibliothèques s'importent correctement.

### 2. Ollama

Installer Ollama pour Windows depuis https://ollama.com/download, puis télécharger les modèles :

```powershell
ollama pull embeddinggemma
ollama pull qwen3:1.7b
```

`embeddinggemma` vectorise les événements et les questions. `qwen3:1.7b` rédige les réponses.

### 3. Fichier `.env` (facultatif)

Les valeurs par défaut sont définies dans `config.py`. Pour les remplacer, créer un fichier `.env` à la racine :

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=embeddinggemma
OLLAMA_CHAT_MODEL=qwen3:1.7b
GEO_CITY=Nantes
GEO_REGION=Pays de la Loire
```

### 4. Construire l'index

Pipeline complet (collecte, prétraitement, indexation) :

```powershell
python -m script.run_pipeline
```

Ou étape par étape :

```powershell
python -m script.api_opendatasoft
python -m script.preprocessing
python -m script.embeddings
```

L'index est enregistré dans `data/faiss_index/`. Il est fourni dans le dépôt : le chatbot peut être lancé sans le reconstruire. Après un changement de modèle d'embedding, il faut le reconstruire.

Pour inspecter l'index sans contacter Ollama :

```powershell
python -m script.read_index --limit 10
```

### 5. Lancer le chatbot

```powershell
# Dans le terminal
python -m script.chatbot

# Dans le navigateur
python -m streamlit run app.py
```

La barre latérale de Streamlit permet de changer le modèle de chat.
