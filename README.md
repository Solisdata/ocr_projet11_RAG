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

### Embeddings : local ou Cloud ?
Le POC a d'abord été écrit avec l'API Mistral, puis basculé sur Ollama. Un index créé avec un modèle d'embedding ne fonctionne pas avec un autre : il faut le reconstruire.

### Base vectorielle

| Solution | Hébergement | Passage à l'échelle | Données locales |
| --- | --- | --- | --- |
| **FAISS (retenu)** | Local, dans l'application Python | Limité par la machine | Oui |
| Weaviate | Local ou Cloud | Adapté à la montée en charge, filtres | Oui si auto-hébergé |
| Milvus | Local ou Cloud, distribué | Très grands volumes | Oui si auto-hébergé |
| Pinecone | Cloud managé | Adapté à la montée en charge | Non |

### Algorithme d'indexation FAISS

| Algorithme | Principe | Compromis |
| --- | --- | --- |
| **IndexFlat (retenu)** | Compare la question à tous les vecteurs. | Résultat exact. Plus lent quand le volume augmente. |
| IndexIVFFlat | Répartit les vecteurs en groupes et n'explore que les plus proches. | Plus rapide, mais peut manquer le meilleur résultat. |
| IndexHNSW | Parcourt un graphe de voisins proches. | Très rapide, mais plus gourmand en mémoire. |

Pour environ 10 000 événements, la recherche exhaustive reste rapide. Elle ne manque jamais le résultat le plus proche.

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
| `tests/test_evaluation.py` | Le jeu de test d'évaluation est complet et cohérent avec les données : chaque `uid` attendu existe dans le CSV et a une séance dans la période de la question, la période détectée correspond à l'annotation. Les notes (Recall, MRR, couverture, similarité, refus) sont vérifiées sur de petits exemples calculés à la main. |

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

### Temps de réponse

L'interface Streamlit affiche sous chaque réponse :

- **Recherche** : vectorisation de la question et recherche FAISS ;
- **Réponse Ollama** : temps de génération du LLM ;
- **Total**.

Exemple mesuré sur la machine de développement, avec `qwen3:1.7b`, pour la question « Un concert ce week-end ? » :

| Étape | Temps |
| --- | --- |
| Recherche (embedding + FAISS) | 3,8 s |
| Génération Ollama | 72,3 s |
| Total | 76,1 s |

La génération représente l'essentiel du temps. Elle dépend du modèle choisi et de la machine (CPU ou GPU). Un modèle plus petit (`qwen3:1.7b`) répond plus vite qu'un modèle plus grand (`qwen3:8b`), mais avec des réponses moins riches.

### Qualité des réponses : évaluation sur un jeu de test annoté

#### Le jeu de test

[`evaluation/jeu_test.json`](evaluation/jeu_test.json) contient **5 questions annotées à la main** à partir du CSV nettoyé. Pour chaque question, on a noté :

| Champ | Rôle |
| --- | --- |
| `question` | La question, écrite comme un utilisateur la poserait. |
| `date_reference` | Le « maintenant » simulé (vendredi 13/11/2026, 10h00). Les réponses dépendent du jour, il faut donc le figer pour que l'évaluation soit reproductible. |
| `periode_attendue` | La période que doit détecter `script/periode.py` (« ce week-end », « demain »... ou aucune). |
| `uids_attendus` | Les événements pertinents (vérité terrain de la recherche). |
| `reponse_attendue` | La réponse de référence, dans le format demandé au LLM. |
| `elements_cles` | Les informations qui doivent figurer dans la réponse : titre, jour, horaire, lieu, prix. |
| `doit_refuser` | `true` si la bonne réponse est « aucun événement ne correspond ». |

Les 5 questions couvrent chacune un cas différent, y compris des cas difficiles :

| Id | Catégorie | Question | Ce qu'elle teste |
| --- | --- | --- | --- |
| q01 | `public` | « Que faire ce week-end avec un enfant de 4 ans ? » | Période + âge du public, 3 événements attendus. |
| q02 | `evenement_precis` | « Comment participer à la Nuit sans fin consacrée à Steve Reich au Conservatoire ? » | Détails pratiques d'un événement (créneaux, réservation, prix). |
| q03 | `periode_et_theme` | « Un spectacle d'improvisation ce soir ? » | Période courte + thème précis, 4 événements attendus. |
| q04 | `periode_et_theme` | « Y a-t-il un concert de Noël le week-end prochain ? » | Période future, un seul bon événement parmi d'autres concerts. |
| q05 | `hors_corpus` | « Quels événements sont prévus à Lyon ce week-end ? » | Le chatbot doit dire qu'il ne couvre que Nantes. |

`tests/test_evaluation.py` vérifie que les annotations restent justes : chaque `uid` attendu existe et a bien une séance dans la période de la question. Si les données sont rechargées depuis l'API, ce test signale les questions à ré-annoter.

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

## Limites du projet

| Limite | Piste d'amélioration |
| --- | --- |
| **Évaluation encore partielle** : 5 questions annotées, une mesure de réponse lexicale (couverture) et une similarité d'embeddings. La fidélité aux sources n'est pas notée directement. | Ajouter un LLM « juge » (RAGAS : *faithfulness*, *answer relevancy*). Enrichir le jeu de test (questions de suivi, fautes de frappe, autres villes). Lancer `--sans-generation` dans la CI pour suivre le Recall@k à chaque modification. |
| **Pas de refus quand rien ne correspond** : le chatbot reçoit toujours *k* événements et en propose même s'ils sont hors sujet (ex. « événements à Lyon »). | Ne pas envoyer au LLM les événements dont le score de similarité est sous un seuil, réglé sur le jeu de test. Ajouter au prompt : « si aucun événement ne correspond, ou si la ville n'est pas Nantes, dis-le ». |
| **Mélange d'événements par le LLM** : `qwen3:1.7b` peut accoler le titre d'un événement au lieu ou au prix d'un autre. | Vérifier après génération que chaque titre cité existe dans le contexte. Tester `qwen3:4b`. |
| **Recherche purement sémantique** : les mots exacts (« impro », sigle « ONPL ») pèsent peu dans l'embedding. | Recherche hybride BM25 + FAISS (`EnsembleRetriever`), ou récupérer 20 candidats et les reclasser avec un cross-encoder. |
| **Dates mal comprises par la recherche** : FAISS cherche par sens, pas par date. Une question comme « ce week-end » peut faire remonter des événements à d'autres dates. | Extraire la période de la question (« ce week-end » → samedi-dimanche), puis filtrer sur les métadonnées `start_date` / `end_date` avant ou après la recherche FAISS. |
| **Couverture limitée** : une seule ville (Nantes) et un plafond de 10 000 événements. | Rendre la ville paramétrable dans l'interface, ajouter `location_city` en filtre de métadonnées et augmenter `API_MAX_RECORDS`. Au-delà de 100 000 événements, passer à un index `IndexHNSW` ou à une base comme Weaviate ou Milvus. |
| **Données figées** : l'index n'est pas mis à jour automatiquement. Le reconstruire oblige à recalculer tous les embeddings. | Planifier la collecte chaque nuit (cron, GitHub Actions). Ne vectoriser que les nouveaux événements et les ajouter avec `add_documents`. Supprimer les événements passés grâce à leur `uid`. |
| **Dépendance au matériel** : les temps de réponse dépendent de la machine. Sans GPU, la génération reste lente (environ 70 s). | Afficher la réponse au fur et à mesure (streaming), réduire `k`, utiliser un GPU ou un serveur dédié, ou une API Cloud européenne (Mistral) pour la production. |
| **Petit modèle de génération** : `qwen3:1.7b` peut mal suivre les consignes ou se tromper sur les dates (ex. un vendredi annoncé comme « mardi »). | Tester `qwen3:4b` / `qwen3:8b`. Donner le jour de la semaine dans le prompt (« jeudi 12/11/2026 ») et dans le texte des événements. |
| **Pas d'historique de conversation** : chaque question est traitée indépendamment des précédentes. | Transmettre les derniers échanges au prompt (`MessagesPlaceholder` de LangChain) et reformuler la question de suivi avant la recherche. |
| **Maintenance** : `langchain-community` est en cours d'abandon (avertissements de dépréciation). `index.pkl` est chargé avec `pickle`. | Migrer vers les paquets dédiés (`langchain-ollama`, intégration FAISS autonome). Ne charger que des index créés par le projet, et enregistrer dans l'index le modèle d'embedding et la date de création. |

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
