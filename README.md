# Projet 11 — Recherche d’événements à Nantes

Ce projet récupère des événements depuis OpenAgenda, nettoie les données et crée un index vectoriel pour pouvoir les rechercher par sens.

## Fonctionnement

Le pipeline suit trois étapes :

1. `script/api_opendatasoft.py` récupère les événements et les enregistre dans `data/evenements_nantes.csv`.
2. `script/preprocessing.py` nettoie les champs, normalise les âges, crée le libellé `age_range` et filtre les événements invalides, trop anciens ou hors de Nantes. Le résultat est enregistré dans `data/evenements_nantes_clean.csv`.
3. `script/embeddings.py` transforme chaque ligne du CSV nettoyé en un document, puis crée un embedding avec Mistral et l’enregistre dans un index FAISS sous `data/faiss_index/`.

Chaque ligne du CSV représente un événement et devient un document. Les métadonnées du document gardent notamment l’identifiant, le titre, les dates, l’âge et l’URL de l’événement.

## Lancer le projet

Créer un environnement virtuel et installer les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Ajouter la clé Mistral dans un fichier `.env` à la racine du projet :

```text
MISTRAL_API_KEY=ta_cle_api
```

Lancer toutes les étapes :

```powershell
python -m script.run_pipeline
```

Pour exécuter une étape seule :

```powershell
python -m script.api_opendatasoft
python -m script.preprocessing
python -m script.embeddings
```

Pour inspecter l'index FAISS local (nombre de vecteurs, dimension, métadonnées et
contenu de quelques documents), sans appeler l'API Mistral :

```powershell
python -m script.read_index
python -m script.read_index --limit 10
```

Le script lit `data/faiss_index/` par défaut. `index.pkl` est chargé avec
`pickle` par LangChain : n'inspecte que les index créés par ce projet ou dont
la provenance est fiable.

## Chatbot de recommandations

Après avoir créé l'index et configuré `MISTRAL_API_KEY`, lancer le chatbot :

```powershell
python -m script.chatbot
```

Le chatbot récupère les cinq événements les plus proches dans FAISS, puis demande
à Mistral de répondre à partir de leur contenu. Il affiche les événements consultés
avec leurs liens disponibles. La conversation ne conserve pas d'historique. Pour
changer le nombre de résultats, utiliser `--k 8`. Le modèle peut être choisi avec
`--model nom-du-modele` ou la variable `MISTRAL_CHAT_MODEL` (défaut :
`mistral-small-latest`). Chaque question déclenche des appels Mistral pour la
recherche vectorielle et la génération de la réponse.

### Interface web Streamlit

Lancer l'interface depuis la racine du projet :

```powershell
python -m streamlit run app.py
```

L'interface affiche les réponses avec les liens des événements consultés. Elle
garde les échanges uniquement pendant la session Streamlit en cours. Le nombre
d'événements utilisés pour la recherche se règle dans la barre latérale. Après
l'ajout de Streamlit dans `requirements.txt`, installer la dépendance avec :

```powershell
python -m pip install -r requirements.txt
```

La création de l’index appelle l’API Mistral. Relancer cette étape recrée l’index et consomme des appels d’API.

## Tests

Lancer tous les tests :

```powershell
python -m pytest tests/ -v
```

`tests/test_embeddings.py` contient des tests locaux : un document est créé par ligne CSV et FAISS crée bien un index. Le test FAISS utilise un faux encodeur : il ne contacte pas Mistral.

Les tests réels de Mistral sont désactivés par défaut. Ils vérifient qu’une requête retourne un vecteur numérique et que deux textes donnent des vecteurs de même dimension. Pour les activer dans PowerShell :

```powershell
$env:RUN_MISTRAL_INTEGRATION = "1"
python -m pytest tests/test_embeddings.py -v
```

Ces tests nécessitent `MISTRAL_API_KEY` et font de vrais appels à l’API. Ils sont donc séparés des tests courants et de la CI.
