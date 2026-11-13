"""Évalue le chatbot sur le jeu de test annoté : python -m script.evaluation

Principe d'un examen corrigé :
1. evaluation/jeu_test.json contient les questions ET leur corrigé ;
2. on pose chaque question au chatbot, à la date simulée de la question ;
3. on compare sa réponse au corrigé et on calcule des notes.

Option : --sans-generation  note seulement la recherche (pas d'appel au LLM, quelques secondes).
"""

import json
import math
import re
import sys
import unicodedata
from datetime import datetime

import pandas as pd

from config import PROJECT_PATH

JEU_TEST_PATH = PROJECT_PATH / "evaluation" / "jeu_test.json"
RESULTATS_PATH = PROJECT_PATH / "evaluation" / "resultats"

# Mots qui montrent que le chatbot dit n'avoir rien trouvé (écrits sans accents).
MOTS_DE_REFUS = ["aucun", "pas trouve", "n'ai pas", "ne dispose pas", "desole", "uniquement"]


def simplifier(texte):
    """Met le texte sous une forme comparable : minuscules, sans accents ni espaces.

    Ainsi "14h00" = "14h", "19 €" = "19€" et "Château" = "chateau".
    """
    texte = unicodedata.normalize("NFKD", str(texte).lower().replace("’", "'"))
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = re.sub(r"(\d)h00", r"\1h", texte)
    return re.sub(r"\s+", "", texte)


# ---------------------------------------------------------------- Notes de la recherche

def recall(uids_trouves, uids_attendus):
    """Part des bons événements retrouvés. Ex. : 1 trouvé sur 4 attendus -> 0,25."""
    if not uids_attendus:
        return None
    return len(set(uids_trouves) & set(uids_attendus)) / len(uids_attendus)


def mrr(uids_trouves, uids_attendus):
    """1 / rang du premier bon événement : 1er -> 1, 2e -> 0,5, absent -> 0."""
    if not uids_attendus:
        return None
    for rang, uid in enumerate(uids_trouves, start=1):
        if uid in uids_attendus:
            return 1 / rang
    return 0.0


# ---------------------------------------------------------------- Notes de la réponse

def couverture(reponse, elements_cles):
    """Part des informations indispensables présentes dans la réponse. Ex. : 1 sur 4 -> 0,25."""
    if not elements_cles:
        return None
    reponse = simplifier(reponse)
    return sum(simplifier(element) in reponse for element in elements_cles) / len(elements_cles)


def similarite(vecteur_a, vecteur_b):
    """Proximité de sens entre deux textes vectorisés (cosinus : 1 = même sens)."""
    produit = sum(a * b for a, b in zip(vecteur_a, vecteur_b))
    normes = math.hypot(*vecteur_a) * math.hypot(*vecteur_b)
    return produit / normes if normes else 0.0


def a_refuse(reponse):
    """Vrai si la réponse dit qu'aucun événement ne correspond."""
    texte = simplifier(reponse)
    return any(simplifier(mot) in texte for mot in MOTS_DE_REFUS)


# ---------------------------------------------------------------- L'examen

def charger_jeu_test():
    with open(JEU_TEST_PATH, encoding="utf-8") as fichier:
        return json.load(fichier)["questions"]


def noter(chatbot, embeddings, item, avec_generation=True):
    """Pose une question au chatbot et renvoie ses notes."""
    maintenant = datetime.fromisoformat(item["date_reference"])
    attendus = item["uids_attendus"]

    if avec_generation:
        reponse, documents, _ = chatbot.ask(item["question"], maintenant=maintenant)
    else:
        _, documents, _ = chatbot.retrieve(item["question"], maintenant=maintenant)
        reponse = ""
    trouves = [str(doc.metadata.get("uid", "")) for doc in documents]

    notes = {
        "id": item["id"],
        "question": item["question"],
        "recall": recall(trouves, attendus),
        "mrr": mrr(trouves, attendus),
    }
    if avec_generation:
        vecteurs = embeddings.embed_documents([reponse, item["reponse_attendue"]])
        notes.update(
            {
                "couverture": couverture(reponse, item["elements_cles"]),
                "similarite": similarite(*vecteurs),
                "refus_correct": a_refuse(reponse) == item["doit_refuser"],
                "reponse": reponse,
                "reponse_attendue": item["reponse_attendue"],
            }
        )
    notes["uids_trouves"] = " ".join(trouves)
    notes["uids_attendus"] = " ".join(attendus)
    return notes


def main():
    avec_generation = "--sans-generation" not in sys.argv

    # Imports ici : ils chargent LangChain et se connectent à Ollama.
    from script.chatbot import EventChatbot
    from script.embeddings import create_embeddings

    chatbot = EventChatbot()
    embeddings = create_embeddings()

    resultats = []
    for item in charger_jeu_test():
        print(f"{item['id']} - {item['question']}")
        resultats.append(noter(chatbot, embeddings, item, avec_generation))

    tableau = pd.DataFrame(resultats)
    RESULTATS_PATH.mkdir(parents=True, exist_ok=True)
    chemin = RESULTATS_PATH / f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    tableau.to_csv(chemin, index=False, encoding="utf-8-sig")

    colonnes = [c for c in ["recall", "mrr", "couverture", "similarite", "refus_correct"] if c in tableau]
    notes = tableau.set_index("id")[colonnes].astype(float)
    notes.loc["moyenne"] = notes.mean()
    print("\n" + notes.round(2).to_string())
    print(f"\nDétail (réponses, événements trouvés) : {chemin}")


if __name__ == "__main__":
    main()
