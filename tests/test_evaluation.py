"""Tests de l'évaluation : le corrigé est-il juste ? les notes sont-elles bien calculées ?

Ces tests n'appellent pas Ollama : ils tournent aussi sur GitHub Actions.
"""

from datetime import datetime

import pandas as pd

from config import CLEANED_DATA_PATH
from script.evaluation import a_refuse, charger_jeu_test, couverture, mrr, recall, similarite
from script.periode import FUSEAU, chevauche, detecter_periode

QUESTIONS = charger_jeu_test()


# ---------------------------------------------------------------- Le corrigé est-il juste ?

def test_chaque_question_a_un_corrige_complet():
    for q in QUESTIONS:
        assert q["question"] and q["reponse_attendue"], q["id"]
        # Soit on attend des événements, soit on attend un refus.
        assert bool(q["uids_attendus"]) != q["doit_refuser"], q["id"]


def test_la_periode_comprise_par_le_code_est_celle_du_corrige():
    for q in QUESTIONS:
        periode = detecter_periode(q["question"], datetime.fromisoformat(q["date_reference"]))
        assert (periode[0] if periode else None) == q["periode_attendue"], q["id"]


def test_les_evenements_attendus_existent_et_ont_lieu_pendant_la_periode():
    """Si les données sont rechargées depuis l'API, ce test signale les questions à ré-annoter."""
    evenements = pd.read_csv(CLEANED_DATA_PATH, encoding="utf-8-sig", dtype=str).fillna("").set_index("uid")

    for q in QUESTIONS:
        maintenant = datetime.fromisoformat(q["date_reference"])
        periode = detecter_periode(q["question"], maintenant)
        debut, fin = (periode[1], periode[2]) if periode else (maintenant, datetime.max.replace(tzinfo=FUSEAU))

        for uid in q["uids_attendus"]:
            assert uid in evenements.index, f"{q['id']} : l'événement {uid} n'existe plus"
            event = evenements.loc[uid]
            seances = {"timings": event["timings"], "start_date": event["firstdate_begin"], "end_date": event["lastdate_end"]}
            assert chevauche(seances, debut, fin), f"{q['id']} : {uid} n'a pas lieu pendant la période"


# ---------------------------------------------------------------- Les notes sont-elles bien calculées ?

def test_notes_de_la_recherche():
    trouves = ["a", "b", "c", "d", "e"]
    assert recall(trouves, ["a", "z"]) == 0.5  # 1 bon événement sur 2
    assert mrr(trouves, ["c"]) == 1 / 3  # le bon événement est 3e
    assert mrr(trouves, ["z"]) == 0  # le bon événement est absent


def test_notes_de_la_reponse():
    reponse = "Samedi 14/11 de 14h00 à 16h30 au Château des ducs, 19 €."
    assert couverture(reponse, ["14h", "19€", "chateau", "Beaujoire"]) == 0.75  # 3 infos sur 4
    assert similarite([1, 0], [1, 0]) == 1  # même sens
    assert a_refuse("Désolé, je n'ai trouvé aucun événement.")
    assert not a_refuse("Rodrigue, samedi 14/11 à 20h00, 29 €.")
