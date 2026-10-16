
# lancer les tests : python -m pytest tests/test_preprocessing.py -v

import logging

import pandas as pd
import pytest

from script.preprocessing import preprocess_data


def test_preprocess_data_cleans_text_and_logs_removed_events(caplog):
    """Nettoie les champs texte et écarte les événements non valides."""
    caplog.set_level(logging.INFO)
    data = pd.DataFrame(
        {
            "uid": [101, 101, 102, 103, None, 104],
            "title_fr": [
                "  Concert   à Nantes ",
                "Concert à Nantes",
                "Événement sans date",
                "   ",
                "Sans identifiant",
                "Dates inversées",
            ],
            "description_fr": ["  Une   description\nutile ", None, "", "", "", ""],
            "location_city": [
                " Nantes ",
                "Nantes",
                "Nantes",
                "Nantes",
                "Nantes",
                "Nantes",
            ],
            "firstdate_begin": [
                "2026-10-16T10:00:00+00:00",
                "2026-10-16T10:00:00+00:00",
                "date invalide",
                "2026-10-16T10:00:00+00:00",
                "2026-10-16T10:00:00+00:00",
                "2026-10-16T12:00:00+00:00",
            ],
            "lastdate_end": [
                "2026-10-16T12:00:00+00:00",
                "2026-10-16T12:00:00+00:00",
                "2026-10-16T12:00:00+00:00",
                "2026-10-16T12:00:00+00:00",
                "2026-10-16T12:00:00+00:00",
                "2026-10-16T10:00:00+00:00",
            ],
        }
    )

    result = preprocess_data(data)

    assert len(result) == 1
    assert result.loc[0, "uid"] == "101"
    assert result.loc[0, "title_fr"] == "Concert à Nantes"
    assert result.loc[0, "description_fr"] == "Une description utile"
    assert result.loc[0, "location_city"] == "Nantes"
    assert "1 doublon(s) trouvé(s) et supprimé(s)." in caplog.text
    assert "Prétraitement terminé : 1 événement(s) conservé(s) sur 6." in caplog.text


def test_preprocess_data_rejects_missing_required_columns():
    """Signale explicitement les colonnes requises absentes."""
    with pytest.raises(ValueError, match="uid"):
        preprocess_data(pd.DataFrame({"title_fr": ["Concert"]}))
