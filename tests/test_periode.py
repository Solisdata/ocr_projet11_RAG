from datetime import datetime

from script.periode import FUSEAU, chevauche, detecter_periode

VENDREDI = datetime(2026, 11, 13, 15, 0, tzinfo=FUSEAU)
DIMANCHE = datetime(2026, 11, 15, 10, 0, tzinfo=FUSEAU)


def test_ce_week_end_depuis_un_vendredi():
    nom, debut, fin, _ = detecter_periode("Que faire ce week-end ?", VENDREDI)
    assert nom == "ce week-end"
    assert debut == datetime(2026, 11, 14, tzinfo=FUSEAU)
    assert fin == datetime(2026, 11, 16, tzinfo=FUSEAU)


def test_ce_weekend_le_dimanche_garde_le_week_end_en_cours():
    _, debut, fin, _ = detecter_periode("un concert ce weekend", DIMANCHE)
    assert debut == DIMANCHE
    assert fin == datetime(2026, 11, 16, tzinfo=FUSEAU)


def test_week_end_prochain():
    nom, debut, _, _ = detecter_periode("le week-end prochain", VENDREDI)
    assert nom == "week-end prochain"
    assert debut == datetime(2026, 11, 21, tzinfo=FUSEAU)


def test_demain_et_apres_demain():
    assert detecter_periode("demain soir", VENDREDI)[1] == datetime(2026, 11, 14, tzinfo=FUSEAU)
    assert detecter_periode("après-demain", VENDREDI)[1] == datetime(2026, 11, 15, tzinfo=FUSEAU)


def test_sans_periode():
    assert detecter_periode("un atelier pour enfants", VENDREDI) is None


def test_chevauche_un_evenement_qui_dure_plusieurs_jours():
    debut, fin = datetime(2026, 11, 14, tzinfo=FUSEAU), datetime(2026, 11, 16, tzinfo=FUSEAU)
    expo = {"start_date": "2026-10-01T08:00:00+00:00", "end_date": "2026-12-31T17:00:00+00:00"}
    passe = {"start_date": "2025-11-13T13:30:00+00:00", "end_date": "2025-11-13T15:00:00+00:00"}
    assert chevauche(expo, debut, fin)
    assert not chevauche(passe, debut, fin)
    assert not chevauche({"start_date": "", "end_date": ""}, debut, fin)


def test_atelier_recurrent_hors_week_end_est_ecarte():
    """Un atelier du vendredi, de septembre à décembre, n'a pas lieu le week-end."""
    debut, fin = datetime(2026, 11, 14, tzinfo=FUSEAU), datetime(2026, 11, 16, tzinfo=FUSEAU)
    atelier = {
        "start_date": "2026-09-04T12:00:00+00:00",
        "end_date": "2026-12-11T16:00:00+00:00",
        "timings": '[{"begin": "2026-11-13T13:00:00+01:00", "end": "2026-11-13T17:00:00+01:00"},'
        ' {"begin": "2026-11-20T13:00:00+01:00", "end": "2026-11-20T17:00:00+01:00"}]',
    }
    assert not chevauche(atelier, debut, fin)
    atelier["timings"] = atelier["timings"][:-1] + ', {"begin": "2026-11-14T10:00:00+01:00", "end": "2026-11-14T12:00:00+01:00"}]'
    assert chevauche(atelier, debut, fin)
