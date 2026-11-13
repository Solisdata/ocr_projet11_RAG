"""Repère une période dans la question ("ce week-end", "demain"...) pour filtrer les événements par date.

La recherche vectorielle compare le SENS des textes : elle ne sait pas que
"ce week-end" veut dire "samedi 14 et dimanche 15 novembre". On traduit donc
ces expressions en un intervalle de dates, appliqué sur les métadonnées.
"""

import json
import re
from functools import lru_cache
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

FUSEAU = ZoneInfo("Europe/Paris")
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

# (expression régulière, nom de la période) — testées dans l'ordre, la première qui correspond gagne.
EXPRESSIONS = [
    (r"apr[eè]s[- ]demain", "après-demain"),
    (r"\bdemain\b", "demain"),
    (r"aujourd'?hui|\bce soir\b|\bcet apr[eè]s[- ]midi\b|\bce matin\b", "aujourd'hui"),
    (r"week[- ]?end prochain|prochain week[- ]?end", "week-end prochain"),
    (r"week[- ]?end|\bwe\b", "ce week-end"),
    (r"semaine prochaine|prochaine semaine", "semaine prochaine"),
    (r"cette semaine", "cette semaine"),
]


def debut_du_jour(moment):
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def detecter_periode(question, maintenant=None):
    """Renvoie (nom, début, fin, expression trouvée), ou None si la question ne cite aucune période.

    L'intervalle est [début, fin[ en heure de Paris.
    """
    maintenant = maintenant or datetime.now(FUSEAU)
    aujourdhui = debut_du_jour(maintenant)
    texte = question.lower().replace("’", "'")

    for motif, nom in EXPRESSIONS:
        trouve = re.search(motif, texte)
        if not trouve:
            continue

        if nom == "aujourd'hui":
            debut, fin = maintenant, aujourdhui + timedelta(days=1)
        elif nom == "demain":
            debut = aujourdhui + timedelta(days=1)
            fin = debut + timedelta(days=1)
        elif nom == "après-demain":
            debut = aujourdhui + timedelta(days=2)
            fin = debut + timedelta(days=1)
        elif nom in ("ce week-end", "week-end prochain"):
            # Samedi de cette semaine (si on est déjà samedi ou dimanche : le week-end en cours).
            samedi = aujourdhui + timedelta(days=5 - aujourdhui.weekday())
            if nom == "week-end prochain":
                samedi += timedelta(days=7)
            debut = max(maintenant, samedi)
            fin = samedi + timedelta(days=2)
        elif nom == "cette semaine":
            debut = maintenant
            fin = aujourdhui + timedelta(days=7 - aujourdhui.weekday())
        else:  # semaine prochaine
            debut = aujourdhui + timedelta(days=7 - aujourdhui.weekday())
            fin = debut + timedelta(days=7)
        return nom, debut, fin, trouve.group(0)

    return None


def lire_date(valeur):
    """Convertit une date ISO du CSV ("2025-11-13T13:30:00+00:00") ; None si vide ou invalide."""
    try:
        date = datetime.fromisoformat(str(valeur))
    except ValueError:
        return None
    return date if date.tzinfo else date.replace(tzinfo=FUSEAU)


def lire_seances(metadata):
    """Liste des séances [(début, fin), ...] de l'événement.

    On utilise le champ "timings" (une entrée par séance) ; s'il est absent, on se
    rabat sur la première et la dernière date, considérées comme une seule séance.
    """
    return _lire_seances(
        metadata.get("timings") or "", metadata.get("start_date", ""), metadata.get("end_date", "")
    )


# Le filtre parcourt les 10 000 événements à chaque question : on ne lit chaque JSON qu'une fois.
@lru_cache(maxsize=None)
def _lire_seances(timings_json, start_date, end_date):
    seances = []
    try:
        timings = json.loads(timings_json or "[]")
    except (TypeError, ValueError):
        timings = []
    for timing in timings if isinstance(timings, list) else []:
        debut_seance = lire_date(timing.get("begin", "")) if isinstance(timing, dict) else None
        if debut_seance:
            seances.append((debut_seance, lire_date(timing.get("end", "")) or debut_seance))
    if seances:
        return tuple(seances)

    debut_evt = lire_date(start_date)
    if debut_evt is None:
        return ()
    return ((debut_evt, lire_date(end_date) or debut_evt),)


def seances_dans(metadata, debut, fin):
    """Les séances qui ont lieu, au moins en partie, dans l'intervalle [début, fin[."""
    return [(d, f) for d, f in lire_seances(metadata) if d < fin and f >= debut]


def chevauche(metadata, debut, fin):
    """Vrai si au moins une séance de l'événement a lieu dans l'intervalle [début, fin[."""
    return bool(seances_dans(metadata, debut, fin))


def formater_seance(debut, fin):
    """Ex. : "samedi 14/11/2026 de 14h00 à 16h30" (heure de Paris)."""
    debut, fin = debut.astimezone(FUSEAU), fin.astimezone(FUSEAU)
    if debut.date() == fin.date():
        return f"{formater_jour(debut)} de {debut:%Hh%M} à {fin:%Hh%M}"
    return f"du {formater_jour(debut)} {debut:%Hh%M} au {formater_jour(fin)} {fin:%Hh%M}"


def formater_jour(moment):
    """Ex. : "vendredi 13/11/2026"."""
    return f"{JOURS[moment.weekday()]} {moment:%d/%m/%Y}"
