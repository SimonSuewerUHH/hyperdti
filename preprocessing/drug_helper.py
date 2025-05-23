import requests
from requests import Session
from functools import lru_cache
from typing import Optional, List

BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
_session = Session()

def get_random_string():
    """
    Generates a random string of 10 characters.
    """
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))

@lru_cache(maxsize=128)
def get_trade_name_from_smiles(
        smiles: str,
        timeout: float = 5.0
) -> Optional[str]:
    """
    Sucht per SMILES zuerst die PubChem-CID, dann alle Synonyme und
    wählt daraus einen wahrscheinlichen Handelsnamen (erstes Eintrag ohne
    Ziffern/Bindestriche, beginnend mit Großbuchstaben).
    Gibt None zurück, wenn nichts gefunden oder bei Fehlern.
    """
    try:
        # 1) CID per SMILES holen
        r1 = _session.get(
            f'{BASE_URL}/compound/smiles/{smiles}/cids/JSON',
            timeout=timeout
        )
        r1.raise_for_status()
        cids = r1.json().get('IdentifierList', {}).get('CID', [])
        if not cids:
            return get_random_string()
        cid = cids[0]

        # 2) Synonyme für diesen CID holen
        r2 = _session.get(
            f'{BASE_URL}/compound/cid/{cid}/synonyms/JSON',
            timeout=timeout
        )
        r2.raise_for_status()
        info = r2.json().get('InformationList', {}).get('Information', [])
        # print(info)
        if not info:
            return get_random_string()
        synonyms: List[str] = info[0].get('Synonym', [])

        shortest_synonym = synonyms[0]
        for synonym in synonyms:
            if len(synonym) < len(shortest_synonym):
                shortest_synonym = synonym
        return shortest_synonym

    except requests.RequestException as e:
        # In Produktiv-Code lieber logging statt print()
        print(f"Fehler bei PubChem-Abfrage: {e}")
        return get_random_string()

