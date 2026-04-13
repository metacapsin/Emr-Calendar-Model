import re
from types import SimpleNamespace
from typing import Any, Dict, List

from bson import ObjectId

from src.database.errors import EntityNotFoundError
from src.database.queries import (
    ensure_database_indexes,
    get_patient_by_name,
    get_patient_data,
    get_provider_by_name,
    get_provider_data,
    get_slot_statistics,
)


class FakeCollection:
    def __init__(self, docs: List[Dict[str, Any]] = None):
        self.docs = docs or []
        self.indexes = []

    def _get_field(self, doc: Dict[str, Any], key: str) -> Any:
        if "." in key:
            parts = key.split(".")
            for part in parts:
                if not isinstance(doc, dict) or part not in doc:
                    return None
                doc = doc[part]
            return doc
        return doc.get(key)

    def _match(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        for key, value in query.items():
            if key == "$and":
                if not all(self._match(doc, sub) for sub in value):
                    return False
                continue
            if key == "$or":
                if not any(self._match(doc, sub) for sub in value):
                    return False
                continue
            field = self._get_field(doc, key)
            if isinstance(value, dict):
                if "$regex" in value:
                    pattern = value["$regex"]
                    flags = re.IGNORECASE if value.get("$options", "") == "i" else 0
                    if not re.search(pattern, str(field or ""), flags):
                        return False
                if "$in" in value:
                    if field not in value["$in"]:
                        return False
                if "$ne" in value:
                    if field == value["$ne"]:
                        return False
            else:
                if field != value:
                    return False
        return True

    def find_one(self, query: Dict[str, Any]) -> Any:
        for doc in self.docs:
            if self._match(doc, query):
                return doc
        return None

    def find(self, query: Dict[str, Any], *args, **kwargs) -> List[Dict[str, Any]]:
        return [doc for doc in self.docs if self._match(doc, query)]

    def insert_one(self, doc: Dict[str, Any]) -> SimpleNamespace:
        doc = dict(doc)
        if "_id" not in doc:
            doc["_id"] = ObjectId()
        self.docs.append(doc)
        return SimpleNamespace(inserted_id=doc["_id"])

    def create_index(self, keys: Any, name: str = None, unique: bool = False) -> None:
        self.indexes.append((tuple(keys), name, unique))

    def update_one(self, filter_: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> None:
        matching = self.find_one(filter_)
        if matching:
            if "$set" in update:
                matching.update(update["$set"])
        elif upsert:
            new_doc = dict(filter_)
            if "$set" in update:
                new_doc.update(update["$set"])
            self.insert_one(new_doc)


class FakeDB:
    def __init__(self, data: Dict[str, List[Dict[str, Any]]]):
        self.data = {k: FakeCollection(v) for k, v in data.items()}

    def __getitem__(self, item: str) -> FakeCollection:
        return self.data[item]


def test_get_patient_by_name_exact():
    patient = {"_id": ObjectId(), "fullName": "Emma Gamez"}
    db = FakeDB({"patient-details": [patient]})

    result = get_patient_by_name(db, "Emma Gamez")
    assert result == str(patient["_id"])


def test_get_patient_by_name_fuzzy():
    patient = {"_id": ObjectId(), "firstName": "Arvind", "lastName": "Mathur"}
    db = FakeDB({"patient-details": [patient]})

    result = get_patient_by_name(db, "Dr. A. Mathur")
    assert result == str(patient["_id"])


def test_get_provider_by_name_exact():
    provider = {"_id": ObjectId(), "firstName": "Arvind", "lastName": "Mathur"}
    db = FakeDB({"users": [provider]})

    result = get_provider_by_name(db, "Arvind Mathur")
    assert result == str(provider["_id"])


def test_get_provider_by_name_fuzzy():
    provider = {"_id": ObjectId(), "firstName": "Arvind", "lastName": "Mathur"}
    db = FakeDB({"users": [provider]})

    result = get_provider_by_name(db, "Dr Arvind Mathur")
    assert result == str(provider["_id"])


def test_get_patient_data_provider_history():
    patient = {"_id": ObjectId(), "fullName": "Emma Gamez", "patient_encoded": 5}
    provider_id = str(ObjectId())
    appts = [
        {"_id": ObjectId(), "patient_id": str(patient["_id"]), "provider_id": provider_id, "appt_date": "2026-04-01", "appt_hour": 9, "status": "Confirmed", "duration_minutes": 60},
        {"_id": ObjectId(), "patient_id": str(patient["_id"]), "provider_id": provider_id, "appt_date": "2026-04-02", "appt_hour": 10, "status": "Cancelled", "duration_minutes": 30},
    ]
    db = FakeDB({"patient-details": [patient], "appointments": appts})

    data = get_patient_data(db, str(patient["_id"]), provider_id=provider_id)
    assert data["patient_provider_history"] == 2
    assert data["patient_provider_success_rate"] == 0.5
    assert data["patient_provider_loyalty"] == 1.0


def test_get_provider_data_with_utilization():
    provider = {"_id": ObjectId(), "firstName": "Arvind", "lastName": "Mathur", "max_daily_slots": 10}
    provider_id = str(provider["_id"])
    appts = [
        {"_id": ObjectId(), "provider_id": provider_id, "patient_id": str(ObjectId()), "appt_date": "2026-04-01", "appt_hour": 9, "status": "Confirmed", "duration_minutes": 45},
        {"_id": ObjectId(), "provider_id": provider_id, "patient_id": str(ObjectId()), "appt_date": "2026-04-02", "appt_hour": 10, "status": "Cancelled", "duration_minutes": 30},
    ]
    db = FakeDB({"users": [provider], "appointments": appts})

    data = get_provider_data(db, provider_id)
    assert data["provider_total_appts"] == 2
    assert data["provider_cancel_rate"] == 0.5
    assert data["provider_avg_duration"] == 37
    assert 0.0 <= data["provider_7day_util"] <= 1.0


def test_get_slot_statistics_fallback():
    provider_id = str(ObjectId())
    appts = [
        {"_id": ObjectId(), "provider_id": provider_id, "patient_id": str(ObjectId()), "appt_date": "2026-04-01", "appt_hour": 9, "status": "Confirmed"},
    ]
    db = FakeDB({"appointments": appts, "slot_statistics": []})

    stats = get_slot_statistics(db, provider_id, weekday=3, hour=9)
    assert stats["success_rate"] == 0.5
    assert stats["popularity_score"] > 0.0
    assert stats["total_count"] == 0


def test_ensure_database_indexes_runs():
    db = FakeDB({"patient-details": [], "users": [], "appointments": [], "slot_statistics": []})
    ensure_database_indexes(db)
    assert len(db["patient-details"].indexes) >= 1
    assert len(db["slot_statistics"].indexes) >= 1
