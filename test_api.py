from pymongo import MongoClient
from pprint import pprint

# =========================
# CONFIG
# =========================
MONGO_URI = "mongodb+srv://SJ:WpLVJ7SHSUSAMAJSISSSEB3n4HiusZL@emr-prod.SHSSHNSHS.mongodb.net/dev"
DB_NAME = "dev"

print("\n🚀 Connecting to MongoDB...\n")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

print("✅ Connected to DB:", DB_NAME)


# =========================
# FETCH DATA
# =========================

# Patients
patients = list(db["patients"].find({}, {"_id": 0}).limit(5))

# Providers
providers = list(db["providers"].find({}, {"_id": 0}).limit(5))

# Schedules
schedules = list(db["schedules"].find({}, {"_id": 0}).limit(5))


# =========================
# PRINT DATA
# =========================

print("\n==================== 🧑 PATIENTS ====================")
for p in patients:
    pprint(p)

print("\n==================== 👨‍⚕️ PROVIDERS ====================")
for p in providers:
    pprint(p)

print("\n==================== 📅 SCHEDULES ====================")
for s in schedules:
    pprint(s)


# =========================
# IMPORTANT IDS (FOR TESTING)
# =========================

print("\n==================== 🔑 IMPORTANT IDS ====================")

patient_ids = [
    p.get("patient_encoded") for p in patients if "patient_encoded" in p
]

provider_ids = [
    p.get("provider_encoded") for p in providers if "provider_encoded" in p
]

print("👉 Patient IDs:", patient_ids)
print("👉 Provider IDs:", provider_ids)


# =========================
# QUICK TEST SAMPLE
# =========================

print("\n==================== 🧪 READY TEST PAYLOAD ====================")

if patient_ids and provider_ids:
    sample_payload = {
        "text": "I want appointment next monday evening",
        "patient_data": {"patient_encoded": patient_ids[0]},
        "provider_data": {"provider_encoded": provider_ids[0]},
        "top_k": 3
    }

    pprint(sample_payload)
else:
    print("❌ No IDs found — check DB data")


print("\n🔥 DONE — use above IDs in your API\n")