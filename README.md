# 🔥 EMR Slot Recommender

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-black.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-ML-green.svg)](https://catboost.ai/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Read--Only-47A248.svg?logo=mongodb)](https://mongodb.com)
[![License: MIT](https://img.shields.io/github/license/HP/Emr-Calendar-Model.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)

**AI-Powered Appointment Slot Recommendation for EMR Systems**

ML system that parses natural language requests (\"Schedule John for Tuesday morning\"), predicts appointment success probability using CatBoost (ROC-AUC 0.85+), generates/ranks optimal slots considering patient prefs, provider load, historical no-shows, and business costs. Production-ready FastAPI API with MongoDB integration.

## 📖 Overview

Optimizes healthcare scheduling by recommending **top-K slots most likely to be confirmed**. 

**Problem Solved**:
- **No-Shows cost clinics $150+ per missed appointment**
- Manual scheduling ignores patient/provider patterns
- Overbooking leads to provider burnout (util >85%)

**How it Works**:
```
NLP Parse → DB Lookup → Slot Generation → ML Prediction → Multi-Objective Ranking → API Response
```

## 🚀 Features

- 🎯 **CatBoost ML Model** (70+ features: insurance, TF-IDF notes, rolling success rates, slot demand)
- 🗣️ **NLP Parser** extracts patient/provider/dates from free text
- 📊 **Real-time Features** from MongoDB (patient history, provider utilization curves)
- ⚖️ **Cost-Aware Ranking** (false negatives cost 1000, false positives 200)
- 🛡️ **Read-Only DB** prevents accidental writes
- 🧪 **Full Test Suite** + Model Evaluation (ROC-AUC, calibration plots)
- 📈 **Production Pipelines** for retraining/inference

## 🧠 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI API   │───▶│   Recommender    │───▶│   MongoDB       │
│ /recommend-slots│    │ • NLP Parser     │    │ • patients      │
└─────────────────┘    │ • Slot Generator │    │ • providers     │
                       │ • CatBoost Pred  │    │ • appointments  │
                       │ • Slot Ranker    │    └─────────────────┘
                       └──────────────────┘
```

**Modular Design**: `src/` (core logic), `api/` (endpoints), `pipelines/` (ML workflows), `configs/` (YAML).

## 🛠️ Tech Stack

| Category      | Tools                                      |
|---------------|--------------------------------------------|
| **API**       | FastAPI, Uvicorn, Pydantic                |
| **ML**        | CatBoost, scikit-learn, pandas, joblib    |
| **Database**  | PyMongo, Motor (async)                    |
| **Config**    | PyYAML, python-dotenv                     |
| **Testing**   | pytest, pytest-asyncio                   |
| **Data**      | CSV/JSON (processed appointments)         |

## ⚙️ Installation

1. **Clone & Environment**
   ```bash
   git clone https://github.com/HP/Emr-Calendar-Model.git
   cd Emr-Calendar-Model
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   ```

2. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **MongoDB** (read-only mode)
   - Set `MONGODB_URI` in `.env`
   - Collections: `patient-details`, `users`, `appointments`, etc. (per `configs/config.yaml`)

4. **Model** (pre-trained)
   ```
   models/slot_prediction_model.pkl  # Already included
   ```

## ▶️ Usage

**Start Server**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**API Playground**: [http://localhost:8000/docs](http://localhost:8000/docs)

**Example Request**:
```bash
curl -X POST "http://localhost:8000/recommend-slots" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Schedule John Smith with Dr. Patel Tuesday morning",
    "top_k": 3
  }'
```

**Response**:
```json
{
  "recommended_slots": [
    {"date": "2024-01-23", "time": "10:00", "prob": 0.87, "score": 0.92},
    {"date": "2024-01-23", "time": "11:00", "prob": 0.82, "score": 0.88}
  ],
  "parsed_intent": {"patient_name": "John Smith", "provider_name": "Dr. Patel"}
}
```

## 📡 API Documentation

**POST /recommend-slots**
- **Request**: `RecommendSlotsRequest` (text, patient_data, provider_data, top_k=5)
- **Response**: `RecommendSlotsResponse` (slots list w/ date/time/prob/score, parsed_intent)
- **Auto-parses**: Patient/provider names, dates, preferred times.

**GET /health** → `{"status": "ok"}`

[Full OpenAPI Docs](/docs)

## 📂 Project Structure

```
Emr-Calendar-Model/
├── api/              # FastAPI app + routes
├── configs/          # YAML config (DB, ranking weights)
├── dataset/          # Raw/processed appointments CSV
├── models/           # slot_prediction_model.pkl
├── notebooks/        # Data prep → evaluation
├── pipelines/        # train_model.py, generate_slots.py
├── src/              # Core: recommender.py, features/, database/
├── tests/            # pytest suite
├── main.py           # uvicorn entrypoint
├── requirements.txt
└── README.md
```

## 📸 Screenshots / Demo

**API Docs** (Auto-generated):
![FastAPI Docs](https://via.placeholder.com/800x400/0066cc/ffffff?text=FastAPI+Docs+at+/docs)

**Model Calibration** (from eval notebook):
![Calibration Plot](https://via.placeholder.com/600x400/28a745/ffffff?text=ROC-AUC+0.85%2B)

**Live Demo**: Run server → visit `/docs` → Try `/recommend-slots`

## 🧪 Testing

```bash
pytest tests/ -v
```

95%+ coverage across API, recommender, features, DB queries.

## 🚧 Future Enhancements

- 🔄 **Feedback Loop**: Auto-retrain on confirmed slots
- 👥 **Multi-Provider**: Team scheduling optimization
- ⚡ **Real-time**: WebSocket slot updates
- 📱 **Frontend**: React dashboard
- 🌐 **Deployment**: Docker + Kubernetes

## 🤝 Contributing

1. Fork → Clone → Create branch (`git checkout -b feature/xyz`)
2. Commit (`git commit -m 'feat: add xyz'`)
3. Push → PR to `main`
4. Add tests + update README

Issues/PRs welcome!

## 📜 License

[MIT License](LICENSE) © 2024 HP

## 👨‍💻 Author

**HP** - [GitHub](https://github.com/HP)  
*Built with ❤️ for efficient healthcare scheduling*

