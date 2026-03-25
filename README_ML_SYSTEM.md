# Healthcare Appointment Slot Recommendation System

Production-level ML system for intelligent appointment slot recommendations.

## Architecture

```
appointment-ai/
├── configs/          # Configuration files
├── dataset/          # Data storage
├── models/           # Trained models
├── src/
│   ├── scheduling/   # Slot generation
│   ├── features/     # Feature engineering
│   ├── models/       # Model training
│   ├── recommendation/ # Recommendation logic
│   └── api/          # FastAPI service
├── pipelines/        # Training & generation pipelines
├── utils/            # Helper functions
└── main.py           # Main orchestrator
```

## Features

- **Slot Generation Engine**: Generates available time slots
- **Success Prediction**: LightGBM model predicts appointment success
- **Intelligent Ranking**: Multi-criteria slot ranking
- **FastAPI Service**: Production-ready REST API
- **Multi-stakeholder**: Patient, Provider, Admin recommendations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Model

```bash
python main.py
```

### 2. Start API Server

```bash
python src/api/app.py
```

### 3. API Endpoints

**Patient Recommendation**
```bash
POST /recommend-slot
{
  "provider_id": 1,
  "days_ahead": 7,
  "top_n": 5,
  "time_preference": "morning"
}
```

**Provider Recommendation**
```bash
POST /recommend-provider-slots?provider_id=1&date=2026-03-15&top_n=5
```

**Admin Dashboard**
```bash
GET /admin/provider-utilization?date=2026-03-15
```

## Model Details

- **Algorithm**: LightGBM Classifier
- **Target**: appointment_success (1=completed, 0=cancelled/no-show)
- **Features**: 80+ temporal, provider, patient, insurance features
- **Metrics**: ROC-AUC, Precision, Recall, F1-Score

## Configuration

Edit `configs/config.yaml` to customize:
- Model hyperparameters
- Working hours
- Slot duration
- Recommendation thresholds

## Future Enhancements

- Reinforcement learning for dynamic scheduling
- Patient preference modeling
- Provider workload optimization
- Dynamic slot pricing
