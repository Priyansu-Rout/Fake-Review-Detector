<<<<<<< HEAD
# Fake-Review-Detector
Fake news check use Transfermer
=======
# 🔍 Fake Review Detector — AI Project

Streamlit + HuggingFace Transformers · No API Key · Runs 100% Offline

## Setup & Run

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Opens at: http://localhost:8501
First run downloads ~250MB of models. After that — fully offline.

## Models Used (No API Key)
- distilbert-base-uncased-finetuned-sst-2-english  → Sentiment Analysis
- typeform/distilbert-base-uncased-mnli             → Zero-Shot Classification
- Custom NLP Rule Engine                            → Linguistic Pattern Detection

## Features
- Fake Score (0-100) per review
- Trust Score for overall product
- Red flag explanations per review
- Donut + bar charts
- Sensitivity slider
- CSV export
- Sample reviews included
>>>>>>> 8508795 (first commit)
