# 🔍 FakeSpot AI — Fake Review Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=black)
![API Key](https://img.shields.io/badge/API%20Key-NOT%20REQUIRED-brightgreen?style=for-the-badge)

**An AI-powered web app that detects fake product reviews using local transformer models and linguistic analysis.**  
No API key. No internet after setup. Runs 100% on your machine.

[Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [How It Works](#-how-it-works) · [Project Structure](#-project-structure) · [Deploy](#-deployment)

</div>

---

## 🌟 Why This Project?

Fake reviews cost consumers **billions of dollars** every year. Studies show that:

- **42%** of Amazon reviews are estimated to be fake *(ReviewMeta, 2023)*
- **93%** of consumers say online reviews influence their buying decisions
- Fake review farms charge as little as **$5 per 10 fake reviews**
- Most people **cannot tell** the difference between real and manufactured reviews

FakeSpot AI solves this by combining the power of **transformer-based NLP** with **rule-based linguistic analysis** to give every review a transparency score — and explain exactly *why* it looks fake or genuine.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **AI Classification** | DistilRoBERTa zero-shot model classifies each review as Fake / Suspicious / Genuine |
| 📊 **Trust Score** | Overall product trust rating from 0–100 based on all reviews combined |
| 🚩 **Red Flag Detection** | Specific reasons why a review looks fake (exclamations, caps, vague language, etc.) |
| ✅ **Positive Signals** | Genuine indicators like specific details, balanced tone, time-based experience |
| 💬 **AI Explanation** | Human-readable explanation for every verdict |
| 📈 **Visual Charts** | Pie chart breakdown + per-review probability bar chart (Plotly) |
| 🎚️ **Sensitivity Slider** | Tune detection aggressiveness from 1 (lenient) to 10 (strict) |
| 🗂️ **Sample Review Sets** | 3 built-in sets (Smartphone, Hotel, Supplements) to demo instantly |
| 💾 **Export Results** | Download full analysis as **JSON** or **CSV** |
| 🌙 **Dark Theme UI** | Custom styled Streamlit interface |

---

## 🎬 Demo

### Input
```
"Absolutely amazing!!! BEST PHONE EVER!! Everyone needs to buy this NOW!!
 You won't regret it I promise!!"

"I've been using this for 3 months. Camera is decent in daylight but struggles
 at night. Battery lasts a full day with moderate use. Good value overall."

"Received this for free in exchange for a review. Simply outstanding product!
 Best purchase of my entire life! Order immediately!"
```

### Output
```
🚨 Review #1 — FAKE           (fake probability: 89%)
   Red flags: Excessive exclamations · ALL CAPS overuse · Extreme language
              Marketing/urgency language · Lacks specific details

✅ Review #2 — GENUINE         (fake probability: 11%)
   Signals: Time-based experience · Specific details · Balanced tone

🚨 Review #3 — FAKE           (fake probability: 78%)
   Red flags: Possible incentivized review · Extreme language · Lacks specifics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Trust Score: 18 / 100  →  ⚠️ VERY LOW TRUST
```

---

## 🛠️ Installation

### Prerequisites
- Python **3.9 or higher**
- **4 GB RAM** minimum (8 GB recommended)
- ~**1.5 GB** free disk space (for model cache)
- Internet connection for **first run only** (downloads models)

---

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/fakespot-ai.git
cd fakespot-ai
```

---

### Step 2 — Create a Virtual Environment

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

---

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

> ⏳ First install downloads PyTorch + HuggingFace models (~1.5 GB total).  
> This only happens once. After that — everything runs fully offline.

---

### Step 4 — Run the App
```bash
streamlit run app.py
```

✅ App opens automatically at **http://localhost:8501**

---

## 🧠 How It Works

FakeSpot AI uses a **two-layer detection system** that combines transformer intelligence with handcrafted linguistic rules:

```
                    ┌─────────────────────────────────────────┐
                    │           Review Text Input              │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┴─────────────────────────┐
              │                                                   │
              ▼                                                   ▼
  ┌───────────────────────┐                      ┌───────────────────────────┐
  │   🤖 TRANSFORMER AI   │                      │  📏 LINGUISTIC ANALYZER   │
  │                       │                      │                           │
  │  cross-encoder/       │                      │  • Exclamation density    │
  │  nli-distilroberta    │                      │  • ALL CAPS ratio         │
  │                       │                      │  • Vocabulary diversity   │
  │  Zero-shot classify:  │                      │  • Specificity score      │
  │  "fake review"   vs   │                      │  • Emotional extremity    │
  │  "genuine review"     │                      │  • Review length          │
  │                       │                      │  • Repetition patterns    │
  └──────────┬────────────┘                      └──────────────┬────────────┘
             │  60% weight                                      │  40% weight
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │    🔀 SCORE FUSION      │
                         │  + Sensitivity Tuning   │
                         └────────────┬───────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │           FINAL VERDICT             │
                    │                                     │
                    │  🚨 FAKE / ⚠️ SUSPICIOUS / ✅ GENUINE │
                    │  + Fake Probability %               │
                    │  + Red Flags List                   │
                    │  + Positive Signals List            │
                    │  + AI Explanation                   │
                    └─────────────────────────────────────┘
```

---

### 🚩 Red Flags (Fake Indicators)

| Signal | What It Means |
|---|---|
| Excessive `!!!` | Genuine reviewers rarely use multiple exclamation marks |
| ALL CAPS words | Manufactured hype language pattern |
| Extreme superlatives | "BEST EVER", "PERFECT IN EVERY WAY", "LIFE CHANGING" |
| Low vocabulary diversity | Repetitive writing suggests low-effort fake content |
| No specific details | Real users mention model numbers, use cases, time periods |
| Very short review | Less than 15 words rarely contains genuine experience |
| Urgency language | "Buy NOW", "Tell all your friends" — marketing, not reviewing |
| Incentivized hints | "Received for free", "given in exchange" |

---

### ✅ Positive Signals (Genuine Indicators)

| Signal | What It Means |
|---|---|
| 40+ words | Adequate length for meaningful feedback |
| Specific product details | Real users remember specifics |
| Time-based experience | "Used for 3 months" — lived experience |
| Balanced tone | Mentions both pros and cons |
| Nuance language | "However", "although", "but" — real opinions have nuance |
| Comparative analysis | Real buyers compare products before purchasing |
| High vocabulary diversity | Natural, unrehearsed writing has varied word choice |

---

## 📁 Project Structure

```
fakespot-ai/
│
├── app.py              ← Streamlit frontend
│                          • Dark theme UI with custom CSS
│                          • Trust Score display
│                          • Per-review breakdown cards
│                          • Plotly charts (pie + bar)
│                          • JSON / CSV export
│                          • Sample review sets
│
├── detector.py         ← AI detection engine
│                          • ReviewDetector class
│                          • Transformer pipeline (zero-shot)
│                          • Linguistic feature extractor
│                          • Red flag & positive signal detectors
│                          • Score fusion logic
│                          • Verdict explainer
│
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🤖 Models Used

| Model | Task | Size | Source |
|---|---|---|---|
| `cross-encoder/nli-distilroberta-base` | Zero-shot fake/genuine classification | ~315 MB | HuggingFace |
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment fallback | ~268 MB | HuggingFace |

Both models are:
- ✅ **Free** — no account, no API key, no credit card
- ✅ **Auto-downloaded** on first run, then cached locally
- ✅ **Offline** after first download
- ✅ **GPU-accelerated** automatically if CUDA is available

---

## 📦 Dependencies

```
streamlit==1.40.0       # Web UI framework
transformers==4.46.3    # HuggingFace model library
torch==2.5.1            # Deep learning backend
pandas==2.2.3           # Data handling + CSV export
plotly==5.24.1          # Interactive charts
sentencepiece==0.2.0    # Tokenizer support
```

---

## 🚀 Deployment

### Option 1 — Streamlit Community Cloud (Free)

1. Push your project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Connect your repo
4. Set **Main file path:** `app.py`
5. Click **Deploy** ✅

> Note: Free tier has memory limits. Models load on first visit (~60 seconds).

---

### Option 2 — Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

```bash
# Build and run
docker build -t fakespot-ai .
docker run -p 8501:8501 fakespot-ai
```

---

### Option 3 — Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Upload all project files
4. Space builds and deploys automatically ✅

---

## 🧪 Testing

Run a quick sanity check without launching the full UI:

```bash
python -c "
from detector import ReviewDetector

detector = ReviewDetector()

reviews = [
    'AMAZING PRODUCT!!! BEST EVER!! BUY NOW!!',
    'Used this for 2 months. Works well but battery drains faster than expected.',
    'Okay product. Nothing special. Arrived on time.',
]

for r in reviews:
    result = detector.analyze(r)
    print(f'Verdict: {result[\"verdict\"]:12} | Fake prob: {result[\"fake_probability\"]:.2f} | {r[:50]}')
"
```

**Expected output:**
```
Verdict: FAKE         | Fake prob: 0.87 | AMAZING PRODUCT!!! BEST EVER!! BUY NOW!!
Verdict: GENUINE      | Fake prob: 0.14 | Used this for 2 months. Works well but batt...
Verdict: SUSPICIOUS   | Fake prob: 0.38 | Okay product. Nothing special. Arrived on ti...
```

---

## 🔮 Future Improvements

- [ ] **Image review analysis** — detect AI-generated product photos
- [ ] **Reviewer history analysis** — flag accounts that only post 5-star reviews
- [ ] **Multi-language support** — detect fake reviews in Hindi, Spanish, French
- [ ] **Browser extension** — analyze reviews directly on Amazon/Flipkart
- [ ] **Fine-tuned model** — train on labeled fake review dataset for higher accuracy
- [ ] **FastAPI backend** — REST API wrapper for e-commerce platform integration
- [ ] **Bulk CSV upload** — analyze thousands of reviews at once

---

## 🤝 Contributing

Contributions are welcome!

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 👨‍💻 Author

Built with ❤️ using HuggingFace Transformers and Streamlit.

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

*No API key. No cloud. No cost. Just AI.*

</div>
