# Grievance Classification & Priority Analysis System

An AI-driven NLP system that automatically classifies civic complaints into departments and assigns priority levels. Built as part of an internship project using the BBMP dataset.

---

## Project Overview

When a citizen files a complaint, this system:
1. Reads the raw complaint text
2. Predicts which **department** should handle it (e.g. Road Maintenance, Water Supply)
3. Assigns a **priority level** (Critical, High, Medium, Low)
4. Returns the result via a REST API

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| NLP & ML | scikit-learn, NLTK |
| Vectorization | TF-IDF |
| Classifier | Logistic Regression |
| API Framework | FastAPI |
| Server | Uvicorn |
| Development | Google Colab |

---

## Project Structure

```
grievance-nlp-system/
│
├── Week1_EDA.ipynb               # Data cleaning and exploratory analysis
├── Week2_Classification.ipynb    # Department classification model
├── Week3_Priority.ipynb          # Priority classification model
│
└── grievance-api/
    ├── main.py                   # FastAPI application
    ├── train.py                  # Model training script
    ├── requirements.txt          # Project dependencies
    ├── department_model.pkl      # Trained department classifier
    ├── department_vectorizer.pkl # TF-IDF vectorizer for departments
    ├── priority_model.pkl        # Trained priority classifier
    └── priority_vectorizer.pkl   # TF-IDF vectorizer for priority
```

---

## Weekly Progress

### Week 1 — Data Cleaning & EDA
- Loaded and cleaned the BBMP civic complaints dataset
- Performed text preprocessing: lowercasing, special character removal, stopword removal, lemmatization
- Conducted exploratory analysis: complaint frequency, department distribution, WordCloud, n-gram analysis

### Week 2 — Department Classification
- Converted cleaned text to numerical vectors using TF-IDF
- Trained a Logistic Regression classifier to predict the responsible department
- Evaluated using accuracy score, classification report, and confusion matrix

### Week 3 — Priority Classification
- Assigned priority labels (Critical, High, Medium, Low) using domain-specific keyword rules
- Trained a second Logistic Regression model for priority prediction
- Documented dataset limitations: 179 unique complaint types, neutral/factual language

### Week 4 — API Deployment
- Built a REST API using FastAPI exposing a `/predict` endpoint
- API accepts raw complaint text and returns predicted department and priority
- Tested via FastAPI's built-in interactive docs at `/docs`

---

## How to Run

### 1. Install dependencies
```bash
pip install fastapi uvicorn scikit-learn nltk pandas
```

### 2. Train the models
```bash
python train.py
```

### 3. Start the API
```bash
uvicorn main:app --reload
```

### 4. Test the API
Open your browser and go to:
```
http://127.0.0.1:8000/docs
```

---

## API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "There is a water leak on the colony road"
}
```

**Response:**
```json
{
  "complaint": "There is a water leak on the colony road",
  "department": "Road Maintenance(Engg)",
  "priority": "Medium"
}
```

---

## Dataset

- **Source:** BBMP (Bruhat Bengaluru Mahanagara Palike) civic complaints dataset 2025
- **Size:** 126,974 rows
- **Columns used:** `Sub Category` (complaint text), `Category` (department)

---

## Author

Built during internship — AI-driven grievance routing system using NLP and REST API deployment.
