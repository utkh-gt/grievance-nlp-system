from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load models
with open('department_model.pkl', 'rb') as f:
    dept_model = pickle.load(f)

with open('department_vectorizer.pkl', 'rb') as f:
    dept_vectorizer = pickle.load(f)

with open('priority_model.pkl', 'rb') as f:
    priority_model = pickle.load(f)

with open('priority_vectorizer.pkl', 'rb') as f:
    priority_vectorizer = pickle.load(f)

# Text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# FastAPI app
app = FastAPI()

class Complaint(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Grievance Classification API is running"}

@app.post("/predict")
def predict(complaint: Complaint):
    clean = clean_text(complaint.text)

    # Predict department
    dept_input = dept_vectorizer.transform([clean])
    department = dept_model.predict(dept_input)[0]

    # Predict priority
    priority_input = priority_vectorizer.transform([clean])
    priority = priority_model.predict(priority_input)[0]

    return {
        "complaint": complaint.text,
        "department": department,
        "priority": priority
    }