import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def get_priority(text):
    text = str(text).lower()
    if any(word in text for word in ['fire', 'electric', 'short circuit', 'electrocution']):
        return "Critical"
    elif any(word in text for word in ['leak', 'damage', 'broken', 'not working', 'flood']):
        return "High"
    elif any(word in text for word in ['garbage', 'clean', 'dirty', 'smell']):
        return "Medium"
    else:
        return "Low"

# Load dataset — put your CSV file in the grievance-api folder
df = pd.read_csv('your_file.csv')
df = df[['Sub Category', 'Category']].rename(columns={
    'Sub Category': 'text',
    'Category': 'department'
})
df['clean_text'] = df['text'].apply(clean_text)
df['priority'] = df['text'].apply(get_priority)

# Train department model
vectorizer = TfidfVectorizer()
X_dept = vectorizer.fit_transform(df['clean_text'])
y_dept = df['department']
dept_model = LogisticRegression(max_iter=1000)
dept_model.fit(X_dept, y_dept)

# Train priority model
vectorizer_priority = TfidfVectorizer()
X_priority = vectorizer_priority.fit_transform(df['clean_text'])
y_priority = df['priority']
priority_model = LogisticRegression(max_iter=1000)
priority_model.fit(X_priority, y_priority)

# Save models
with open('department_model.pkl', 'wb') as f:
    pickle.dump(dept_model, f)
with open('department_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('priority_model.pkl', 'wb') as f:
    pickle.dump(priority_model, f)
with open('priority_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer_priority, f)

print("Models trained and saved successfully")