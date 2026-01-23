import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("GENAI NEWS INTELLIGENCE SYSTEM - MODEL TRAINING")
print("=" * 60)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
print(f"\nProject root: {project_root}")

sys.path.append(project_root)

DATA_DIR = os.path.join(project_root, "data", "raw")
MODEL_DIR = os.path.join(project_root, "models")
PROCESSED_DIR = os.path.join(project_root, "data", "processed")

print(f"\nDirectories:")
print(f"Data directory: {DATA_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Processed directory: {PROCESSED_DIR}")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

fake_path = os.path.join(DATA_DIR, "Fake.csv")
real_path = os.path.join(DATA_DIR, "True.csv")

print(f"\nData files:")
print(f"Fake.csv: {fake_path} - Exists: {os.path.exists(fake_path)}")
print(f"True.csv: {real_path} - Exists: {os.path.exists(real_path)}")

try:
    from src.preprocess import clean_text
    print("\n✓ Imported clean_text from src.preprocess")
except ImportError as e:
    print(f"\n✗ Could not import clean_text: {e}")
    print("Creating simple clean_text function...")
    import re
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        cleaned_words = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(cleaned_words)

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

try:
    print("\nLoading Fake.csv...")
    fake_data = pd.read_csv(fake_path)
    print(f"✓ Loaded Fake.csv: {len(fake_data)} rows")
    
    print("\nLoading True.csv...")
    real_data = pd.read_csv(real_path)
    print(f"✓ Loaded True.csv: {len(real_data)} rows")
    
except Exception as e:
    print(f"\n✗ Error loading data: {e}")
    exit(1)

fake_data['label'] = 0
real_data['label'] = 1

data = pd.concat([fake_data, real_data], ignore_index=True)
print(f"\n✓ Combined dataset: {len(data)} total rows")

if 'text' in data.columns:
    print("✓ Using 'text' column")
    text_col = 'text'
elif 'title' in data.columns:
    print("✓ Using 'title' column")
    text_col = 'title'
else:
    print("✗ No text column found")
    exit(1)

data = data[[text_col, 'label']].rename(columns={text_col: 'text'})
print(f"✓ Final dataset shape: {data.shape}")

print("\n" + "=" * 60)
print("PREPROCESSING DATA")
print("=" * 60)

print("\nCleaning text...")
try:
    data['cleaned_text'] = data['text'].apply(clean_text)
    print("✓ Text cleaning completed")
except Exception as e:
    print(f"✗ Error cleaning text: {e}")
    data['cleaned_text'] = data['text'].astype(str).str.lower()

print("Removing empty texts...")
initial_count = len(data)
data = data[data['cleaned_text'].str.strip() != '']
print(f"✓ Removed {initial_count - len(data)} empty texts")
print(f"✓ Remaining samples: {len(data)}")

print("\nClass distribution:")
class_dist = data['label'].value_counts()
print(class_dist)
print(f"Fake (0): {(data['label'] == 0).sum()}")
print(f"Real (1): {(data['label'] == 1).sum()}")

cleaned_path = os.path.join(PROCESSED_DIR, "news_cleaned.csv")
data.to_csv(cleaned_path, index=False)
print(f"\n✓ Saved cleaned data to: {cleaned_path}")

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

X = data['cleaned_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

print("\n" + "=" * 60)
print("FEATURE EXTRACTION")
print("=" * 60)

print("\nCreating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✓ Vocabulary size: {len(vectorizer.get_feature_names_out())}")

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

print("\nTraining Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)
print("✓ Model training completed")

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

model_path = os.path.join(MODEL_DIR, "fake_news_model.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✓ Model saved to: {model_path}")
print(f"✓ Vectorizer saved to: {vectorizer_path}")

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

feature_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
})

top_real = feature_df.nlargest(10, 'coefficient')
top_fake = feature_df.nsmallest(10, 'coefficient')

print("\n🔝 Top 10 features for REAL news:")
print(top_real[['feature', 'coefficient']].to_string(index=False))

print("\n🔝 Top 10 features for FAKE news:")
print(top_fake[['feature', 'coefficient']].to_string(index=False))

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"\n📊 Summary:")
print(f"• Total samples: {len(data)}")
print(f"• Training samples: {len(X_train)}")
print(f"• Testing samples: {len(X_test)}")
print(f"• Model accuracy: {accuracy:.4f}")
print(f"• Vocabulary size: {len(feature_names)}")
print(f"\n✅ Model files are ready in: {MODEL_DIR}")
print(f"✅ You can now run: streamlit run app.py")