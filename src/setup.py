import subprocess
import sys

packages = [
    'matplotlib==3.7.2',
    'seaborn==0.12.2',
    'scikit-learn==1.3.0',
    'pandas==2.0.3',
    'numpy==1.24.3',
    'nltk==3.8.1',
    'streamlit==1.28.0',
    'textblob==0.18.0',
    'joblib==1.3.2'
]

print("Installing required packages...")
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("\nAll packages installed successfully!")
print("\nNow run: python notebooks/model_training.py")