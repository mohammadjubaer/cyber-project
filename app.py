from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)

# --- CONFIGURATION ---
# IMPORTANT: Ensure your .pkl files are inside a folder named 'models' next to this script
MODEL_DIR = 'models' 

# Global variables
clf = None
word_vect = None
char_vect = None

def load_models():
    """Load models at startup (Crucial for Heroku/Gunicorn)"""
    global clf, word_vect, char_vect
    try:
        print("Loading models...")
        clf = joblib.load(os.path.join(MODEL_DIR, 'model_svm_95.pkl'))
        word_vect = joblib.load(os.path.join(MODEL_DIR, 'word_vect.pkl'))
        char_vect = joblib.load(os.path.join(MODEL_DIR, 'char_vect.pkl'))
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# Load immediately
load_models()

def clean_text(text):
    """Must match training script exactly"""
    if not isinstance(text, str): return ""
    text = str(text)
    text = re.sub(r"you're", "you are", text, flags=re.IGNORECASE)
    text = re.sub(r"youre", "you are", text, flags=re.IGNORECASE)
    text = re.sub(r"ur", "your", text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?\']', '', text) 
    return re.sub(r'\s+', ' ', text).strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_text = data.get('text', '').strip()
        
        if not raw_text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        # Preprocess & Vectorize
        cleaned_text = clean_text(raw_text)
        word_features = word_vect.transform([cleaned_text])
        char_features = char_vect.transform([cleaned_text])
        features = hstack([word_features, char_features])
        
        # Predict
        probs = clf.predict_proba(features)[0]
        prediction = clf.predict(features)[0]
        
        is_aggressive = int(prediction) == 1
        
        return jsonify({
            'success': True,
            'data': {
                'prediction': 'Aggressive' if is_aggressive else 'Non-Aggressive',
                'is_aggressive': is_aggressive,
                'probabilities': {
                    'non_aggressive': round(probs[0] * 100, 1),
                    'aggressive': round(probs[1] * 100, 1)
                },
                'cleaned_text': cleaned_text
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Local testing
    app.run(debug=True, port=5000)