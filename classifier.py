import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessor import TextPreprocessor
import joblib
import os

class ToxicClassifier:
    def __init__(self, keyword_list=None):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression()
        self.keyword_list = keyword_list or ['hate', 'ugly', 'stupid', 'loser', 'idiot', 'worthless', 'kill']
        self.is_trained = False

    def check_keywords(self, text):
        """Simple rule-based check for high-risk words (Responsible AI)"""
        text_lower = text.lower()
        found_keywords = [word for word in self.keyword_list if word in text_lower]
        return len(found_keywords) > 0, found_keywords

    def train(self, df):
        print("Cleaning text...")
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)
        
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['label']
        
        print("Training Logistic Regression model...")
        self.model.fit(X, y)
        self.is_trained = True
        print("Model trained successfully.")

    def predict(self, text):
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        
        # Responsible AI: Keyword check first
        keyword_flag, found = self.check_keywords(text)
        
        # ML Prediction
        cleaned = self.preprocessor.clean_text(text)
        if not cleaned: # Handle empty strings after cleaning
             ml_prob = 0.0
             ml_pred = 0
        else:
            X_vec = self.vectorizer.transform([cleaned])
            ml_prob = self.model.predict_proba(X_vec)[0][1]
            ml_pred = self.model.predict(X_vec)[0]
        
        # Hybrid decision: Flag if either flags it (conservative approach)
        final_flag = 1 if (keyword_flag or ml_pred == 1) else 0
        
        return {
            'text': text,
            'keyword_flag': int(keyword_flag),
            'found_keywords': found,
            'ml_probability': float(ml_prob),
            'ml_prediction': int(ml_pred),
            'final_prediction': final_flag
        }

    def evaluate(self, df):
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)
        X = self.vectorizer.transform(df['cleaned_text'])
        y_true = df['label']
        y_pred = self.model.predict(X)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Toxic', 'Toxic'], 
                    yticklabels=['Non-Toxic', 'Toxic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Toxic Comment Classifier')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Non-Toxic', 'Toxic']))
        print("Confusion Matrix saved as 'confusion_matrix.png'")

    def save_model(self, path='model_assets'):
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self.model, os.path.join(path, 'classifier.pkl'))
        joblib.dump(self.vectorizer, os.path.join(path, 'vectorizer.pkl'))
        print(f"Model saved to {path}")

    def load_model(self, path='model_assets'):
        self.model = joblib.load(os.path.join(path, 'classifier.pkl'))
        self.vectorizer = joblib.load(os.path.join(path, 'vectorizer.pkl'))
        self.is_trained = True
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    # Test script
    from generate_data import generate_sample_data
    if not os.path.exists('toxic_dataset.csv'):
        generate_sample_data()
    
    df = pd.read_csv('toxic_dataset.csv')
    classifier = ToxicClassifier()
    classifier.train(df)
    classifier.evaluate(df)
    
    test_text = "You are an absolute idiot."
    print(f"\nTesting prediction: '{test_text}'")
    print(classifier.predict(test_text))
