# Toxic Comment Classifier - Responsible AI

This repository contains a hybrid system for detecting toxic language in comments. It follows **Responsible AI** principles by combining transparent rules with statistical learning.

## ✨ Features
- **Hybrid Detection**: Uses both a simple keyword-based engine (high transparency) and a Logistic Regression ML model (contextual understanding).
- **Text Cleaning Pipeline**: Automated normalization including punctuation removal, stop-word filtering, and lemmatization.
- **TF-IDF Vectorization**: Converts text to numerical features while preserving word significance.
- **Interpretability**: Logistic Regression coefficients can be inspected to understand model decisions.
- **Evaluation**: Generates detailed classification reports and a visual **Confusion Matrix**.

## 🛡️ Responsible AI Design
1. **Explainability**: By checking for keywords first, the system can provide a clear reason (e.g., "Contains flagged word: 'idiot'") before falling back to the ML prediction.
2. **Deterministic Fallback**: Keywords act as a safety net for blatant toxicity that the ML model might occasionally miss due to data variance.
3. **Privacy**: The preprocessing steps ensure that noise and potential PII (like specific numbers) are removed before processing.

## 🚀 Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Option A: Run the Notebook (Recommended for Exploration)**
   Open `Toxic_Comment_Classifier.ipynb` in Jupyter Lab or VS Code to see the interactive explanation and visualizations.

3. **Option B: Run the CLI application**
   ```bash
   python main.py
   ```

## 📊 Evaluation
After training, the model generates a `confusion_matrix.png` which helps analyze:
- **True Positives**: Correctly identified toxic comments.
- **False Negatives**: Toxic comments missed by the system (Safety gap).
- **False Positives**: Safe comments incorrectly flagged (Censorship risk).
