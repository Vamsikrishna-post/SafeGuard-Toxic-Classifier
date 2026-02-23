import pandas as pd
import os
from classifier import ToxicClassifier
from generate_data import generate_sample_data

def main():
    print("=== Toxic Comment Classifier (Responsible AI Edition) ===")
    
    # 1. Ensure data exists
    if not os.path.exists('toxic_dataset.csv'):
        print("Generating sample dataset...")
        generate_sample_data()
    
    df = pd.read_csv('toxic_dataset.csv')
    
    # 2. Initialize and train
    classifier = ToxicClassifier()
    classifier.train(df)
    
    # 3. Evaluate
    print("\n--- Evaluating Model ---")
    classifier.evaluate(df)
    
    # 4. Interactive Testing
    print("\n--- Interactive Mode (type 'exit' to quit) ---")
    while True:
        user_input = input("\nEnter a comment to test: ")
        if user_input.lower() == 'exit':
            break
        
        result = classifier.predict(user_input)
        
        status = "🚩 TOXIC" if result['final_prediction'] == 1 else "✅ SAFE"
        print(f"Result: {status}")
        print(f"- ML Confidence: {result['ml_probability']:.2f}")
        if result['keyword_flag']:
            print(f"- Keyword Match: Found {result['found_keywords']}")
        
    print("\nExiting. Thank you for using the Responsible AI Classifier!")

if __name__ == "__main__":
    main()
