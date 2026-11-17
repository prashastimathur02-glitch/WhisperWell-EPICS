#!/usr/bin/env python3
"""
Quick validation of Ultimate Hybrid Model
"""

import pickle
import pandas as pd

def validate_model():
    print("🔍 Validating Ultimate Hybrid Model Files")
    print("=" * 40)
    
    # Check pickle file
    try:
        with open('ultimate_hybrid_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("✅ Model pickle file loaded successfully")
        print(f"   • Models: {list(model_data.get('models', {}).keys())}")
        print(f"   • Features: {len(model_data.get('feature_names', []))}")
        print(f"   • Diseases: {len(model_data.get('disease_names', []))}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    # Check CSV file
    try:
        df = pd.read_csv('Final_Augmented_dataset_Diseases_and_Symptoms.csv')
        print("✅ Dataset CSV loaded successfully")
        print(f"   • Rows: {len(df):,}")
        print(f"   • Columns: {len(df.columns)}")
        print(f"   • Diseases: {df['disease'].nunique()}")
        print(f"   • Sample diseases: {list(df['disease'].unique()[:5])}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
    
    print("\n🎯 Files ready for use!")

if __name__ == "__main__":
    validate_model()