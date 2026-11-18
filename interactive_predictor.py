#!/usr/bin/env python3
"""
Interactive Disease Prediction System
Ultimate Hybrid Model - Python Implementation
"""

import sys
import os

# Add current directory to path to import our model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultimate_hybrid_model import UltimateHybridDiseasePredictor

def interactive_prediction():
    """Interactive disease prediction interface"""
    print("🩺 Interactive Disease Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    print("Loading Ultimate Hybrid Model...")
    predictor = UltimateHybridDiseasePredictor()
    
    print("\n✅ Model loaded successfully!")
    print(f"📊 Dataset: {len(predictor.dataset)} samples")
    print(f"🔬 Features: {len(predictor.feature_names)} symptoms")
    print(f"🏥 Diseases: {len(predictor.disease_names)} conditions")
    
    while True:
        print("\n" + "="*50)
        print("🔍 Disease Prediction Options:")
        print("1. Enter symptoms manually")
        print("2. Use example cases")
        print("3. Show available symptoms")
        print("4. Show available diseases")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            manual_prediction(predictor)
        elif choice == '2':
            example_cases(predictor)
        elif choice == '3':
            show_symptoms(predictor)
        elif choice == '4':
            show_diseases(predictor)
        elif choice == '5':
            print("👋 Thank you for using the Disease Prediction System!")
            break
        else:
            print("❌ Invalid option. Please try again.")

def manual_prediction(predictor):
    """Manual symptom input and prediction"""
    print("\n📝 Manual Symptom Input")
    print("-" * 30)
    print("Enter symptoms separated by commas (e.g., fever, cough, headache)")
    print("Type 'help' to see available symptoms")
    
    while True:
        user_input = input("\nSymptoms: ").strip()
        
        if user_input.lower() == 'help':
            print("\nAvailable symptoms:")
            for i, symptom in enumerate(predictor.feature_names, 1):
                print(f"{i:2d}. {symptom.replace('_', ' ')}")
            continue
        elif user_input.lower() == 'back':
            break
        elif not user_input:
            print("❌ Please enter at least one symptom")
            continue
        
        # Parse symptoms
        symptoms = [s.strip().lower().replace(' ', '_') for s in user_input.split(',')]
        
        # Validate symptoms
        valid_symptoms = []
        invalid_symptoms = []
        
        for symptom in symptoms:
            if symptom in predictor.feature_names:
                valid_symptoms.append(symptom)
            else:
                # Try to find close matches
                close_match = find_close_symptom(symptom, predictor.feature_names)
                if close_match:
                    valid_symptoms.append(close_match)
                    print(f"💡 Did you mean '{close_match.replace('_', ' ')}'? Using that instead.")
                else:
                    invalid_symptoms.append(symptom)
        
        if invalid_symptoms:
            print(f"⚠️  Unknown symptoms: {', '.join(invalid_symptoms)}")
        
        if valid_symptoms:
            print(f"\n🔍 Analyzing symptoms: {', '.join([s.replace('_', ' ') for s in valid_symptoms])}")
            
            # Get predictions
            predictions = predictor.predict_disease(valid_symptoms)
            
            # Display results
            display_predictions(predictions, predictor)
        else:
            print("❌ No valid symptoms found. Please try again.")

def find_close_symptom(input_symptom, symptom_list):
    """Find close symptom matches using simple string matching"""
    input_words = set(input_symptom.replace('_', ' ').split())
    
    for symptom in symptom_list:
        symptom_words = set(symptom.replace('_', ' ').split())
        
        # Check for word overlap
        if input_words & symptom_words:
            return symptom
        
        # Check for substring matches
        if input_symptom in symptom or symptom.replace('_', '') in input_symptom.replace('_', ''):
            return symptom
    
    return None

def example_cases(predictor):
    """Show predefined example cases"""
    print("\n📋 Example Cases")
    print("-" * 20)
    
    examples = {
        "1": {
            "name": "Respiratory Infection",
            "symptoms": ["fever", "cough", "breathing_difficulty", "fatigue", "chest_pain"]
        },
        "2": {
            "name": "Diabetes Symptoms",
            "symptoms": ["frequent_urination", "extreme_thirst", "blurred_vision", "fatigue", "weight_loss"]
        },
        "3": {
            "name": "Flu-like Symptoms",
            "symptoms": ["fever", "muscle_aches", "headache", "fatigue", "chills"]
        },
        "4": {
            "name": "Gastrointestinal Issues",
            "symptoms": ["stomach_pain", "nausea", "vomiting", "diarrhea"]
        },
        "5": {
            "name": "Cardiovascular Symptoms",
            "symptoms": ["chest_pain", "palpitations", "dizziness", "fatigue"]
        }
    }
    
    for key, case in examples.items():
        print(f"{key}. {case['name']}: {', '.join([s.replace('_', ' ') for s in case['symptoms']])}")
    
    choice = input("\nSelect example (1-5) or 'back': ").strip()
    
    if choice in examples:
        case = examples[choice]
        print(f"\n🔍 Analyzing: {case['name']}")
        print(f"Symptoms: {', '.join([s.replace('_', ' ') for s in case['symptoms']])}")
        
        predictions = predictor.predict_disease(case['symptoms'])
        display_predictions(predictions, predictor)
    elif choice.lower() != 'back':
        print("❌ Invalid selection")

def display_predictions(predictions, predictor):
    """Display prediction results in a formatted way"""
    print("\n🎯 Prediction Results")
    print("=" * 40)
    
    if not predictions:
        print("❌ No predictions available")
        return
    
    # Show top 5 predictions
    top_predictions = predictions[:5]
    
    for i, (disease, confidence) in enumerate(top_predictions, 1):
        confidence_pct = confidence * 100
        
        # Get disease info
        disease_info = predictor.get_disease_info(disease)
        
        print(f"\n{i}. {disease.replace('_', ' ')}")
        print(f"   Confidence: {confidence_pct:.1f}%")
        
        if disease_info:
            print(f"   Category: {disease_info.get('category', 'Unknown')}")
            print(f"   Severity: {disease_info.get('severity', 'Unknown').replace('_', ' ').title()}")
            
        # Confidence bar
        bar_length = 20
        filled_length = int(bar_length * confidence)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"   [{bar}] {confidence_pct:.1f}%")
    
    # Show detailed info for top prediction
    if top_predictions:
        top_disease, top_confidence = top_predictions[0]
        disease_info = predictor.get_disease_info(top_disease)
        
        if disease_info and top_confidence > 0.1:  # Only show if confidence > 10%
            print(f"\n📋 Details for {top_disease.replace('_', ' ')}:")
            print(f"   • Key symptoms: {', '.join([s.replace('_', ' ') for s in disease_info['symptoms'][:5]])}")
            print(f"   • Population prevalence: {disease_info['prevalence']*100:.1f}%")
    
    print("\n⚠️  Medical Disclaimer:")
    print("   This AI prediction is for informational purposes only.")
    print("   Always consult healthcare professionals for proper diagnosis.")

def show_symptoms(predictor):
    """Display all available symptoms"""
    print("\n📊 Available Symptoms")
    print("=" * 30)
    
    # Group by category
    categories = {}
    for symptom, info in predictor.symptom_database.items():
        category = info.get('category', 'other')
        if category not in categories:
            categories[category] = []
        categories[category].append(symptom)
    
    for category, symptoms in categories.items():
        print(f"\n{category.upper()}:")
        for symptom in sorted(symptoms):
            print(f"  • {symptom.replace('_', ' ')}")

def show_diseases(predictor):
    """Display all available diseases"""
    print("\n🏥 Available Diseases")
    print("=" * 25)
    
    # Group by category
    categories = {}
    for disease, info in predictor.disease_database.items():
        category = info.get('category', 'other')
        if category not in categories:
            categories[category] = []
        categories[category].append((disease, info))
    
    for category, diseases in categories.items():
        print(f"\n{category.upper()}:")
        for disease, info in sorted(diseases):
            prevalence = info.get('prevalence', 0) * 100
            severity = info.get('severity', 'unknown').replace('_', ' ')
            print(f"  • {disease.replace('_', ' ')} (Prevalence: {prevalence:.1f}%, Severity: {severity})")

if __name__ == "__main__":
    try:
        interactive_prediction()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure all required libraries are installed:")
        print("pip install pandas numpy scikit-learn")