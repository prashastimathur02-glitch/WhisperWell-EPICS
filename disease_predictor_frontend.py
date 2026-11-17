# =============================================================================
# INTERACTIVE DISEASE PREDICTION FRONTEND
# Add this as a new cell in your Jupyter notebook
# =============================================================================

# Import widgets for interactive interface
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    import matplotlib.pyplot as plt
    
    print("✅ Creating Interactive Disease Prediction Interface...")
    
    class InteractiveDiseasePredictor:
        def __init__(self, model_dict, symptom_columns, label_encoder):
            self.model_dict = model_dict
            self.symptom_columns = symptom_columns
            self.label_encoder = label_encoder
            self.predictions_output = widgets.Output()
            self.create_interface()
        
        def create_interface(self):
            # Title
            title = widgets.HTML(
                value="""<div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <h1 style='color: white; margin: 0; font-size: 2.5em;'>🧠 AI Disease Predictor</h1>
                <p style='color: white; margin: 10px 0 0 0; font-size: 1.2em;'>Select your symptoms to get AI-powered predictions</p>
                </div>"""
            )
            
            # Instructions
            instructions = widgets.HTML(
                value="""<div style='background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #2196f3;'>
                <h4 style='color: #1976d2; margin: 0 0 10px 0;'>📋 How to use:</h4>
                <p style='margin: 0; color: #1976d2;'>1. Select the symptoms you are experiencing<br>
                2. Click "Predict Disease" to get AI analysis<br>
                3. Review the results and consult a healthcare professional</p>
                </div>"""
            )
            
            # Symptom selection
            symptom_title = widgets.HTML(
                value="<h3 style='color: #4a90e2; margin: 20px 0 15px 0;'>🩺 Select Your Symptoms:</h3>"
            )
            
            # Create checkboxes for symptoms (display first 20)
            self.symptom_checkboxes = {}
            display_symptoms = self.symptom_columns[:20] if len(self.symptom_columns) > 20 else self.symptom_columns
            
            # Create symptom checkboxes in rows of 3
            symptom_rows = []
            for i in range(0, len(display_symptoms), 3):
                row_symptoms = display_symptoms[i:i+3]
                row_widgets = []
                
                for symptom in row_symptoms:
                    checkbox = widgets.Checkbox(
                        value=False,
                        description=symptom.replace('_', ' ').title(),
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(width='300px', margin='8px')
                    )
                    self.symptom_checkboxes[symptom] = checkbox
                    row_widgets.append(checkbox)
                
                # Fill row if needed
                while len(row_widgets) < 3:
                    row_widgets.append(widgets.HTML(""))
                
                symptom_rows.append(widgets.HBox(row_widgets))
            
            # Prediction button
            self.predict_button = widgets.Button(
                description='🔮 Predict Disease',
                style={'button_color': '#4CAF50'},
                layout=widgets.Layout(width='220px', height='45px', margin='20px 10px')
            )
            self.predict_button.on_click(self.predict_disease)
            
            # Clear button
            self.clear_button = widgets.Button(
                description='🗑️ Clear All',
                style={'button_color': '#f44336'},
                layout=widgets.Layout(width='150px', height='45px', margin='20px 10px')
            )
            self.clear_button.on_click(self.clear_symptoms)
            
            # Example button
            self.example_button = widgets.Button(
                description='💡 Load Example',
                style={'button_color': '#ff9800'},
                layout=widgets.Layout(width='180px', height='45px', margin='20px 10px')
            )
            self.example_button.on_click(self.load_example)
            
            # Button container
            button_container = widgets.HBox([self.predict_button, self.clear_button, self.example_button])
            
            # Results area
            results_title = widgets.HTML(
                value="<h3 style='color: #4a90e2; margin: 30px 0 15px 0;'>🎯 Prediction Results:</h3>"
            )
            
            # Main interface
            self.interface = widgets.VBox([
                title,
                instructions,
                symptom_title,
                *symptom_rows,
                button_container,
                results_title,
                self.predictions_output
            ])
            
            display(self.interface)
        
        def predict_disease(self, button):
            with self.predictions_output:
                clear_output(wait=True)
                
                # Get selected symptoms
                selected_symptoms = []
                for symptom, checkbox in self.symptom_checkboxes.items():
                    if checkbox.value:
                        selected_symptoms.append(symptom)
                
                if not selected_symptoms:
                    display(HTML("<div style='color: red; padding: 15px; background: #ffebee; border-radius: 8px;'>⚠️ Please select at least one symptom!</div>"))
                    return
                
                # Show loading message
                display(HTML("<div style='text-align: center; padding: 20px;'><h4>🔄 Analyzing symptoms...</h4></div>"))
                
                # Create feature vector
                feature_vector = np.zeros((1, len(self.symptom_columns)))
                for symptom in selected_symptoms:
                    if symptom in self.symptom_columns:
                        idx = self.symptom_columns.index(symptom)
                        feature_vector[0, idx] = 1
                
                # Make prediction
                try:
                    clear_output(wait=True)  # Clear loading message
                    
                    # Get probabilities for top predictions
                    if self.model_dict['type'] == "XGBoost Stacking Ensemble":
                        # For stacking ensemble
                        stack_features = []
                        for model_name, model in self.model_dict['models'].items():
                            if self.model_dict['requires_scaling'][model_name]:
                                X_scaled = self.model_dict['scaler'].transform(feature_vector)
                                probs = model.predict_proba(X_scaled)
                            else:
                                probs = model.predict_proba(feature_vector)
                            stack_features.append(probs)
                        
                        X_stack = np.hstack(stack_features)
                        probabilities = self.model_dict['meta_model'].predict_proba(X_stack)[0]
                    else:
                        # For weighted ensemble
                        X_scaled = self.model_dict['scaler'].transform(feature_vector)
                        
                        rf_probs = self.model_dict['models']['random_forest'].predict_proba(feature_vector)[0]
                        xgb_probs = self.model_dict['models']['xgboost'].predict_proba(feature_vector)[0]
                        nb_probs = self.model_dict['models']['naive_bayes'].predict_proba(X_scaled)[0]
                        knn_probs = self.model_dict['models']['knn'].predict_proba(X_scaled)[0]
                        lr_probs = self.model_dict['models']['logistic_regression'].predict_proba(X_scaled)[0]
                        
                        weights = self.model_dict['model_weights']
                        probabilities = np.zeros_like(rf_probs)
                        probabilities += rf_probs * weights['random_forest']
                        probabilities += xgb_probs * weights['xgboost']
                        probabilities += nb_probs * weights['naive_bayes']
                        probabilities += knn_probs * weights['knn']
                        probabilities += lr_probs * weights['logistic_regression']
                    
                    # Get top 5 predictions
                    top_indices = np.argsort(probabilities)[::-1][:5]
                    
                    # Display results
                    results_html = """<div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>"""
                    
                    results_html += f"""<h4 style='color: #2e7d32; margin-top: 0; font-size: 1.3em;'>
                    🩺 Selected Symptoms:</h4>
                    <div style='background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
                    <p style='margin: 0; font-size: 1.1em; color: #333;'>
                    {', '.join([s.replace('_', ' ').title() for s in selected_symptoms])}</p></div>"""
                    
                    results_html += "<h4 style='color: #1976d2; margin: 20px 0 15px 0; font-size: 1.3em;'>🎯 AI Predictions:</h4>"
                    
                    for i, idx in enumerate(top_indices):
                        disease = self.label_encoder.classes_[idx]
                        confidence = probabilities[idx] * 100
                        
                        # Color coding based on confidence
                        if confidence > 60:
                            color = "#d32f2f"  # Red for high confidence
                            bg_color = "#ffebee"
                        elif confidence > 30:
                            color = "#f57c00"  # Orange for medium confidence
                            bg_color = "#fff8e1"
                        else:
                            color = "#388e3c"  # Green for low confidence
                            bg_color = "#e8f5e9"
                        
                        # Confidence bar width
                        bar_width = max(confidence, 5)  # Minimum 5% for visibility
                        
                        results_html += f"""
                        <div style='margin: 15px 0; padding: 20px; background: {bg_color}; 
                        border-radius: 10px; border-left: 5px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div>
                                    <strong style='color: {color}; font-size: 1.2em;'>
                                    {i+1}. {disease.replace('_', ' ').title()}</strong><br>
                                    <span style='color: #666; font-size: 1.1em;'>Confidence: {confidence:.1f}%</span>
                                </div>
                                <div style='font-size: 2em;'>{'🔴' if confidence > 60 else '🟡' if confidence > 30 else '🟢'}</div>
                            </div>
                            <div style='background: #e0e0e0; height: 12px; border-radius: 6px; margin-top: 15px; overflow: hidden;'>
                                <div style='background: {color}; height: 12px; width: {bar_width}%; border-radius: 6px; 
                                transition: width 1s ease-in-out;'></div>
                            </div>
                        </div>"""
                    
                    results_html += "</div>"
                    
                    # Add medical disclaimer
                    disclaimer_html = """
                    <div style='background: #fff3cd; padding: 20px; border-radius: 10px; margin-top: 25px; 
                    border-left: 5px solid #ffc107; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                        <h4 style='color: #856404; margin-top: 0; font-size: 1.2em;'>⚠️ Important Medical Disclaimer</h4>
                        <p style='color: #856404; margin-bottom: 0; font-size: 1.1em; line-height: 1.5;'>
                        This AI prediction is for <strong>informational purposes only</strong>. 
                        Always consult qualified healthcare professionals for proper medical diagnosis and treatment. 
                        Do not use this tool for emergency medical decisions.</p>
                    </div>"""
                    
                    display(HTML(results_html + disclaimer_html))
                    
                except Exception as e:
                    display(HTML(f"<div style='color: red; padding: 20px; background: #ffebee; border-radius: 8px;'>❌ Error making prediction: {str(e)}</div>"))
        
        def clear_symptoms(self, button):
            for checkbox in self.symptom_checkboxes.values():
                checkbox.value = False
            
            with self.predictions_output:
                clear_output(wait=True)
                display(HTML("<div style='color: green; padding: 15px; background: #e8f5e9; border-radius: 8px; text-align: center;'>✅ All symptoms cleared!</div>"))
        
        def load_example(self, button):
            # Clear all first
            for checkbox in self.symptom_checkboxes.values():
                checkbox.value = False
            
            # Load example symptoms (flu-like symptoms)
            example_symptoms = ['fever', 'cough', 'fatigue', 'headache', 'muscle_aches']
            for symptom in example_symptoms:
                if symptom in self.symptom_checkboxes:
                    self.symptom_checkboxes[symptom].value = True
            
            with self.predictions_output:
                clear_output(wait=True)
                display(HTML("<div style='color: blue; padding: 15px; background: #e3f2fd; border-radius: 8px; text-align: center;'>💡 Example symptoms loaded! Click 'Predict Disease' to see results.</div>"))
    
    # Create and display the interactive interface
    if 'hybrid_model' in globals() and 'symptom_cols' in globals() and 'le' in globals():
        print("🚀 Launching Interactive Disease Prediction Interface...")
        predictor_interface = InteractiveDiseasePredictor(hybrid_model, symptom_cols, le)
    else:
        print("❌ Cannot create interface: Required variables not found")
        print("Please make sure the model training completed successfully")
        print("Required variables: hybrid_model, symptom_cols, le")

except ImportError as e:
    print("❌ Required packages not available. Installing...")
    print("Please run: !pip install ipywidgets")
    print("Then restart the kernel and run this cell again")

except Exception as e:
    print(f"❌ Error creating interface: {e}")
    print("\n📝 Alternative: Text-based prediction")
    print("You can still use the predict_with_hybrid_model() function directly")