import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Model Testing with PKL Files
def load_models_and_scalers(model_dir, water_level):
    """Load all models and scalers for a specific water level"""
    nutrients = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH']
    models = {}
    scalers = {}
    
    for nutrient in nutrients:
        model_path = os.path.join(model_dir, water_level, f"{water_level}_{nutrient}_model.pkl")
        scaler_path = os.path.join(model_dir, water_level, f"{water_level}_{nutrient}_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Warning: Model or scaler not found for {nutrient}")
            continue
            
        models[nutrient] = joblib.load(model_path)
        scalers[nutrient] = joblib.load(scaler_path)
    
    return models, scalers

def predict_soil_nutrients(spectral_data, models, scalers, wavelengths):
    """Predict soil nutrients using loaded models"""
    predictions = {}
    
    for nutrient, model in models.items():
        scaler = scalers[nutrient]
        X = spectral_data[wavelengths]
        X_scaled = scaler.transform(X)
        predictions[nutrient] = model.predict(X_scaled)
    
    return pd.DataFrame(predictions)

# 2. Frequency Analysis
def analyze_frequency_importance(models_dict, wavelengths):
    """Analyze feature importance across all models"""
    importance_data = []
    
    for water_level, models in models_dict.items():
        for nutrient, model in models.items():
            if hasattr(model, 'feature_importances_'):
                for i, wavelength in enumerate(wavelengths):
                    importance_data.append({
                        'Wavelength': wavelength,
                        'Importance': model.feature_importances_[i],
                        'Nutrient': nutrient,
                        'WaterLevel': water_level
                    })
    
    return pd.DataFrame(importance_data)

def plot_frequency_importance(importance_df):
    """Visualize frequency importance results"""
    plt.figure(figsize=(15, 8))
    
    # Overall importance
    plt.subplot(2, 2, 1)
    overall = importance_df.groupby('Wavelength')['Importance'].mean().sort_values()
    overall.plot(kind='barh')
    plt.title('Overall Wavelength Importance')
    plt.xlabel('Mean Importance')
    
    # By nutrient
    plt.subplot(2, 2, 2)
    for nutrient, group in importance_df.groupby('Nutrient'):
        group.groupby('Wavelength')['Importance'].mean().sort_values().plot(
            kind='barh', alpha=0.5, label=nutrient)
    plt.title('Importance by Nutrient')
    plt.xlabel('Mean Importance')
    plt.legend()
    
    # By water level
    plt.subplot(2, 2, 3)
    for wl, group in importance_df.groupby('WaterLevel'):
        group.groupby('Wavelength')['Importance'].mean().sort_values().plot(
            kind='barh', alpha=0.5, label=wl)
    plt.title('Importance by Water Level')
    plt.xlabel('Mean Importance')
    plt.legend()
    
    # Heatmap
    plt.subplot(2, 2, 4)
    heatmap_data = importance_df.pivot_table(
        index='Wavelength', 
        columns='Nutrient', 
        values='Importance',
        aggfunc='mean'
    )
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Wavelength Importance Heatmap')
    
    plt.tight_layout()
    plt.show()

# 3. Complete Testing Workflow for 50ml
def test_50ml_soil_nutrient_models(test_data_path, model_dir='soil_nutrient_models'):
    """Complete testing workflow for 50ml water level"""
    # Load test data
    test_data = pd.read_csv(test_data_path)
    wavelengths = ['410','435','460','485','510','535','560','585','610',
                  '645','680','705','730','760','810','860','900','940']
    
    # Check data format
    missing = [wl for wl in wavelengths if wl not in test_data.columns]
    if missing:
        raise ValueError(f"Test data missing wavelengths: {missing}")
    
    # Load 50ml models
    water_level = '50ml'
    models, scalers = load_models_and_scalers(model_dir, water_level)
    
    if not models:
        raise ValueError(f"No valid models found for {water_level} in the model directory")
    
    # Make predictions
    print(f"\nMaking predictions for {water_level} water level...")
    predictions = predict_soil_nutrients(test_data, models, scalers, wavelengths)
    print(predictions.describe())
    
    # Analyze frequency importance if possible
    try:
        print("\nAnalyzing frequency importance...")
        models_dict = {water_level: models}  # Create dict with just 50ml models
        importance_df = analyze_frequency_importance(models_dict, wavelengths)
        
        # Plot importance for 50ml only
        plt.figure(figsize=(12, 6))
        
        # Importance by nutrient
        plt.subplot(1, 2, 1)
        for nutrient, group in importance_df.groupby('Nutrient'):
            group.groupby('Wavelength')['Importance'].mean().sort_values().plot(
                kind='barh', alpha=0.7, label=nutrient)
        plt.title(f'Importance by Nutrient ({water_level})')
        plt.xlabel('Mean Importance')
        plt.legend()
        
        # Heatmap
        plt.subplot(1, 2, 2)
        heatmap_data = importance_df.pivot_table(
            index='Wavelength', 
            columns='Nutrient', 
            values='Importance',
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.2f')
        plt.title(f'Wavelength Importance Heatmap ({water_level})')
        
        plt.tight_layout()
        plt.show()
        
        print("\nTop 5 most important wavelengths for 50ml:")
        print(importance_df.groupby('Wavelength')['Importance'].mean().nlargest(5))
    except Exception as e:
        print(f"\nCould not analyze frequency importance: {e}")
    
    return predictions

# 4. Example Usage for 50ml
if __name__ == "__main__":
    # Configuration
    TEST_DATA_PATH = r"E:\Nirma hackthon\training\Soil_50ml.csv"  # Make sure to use 50ml test data
    MODEL_DIR = "soil_nutrient_models"  # Directory containing your PKL files
    
    try:
        print("Starting 50ml soil nutrient model testing...")
        predictions = test_50ml_soil_nutrient_models(TEST_DATA_PATH, MODEL_DIR)
        
        # Example of accessing predictions
        print("\nSample predictions for 50ml water level:")
        print(predictions.head())
        
    except Exception as e:
        print(f"Error during testing: {e}")