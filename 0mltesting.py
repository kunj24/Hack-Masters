import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Model Testing with PKL Files
def load_models_and_scalers(model_dir, water_level):
    """Load all models and scalers for a specific water level with verification"""
    nutrients = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH']
    models = {}
    scalers = {}
    missing_files = []
    
    for nutrient in nutrients:
        model_file = f"{water_level}_{nutrient}_model.pkl"
        scaler_file = f"{water_level}_{nutrient}_scaler.pkl"
        model_path = os.path.join(model_dir, water_level, model_file)
        scaler_path = os.path.join(model_dir, water_level, scaler_file)
        
        if not os.path.exists(model_path):
            missing_files.append(model_file)
            continue
        if not os.path.exists(scaler_path):
            missing_files.append(scaler_file)
            continue
            
        try:
            models[nutrient] = joblib.load(model_path)
            scalers[nutrient] = joblib.load(scaler_path)
            print(f"Loaded {nutrient} model and scaler successfully")
        except Exception as e:
            print(f"Error loading {nutrient} model: {e}")
            missing_files.extend([model_file, scaler_file])
    
    if missing_files:
        print("\nMissing or corrupted files:")
        for file in missing_files:
            print(f"- {file}")
        
        # Show what files are actually present
        existing_files = os.listdir(os.path.join(model_dir, water_level))
        print("\nFiles found in directory:")
        for f in existing_files:
            print(f"- {f}")
    
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

# 2. Accuracy Evaluation
def evaluate_accuracy(y_true, y_pred, nutrient_name):
    """Calculate and display accuracy metrics"""
    metrics = {
        'R² Score': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred)
    }
    
    print(f"\nAccuracy Metrics for {nutrient_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

# 3. Enhanced Frequency Analysis
def analyze_frequency_importance(models_dict, wavelengths):
    """Enhanced frequency importance analysis with statistics"""
    importance_data = []
    
    for water_level, models in models_dict.items():
        for nutrient, model in models.items():
            if hasattr(model, 'feature_importances_'):
                for i, wavelength in enumerate(wavelengths):
                    importance_data.append({
                        'Wavelength': int(wavelength),
                        'Importance': model.feature_importances_[i],
                        'Nutrient': nutrient,
                        'WaterLevel': water_level
                    })
    
    importance_df = pd.DataFrame(importance_data)
    
    # Calculate statistics
    stats_df = importance_df.groupby(['Nutrient', 'Wavelength'])['Importance'].agg(
        ['mean', 'std', 'min', 'max']).reset_index()
    
    return importance_df, stats_df

def plot_enhanced_frequency_importance(importance_df, water_level):
    """Enhanced visualization of frequency importance"""
    plt.figure(figsize=(18, 12))
    
    # 1. Overall Importance
    plt.subplot(2, 2, 1)
    overall = importance_df.groupby('Wavelength')['Importance'].mean().sort_values()
    overall.plot(kind='barh', color='teal')
    plt.title(f'Overall Wavelength Importance ({water_level})')
    plt.xlabel('Mean Importance')
    
    # 2. Nutrient-Specific Importance
    plt.subplot(2, 2, 2)
    nutrients = importance_df['Nutrient'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(nutrients)))
    
    for nutrient, color in zip(nutrients, colors):
        nutrient_data = importance_df[importance_df['Nutrient'] == nutrient]
        nutrient_mean = nutrient_data.groupby('Wavelength')['Importance'].mean().sort_values()
        nutrient_mean.plot(kind='barh', alpha=0.7, label=nutrient, color=color)
    
    plt.title(f'Nutrient-Specific Importance ({water_level})')
    plt.xlabel('Mean Importance')
    plt.legend()
    
    # 3. Top Wavelengths
    plt.subplot(2, 2, 3)
    top_wavelengths = []
    for nutrient in nutrients:
        top = importance_df[importance_df['Nutrient'] == nutrient].nlargest(3, 'Importance')
        top_wavelengths.extend(top['Wavelength'].unique())
    
    top_data = importance_df[importance_df['Wavelength'].isin(top_wavelengths)]
    sns.boxplot(x='Wavelength', y='Importance', hue='Nutrient', 
                data=top_data, palette='viridis')
    plt.title(f'Top Wavelengths by Nutrient ({water_level})')
    plt.xticks(rotation=45)
    
    # 4. Heatmap
    plt.subplot(2, 2, 4)
    heatmap_data = importance_df.pivot_table(
        index='Wavelength', 
        columns='Nutrient', 
        values='Importance',
        aggfunc='mean'
    ).sort_index()
    
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.2f')
    plt.title(f'Wavelength Importance Heatmap ({water_level})')
    
    plt.tight_layout()
    plt.show()

def print_frequency_stats(stats_df, water_level):
    """Print detailed frequency statistics"""
    print(f"\n{'='*60}")
    print(f"Frequency Importance Statistics for {water_level}")
    print(f"{'='*60}")
    
    for nutrient in stats_df['Nutrient'].unique():
        print(f"\n{nutrient}:")
        print("Top 5 Wavelengths (by importance):")
        top5 = stats_df[stats_df['Nutrient'] == nutrient].nlargest(5, 'mean')
        print(top5[['Wavelength', 'mean', 'std']].to_string(index=False))
        
        print("\nMost Consistent Wavelength (lowest std dev):")
        consistent = stats_df[stats_df['Nutrient'] == nutrient].nsmallest(1, 'std')
        print(consistent[['Wavelength', 'mean', 'std']].to_string(index=False))

# 4. Testing Workflow for 0ml
def test_0ml_models(test_data_path, model_dir='soil_nutrient_models'):
    """Complete testing workflow for 0ml water level"""
    # Define target columns
    target_columns = {
        'Nitrogen': 'Nitro (mg/10 g)',
        'Phosphorus': 'Posh Nitro (mg/10 g)',
        'Potassium': 'Pota Nitro (mg/10 g)',
        'pH': 'Ph'
    }
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    wavelengths = ['410','435','460','485','510','535','560','585','610',
                 '645','680','705','730','760','810','860','900','940']
    
    # Check data format
    missing = [wl for wl in wavelengths if wl not in test_data.columns]
    if missing:
        raise ValueError(f"Test data missing wavelengths: {missing}")
    
    # Verify target columns exist
    for nutrient, col in target_columns.items():
        if col not in test_data.columns:
            raise ValueError(f"Target column '{col}' for {nutrient} not found in test data")
    
    # Load 0ml models
    water_level = '0ml'
    print(f"\nLoading models for {water_level} water level...")
    models, scalers = load_models_and_scalers(model_dir, water_level)
    if not models:
        raise ValueError(f"No valid models found for {water_level} water level")
    
    # Filter data for 0ml
    test_data_0ml = test_data[test_data['Records'].str.contains(water_level, na=False)]
    if test_data_0ml.empty:
        raise ValueError(f"No test data available for {water_level} water level")
    
    # Make predictions
    print(f"\nMaking predictions for {water_level} water level...")
    predictions = predict_soil_nutrients(test_data_0ml, models, scalers, wavelengths)
    
    # Evaluate accuracy
    print(f"\n{'='*40}")
    print(f"Accuracy Evaluation for {water_level} water level")
    print(f"{'='*40}")
    
    accuracy_results = {}
    for nutrient, pred in predictions.items():
        true_col = target_columns[nutrient]
        y_true = test_data_0ml[true_col]
        y_pred = pred
        
        accuracy_results[nutrient] = evaluate_accuracy(y_true, y_pred, nutrient)
    
    # Enhanced Frequency Analysis
    try:
        importance_df, stats_df = analyze_frequency_importance({water_level: models}, wavelengths)
        plot_enhanced_frequency_importance(importance_df, water_level)
        print_frequency_stats(stats_df, water_level)
    except Exception as e:
        print(f"\nCould not analyze frequency importance: {e}")
    
    return {
        'predictions': predictions,
        'accuracy': accuracy_results,
        'frequency_stats': stats_df if 'stats_df' in locals() else None
    }

# 5. Main Execution
if __name__ == "__main__":
    # Configuration
    TEST_DATA_PATH = r"E:\Nirma hackthon\training\Soil_0ml.csv"
    MODEL_DIR = r"E:\Nirma hackthon\training\soil_nutrient_models"
    
    try:
        print("Starting 0ml water level model testing...")
        
        # Verify model directory exists
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
        
        # Verify 0ml subdirectory exists
        water_level_dir = os.path.join(MODEL_DIR, '0ml')
        if not os.path.exists(water_level_dir):
            raise FileNotFoundError(
                f"0ml model directory not found at: {water_level_dir}\n"
                f"Please ensure you have trained models for 0ml water level"
            )
        
        results = test_0ml_models(TEST_DATA_PATH, MODEL_DIR)
        
        # Print final summary
        print("\n\n=== Final Results ===")
        print("\nSample predictions:")
        print(results['predictions'].head())
        
        print("\nAccuracy Summary:")
        for nutrient, metrics in results['accuracy'].items():
            print(f"\n{nutrient}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("\nTroubleshooting Tips:")
        print(f"1. Verify the file exists: {TEST_DATA_PATH}")
        print(f"2. Check model directory contains 0ml subfolder: {MODEL_DIR}")
        print("3. Ensure all required .pkl files are present in 0ml folder")
        print("4. Confirm your test data contains 0ml samples")