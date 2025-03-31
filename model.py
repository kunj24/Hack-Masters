# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# import joblib

# # Load Dataset
# df = pd.read_csv("Soil_25ml.csv")  # Replace with actual file path

# # Drop non-numeric or unnecessary columns
# df = df.drop(columns=['Records'], errors='ignore')  # Remove 'Records' column if present

# # Define features (wavelengths) and targets (pH, N, P, K)
# X = df.drop(columns=['Ph', 'Nitro (mg/10 g)', 'Phos (mg/10 g)', 'Pota (mg/10 g)'])
# y_ph = df['Ph']
# y_n = df['Nitro (mg/10 g)']
# y_p = df['Phos (mg/10 g)']
# y_k = df['Pota (mg/10 g)']

# # Normalize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train_ph, y_test_ph = train_test_split(X_scaled, y_ph, test_size=0.2, random_state=42)
# X_train, X_test, y_train_n, y_test_n = train_test_split(X_scaled, y_n, test_size=0.2, random_state=42)
# X_train, X_test, y_train_p, y_test_p = train_test_split(X_scaled, y_p, test_size=0.2, random_state=42)
# X_train, X_test, y_train_k, y_test_k = train_test_split(X_scaled, y_k, test_size=0.2, random_state=42)

# # Train models
# models = {}
# targets = {'pH': y_train_ph, 'Nitrogen': y_train_n, 'Phosphorus': y_train_p, 'Potassium': y_train_k}

# def train_model(y_train, y_test, target_name):
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     print(f"{target_name}: MAE = {mae:.4f}, R² = {r2:.4f}")
#     models[target_name] = model
#     return model, y_test, y_pred

# # Train and evaluate each model
# results = {}
# ph_model, y_test_ph, y_pred_ph = train_model(y_train_ph, y_test_ph, "pH")
# n_model, y_test_n, y_pred_n = train_model(y_train_n, y_test_n, "Nitrogen")
# p_model, y_test_p, y_pred_p = train_model(y_train_p, y_test_p, "Phosphorus")
# k_model, y_test_k, y_pred_k = train_model(y_train_k, y_test_k, "Potassium")

# # Plot actual vs predicted values
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # If not installed, run: pip install seaborn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. Load Datasets
def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        return None
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return None

# 2. Prepare Data
def prepare_data(df, water_level):
    """Prepares the data for model training."""
    if df is None:
        print(f"Error: DataFrame is None. Check data loading.")
        return None, None, None
    
    # Define spectral wavelengths (features)
    wavelengths = ['410','435','460','485','510','535','560','585','610',
                  '645','680','705','730','760','810','860','900','940']
    
    # Define target variables
    targets = {
        'Nitrogen': 'Nitro (mg/10 g)',
        'Phosphorus': 'Posh Nitro (mg/10 g)',
        'Potassium': 'Pota Nitro (mg/10 g)',
        'pH': 'Ph'
    }
    
    # Filter data based on water level
    df_filtered = df[df['Records'].str.contains(water_level, na=False)].copy()
    
    if df_filtered.empty:
        print(f"Warning: No data found for water level '{water_level}'")
        return None, wavelengths, targets
    
    # Handle missing values
    for target in targets.values():
        if target in df_filtered.columns and df_filtered[target].isna().any():
            median_val = df_filtered[target].median()
            if pd.isna(median_val):  # Check if median is also NaN
                print(f"Warning: Target '{target}' contains only NaN values.")
                df_filtered[target].fillna(0, inplace=True)
            else:
                df_filtered[target].fillna(median_val, inplace=True)
    
    # Check for missing values in features
    for wavelength in wavelengths:
        if wavelength in df_filtered.columns and df_filtered[wavelength].isna().any():
            median_val = df_filtered[wavelength].median()
            if pd.isna(median_val):
                print(f"Warning: Feature '{wavelength}' contains only NaN values.")
                df_filtered[wavelength].fillna(0, inplace=True)
            else:
                df_filtered[wavelength].fillna(median_val, inplace=True)
    
    return df_filtered, wavelengths, targets

# 3. Train models with hyperparameter tuning
def train_model(X_train, y_train, nutrient_name, cv=5):
    """Trains a Random Forest model with hyperparameter tuning."""
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create base model
    rf = RandomForestRegressor(random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=0
    )
    
    print(f"Training model for {nutrient_name} with hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {nutrient_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

# 4. Evaluate model
def evaluate_model(model, X_test, y_test, nutrient_name):
    """Evaluates a trained model."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"Evaluation metrics for {nutrient_name}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics

# 5. Plot feature importance
def plot_feature_importance(models, wavelengths, water_level, output_dir=None):
    """Plots feature importance for each model."""
    if not models:
        print(f"No models to plot feature importance for {water_level}.")
        return
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, model) in enumerate(models.items(), 1):
        plt.subplot(2, 2, i)
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        # Plot horizontal bar chart
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [wavelengths[i] for i in indices])
        plt.title(f'Feature Importance for {name} ({water_level})')
        plt.xlabel('Relative Importance')
    
    plt.tight_layout()
    
    # Save plot if output_dir is provided
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f'{water_level}_feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()

# 6. Analyze spectral frequencies
def analyze_spectral_frequencies(models_dict, wavelengths, output_dir="frequency_analysis"):
    """
    Analyzes the importance of different spectral frequencies across trained models.
    
    Args:
        models_dict (dict): Dictionary of trained models (as returned by create_soil_nutrient_models)
        wavelengths (list): List of wavelength names
        output_dir (str): Directory to save analysis results
        
    Returns:
        pandas.DataFrame: DataFrame with frequency importance analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store importance scores
    importance_data = {
        'Wavelength': [],
        'Nutrient': [],
        'Water_Level': [],
        'Importance': []
    }
    
    # Extract importance from each model
    for water_level, models in models_dict.items():
        for nutrient, model in models.items():
            # Get feature importances
            importances = model.feature_importances_
            
            # Store importance for each wavelength
            for i, wavelength in enumerate(wavelengths):
                importance_data['Wavelength'].append(int(wavelength))
                importance_data['Nutrient'].append(nutrient)
                importance_data['Water_Level'].append(water_level)
                importance_data['Importance'].append(importances[i])
    
    # Create DataFrame
    importance_df = pd.DataFrame(importance_data)
    
    return importance_df

# 7. Visualize frequency importance
def visualize_frequency_importance(importance_df, output_dir="frequency_analysis"):
    """
    Generates visualizations and reports of spectral frequency importance.
    
    Args:
        importance_df (pandas.DataFrame): DataFrame with importance scores
        output_dir (str): Directory to save analysis results
    """
    # 1. Overall most important frequencies across all models
    overall_importance = importance_df.groupby('Wavelength')['Importance'].mean().reset_index()
    overall_importance = overall_importance.sort_values('Importance', ascending=False)
    
    print("\n*Overall Most Important Wavelengths:*")
    for i, row in overall_importance.head(5).iterrows():
        print(f"  {row['Wavelength']} nm: {row['Importance']:.4f}")
    
    # Plot overall importance
    plt.figure(figsize=(12, 6))
    plt.bar(overall_importance['Wavelength'].astype(str), overall_importance['Importance'])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Average Importance')
    plt.title('Overall Importance of Different Spectral Frequencies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_frequency_importance.png'), dpi=300)
    plt.show()
    
    # 2. Most important frequencies by nutrient
    nutrient_importance = importance_df.groupby(['Nutrient', 'Wavelength'])['Importance'].mean().reset_index()
    
    for nutrient in importance_df['Nutrient'].unique():
        nutrient_data = nutrient_importance[nutrient_importance['Nutrient'] == nutrient]
        top_wavelengths = nutrient_data.sort_values('Importance', ascending=False).head(5)
        
        print(f"\n*Most Important Wavelengths for {nutrient}:*")
        for i, row in top_wavelengths.iterrows():
            print(f"  {row['Wavelength']} nm: {row['Importance']:.4f}")
        
        # Plot nutrient-specific importance
        plt.figure(figsize=(12, 6))
        nutrient_data = nutrient_data.sort_values('Wavelength')
        plt.bar(nutrient_data['Wavelength'].astype(str), nutrient_data['Importance'])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Average Importance')
        plt.title(f'Spectral Frequency Importance for {nutrient}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{nutrient}_frequency_importance.png'), dpi=300)
        plt.show()
    
    # 3. Effect of water level on frequency importance
    water_level_importance = importance_df.groupby(['Water_Level', 'Wavelength'])['Importance'].mean().reset_index()
    
    for water_level in importance_df['Water_Level'].unique():
        wl_data = water_level_importance[water_level_importance['Water_Level'] == water_level]
        top_wavelengths = wl_data.sort_values('Importance', ascending=False).head(5)
        
        print(f"\n*Most Important Wavelengths for {water_level} Water Level:*")
        for i, row in top_wavelengths.iterrows():
            print(f"  {row['Wavelength']} nm: {row['Importance']:.4f}")
    
    # 4. Frequency bands analysis
    # Group wavelengths into bands
    def get_band(wavelength):
        if wavelength < 450:
            return 'Violet-Blue (400-450nm)'
        elif wavelength < 500:
            return 'Blue (450-500nm)'
        elif wavelength < 570:
            return 'Green (500-570nm)'
        elif wavelength < 620:
            return 'Yellow (570-620nm)'
        elif wavelength < 750:
            return 'Red (620-750nm)'
        else:
            return 'NIR (750-940nm)'
    
    importance_df['Band'] = importance_df['Wavelength'].apply(get_band)
    band_importance = importance_df.groupby(['Band', 'Nutrient'])['Importance'].mean().reset_index()
    
    print("\n*Spectral Band Importance by Nutrient:*")
    for nutrient in importance_df['Nutrient'].unique():
        nutrient_bands = band_importance[band_importance['Nutrient'] == nutrient].sort_values('Importance', ascending=False)
        print(f"\n  {nutrient}:")
        for i, row in nutrient_bands.iterrows():
            print(f"    {row['Band']}: {row['Importance']:.4f}")
    
    # 5. Save detailed analysis to CSV
    importance_df.to_csv(os.path.join(output_dir, 'frequency_importance_details.csv'), index=False)
    
    # 6. Create heatmap of importance across nutrients and water levels
    plt.figure(figsize=(15, 10))
    pivot_table = importance_df.pivot_table(
        index='Wavelength', 
        columns='Nutrient', 
        values='Importance',
        aggfunc='mean'
    )
    plt.imshow(pivot_table, cmap='viridis', aspect='auto')
    plt.colorbar(label='Importance')
    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=45)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    plt.title('Wavelength Importance Heatmap by Nutrient')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wavelength_nutrient_heatmap.png'), dpi=300)
    plt.show()

# 8. Prediction function for new samples
def predict_soil_nutrients(spectral_data, water_level='0ml', model_dir=None):
    """Predicts soil nutrients from spectral data."""
    if model_dir is None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define wavelengths
    wavelengths = ['410','435','460','485','510','535','560','585','610',
                  '645','680','705','730','760','810','860','900','940']
    
    # Check if all wavelengths are present
    missing_wavelengths = [wl for wl in wavelengths if wl not in spectral_data.columns]
    if missing_wavelengths:
        print(f"Error: Missing wavelengths in input data: {missing_wavelengths}")
        return None
    
    # Get only the wavelength columns
    X = spectral_data[wavelengths]
    
    # Define targets
    nutrients = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH']
    
    # Make predictions for each nutrient
    predictions = []
    
    for i, row in X.iterrows():
        row_data = row.values.reshape(1, -1)
        row_predictions = {}
        
        for nutrient in nutrients:
            try:
                # Load model and scaler
                model_path = os.path.join(model_dir, f'{water_level}_{nutrient}_model.pkl')
                scaler_path = os.path.join(model_dir, f'{water_level}_{nutrient}_scaler.pkl')
                
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    print(f"Error: Model or scaler file not found for {nutrient}.")
                    row_predictions[nutrient] = None
                    continue
                
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Scale data
                scaled_data = scaler.transform(row_data)
                
                # Predict
                pred = model.predict(scaled_data)[0]
                row_predictions[nutrient] = pred
                
            except Exception as e:
                print(f"Error predicting {nutrient}: {e}")
                row_predictions[nutrient] = None
        
        predictions.append(row_predictions)
    
    # Create results dataframe
    results_df = pd.DataFrame(predictions)
    
    return results_df

# 9. Create and train models for each water level
def create_soil_nutrient_models(output_dir=None):
    """Creates and trains models for soil nutrient prediction."""
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths (relative to the script's location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_0ml_path = os.path.join(script_dir, 'Soil_0ml.csv')
    df_25ml_path = os.path.join(script_dir, 'Soil_25ml.csv')
    df_50ml_path = os.path.join(script_dir, 'Soil_50ml.csv')
    
    # Load the datasets
    df_0ml = load_data(df_0ml_path)
    df_25ml = load_data(df_25ml_path)
    df_50ml = load_data(df_50ml_path)
    
    # Check if any datasets failed to load
    if df_0ml is None or df_25ml is None or df_50ml is None:
        print("Error: Failed to load one or more datasets.")
        return None
    
    # Process each water level
    water_levels = {
        '0ml': df_0ml,
        '25ml': df_25ml,
        '50ml': df_50ml
    }
    
    all_models = {}
    all_results = {}
    all_scalers = {}
    
    for water_level, df in water_levels.items():
        print(f"\n{'='*50}")
        print(f"Processing {water_level} water level data")
        print(f"{'='*50}")
        
        # Prepare data
        data, wavelengths, targets = prepare_data(df, water_level)
        
        if data is None:
            print(f"Skipping {water_level} due to data preparation issues.")
            continue
        
        X = data[wavelengths]
        
        models = {}
        scalers = {}
        results = []
        
        # Train models for each target
        for nutrient_name, target_col in targets.items():
            if target_col not in data.columns:
                print(f"Warning: Target column '{target_col}' not found. Skipping.")
                continue
            
            y = data[target_col]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model with hyperparameter tuning
            model, best_params = train_model(X_train_scaled, y_train, nutrient_name)
            
            # Evaluate model
            metrics = evaluate_model(model, X_test_scaled, y_test, nutrient_name)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, scaler.transform(X), y, 
                cv=5, scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            print(f"Cross-validation RMSE: {cv_rmse:.4f}")
            
            # Save model details
            models[nutrient_name] = model
            scalers[nutrient_name] = scaler
            results.append([
                nutrient_name, 
                metrics['MSE'], 
                metrics['RMSE'],
                metrics['MAE'],
                metrics['R2'], 
                best_params
            ])
            
            # Save model to disk if output_dir is provided
            if output_dir is not None:
                model_path = os.path.join(output_dir, f'{water_level}_{nutrient_name}_model.pkl')
                scaler_path = os.path.join(output_dir, f'{water_level}_{nutrient_name}_scaler.pkl')
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                print(f"Model and scaler saved to {output_dir}")
        
        all_models[water_level] = models
        all_scalers[water_level] = scalers
        all_results[water_level] = results
        
        # Display results table
        if results:
            results_df = pd.DataFrame(
                results, 
                columns=['Nutrient', 'MSE', 'RMSE', 'MAE', 'R2', 'Best Parameters']
            )
            print(f"\nModel Performance Summary for {water_level}:")
            print(results_df[['Nutrient', 'MSE', 'RMSE', 'R2']].to_string(index=False))
        
        # Plot feature importance
        if models:
            plot_feature_importance(models, wavelengths, water_level, output_dir)
    
    return {
        'models': all_models,
        'scalers': all_scalers,
        'results': all_results
    }

# Main execution
if __name__ == "__main__":
    # Create output directory for models and plots
    output_dir = "soil_nutrient_models"
    
    # Train models and save to output directory
    results = create_soil_nutrient_models(output_dir)
    
    if results is not None:
        print("\nModel training complete. Models saved to disk.")
        
        # Analyze spectral frequency importance
        print("\nAnalyzing spectral frequency importance...")
        # Extract wavelengths list
        wavelengths = ['410','435','460','485','510','535','560','585','610',
                      '645','680','705','730','760','810','860','900','940']
        
        # Analyze frequency importance
        importance_df = analyze_spectral_frequencies(results['models'], wavelengths)
        visualize_frequency_importance(importance_df)
        
        print("\n*Practical Applications:*")
        print("  1. Sensor Design: Focus on the top 5 wavelengths for more cost-effective sensor development")
        print("  2. Soil Monitoring: Prioritize the bands which show highest importance for each nutrient")
        print("  3. Soil Moisture Compensation: Adjust models based on soil moisture levels")
        print("  4. Targeted Measurements: Different nutrients show sensitivity to different spectral regions")
        
        # Example of how to use the prediction function
        print("\nExample prediction:")
        print("To predict soil nutrients for new samples, use the following code:")
        print("")
        print("# Load your new spectral data")
        print("new_samples = pd.read_csv('new_samples.csv')")
        print("# Make predictions")
        print("predictions = predict_soil_nutrients(new_samples, water_level='25ml', model_dir='soil_nutrient_models')")
        print("print(predictions)")
        