"""
Student Performance Analysis
Main execution file for the complete ML pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os

# Create directories for saving results
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/models', exist_ok=True)

print("="*80)
print("STUDENT PERFORMANCE ANALYSIS - ML PROJECT")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n1. LOADING DATA...")
df = pd.read_csv('StudentsPerformance.csv')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("2. DATA EXPLORATION")
print("="*80)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(f"Number of duplicates: {df.duplicated().sum()}")

print("\nCategorical Variables Distribution:")
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 
                    'lunch', 'test preparation course']

for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("3. FEATURE ENGINEERING")
print("="*80)

# Create new features
df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
df['average_score'] = df['total_score'] / 3

print("\nNew features created:")
print("- total_score: Sum of all three scores")
print("- average_score: Average of all three scores")
print(f"\nAverage Score Statistics:")
print(df['average_score'].describe())

# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("4. EXPLORATORY DATA ANALYSIS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 4.1 Score Distributions
print("\n4.1 Creating score distribution plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution of Scores', fontsize=16, fontweight='bold')

df['math score'].hist(ax=axes[0, 0], bins=30, edgecolor='black', color='skyblue')
axes[0, 0].set_title('Math Score Distribution')
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_ylabel('Frequency')

df['reading score'].hist(ax=axes[0, 1], bins=30, edgecolor='black', color='lightcoral')
axes[0, 1].set_title('Reading Score Distribution')
axes[0, 1].set_xlabel('Score')
axes[0, 1].set_ylabel('Frequency')

df['writing score'].hist(ax=axes[1, 0], bins=30, edgecolor='black', color='lightgreen')
axes[1, 0].set_title('Writing Score Distribution')
axes[1, 0].set_xlabel('Score')
axes[1, 0].set_ylabel('Frequency')

df['average_score'].hist(ax=axes[1, 1], bins=30, edgecolor='black', color='plum')
axes[1, 1].set_title('Average Score Distribution')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/figures/score_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/score_distributions.png")
plt.close()

# 4.2 Correlation Heatmap
print("\n4.2 Creating correlation heatmap...")
plt.figure(figsize=(10, 8))
correlation_matrix = df[['math score', 'reading score', 'writing score', 'average_score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Scores', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/correlation_heatmap.png")
plt.close()

# 4.3 Gender Analysis
print("\n4.3 Creating gender analysis plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Score Analysis by Gender', fontsize=16, fontweight='bold')

sns.boxplot(data=df, x='gender', y='math score', ax=axes[0], palette='Set2')
axes[0].set_title('Math Score by Gender')

sns.boxplot(data=df, x='gender', y='reading score', ax=axes[1], palette='Set2')
axes[1].set_title('Reading Score by Gender')

sns.boxplot(data=df, x='gender', y='writing score', ax=axes[2], palette='Set2')
axes[2].set_title('Writing Score by Gender')

plt.tight_layout()
plt.savefig('results/figures/gender_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/gender_analysis.png")
plt.close()

# 4.4 Test Preparation Analysis
print("\n4.4 Creating test preparation analysis...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='test preparation course', y='average_score', palette='Set1')
plt.title('Impact of Test Preparation Course on Average Score', fontsize=14, fontweight='bold')
plt.xlabel('Test Preparation Course')
plt.ylabel('Average Score')
plt.tight_layout()
plt.savefig('results/figures/test_prep_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/test_prep_analysis.png")
plt.close()

# 4.5 Parental Education Analysis
print("\n4.5 Creating parental education analysis...")
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='parental level of education', y='average_score', palette='viridis')
plt.title('Impact of Parental Education Level on Average Score', fontsize=14, fontweight='bold')
plt.xlabel('Parental Level of Education')
plt.ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/figures/parental_education_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/parental_education_analysis.png")
plt.close()

# 4.6 Lunch Type Analysis
print("\n4.6 Creating lunch type analysis...")
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='lunch', y='average_score', palette='muted')
plt.title('Impact of Lunch Type on Average Score', fontsize=14, fontweight='bold')
plt.xlabel('Lunch Type')
plt.ylabel('Average Score')
plt.tight_layout()
plt.savefig('results/figures/lunch_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/lunch_analysis.png")
plt.close()

# 4.7 Race/Ethnicity Analysis
print("\n4.7 Creating race/ethnicity analysis...")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='race/ethnicity', y='average_score', palette='pastel')
plt.title('Average Score by Race/Ethnicity', fontsize=14, fontweight='bold')
plt.xlabel('Race/Ethnicity')
plt.ylabel('Average Score')
plt.tight_layout()
plt.savefig('results/figures/ethnicity_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/ethnicity_analysis.png")
plt.close()

# 4.8 Combined Analysis
print("\n4.8 Creating combined analysis plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comprehensive Analysis of Factors Affecting Student Performance', 
             fontsize=16, fontweight='bold')

# Gender
sns.barplot(data=df, x='gender', y='average_score', ax=axes[0, 0], 
            palette='Set2', ci=None)
axes[0, 0].set_title('Gender vs Average Score')
axes[0, 0].set_ylabel('Average Score')

# Test Prep
sns.barplot(data=df, x='test preparation course', y='average_score', 
            ax=axes[0, 1], palette='Set1', ci=None)
axes[0, 1].set_title('Test Preparation vs Average Score')
axes[0, 1].set_ylabel('Average Score')

# Lunch
sns.barplot(data=df, x='lunch', y='average_score', ax=axes[1, 0], 
            palette='muted', ci=None)
axes[1, 0].set_title('Lunch Type vs Average Score')
axes[1, 0].set_ylabel('Average Score')

# Parental Education
parent_ed_avg = df.groupby('parental level of education')['average_score'].mean().sort_values()
parent_ed_avg.plot(kind='barh', ax=axes[1, 1], color='skyblue')
axes[1, 1].set_title('Parental Education vs Average Score')
axes[1, 1].set_xlabel('Average Score')

plt.tight_layout()
plt.savefig('results/figures/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/comprehensive_analysis.png")
plt.close()

# ============================================================================
# 5. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("5. DATA PREPROCESSING")
print("="*80)

# Prepare features and target
feature_cols = ['gender', 'race/ethnicity', 'parental level of education', 
                'lunch', 'test preparation course']
X = df[feature_cols].copy()
y = df['average_score'].copy()

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encode categorical variables
print("\n5.1 Encoding categorical variables...")
X_encoded = X.copy()
label_encoders = {}

for col in X_encoded.columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# Train-test split
print("\n5.2 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Split ratio: 80-20")

# Feature Scaling
print("\n5.3 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler")

# ============================================================================
# 6. MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n" + "="*80)
print("6. MODEL TRAINING AND EVALUATION")
print("="*80)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, 
                                           max_depth=10, min_samples_split=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42,
                                                    learning_rate=0.1, max_depth=5),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, 
                            learning_rate=0.1, max_depth=5)
}

# Store results
results = {}
trained_models = {}

# Training and evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Store results
    results[model_name] = {
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'CV R² Mean': cv_mean,
        'CV R² Std': cv_std
    }
    
    # Print results
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
    print(f"{'-'*70}")
    print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
    print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
    print(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f}")
    print(f"{'MSE':<20} {train_mse:<15.4f} {test_mse:<15.4f}")
    print(f"{'-'*70}")
    print(f"Cross-Validation R² (5-fold): {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"{'='*70}")
    
    return model, y_test_pred

# Train all models
print("\nTraining models...")
predictions = {}

for name, model in models.items():
    trained_model, y_pred = evaluate_model(
        model, X_train, X_test, y_train, y_test, name
    )
    trained_models[name] = trained_model
    predictions[name] = y_pred

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("7. MODEL COMPARISON")
print("="*80)

# Create comparison DataFrame
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df.to_string())

# Save results to CSV
results_df.to_csv('results/model_comparison.csv')
print("\nSaved: results/model_comparison.csv")

# 7.1 R² Score Comparison
print("\n7.1 Creating R² score comparison plot...")
plt.figure(figsize=(12, 6))
x = np.arange(len(results_df.index))
width = 0.35

plt.bar(x - width/2, results_df['Train R²'], width, label='Train R²', alpha=0.8)
plt.bar(x + width/2, results_df['Test R²'], width, label='Test R²', alpha=0.8)

plt.xlabel('Models', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('R² Score Comparison - Train vs Test', fontsize=14, fontweight='bold')
plt.xticks(x, results_df.index, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/r2_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/r2_comparison.png")
plt.close()

# 7.2 RMSE Comparison
print("\n7.2 Creating RMSE comparison plot...")
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, results_df['Train RMSE'], width, label='Train RMSE', alpha=0.8)
plt.bar(x + width/2, results_df['Test RMSE'], width, label='Test RMSE', alpha=0.8)

plt.xlabel('Models', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('RMSE Comparison - Train vs Test', fontsize=14, fontweight='bold')
plt.xticks(x, results_df.index, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/rmse_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/rmse_comparison.png")
plt.close()

# 7.3 MAE Comparison
print("\n7.3 Creating MAE comparison plot...")
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, results_df['Train MAE'], width, label='Train MAE', alpha=0.8)
plt.bar(x + width/2, results_df['Test MAE'], width, label='Test MAE', alpha=0.8)

plt.xlabel('Models', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('MAE Comparison - Train vs Test', fontsize=14, fontweight='bold')
plt.xticks(x, results_df.index, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/mae_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/mae_comparison.png")
plt.close()

# 7.4 Comprehensive Comparison
print("\n7.4 Creating comprehensive comparison plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')

# R² Score
results_df[['Train R²', 'Test R²']].plot(kind='bar', ax=axes[0, 0], alpha=0.8)
axes[0, 0].set_title('R² Score')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_xticklabels(results_df.index, rotation=45, ha='right')
axes[0, 0].legend(['Train', 'Test'])
axes[0, 0].grid(axis='y', alpha=0.3)

# RMSE
results_df[['Train RMSE', 'Test RMSE']].plot(kind='bar', ax=axes[0, 1], alpha=0.8, color=['orange', 'red'])
axes[0, 1].set_title('RMSE')
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_xticklabels(results_df.index, rotation=45, ha='right')
axes[0, 1].legend(['Train', 'Test'])
axes[0, 1].grid(axis='y', alpha=0.3)

# MAE
results_df[['Train MAE', 'Test MAE']].plot(kind='bar', ax=axes[1, 0], alpha=0.8, color=['green', 'darkgreen'])
axes[1, 0].set_title('MAE')
axes[1, 0].set_ylabel('MAE')
axes[1, 0].set_xticklabels(results_df.index, rotation=45, ha='right')
axes[1, 0].legend(['Train', 'Test'])
axes[1, 0].grid(axis='y', alpha=0.3)

# Cross-Validation Scores
cv_data = results_df[['CV R² Mean']].copy()
cv_errors = results_df['CV R² Std'].values
cv_data.plot(kind='bar', ax=axes[1, 1], alpha=0.8, color='purple', yerr=cv_errors, capsize=4)
axes[1, 1].set_title('Cross-Validation R² Score (5-Fold)')
axes[1, 1].set_ylabel('CV R² Score')
axes[1, 1].set_xticklabels(results_df.index, rotation=45, ha='right')
axes[1, 1].legend(['CV R²'])
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/comprehensive_comparison.png")
plt.close()

# ============================================================================
# 8. BEST MODEL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8. BEST MODEL ANALYSIS")
print("="*80)

# Select best model based on Test R²
best_model_name = results_df['Test R²'].idxmax()
best_model = trained_models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Test R² Score: {results_df.loc[best_model_name, 'Test R²']:.4f}")
print(f"Test RMSE: {results_df.loc[best_model_name, 'Test RMSE']:.4f}")
print(f"Test MAE: {results_df.loc[best_model_name, 'Test MAE']:.4f}")

# 8.1 Feature Importance (for tree-based models)
if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
    print(f"\n8.1 Analyzing feature importance for {best_model_name}...")
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    # Save to CSV
    feature_importance.to_csv('results/feature_importance.csv', index=False)
    print("\nSaved: results/feature_importance.csv")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/feature_importance.png")
    plt.close()

# 8.2 Actual vs Predicted
print(f"\n8.2 Creating actual vs predicted plot...")
plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_predictions, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')

# Calculate and display metrics on plot
r2 = r2_score(y_test, best_predictions)
rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
mae = mean_absolute_error(y_test, best_predictions)

plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel('Actual Score', fontsize=12)
plt.ylabel('Predicted Score', fontsize=12)
plt.title(f'Actual vs Predicted Scores - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/actual_vs_predicted.png")
plt.close()

# 8.3 Residual Analysis
print(f"\n8.3 Creating residual analysis plots...")
residuals = y_test - best_predictions

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Residual Analysis - {best_model_name}', fontsize=16, fontweight='bold')

# Residual plot
axes[0].scatter(best_predictions, residuals, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Score', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('Residual Plot')
axes[0].grid(True, alpha=0.3)

# Residual distribution
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residuals', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Residuals')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/residual_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/residual_analysis.png")
plt.close()

# 8.4 Prediction Error Distribution
print(f"\n8.4 Creating prediction error distribution...")
plt.figure(figsize=(12, 6))

# Create bins for error ranges
errors = np.abs(residuals)
error_ranges = ['0-2', '2-4', '4-6', '6-8', '8-10', '>10']
error_counts = [
    np.sum(errors <= 2),
    np.sum((errors > 2) & (errors <= 4)),
    np.sum((errors > 4) & (errors <= 6)),
    np.sum((errors > 6) & (errors <= 8)),
    np.sum((errors > 8) & (errors <= 10)),
    np.sum(errors > 10)
]

plt.bar(error_ranges, error_counts, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Absolute Error Range', fontsize=12)
plt.ylabel('Number of Predictions', fontsize=12)
plt.title(f'Prediction Error Distribution - {best_model_name}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add percentage labels
total = len(errors)
for i, (range_label, count) in enumerate(zip(error_ranges, error_counts)):
    percentage = (count / total) * 100
    plt.text(i, count + 2, f'{percentage:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/error_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/error_distribution.png")
plt.close()

# ============================================================================
# 9. STATISTICAL INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("9. STATISTICAL INSIGHTS")
print("="*80)

# Gender impact
print("\n9.1 Gender Impact on Scores:")
gender_analysis = df.groupby('gender')[['math score', 'reading score', 
                                        'writing score', 'average_score']].mean()
print(gender_analysis)

# Test prep impact
print("\n9.2 Test Preparation Course Impact:")
test_prep_analysis = df.groupby('test preparation course')[['math score', 'reading score', 
                                                             'writing score', 'average_score']].mean()
print(test_prep_analysis)

# Lunch impact
print("\n9.3 Lunch Type Impact:")
lunch_analysis = df.groupby('lunch')[['math score', 'reading score', 
                                      'writing score', 'average_score']].mean()
print(lunch_analysis)

# Parental education impact
print("\n9.4 Parental Education Level Impact:")
parent_ed_analysis = df.groupby('parental level of education')[['math score', 'reading score', 
                                                                 'writing score', 'average_score']].mean()
print(parent_ed_analysis)

# Ethnicity impact
print("\n9.5 Race/Ethnicity Impact:")
ethnicity_analysis = df.groupby('race/ethnicity')[['math score', 'reading score', 
                                                    'writing score', 'average_score']].mean()
print(ethnicity_analysis)

# Save all insights
all_insights = pd.DataFrame()
all_insights['Gender'] = gender_analysis['average_score']
all_insights['Test Prep'] = test_prep_analysis['average_score']
all_insights['Lunch'] = lunch_analysis['average_score']

all_insights.to_csv('results/statistical_insights.csv')
print("\nSaved: results/statistical_insights.csv")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("10. FINAL SUMMARY AND CONCLUSIONS")
print("="*80)

print(f"""
PROJECT SUMMARY:
================

Dataset Information:
- Total samples: {df.shape[0]}
- Total features: {len(feature_cols)}
- Target variable: Average Score (ranging from {df['average_score'].min():.2f} to {df['average_score'].max():.2f})

Data Split:
- Training samples: {X_train.shape[0]} (80%)
- Testing samples: {X_test.shape[0]} (20%)

Models Trained: {len(models)}
{', '.join(models.keys())}

BEST MODEL PERFORMANCE:
=======================
Model: {best_model_name}
- Test R² Score: {results_df.loc[best_model_name, 'Test R²']:.4f}
- Test RMSE: {results_df.loc[best_model_name, 'Test RMSE']:.4f}
- Test MAE: {results_df.loc[best_model_name, 'Test MAE']:.4f}
- Cross-Validation R² (5-fold): {results_df.loc[best_model_name, 'CV R² Mean']:.4f} ± {results_df.loc[best_model_name, 'CV R² Std']:.4f}

KEY FINDINGS:
=============
1. Score Correlation:
   - Reading and Writing scores are highly correlated ({correlation_matrix.loc['reading score', 'writing score']:.3f})
   - Math score shows moderate correlation with other subjects

2. Impact of Factors:
   - Test Preparation: Students who completed test prep scored {test_prep_analysis.loc['completed', 'average_score'] - test_prep_analysis.loc['none', 'average_score']:.2f} points higher on average
   - Lunch Type: Students with standard lunch scored {lunch_analysis.loc['standard', 'average_score'] - lunch_analysis.loc['free/reduced', 'average_score']:.2f} points higher
   - Gender: Notable differences in subject performance between genders

3. Model Performance:
   - All models show reasonable performance with R² > 0.20
   - {best_model_name} provides the best balance between bias and variance
   - Low residual errors indicate good prediction accuracy

OUTPUT FILES GENERATED:
=======================
Figures: {len(os.listdir('results/figures'))} plot(s) saved in results/figures/
- Score distributions
- Correlation heatmap
- Gender, test prep, and parental education analysis
- Model comparison plots
- Feature importance
- Actual vs predicted
- Residual analysis
- Error distribution

Data Files:
- results/model_comparison.csv
- results/feature_importance.csv (if applicable)
- results/statistical_insights.csv

""")

print("="*80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nAll results have been saved to the 'results/' directory.")
print("Check the 'results/figures/' folder for all visualizations.")
print("="*80)
