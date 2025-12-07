"""
COMPREHENSIVE CLUSTERING ANALYSIS SCRIPT FOR MODCLOTH FASHION DATASET

This script performs an extensive unsupervised learning analysis using three different clustering algorithms:
1. K-Means Clustering - Partition-based algorithm that divides data into K clusters
2. Hierarchical Agglomerative Clustering (HAC) - Bottom-up hierarchical clustering approach
3. DBSCAN - Density-based clustering that can identify arbitrary-shaped clusters and outliers

The analysis includes:
- Complete data preprocessing and feature engineering
- Exploratory Data Analysis (EDA) with comprehensive visualizations
- Optimal cluster number determination using multiple metrics
- Comparative performance evaluation of all three algorithms
- Automated selection of the best algorithm based on normalized global scores

Author: Data Science Team
Date: December 2024
Dataset: ModCloth Fashion Rental Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
import re

# Suppress warnings for cleaner output during execution
warnings.filterwarnings('ignore')

# Configure matplotlib style for consistent, professional-looking visualizations
# seaborn-v0_8-darkgrid provides a clean grid background for better readability
plt.style.use('seaborn-v0_8-darkgrid')
# husl color palette provides visually distinct colors for categorical data
sns.set_palette("husl")


# ==================== DATA CONVERSION FUNCTIONS ====================
# These functions handle the conversion of raw string data into numerical format
# necessary for machine learning algorithms

def convert_weight_to_kg(weight_str):
    """
    Convert weight from pounds (lbs) to kilograms (kg).

    This function extracts numerical values from strings like '137lbs' and converts
    them to the metric system for standardization. Weight in kg allows for better
    international comparison and is more commonly used in scientific analyses.

    Args:
        weight_str: String containing weight in pounds format (e.g., '137lbs', '145')

    Returns:
        float: Weight converted to kilograms, or np.nan if conversion fails

    Conversion factor: 1 pound = 0.453592 kilograms
    """
    if pd.isna(weight_str):
        return np.nan
    weight_str = str(weight_str).lower()
    match = re.search(r'(\d+)', weight_str)
    if match:
        lbs = float(match.group(1))
        return lbs * 0.453592  # Standard conversion factor from lbs to kg
    return np.nan


def convert_height_to_cm(height_str):
    """
    Convert height from feet-inches format to centimeters.

    This function parses strings like "5' 8\"" (5 feet 8 inches) and converts them
    to centimeters for standardization. The metric system is preferred for consistency
    in machine learning models and international applicability.

    Args:
        height_str: String containing height in feet-inches format (e.g., "5' 8\"", "5'10")

    Returns:
        float: Height converted to centimeters, or np.nan if conversion fails

    Conversion process:
    1. Extract feet and inches using regex pattern
    2. Convert to total inches: (feet * 12) + inches
    3. Convert to centimeters: inches * 2.54
    """
    if pd.isna(height_str):
        return np.nan
    height_str = str(height_str)
    match = re.search(r"(\d+)'?\s*(\d+)", height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        total_inches = feet * 12 + inches
        return total_inches * 2.54  # Convert inches to centimeters
    return np.nan


def convert_bust_size(bust_str):
    """
    Extract numerical bust size from bra size notation.

    This function extracts the band size (numerical part) from bust size strings
    like '34D' or '36C'. The band size represents the chest circumference in inches
    and is useful for clustering similar body types together.

    Args:
        bust_str: String containing bust size (e.g., '34d', '36C', '32DD')

    Returns:
        float: Numerical bust band size, or np.nan if extraction fails

    Note: This function only extracts the band size (number) and ignores the cup size (letter)
    as the numerical component is more relevant for clustering purposes.
    """
    if pd.isna(bust_str):
        return np.nan
    bust_str = str(bust_str).lower()
    match = re.search(r'(\d+)', bust_str)
    if match:
        return float(match.group(1))
    return np.nan


# ==================== DATA LOADING AND INITIAL EXPLORATION ====================
"""
DATA LOADING PHASE:
This section loads the ModCloth dataset from an Excel file and performs initial
exploratory analysis to understand the data structure, types, and quality.

The dataset contains fashion rental information including customer body measurements,
clothing fit ratings, and rental details. Understanding these characteristics is crucial
for effective clustering analysis.
"""
print("=" * 80)
print("DATA LOADING AND PREPARATION")
print("=" * 80)

# Load the dataset from Excel file
# Specify the full file path to ensure correct data loading
file_path = r"C:\Users\pc msi\Downloads\archive\modcloth_dataaaaaaaaa_clean.xlsx"
df = pd.read_excel(file_path)

# Display basic dataset information for initial assessment
print(f"\nDataset dimensions: {df.shape}")
print(f"  - Number of observations (rows): {df.shape[0]}")
print(f"  - Number of features (columns): {df.shape[1]}")

# Show first few rows to understand data structure and format
print(f"\nData preview (first 5 rows):")
print(df.head())

# Inspect data types to identify numerical vs categorical features
# This helps determine which features need encoding or transformation
print(f"\nData types for each column:")
print(df.dtypes)

# Check for missing values which require handling before clustering
# Missing data can significantly impact clustering algorithm performance
print(f"\nMissing values count per column:")
print(df.isnull().sum())

# ==================== FEATURE ENGINEERING: VARIABLE CONVERSION ====================
"""
FEATURE ENGINEERING PHASE:
This section converts raw string-based measurements into numerical values suitable
for machine learning algorithms. Clustering algorithms require numerical inputs,
so all measurements must be standardized and converted to appropriate units.

Key transformations:
1. Weight: pounds (lbs) → kilograms (kg)
2. Height: feet-inches → centimeters (cm)
3. Bust size: alphanumeric (e.g., '34D') → numerical (34)

These conversions ensure consistency, enable mathematical operations, and improve
the interpretability of clustering results across different measurement systems.
"""
print("\n" + "=" * 80)
print("FEATURE ENGINEERING: CONVERTING VARIABLES TO NUMERICAL FORMAT")
print("=" * 80)

# Create a copy of the dataframe to preserve original data
# This allows us to compare original and processed data if needed
df_encoded = df.copy()

# Convert weight from pounds to kilograms
# Kilograms is the international standard unit and provides better comparability
if 'weight' in df_encoded.columns:
    print("\nConverting 'weight' from pounds (lbs) to kilograms (kg)...")
    df_encoded['weight_kg'] = df_encoded['weight'].apply(convert_weight_to_kg)
    print(f"  Original examples: {df['weight'].head().tolist()}")
    print(f"  Converted to kg: {df_encoded['weight_kg'].head().tolist()}")
    print(f"  Conversion successful: {df_encoded['weight_kg'].notna().sum()} values converted")

# Convert height from feet-inches to centimeters
# Centimeters provide finer granularity and international compatibility
if 'height' in df_encoded.columns:
    print("\nConverting 'height' from feet-inches to centimeters (cm)...")
    df_encoded['height_cm'] = df_encoded['height'].apply(convert_height_to_cm)
    print(f"  Original examples: {df['height'].head().tolist()}")
    print(f"  Converted to cm: {df_encoded['height_cm'].head().tolist()}")
    print(f"  Conversion successful: {df_encoded['height_cm'].notna().sum()} values converted")

# Extract numerical bust size from alphanumeric format
# The band size (number) represents chest circumference and is key for body type clustering
if 'bust size' in df_encoded.columns:
    print("\nExtracting numerical 'bust size' (band measurement)...")
    df_encoded['bust_size_num'] = df_encoded['bust size'].apply(convert_bust_size)
    print(f"  Original examples: {df['bust size'].head().tolist()}")
    print(f"  Extracted numerical values: {df_encoded['bust_size_num'].head().tolist()}")
    print(f"  Extraction successful: {df_encoded['bust_size_num'].notna().sum()} values extracted")

# ==================== CATEGORICAL VARIABLE ENCODING ====================
"""
CATEGORICAL ENCODING PHASE:
Machine learning algorithms require numerical inputs, but many features in our dataset
are categorical (text-based). This section uses Label Encoding to convert categorical
variables into numerical format while preserving their distinct categories.

Label Encoding Process:
- Each unique category is assigned a unique integer (0, 1, 2, ...)
- For example: 'fit' might have values: small=0, fit=1, large=2
- This maintains the categorical nature while enabling mathematical operations

Key categorical features being encoded:
1. fit: How the garment fits (small, fit, large)
2. body type: Customer's body shape (pear, hourglass, athletic, etc.)
3. category: Clothing category (dresses, tops, bottoms, etc.)
4. rented for: Occasion for rental (wedding, party, everyday, etc.)

Note: Label encoding assumes ordinal relationships. For nominal data without
natural order, one-hot encoding might be preferred, but label encoding is used
here for dimensionality efficiency in clustering.
"""
print("\n" + "=" * 80)
print("CATEGORICAL VARIABLE ENCODING")
print("=" * 80)

# Define which columns need encoding
# Exclude already converted numerical columns and free-text review columns
cols_to_encode = ['fit', 'body type', 'category', 'rented for']
categorical_cols = [col for col in cols_to_encode if col in df_encoded.columns]

print(f"\nCategorical columns selected for encoding: {categorical_cols}")
print("These text-based features will be converted to numerical format for clustering.")

# Apply Label Encoding to each categorical column
# Store encoders in a dictionary for potential reverse transformation later
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Create new column with '_encoded' suffix to preserve original data
    df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

    # Display encoding information for transparency and validation
    print(f"\n{col}:")
    print(f"  Number of unique categories: {df_encoded[col].nunique()}")
    unique_values = df_encoded[col].unique()[:5]  # Show first 5 examples
    print(f"  Example categories: {list(unique_values)}")
    # Show the encoding mapping for the first few values
    print(
        f"  Encoding example: {dict(zip(unique_values[:3], df_encoded[df_encoded[col].isin(unique_values[:3])][f'{col}_encoded'].unique()[:3]))}")
    print(f"  Encoded column created: '{col}_encoded'")

# ==================== STATISTICAL PROFILING ====================
"""
STATISTICAL PROFILING PHASE:
Before applying clustering algorithms, it's essential to understand the statistical
properties of our features. This section provides descriptive statistics and
correlation analysis to identify:

1. Distribution characteristics (mean, median, std deviation, min/max)
2. Potential outliers or data quality issues
3. Relationships between variables (correlations)
4. Feature scales and ranges (important for distance-based algorithms)

Understanding these properties helps in:
- Identifying features that may need scaling or transformation
- Detecting multicollinearity (highly correlated features)
- Making informed decisions about feature selection
- Interpreting clustering results in context
"""
print("\n" + "=" * 80)
print("STATISTICAL PROFILING")
print("=" * 80)

# Generate descriptive statistics for all numerical features
# This includes count, mean, std, min, 25%, 50%, 75%, max
print("\nDescriptive statistics for numerical variables:")
print("=" * 60)
numeric_features = ['weight_kg', 'height_cm', 'bust_size_num', 'size', 'age', 'rating']
available_numeric = [col for col in numeric_features if col in df_encoded.columns]
print(df_encoded[available_numeric].describe())
print("\nKey insights from descriptive statistics:")
print("  - Check for unusual min/max values that might indicate data quality issues")
print("  - Large standard deviations suggest high variability in that feature")
print("  - Compare mean vs median to assess skewness in distributions")

# Calculate correlation matrix to understand feature relationships
# Correlations range from -1 (perfect negative) to +1 (perfect positive)
# High correlations (>0.7 or <-0.7) may indicate redundant features
print("\n" + "=" * 60)
print("Correlation matrix for key variables:")
print("=" * 60)
if len(available_numeric) > 1:
    corr_data = df_encoded[available_numeric].dropna()
    if len(corr_data) > 0:
        correlation_matrix = corr_data.corr()
        print(correlation_matrix)
        print("\nInterpretation guide:")
        print("  - Values close to +1: Strong positive correlation")
        print("  - Values close to -1: Strong negative correlation")
        print("  - Values close to 0: Little to no linear correlation")
        print("  - High correlations may indicate feature redundancy")

# ==================== EXPLORATORY DATA ANALYSIS (EDA) ====================
"""
EXPLORATORY DATA ANALYSIS (EDA) PHASE:
Visual exploration is crucial for understanding data patterns, distributions, and
relationships before applying machine learning algorithms. This comprehensive EDA
includes multiple visualization types to reveal insights from different perspectives.

Visualization Strategy:
1. Univariate Analysis: Distribution of individual features (histograms)
2. Bivariate Analysis: Relationships between two variables (scatter plots)
3. Multivariate Analysis: Patterns across multiple features (box plots by category)
4. Correlation Visualization: Heatmap showing feature relationships

These visualizations help identify:
- Normal vs skewed distributions
- Outliers and anomalies
- Natural groupings in the data
- Feature importance for clustering
- Data quality issues requiring attention
"""
print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# ========== FIGURE 1: DISTRIBUTION ANALYSIS ==========
"""
Distribution histograms reveal the shape and spread of each numerical feature.
Understanding distributions is crucial because:
- Skewed distributions may need transformation
- Outliers can significantly impact clustering results
- Distribution shapes suggest appropriate clustering approaches
- Bimodal/multimodal distributions hint at natural clusters

Each histogram shows:
- X-axis: Feature values (range of the variable)
- Y-axis: Frequency (number of observations)
- Bins: 30 intervals for detailed distribution visualization
"""
fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
fig1.suptitle('Distribution of Key Numerical Variables', fontsize=16, fontweight='bold')

variables_to_plot = ['weight_kg', 'height_cm', 'bust_size_num', 'size', 'age', 'rating']
for idx, var in enumerate(variables_to_plot):
    if var in df_encoded.columns:
        ax = axes[idx // 3, idx % 3]
        # Create histogram with 30 bins for detailed distribution view
        df_encoded[var].dropna().hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {var}', fontweight='bold')
        ax.set_xlabel(var)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # Add statistical information as text on the plot
        mean_val = df_encoded[var].mean()
        median_val = df_encoded[var].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        ax.legend()

plt.tight_layout()
plt.savefig('01_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Distribution plots saved: 01_distributions.png")
print("  These histograms show the frequency distribution of each numerical variable")
print("  Look for: skewness, outliers, bimodal distributions, and data quality issues")
plt.close()

# ========== FIGURE 2: BOX PLOTS BY FIT CATEGORY ==========
"""
Box plots by FIT category reveal how body measurements vary across different fit ratings.
This analysis is particularly important because:
- Fit ratings (small, fit, large) are subjective assessments of garment sizing
- Comparing measurements across fit categories reveals sizing patterns
- Outliers in box plots indicate unusual body-garment combinations
- Median differences suggest systematic size variations

Box plot components:
- Box: Interquartile range (IQR) containing middle 50% of data
- Line in box: Median (50th percentile)
- Whiskers: Extend to 1.5 × IQR or min/max data point
- Dots: Outliers beyond whiskers
"""
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle('Variable Comparison Across FIT Categories', fontsize=16, fontweight='bold')

if 'fit' in df_encoded.columns:
    for idx, var in enumerate(['weight_kg', 'height_cm', 'bust_size_num', 'size', 'age', 'rating']):
        if var in df_encoded.columns:
            ax = axes[idx // 3, idx % 3]
            df_encoded.boxplot(column=var, by='fit', ax=ax)
            ax.set_title(f'{var} by Fit Category')
            ax.set_xlabel('Fit Rating')
            ax.set_ylabel(var)
            plt.sca(ax)
            plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('02_boxplots_by_fit.png', dpi=300, bbox_inches='tight')
print("✓ Box plots by FIT saved: 02_boxplots_by_fit.png")
print("  These box plots show how measurements differ across fit categories")
print("  Useful for: understanding size-fit relationships and identifying outliers")
plt.close()

# ========== FIGURE 3: BOX PLOTS BY BODY TYPE ==========
"""
Box plots by BODY TYPE reveal physical measurement patterns across different body shapes.
Body type classification (pear, hourglass, athletic, etc.) provides insight into:
- Natural groupings in the customer population
- How body measurements cluster by shape category
- Potential features that distinguish body types
- Expected variation within each body type category

This analysis validates whether body type categories align with actual measurements
and helps identify which features are most distinctive for each body shape.
"""
fig3, axes = plt.subplots(2, 3, figsize=(18, 12))
fig3.suptitle('Variable Comparison Across BODY TYPES', fontsize=16, fontweight='bold')

if 'body type' in df_encoded.columns:
    for idx, var in enumerate(['weight_kg', 'height_cm', 'bust_size_num', 'size', 'age', 'rating']):
        if var in df_encoded.columns:
            ax = axes[idx // 3, idx % 3]
            df_encoded.boxplot(column=var, by='body type', ax=ax)
            ax.set_title(f'{var} by Body Type')
            ax.set_xlabel('Body Type Category')
            ax.set_ylabel(var)
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('03_boxplots_by_bodytype.png', dpi=300, bbox_inches='tight')
print("✓ Box plots by BODY TYPE saved: 03_boxplots_by_bodytype.png")
print("  These visualizations show measurement distributions across body shape categories")
print("  Useful for: validating body type classifications and feature importance")
plt.close()

# ========== FIGURE 4: BOX PLOTS BY CLOTHING CATEGORY ==========
"""
Box plots by CATEGORY examine how measurements vary across different clothing types.
Different garment categories (dresses, tops, bottoms) may be preferred by customers
with different body characteristics. This analysis reveals:
- Whether certain body types prefer specific clothing categories
- Size ranges typical for each garment category
- Age demographics for different clothing types
- Rating patterns across categories

Understanding these patterns helps in customer segmentation and inventory planning.
"""
fig4, axes = plt.subplots(2, 3, figsize=(18, 12))
fig4.suptitle('Variable Comparison Across CLOTHING CATEGORIES', fontsize=16, fontweight='bold')

if 'category' in df_encoded.columns:
    for idx, var in enumerate(['weight_kg', 'height_cm', 'bust_size_num', 'size', 'age', 'rating']):
        if var in df_encoded.columns:
            ax = axes[idx // 3, idx % 3]
            df_encoded.boxplot(column=var, by='category', ax=ax)
            ax.set_title(f'{var} by Clothing Category')
            ax.set_xlabel('Garment Category')
            ax.set_ylabel(var)
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('04_boxplots_by_category.png', dpi=300, bbox_inches='tight')
print("✓ Box plots by CATEGORY saved: 04_boxplots_by_category.png")
print("  These plots reveal customer characteristic patterns across garment types")
print("  Useful for: product-customer matching and inventory optimization")
plt.close()

# ========== FIGURE 5: CORRELATION HEATMAP ==========
"""
Correlation heatmap provides a comprehensive view of linear relationships between all
numerical and encoded categorical variables. This visualization is critical for:

1. Feature Selection: Highly correlated features may be redundant
2. Multicollinearity Detection: Strong correlations can impact some algorithms
3. Feature Relationships: Understanding which variables move together
4. Clustering Insights: Correlated features form natural groupings

Color Interpretation:
- Red/Warm colors: Positive correlation (variables increase together)
- Blue/Cool colors: Negative correlation (one increases, other decreases)
- White/Neutral: No correlation (variables are independent)

Correlation coefficient ranges:
- 0.7 to 1.0: Strong positive correlation
- 0.3 to 0.7: Moderate positive correlation
- -0.3 to 0.3: Weak or no correlation
- -0.7 to -0.3: Moderate negative correlation
- -1.0 to -0.7: Strong negative correlation
"""
fig5, ax = plt.subplots(figsize=(14, 10))
# Combine numerical features with encoded categorical features
corr_cols = ['weight_kg', 'height_cm', 'bust_size_num', 'size', 'age', 'rating']
corr_cols += [col for col in df_encoded.columns if '_encoded' in col]
corr_cols = [col for col in corr_cols if col in df_encoded.columns]

# Calculate correlation matrix
correlation_matrix = df_encoded[corr_cols].corr()

# Create heatmap with annotations
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Heatmap: Feature Relationships', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved: 05_correlation_heatmap.png")
print("  This heatmap shows linear correlations between all numerical features")
print("  Values range from -1 (negative) to +1 (positive correlation)")
print("  Use this to identify: redundant features, feature dependencies, clustering patterns")
plt.close()

# ========== FIGURE 6: SCATTER PLOT RELATIONSHIPS ==========
"""
Scatter plots reveal non-linear relationships and clustering patterns that correlation
coefficients might miss. Each plot shows the relationship between two continuous variables,
colored by a categorical variable to reveal group patterns.

Why scatter plots are essential:
1. Visual Cluster Detection: Natural groupings appear as distinct point clouds
2. Outlier Identification: Unusual data points are immediately visible
3. Non-linear Relationships: Curves and patterns not captured by correlation
4. Category Separation: How well categories separate in feature space

Each scatter plot is colored by a different categorical variable to explore:
- How fit ratings relate to body measurements
- How body types cluster in measurement space
- Whether clothing categories have distinct customer profiles
"""
fig6, axes = plt.subplots(2, 2, figsize=(16, 12))
fig6.suptitle('Bivariate Relationships Between Key Features', fontsize=16, fontweight='bold')

# Define feature pairs and their coloring variables for multi-perspective analysis
scatter_pairs = [
    ('height_cm', 'weight_kg', 'fit'),  # Height vs Weight by Fit
    ('bust_size_num', 'weight_kg', 'fit'),  # Bust vs Weight by Fit
    ('size', 'weight_kg', 'body type'),  # Size vs Weight by Body Type
    ('height_cm', 'bust_size_num', 'category')  # Height vs Bust by Category
]

for idx, (x_var, y_var, hue_var) in enumerate(scatter_pairs):
    if all(var in df_encoded.columns for var in [x_var, y_var, hue_var]):
        ax = axes[idx // 2, idx % 2]
        plot_data = df_encoded[[x_var, y_var, hue_var]].dropna()

        # Limit to 10 categories for visual clarity
        for category in plot_data[hue_var].unique()[:10]:
            mask = plot_data[hue_var] == category
            ax.scatter(plot_data[mask][x_var], plot_data[mask][y_var],
                       label=str(category), alpha=0.6, s=50)

        ax.set_xlabel(x_var, fontweight='bold')
        ax.set_ylabel(y_var, fontweight='bold')
        ax.set_title(f'{y_var} vs {x_var} (colored by {hue_var})')
        ax.legend(title=hue_var, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_scatter_relationships.png', dpi=300, bbox_inches='tight')
print("✓ Scatter plots saved: 06_scatter_relationships.png")
print("  These scatter plots reveal relationships between pairs of variables")
print("  Look for: natural clusters, outliers, linear/non-linear patterns, category separation")
plt.close()

# ==================== DATA PREPARATION FOR CLUSTERING ====================
"""
CLUSTERING DATA PREPARATION PHASE:
This critical step prepares the data for clustering algorithms by:

1. Feature Selection: Choosing relevant variables for clustering
   - Include body measurements (continuous variables)
   - Include encoded categorical variables (discrete variables)
   - Exclude text fields and identifiers

2. Missing Value Handling: Algorithms cannot process NaN values
   - Strategy: Median imputation (robust to outliers)
   - Alternative strategies: mean, mode, or sophisticated imputation

3. Feature Scaling: Essential for distance-based algorithms
   - StandardScaler: Transforms features to mean=0, std=1
   - Why needed: Features with larger scales dominate distance calculations
   - Example: Weight (50-100 kg) would dominate height (150-180 cm) without scaling

Feature Selection Rationale:
- Body measurements capture physical characteristics
- Encoded categorical variables capture behavioral patterns
- Combined features provide comprehensive customer profiles
"""
print("\n" + "=" * 80)
print("DATA PREPARATION FOR CLUSTERING ALGORITHMS")
print("=" * 80)

# Select features for clustering analysis
# Combine continuous and categorical (encoded) features
clustering_features = []

# Add continuous numerical features
for col in ['weight_kg', 'height_cm', 'bust_size_num', 'size', 'age', 'rating']:
    if col in df_encoded.columns:
        clustering_features.append(col)

# Add encoded categorical features
for col in ['fit_encoded', 'body type_encoded', 'category_encoded']:
    if col in df_encoded.columns:
        clustering_features.append(col)

print(f"\nSelected features for clustering ({len(clustering_features)} total):")
print(f"  {clustering_features}")
print("\nFeature types included:")
print("  - Body measurements: weight_kg, height_cm, bust_size_num")
print("  - Sizing information: size")
print("  - Demographics: age")
print("  - Behavioral: rating, fit preferences, category choices")

# Create clustering dataset with selected features only
df_clustering = df_encoded[clustering_features].copy()

# Handle missing values before clustering
print(f"\nMissing values BEFORE imputation: {df_clustering.isnull().sum().sum()}")
print("  Strategy: Median imputation (robust to outliers)")
df_clustering = df_clustering.fillna(df_clustering.median())
print(f"Missing values AFTER imputation: {df_clustering.isnull().sum().sum()}")
print("  ✓ All missing values successfully handled")

print(f"\nFinal clustering dataset dimensions: {df_clustering.shape}")
print(f"  - Observations: {df_clustering.shape[0]}")
print(f"  - Features: {df_clustering.shape[1]}")

# Standardize features for distance-based clustering algorithms
# StandardScaler transforms each feature to have mean=0 and standard deviation=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clustering)

print("\n✓ Feature scaling completed using StandardScaler")
print("  Transformation: X_scaled = (X - mean) / std_deviation")
print("  Result: All features now have mean ≈ 0 and std ≈ 1")
print("  Benefits:")
print("    - Equal weight for all features in distance calculations")
print("    - Improved algorithm convergence")
print("    - Better clustering quality")
print(f"\nScaled data shape: {X_scaled.shape}")
print(f"Scaled data type: {type(X_scaled)} (numpy array for efficient computation)")

# ==================== OPTIMAL CLUSTER NUMBER DETERMINATION ====================
"""
DETERMINING OPTIMAL NUMBER OF CLUSTERS (K):
Finding the right number of clusters is crucial for meaningful segmentation.
Too few clusters oversimplify patterns; too many create noise without insight.

We use three complementary metrics to determine optimal K:

1. ELBOW METHOD (Inertia):
   - Measures within-cluster sum of squared distances
   - Look for "elbow" where adding clusters yields diminishing returns
   - Lower is better, but we seek the inflection point

2. SILHOUETTE SCORE:
   - Measures how similar objects are to their own cluster vs other clusters
   - Range: -1 to +1 (higher is better)
   - >0.7: Strong structure, >0.5: Reasonable, <0.25: No substantial structure
   - Considers both cohesion (within-cluster) and separation (between-cluster)

3. DAVIES-BOULDIN INDEX:
   - Ratio of within-cluster to between-cluster distances
   - Lower scores indicate better clustering (0 is perfect)
   - Penalizes clusters that are too close together

Strategy: Test K from 2 to 10 clusters and evaluate all three metrics
to make an informed decision about optimal cluster count.
"""
print("\n" + "=" * 80)
print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 80)

# Initialize lists to store evaluation metrics for each K value
inertias = []  # Within-cluster sum of squares (lower is better at elbow)
silhouette_scores = []  # Cluster separation quality (higher is better)
db_scores = []  # Davies-Bouldin index (lower is better)
K_range = range(2, 11)  # Test from 2 to 10 clusters

print("\nTesting different values of K (number of clusters)...")
print("=" * 70)
print(f"{'K':<5} {'Inertia':<15} {'Silhouette':<15} {'Davies-Bouldin':<15}")
print("=" * 70)

for k in K_range:
    # Apply K-Means with current K value
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Calculate evaluation metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)

    # Store metrics for visualization
    inertias.append(inertia)
    silhouette_scores.append(silhouette)
    db_scores.append(davies_bouldin)

    # Display results for this K value
    print(f"{k:<5} {inertia:<15.2f} {silhouette:<15.3f} {davies_bouldin:<15.3f}")

print("=" * 70)
print("\nMetric Interpretation Guide:")
print("  - Inertia: Look for 'elbow' where curve bends")
print("  - Silhouette: Higher values indicate better-defined clusters")
print("  - Davies-Bouldin: Lower values indicate better separation between clusters")

# ========== FIGURE 7: OPTIMAL CLUSTER VISUALIZATION ==========
"""
Visualizing clustering metrics helps identify the optimal K through pattern recognition.
These three plots provide complementary perspectives on cluster quality:

PLOT 1 - ELBOW METHOD:
- Shows how inertia decreases as K increases
- "Elbow" point indicates diminishing returns from additional clusters
- Steep drops suggest significant improvement; gentle slopes indicate marginal gains

PLOT 2 - SILHOUETTE SCORE:
- Peak indicates best cluster separation and cohesion
- Look for the maximum value
- Consistent high scores suggest stable clustering across K values

PLOT 3 - DAVIES-BOULDIN INDEX:
- Minimum indicates best cluster compactness and separation
- Lower valleys show optimal clustering configurations
- Helps confirm findings from other metrics

Decision Strategy: Choose K where:
1. Elbow curve shows significant bend
2. Silhouette score is maximized
3. Davies-Bouldin score is minimized
4. All three metrics converge on similar K value
"""
fig7, axes = plt.subplots(1, 3, figsize=(18, 5))
fig7.suptitle('Optimal Cluster Number Determination Using Multiple Metrics',
              fontsize=16, fontweight='bold')

# Plot 1: Elbow Method (Inertia)
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontweight='bold')
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontweight='bold')
axes[0].set_title('Elbow Method: Inertia vs K')
axes[0].grid(True, alpha=0.3)
axes[0].annotate('Look for the "elbow"',
                 xy=(K_range[3], inertias[3]),
                 xytext=(K_range[5], inertias[1]),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=10, color='red')

# Plot 2: Silhouette Score (Higher is Better)
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontweight='bold')
axes[1].set_title('Silhouette Score: Higher = Better')
axes[1].grid(True, alpha=0.3)
# Mark the maximum silhouette score
max_silhouette_idx = np.argmax(silhouette_scores)
axes[1].scatter(K_range[max_silhouette_idx], silhouette_scores[max_silhouette_idx],
                s=200, c='red', marker='*', edgecolors='black', linewidths=2,
                label=f'Max at K={K_range[max_silhouette_idx]}')
axes[1].legend()

# Plot 3: Davies-Bouldin Index (Lower is Better)
axes[2].plot(K_range, db_scores, 'ro-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (K)', fontweight='bold')
axes[2].set_ylabel('Davies-Bouldin Index', fontweight='bold')
axes[2].set_title('Davies-Bouldin: Lower = Better')
axes[2].grid(True, alpha=0.3)
# Mark the minimum Davies-Bouldin score
min_db_idx = np.argmin(db_scores)
axes[2].scatter(K_range[min_db_idx], db_scores[min_db_idx],
                s=200, c='green', marker='*', edgecolors='black', linewidths=2,
                label=f'Min at K={K_range[min_db_idx]}')
axes[2].legend()

plt.tight_layout()
plt.savefig('07_optimal_clusters.png', dpi=300, bbox_inches='tight')
print("\n✓ Optimal cluster metrics visualization saved: 07_optimal_clusters.png")
print("  This figure helps determine the best number of clusters")
print("  Analyze all three plots together for informed decision-making")
plt.close()

# Determine optimal K based on silhouette score (most reliable single metric)
optimal_k = K_range[np.argmax(silhouette_scores)]
print("\n" + "=" * 80)
print(f">>> RECOMMENDED OPTIMAL NUMBER OF CLUSTERS: K = {optimal_k}")
print("=" * 80)
print(f"Decision based on maximum Silhouette Score: {max(silhouette_scores):.3f}")
print(f"This K value provides the best balance of:")
print(f"  - Within-cluster cohesion (points close to cluster center)")
print(f"  - Between-cluster separation (distinct, well-separated groups)")
print(f"  - Interpretability (meaningful customer segments)")

# ==================== K-MEANS CLUSTERING IMPLEMENTATION ====================
"""
K-MEANS CLUSTERING ALGORITHM:
K-Means is a partition-based clustering algorithm that divides data into K clusters
by iteratively assigning points to the nearest centroid and updating centroids.

Algorithm Steps:
1. Initialize K random centroids (cluster centers)
2. Assign each point to nearest centroid (forming K clusters)
3. Recalculate centroids as mean of assigned points
4. Repeat steps 2-3 until convergence (centroids stop moving)

Advantages of K-Means:
- Fast and efficient (O(n*k*i) complexity)
- Scales well to large datasets
- Simple to understand and interpret
- Produces spherical, evenly-sized clusters
- Reproducible with fixed random seed

Parameters Used:
- n_clusters: Optimal K determined from previous analysis
- random_state=42: Ensures reproducibility
- n_init=10: Run algorithm 10 times with different initializations, keep best result

This implementation will create customer segments based on body measurements,
demographics, and behavioral patterns for targeted marketing and inventory optimization.
"""
print("\n" + "=" * 80)
print(f"K-MEANS CLUSTERING WITH K={optimal_k} CLUSTERS")
print("=" * 80)

print("\nInitializing K-Means algorithm...")
print(f"  Algorithm: K-Means (partition-based clustering)")
print(f"  Number of clusters: {optimal_k}")
print(f"  Random state: 42 (for reproducibility)")
print(f"  Number of initializations: 10 (best result selected)")

# Apply K-Means clustering with optimal K
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

print(f"\n✓ K-Means clustering completed successfully")
print(f"  Iterations to convergence: {kmeans_final.n_iter_}")
print(f"  Final inertia (within-cluster variance): {kmeans_final.inertia_:.2f}")

# Add cluster assignments to the original dataframe
# Use .loc to ensure proper index alignment
df_encoded['cluster'] = np.nan
df_encoded.loc[df_clustering.index, 'cluster'] = cluster_labels

print(f"\n✓ Cluster labels added to dataframe")
print(f"\nCluster distribution (number of customers per cluster):")
print("=" * 60)
cluster_distribution = df_encoded['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_distribution.items():
    percentage = (count / len(df_encoded['cluster'].dropna())) * 100
    print(f"  Cluster {int(cluster_id)}: {count:,} customers ({percentage:.1f}%)")
print("=" * 60)

# Check for cluster balance
# Balanced clusters are generally preferred for actionable insights
max_cluster = cluster_distribution.max()
min_cluster = cluster_distribution.min()
balance_ratio = max_cluster / min_cluster if min_cluster > 0 else float('inf')
print(f"\nCluster balance analysis:")
print(f"  Largest cluster: {max_cluster:,} customers")
print(f"  Smallest cluster: {min_cluster:,} customers")
print(f"  Balance ratio: {balance_ratio:.2f} (closer to 1.0 is more balanced)")
if balance_ratio < 3:
    print("  ✓ Clusters are reasonably balanced")
else:
    print("  ⚠ Some cluster imbalance detected - may need investigation")

# ==================== CLUSTER PROFILING AND ANALYSIS ====================
"""
CLUSTER PROFILING:
Understanding what distinguishes each cluster is essential for actionable insights.
This analysis creates detailed profiles of each customer segment by examining:

1. Numerical Features: Mean and median values reveal typical characteristics
   - Body measurements (weight, height, bust size)
   - Demographic info (age)
   - Behavioral metrics (size preference, ratings)

2. Categorical Features: Mode (most common value) shows dominant patterns
   - Fit preferences (how garments typically fit)
   - Body type associations
   - Category preferences (clothing types)

Profile Interpretation:
- Clusters with similar measurements but different behaviors suggest style preferences
- Clusters with different measurements indicate distinct body type segments
- Age differences reveal generational preferences
- Rating patterns show satisfaction levels

These profiles enable:
- Targeted marketing campaigns
- Personalized product recommendations
- Inventory optimization by segment
- Customer experience improvements
"""
print("\n" + "=" * 80)
print("DETAILED CLUSTER PROFILING")
print("=" * 80)
print("\nGenerating comprehensive profiles for each customer segment...")
print("Each profile includes statistical summaries and dominant characteristics.\n")

for cluster_id in range(optimal_k):
    cluster_data = df_encoded[df_encoded['cluster'] == cluster_id]
    cluster_size = len(cluster_data)
    cluster_percentage = (cluster_size / len(df_encoded['cluster'].dropna())) * 100

    print("=" * 80)
    print(f"CLUSTER {cluster_id} PROFILE")
    print("=" * 80)
    print(f"Segment Size: {cluster_size:,} customers ({cluster_percentage:.1f}% of total)")
    print(f"{'─' * 80}")

    # Analyze numerical features
    print("\n📊 NUMERICAL CHARACTERISTICS:")
    print(f"{'─' * 80}")
    numerical_vars = ['weight_kg', 'height_cm', 'bust_size_num', 'size', 'age', 'rating']
    for col in numerical_vars:
        if col in cluster_data.columns:
            mean_val = cluster_data[col].mean()
            median_val = cluster_data[col].median()
            std_val = cluster_data[col].std()
            min_val = cluster_data[col].min()
            max_val = cluster_data[col].max()

            if not np.isnan(mean_val):
                print(f"\n  {col.upper().replace('_', ' ')}:")
                print(f"    Mean:   {mean_val:>8.2f}  |  Median: {median_val:>8.2f}")
                print(f"    Std:    {std_val:>8.2f}  |  Range:  {min_val:.2f} - {max_val:.2f}")

    # Analyze categorical features
    print(f"\n{'─' * 80}")
    print("👥 CATEGORICAL PATTERNS:")
    print(f"{'─' * 80}")
    categorical_vars = ['fit', 'body type', 'category', 'rented for']
    for col in categorical_vars:
        if col in cluster_data.columns:
            mode_val = cluster_data[col].mode()
            if len(mode_val) > 0:
                value_counts = cluster_data[col].value_counts()
                top_value = value_counts.index[0]
                top_percentage = (value_counts.iloc[0] / cluster_size) * 100

                print(f"\n  {col.upper()}:")
                print(f"    Dominant: {top_value} ({top_percentage:.1f}% of cluster)")

                # Show top 3 categories if available
                if len(value_counts) > 1:
                    print(f"    Distribution:")
                    for i, (category, count) in enumerate(value_counts.head(3).items()):
                        pct = (count / cluster_size) * 100
                        print(f"      {i + 1}. {category}: {count} ({pct:.1f}%)")

    print(f"\n{'═' * 80}\n")

print("\n✓ Cluster profiling completed")
print("  Use these profiles to understand customer segments and tailor strategies")

# ==================== CLUSTER VISUALIZATION ====================
"""
CLUSTER VISUALIZATION USING PCA:
High-dimensional data (9+ features) cannot be directly visualized. We use Principal
Component Analysis (PCA) to project data into 2D space while preserving maximum variance.

PCA (Principal Component Analysis):
- Transforms correlated features into uncorrelated principal components
- PC1 captures the most variance in the data
- PC2 captures the second-most variance (orthogonal to PC1)
- Together, PC1 and PC2 often capture 40-70% of total variance

Visualization Components:
1. SCATTER PLOT: Shows cluster distribution in 2D PCA space
   - Each point represents one customer
   - Colors distinguish different clusters
   - Centroids marked with red X symbols
   - Cluster separation indicates quality

2. BAR CHART: Shows cluster size distribution
   - Height indicates number of customers
   - Helps identify dominant segments
   - Reveals cluster balance

Interpretation Tips:
- Well-separated clusters suggest distinct customer segments
- Overlapping clusters may share similar characteristics
- Cluster shapes reveal within-cluster homogeneity
- Centroid positions show typical segment characteristics
"""
print("\n" + "=" * 80)
print("CLUSTER VISUALIZATION WITH DIMENSIONALITY REDUCTION")
print("=" * 80)

# Apply PCA to reduce dimensionality from 9+ features to 2 dimensions
print("\nApplying PCA (Principal Component Analysis)...")
print(f"  Original dimensionality: {X_scaled.shape[1]} features")
print(f"  Target dimensionality: 2 components (for visualization)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Calculate variance explained by each principal component
variance_explained = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print(f"\n✓ PCA transformation completed")
print(f"  PC1 explains: {variance_explained[0] * 100:.1f}% of variance")
print(f"  PC2 explains: {variance_explained[1] * 100:.1f}% of variance")
print(f"  Total variance captured: {cumulative_variance[1] * 100:.1f}%")
print(f"\nInterpretation: These 2 components capture {cumulative_variance[1] * 100:.1f}% of")
print(f"                the information from all {X_scaled.shape[1]} original features")

# Create comprehensive visualization
fig8, axes = plt.subplots(1, 2, figsize=(18, 7))
fig8.suptitle(f'K-Means Clustering Visualization: {optimal_k} Customer Segments',
              fontsize=16, fontweight='bold')

# LEFT PLOT: PCA Scatter Plot with Clusters
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                          cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

# Transform and plot cluster centroids
centers_pca = pca.transform(kmeans_final.cluster_centers_)
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1],
                c='red', s=300, marker='X', edgecolors='black', linewidths=2,
                label='Cluster Centroids', zorder=10)

axes[0].set_xlabel(f'Principal Component 1 ({variance_explained[0] * 100:.1f}% variance)',
                   fontweight='bold')
axes[0].set_ylabel(f'Principal Component 2 ({variance_explained[1] * 100:.1f}% variance)',
                   fontweight='bold')
axes[0].set_title('Customer Segments in PCA Space', fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Add colorbar for cluster identification
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('Cluster ID', fontweight='bold')

# RIGHT PLOT: Cluster Size Distribution
cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))
bars = axes[1].bar(cluster_sizes.index, cluster_sizes.values, color=colors,
                   edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontweight='bold')

axes[1].set_xlabel('Cluster ID', fontweight='bold')
axes[1].set_ylabel('Number of Customers', fontweight='bold')
axes[1].set_title('Cluster Size Distribution', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_xticks(range(optimal_k))

plt.tight_layout()
plt.savefig('08_clusters_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Cluster visualization saved: 08_clusters_visualization.png")
print("  Left plot: Spatial distribution of clusters (PCA projection)")
print("  Right plot: Size comparison of customer segments")
plt.close()

# ========== FIGURE 9: CLUSTER COMPARISON BY FEATURE ==========
"""
COMPARATIVE ANALYSIS ACROSS CLUSTERS:
Box plots comparing clusters across all features provide deep insights into what
makes each segment unique. This multi-faceted comparison reveals:

Feature-by-Feature Analysis:
- Median differences: Central tendency of each cluster
- IQR spread: Variability within clusters
- Outlier patterns: Unusual cases in each segment
- Range overlaps: Similarities between clusters

Strategic Insights from This Analysis:
1. Body Measurements: Physical characteristics defining segments
2. Size Preferences: Typical size choices per segment
3. Age Demographics: Generational differences between segments
4. Satisfaction Levels: Rating patterns indicating segment happiness
5. Behavioral Patterns: Encoded categorical preferences

Business Applications:
- Size chart recommendations per segment
- Age-appropriate marketing messages
- Quality improvement focus areas (low-rating segments)
- Inventory allocation by segment characteristics
"""
fig9, axes = plt.subplots(2, 3, figsize=(18, 12))
fig9.suptitle('Feature-by-Feature Cluster Comparison', fontsize=16, fontweight='bold')

feature_vars = ['weight_kg', 'height_cm', 'bust_size_num', 'size', 'age', 'rating']
for idx, var in enumerate(feature_vars):
    if var in df_encoded.columns:
        ax = axes[idx // 3, idx % 3]

        # Create box plot for this feature across all clusters
        df_encoded[df_encoded['cluster'].notna()].boxplot(column=var, by='cluster', ax=ax)

        # Customize appearance
        ax.set_title(f'{var.replace("_", " ").title()} Distribution by Cluster', fontweight='bold')
        ax.set_xlabel('Cluster ID', fontweight='bold')
        ax.set_ylabel(var.replace('_', ' ').title(), fontweight='bold')

        # Add mean line for reference
        cluster_means = df_encoded[df_encoded['cluster'].notna()].groupby('cluster')[var].mean()
        for cluster_id, mean_val in cluster_means.items():
            ax.plot(cluster_id + 1, mean_val, 'r*', markersize=15, label='Mean' if cluster_id == 0 else '')

        if idx == 0:
            ax.legend(loc='best')

plt.tight_layout()
plt.savefig('09_clusters_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Cluster comparison visualization saved: 09_clusters_comparison.png")
print("  These box plots reveal feature distributions within each cluster")
print("  Compare medians, spreads, and outliers across segments")
plt.close()

# ==================== CLUSTERING AVEC CAH (CLASSIFICATION ASCENDANTE HIÉRARCHIQUE) ====================
print("\n" + "=" * 80)
print("CLUSTERING AVEC CAH (CLASSIFICATION ASCENDANTE HIÉRARCHIQUE)")
print("=" * 80)

# Dendrogramme (sur échantillon pour performance)
print("\nCréation du dendrogramme...")
sample_size = min(500, len(X_scaled))  # Limiter à 500 pour la visualisation
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]

fig10, ax = plt.subplots(figsize=(16, 8))
linkage_matrix = linkage(X_sample, method='ward')
dendrogram(linkage_matrix, ax=ax, truncate_mode='lastp', p=30, show_leaf_counts=True)
ax.set_title('Dendrogramme CAH (Classification Ascendante Hiérarchique)', fontsize=16, fontweight='bold')
ax.set_xlabel('Index des échantillons ou (taille du cluster)', fontweight='bold')
ax.set_ylabel('Distance', fontweight='bold')
plt.tight_layout()
plt.savefig('10_cah_dendrogram.png', dpi=300, bbox_inches='tight')
print("✓ Dendrogramme CAH sauvegardé: 10_cah_dendrogram.png")
plt.close()

# Clustering CAH avec le même nombre de clusters
# IMPORTANT: CAH est très gourmand en mémoire, on utilise un échantillon si le dataset est trop grand
print(f"\nClustering CAH avec k={optimal_k} clusters...")

# Vérifier la taille des données et échantillonner si nécessaire
max_samples_cah = 5000  # Limite pour éviter les problèmes de mémoire
if len(X_scaled) > max_samples_cah:
    print(f"  ⚠️ Dataset trop grand ({len(X_scaled)} échantillons) pour CAH complet")
    print(f"  → Utilisation d'un échantillon de {max_samples_cah} observations")

    # Échantillonner de manière stratifiée si possible
    sample_indices_cah = np.random.choice(len(X_scaled), max_samples_cah, replace=False)
    X_scaled_cah = X_scaled[sample_indices_cah]

    # Appliquer CAH sur l'échantillon
    cah_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    cah_labels_sample = cah_model.fit_predict(X_scaled_cah)

    # Prédire les clusters pour tout le dataset en utilisant le plus proche voisin
    print("  → Prédiction des clusters pour l'ensemble du dataset...")
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled_cah, cah_labels_sample)
    cah_labels = knn.predict(X_scaled)
    print(f"  ✓ Prédiction terminée pour {len(X_scaled)} observations")
else:
    # Dataset assez petit, on peut appliquer CAH directement
    cah_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    cah_labels = cah_model.fit_predict(X_scaled)

# Métriques CAH
cah_silhouette = silhouette_score(X_scaled, cah_labels)
cah_davies_bouldin = davies_bouldin_score(X_scaled, cah_labels)
cah_calinski = calinski_harabasz_score(X_scaled, cah_labels)

print(f"\nMétriques CAH:")
print(f"  Silhouette Score: {cah_silhouette:.3f}")
print(f"  Davies-Bouldin Score: {cah_davies_bouldin:.3f}")
print(f"  Calinski-Harabasz Score: {cah_calinski:.2f}")
print(f"\nRépartition des clusters CAH:")
print(pd.Series(cah_labels).value_counts().sort_index())

# Ajouter les clusters CAH au dataframe
df_encoded['cluster_cah'] = np.nan
df_encoded.loc[df_clustering.index, 'cluster_cah'] = cah_labels

# Visualisation CAH
fig11, axes = plt.subplots(1, 2, figsize=(18, 7))
fig11.suptitle(f'Visualisation CAH - {optimal_k} Clusters', fontsize=16, fontweight='bold')

scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cah_labels,
                          cmap='plasma', s=50, alpha=0.6, edgecolors='black')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontweight='bold')
axes[0].set_title('Projection PCA des Clusters CAH')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label='Cluster CAH')

cluster_sizes_cah = pd.Series(cah_labels).value_counts().sort_index()
axes[1].bar(cluster_sizes_cah.index, cluster_sizes_cah.values,
            color=plt.cm.plasma(np.linspace(0, 1, optimal_k)),
            edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Cluster ID', fontweight='bold')
axes[1].set_ylabel('Nombre d\'individus', fontweight='bold')
axes[1].set_title('Taille des Clusters CAH')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('11_cah_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualisation CAH sauvegardée: 11_cah_visualization.png")
plt.close()

# ==================== CLUSTERING AVEC DBSCAN ====================
print("\n" + "=" * 80)
print("CLUSTERING AVEC DBSCAN")
print("=" * 80)

# Test de différents paramètres eps pour DBSCAN
print("\nRecherche des meilleurs paramètres DBSCAN...")
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
best_dbscan_score = -1
best_eps = 0.5
best_min_samples = 5

for eps in eps_values:
    for min_samples in [5, 10, 15]:
        dbscan_temp = DBSCAN(eps=eps, min_samples=min_samples)
        labels_temp = dbscan_temp.fit_predict(X_scaled)
        n_clusters = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
        n_noise = list(labels_temp).count(-1)

        if n_clusters > 1 and n_noise < len(X_scaled) * 0.5:  # Au moins 2 clusters et moins de 50% de bruit
            try:
                score = silhouette_score(X_scaled, labels_temp)
                if score > best_dbscan_score:
                    best_dbscan_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                print(
                    f"  eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} points de bruit, Silhouette={score:.3f}")
            except:
                pass

print(f"\nMeilleurs paramètres DBSCAN: eps={best_eps}, min_samples={best_min_samples}")

# Clustering DBSCAN avec les meilleurs paramètres
dbscan_model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = dbscan_model.fit_predict(X_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_dbscan = list(dbscan_labels).count(-1)

print(f"\nRésultats DBSCAN:")
print(f"  Nombre de clusters: {n_clusters_dbscan}")
print(f"  Nombre de points de bruit: {n_noise_dbscan}")

# Métriques DBSCAN (en excluant le bruit si possible)
if n_clusters_dbscan > 1 and n_noise_dbscan < len(X_scaled):
    try:
        dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
        dbscan_davies_bouldin = davies_bouldin_score(X_scaled, dbscan_labels)
        dbscan_calinski = calinski_harabasz_score(X_scaled, dbscan_labels)

        print(f"\nMétriques DBSCAN:")
        print(f"  Silhouette Score: {dbscan_silhouette:.3f}")
        print(f"  Davies-Bouldin Score: {dbscan_davies_bouldin:.3f}")
        print(f"  Calinski-Harabasz Score: {dbscan_calinski:.2f}")
    except:
        print("  Impossible de calculer les métriques (trop de bruit ou clusters invalides)")
        dbscan_silhouette = -1
        dbscan_davies_bouldin = 999
        dbscan_calinski = 0
else:
    print("  Clustering DBSCAN non optimal (trop peu de clusters ou trop de bruit)")
    dbscan_silhouette = -1
    dbscan_davies_bouldin = 999
    dbscan_calinski = 0

print(f"\nRépartition des clusters DBSCAN:")
print(pd.Series(dbscan_labels).value_counts().sort_index())

# Ajouter les clusters DBSCAN au dataframe
df_encoded['cluster_dbscan'] = np.nan
df_encoded.loc[df_clustering.index, 'cluster_dbscan'] = dbscan_labels

# Visualisation DBSCAN
fig12, axes = plt.subplots(1, 2, figsize=(18, 7))
fig12.suptitle(f'Visualisation DBSCAN - {n_clusters_dbscan} Clusters', fontsize=16, fontweight='bold')

scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels,
                          cmap='tab10', s=50, alpha=0.6, edgecolors='black')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontweight='bold')
axes[0].set_title('Projection PCA des Clusters DBSCAN (bruit = -1)')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label='Cluster DBSCAN')

cluster_sizes_dbscan = pd.Series(dbscan_labels).value_counts().sort_index()
colors = ['red' if idx == -1 else plt.cm.tab10(idx / max(1, n_clusters_dbscan))
          for idx in cluster_sizes_dbscan.index]
axes[1].bar(cluster_sizes_dbscan.index, cluster_sizes_dbscan.values,
            color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Cluster ID (-1 = bruit)', fontweight='bold')
axes[1].set_ylabel('Nombre d\'individus', fontweight='bold')
axes[1].set_title('Taille des Clusters DBSCAN')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('12_dbscan_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualisation DBSCAN sauvegardée: 12_dbscan_visualization.png")
plt.close()

# ==================== COMPREHENSIVE ALGORITHM COMPARISON ====================
"""
MULTI-ALGORITHM COMPARISON FRAMEWORK:
This section provides a rigorous, quantitative comparison of all three clustering
algorithms using multiple evaluation metrics. The goal is to identify which algorithm
produces the most meaningful, well-separated customer segments for this dataset.

Comparison Methodology:
1. Collect performance metrics for all algorithms
2. Create comparative visualizations
3. Calculate normalized global scores
4. Make data-driven recommendation

Why Compare Multiple Algorithms?
- Different algorithms have different strengths
- Dataset characteristics favor certain approaches
- No single "best" algorithm for all cases
- Rigorous comparison ensures confidence in final choice

Evaluation Metrics Explained:

1. SILHOUETTE SCORE (-1 to +1, higher is better):
   - Measures cluster cohesion (how close points are within cluster)
   - Measures cluster separation (how far apart clusters are)
   - >0.7: Strong, well-defined clusters
   - 0.5-0.7: Reasonable structure
   - <0.3: Weak or artificial structure

2. DAVIES-BOULDIN INDEX (0 to ∞, lower is better):
   - Ratio of within-cluster to between-cluster distances
   - Penalizes overlapping clusters
   - Lower values indicate better separation
   - Close to 0: Ideal separation

3. CALINSKI-HARABASZ SCORE (0 to ∞, higher is better):
   - Ratio of between-cluster to within-cluster variance
   - Higher values: Dense, well-separated clusters
   - Considers both compactness and separation
   - Also known as Variance Ratio Criterion
"""
print("\n" + "=" * 80)
print("COMPREHENSIVE THREE-ALGORITHM COMPARISON")
print("=" * 80)

print("\nCollecting performance metrics for all algorithms...")

# Calculate K-Means metrics
kmeans_silhouette = silhouette_score(X_scaled, cluster_labels)
kmeans_davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
kmeans_calinski = calinski_harabasz_score(X_scaled, cluster_labels)

print("✓ K-Means metrics calculated")
print("✓ HAC metrics calculated (from previous section)")
print("✓ DBSCAN metrics calculated (from previous section)")

# Create comprehensive comparison dataframe
comparison_data = {
    'Algorithm': ['K-Means', 'HAC (Hierarchical)', 'DBSCAN'],
    'Silhouette Score': [kmeans_silhouette, cah_silhouette, dbscan_silhouette],
    'Davies-Bouldin Score': [kmeans_davies_bouldin, cah_davies_bouldin, dbscan_davies_bouldin],
    'Calinski-Harabasz Score': [kmeans_calinski, cah_calinski, dbscan_calinski],
    'Number of Clusters': [optimal_k, optimal_k, n_clusters_dbscan],
    'Noise Points': [0, 0, n_noise_dbscan]
}

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "=" * 90)
print("PERFORMANCE COMPARISON TABLE")
print("=" * 90)
print(comparison_df.to_string(index=False))
print("=" * 90)

print("\n📊 Metric Interpretation Guide:")
print("  Silhouette Score: Higher is better (max = 1.0)")
print("    ✓ Best algorithm has highest value")
print("  Davies-Bouldin Index: Lower is better (min = 0.0)")
print("    ✓ Best algorithm has lowest value")
print("  Calinski-Harabasz Score: Higher is better")
print("    ✓ Best algorithm has highest value")
print("  Noise Points: Lower is usually better (indicates fewer outliers)")

# Identify best performer for each metric
print("\n🏆 Best Performer by Metric:")
best_silhouette_idx = np.nanargmax(comparison_df['Silhouette Score'])
best_db_idx = np.nanargmin(comparison_df['Davies-Bouldin Score'])
best_ch_idx = np.nanargmax(comparison_df['Calinski-Harabasz Score'])

print(f"  Silhouette Score:        {comparison_df.iloc[best_silhouette_idx]['Algorithm']}")
print(f"  Davies-Bouldin Index:    {comparison_df.iloc[best_db_idx]['Algorithm']}")
print(f"  Calinski-Harabasz Score: {comparison_df.iloc[best_ch_idx]['Algorithm']}")

# Visualisation comparative
fig13, axes = plt.subplots(2, 2, figsize=(18, 14))
fig13.suptitle('Comparaison des Algorithmes de Clustering', fontsize=18, fontweight='bold')

# 1. Silhouette Score (plus élevé = meilleur)
axes[0, 0].bar(comparison_df['Algorithme'], comparison_df['Silhouette Score'],
               color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
axes[0, 0].set_ylabel('Score', fontweight='bold')
axes[0, 0].set_title('Silhouette Score (↑ meilleur)', fontweight='bold', fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(comparison_df['Silhouette Score']):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# 2. Davies-Bouldin Score (plus bas = meilleur)
axes[0, 1].bar(comparison_df['Algorithme'], comparison_df['Davies-Bouldin Score'],
               color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
axes[0, 1].set_ylabel('Score', fontweight='bold')
axes[0, 1].set_title('Davies-Bouldin Score (↓ meilleur)', fontweight='bold', fontsize=14)
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(comparison_df['Davies-Bouldin Score']):
    axes[0, 1].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')

# 3. Calinski-Harabasz Score (plus élevé = meilleur)
axes[1, 0].bar(comparison_df['Algorithme'], comparison_df['Calinski-Harabasz Score'],
               color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
axes[1, 0].set_ylabel('Score', fontweight='bold')
axes[1, 0].set_title('Calinski-Harabasz Score (↑ meilleur)', fontweight='bold', fontsize=14)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(comparison_df['Calinski-Harabasz Score']):
    axes[1, 0].text(i, v + 50, f'{v:.0f}', ha='center', fontweight='bold')

# 4. Comparaison visuelle des trois clustering
ax4 = axes[1, 1]
scatter1 = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                       cmap='viridis', s=30, alpha=0.3, label='K-Means', marker='o')
scatter2 = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=cah_labels,
                       cmap='plasma', s=20, alpha=0.3, label='CAH', marker='s')
scatter3 = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels,
                       cmap='tab10', s=10, alpha=0.3, label='DBSCAN', marker='^')
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontweight='bold')
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontweight='bold')
ax4.set_title('Superposition des Trois Algorithmes', fontweight='bold', fontsize=14)
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('13_algorithms_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique de comparaison sauvegardé: 13_algorithms_comparison.png")
plt.close()

# ==================== ANALYSE ET RECOMMANDATION ====================
print("\n" + "=" * 80)
print("ANALYSE ET RECOMMANDATION FINALE")
print("=" * 80)


# Calculer un score global pour chaque algorithme
# Normaliser les scores pour la comparaison (entre 0 et 1)
def normalize_score(scores, higher_better=True):
    scores = np.array(scores)
    scores = np.where(scores == -1, np.nan, scores)  # Remplacer -1 par NaN
    scores = np.where(scores == 999, np.nan, scores)  # Remplacer 999 par NaN

    if higher_better:
        return (scores - np.nanmin(scores)) / (np.nanmax(scores) - np.nanmin(scores) + 1e-10)
    else:
        return 1 - (scores - np.nanmin(scores)) / (np.nanmax(scores) - np.nanmin(scores) + 1e-10)


silhouette_norm = normalize_score(comparison_df['Silhouette Score'], higher_better=True)
db_norm = normalize_score(comparison_df['Davies-Bouldin Score'], higher_better=False)
ch_norm = normalize_score(comparison_df['Calinski-Harabasz Score'], higher_better=True)

# Score global (moyenne des scores normalisés)
global_scores = np.nanmean([silhouette_norm, db_norm, ch_norm], axis=0)
comparison_df['Score Global'] = global_scores

print("\nScores globaux (normalisés):")
for i, algo in enumerate(comparison_df['Algorithme']):
    print(f"  {algo}: {global_scores[i]:.3f}")

# Identifier le meilleur algorithme
best_algo_idx = np.nanargmax(global_scores)
best_algo = comparison_df.iloc[best_algo_idx]['Algorithme']

print("\n" + "=" * 80)
print(f"🏆 ALGORITHME RECOMMANDÉ: {best_algo}")
print("=" * 80)

print(f"\nJustification:")
if best_algo == 'K-Means':
    print("  ✓ K-Means présente les meilleures performances globales")
    print("  ✓ Silhouette Score élevé: clusters bien séparés et compacts")
    print("  ✓ Davies-Bouldin bas: faible chevauchement entre clusters")
    print("  ✓ Calinski-Harabasz élevé: variance inter-clusters forte")
    print("  ✓ Algorithme efficace, rapide et scalable")
    print("  ✓ Résultats reproductibles et interprétables")
elif best_algo == 'CAH':
    print("  ✓ CAH (Classification Ascendante Hiérarchique) offre une structure hiérarchique")
    print("  ✓ Permet de visualiser les relations entre clusters via dendrogramme")
    print("  ✓ Ne nécessite pas de spécifier k à l'avance")
elif best_algo == 'DBSCAN':
    print("  ✓ DBSCAN détecte les clusters de forme arbitraire")
    print("  ✓ Identifie automatiquement les points de bruit")
    print("  ✓ Ne nécessite pas de spécifier k à l'avance")

print("\nComparaison détaillée:")
print(f"  K-Means: Score global = {global_scores[0]:.3f}")
print(f"  CAH:     Score global = {global_scores[1]:.3f}")
print(f"  DBSCAN:  Score global = {global_scores[2]:.3f}")

# ==================== EXPORT DES RÉSULTATS ====================
print("\n" + "=" * 80)
print("EXPORT DES RÉSULTATS")
print("=" * 80)

# Sauvegarder le dataset avec les clusters
output_file = 'modcloth_avec_clusters.xlsx'
df_encoded.to_excel(output_file, index=False)
print(f"\n✓ Dataset avec clusters sauvegardé: {output_file}")

# Résumé des clusters
summary = df_encoded[df_encoded['cluster'].notna()].groupby('cluster')[clustering_features].mean()
summary.to_excel('clusters_summary.xlsx')
print("✓ Résumé des clusters sauvegardé: clusters_summary.xlsx")

# Sauvegarder la comparaison des algorithmes
comparison_df.to_excel('algorithms_comparison.xlsx', index=False)
print("✓ Comparaison des algorithmes sauvegardée: algorithms_comparison.xlsx")

print("\n" + "=" * 80)
print("ANALYSE TERMINÉE AVEC SUCCÈS!")
print("=" * 80)
print("\nFichiers générés:")
print("  - 01_distributions.png")
print("  - 02_boxplots_by_fit.png")
print("  - 03_boxplots_by_bodytype.png")
print("  - 04_boxplots_by_category.png")
print("  - 05_correlation_heatmap.png")
print("  - 06_scatter_relationships.png")
print("  - 07_optimal_clusters.png")
print("  - 08_clusters_visualization.png")
print("  - 09_clusters_comparison.png")
print("  - 10_cah_dendrogram.png (NOUVEAU)")
print("  - 11_cah_visualization.png (NOUVEAU)")
print("  - 12_dbscan_visualization.png (NOUVEAU)")
print("  - 13_algorithms_comparison.png (NOUVEAU)")
print("  - modcloth_avec_clusters.xlsx")
print("  - clusters_summary.xlsx")
print("  - algorithms_comparison.xlsx (NOUVEAU)")

print("\n" + "=" * 80)
print(f"🎯 CONCLUSION: {best_algo} est l'algorithme optimal pour ce dataset")
print("=" * 80)