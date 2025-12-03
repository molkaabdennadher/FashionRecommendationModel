import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données train et test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("=== ANALYSE DES DONNÉES ===")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nColonnes: {train.columns.tolist()}")

# Les valeurs négatives sont conservées (elles sont intentionnelles)
print("\n=== STATISTIQUES DES DONNÉES ===")
numeric_cols = ['weight_kg', 'height_cm', 'bust_size_num', 'size_num']
for col in numeric_cols:
    print(f"{col}: min={train[col].min():.4f}, max={train[col].max():.4f}")

# Séparer features et target
# Supposons que 'rating' est votre variable cible
X_train = train.drop(['rating'], axis=1)
y_train = train['rating']
X_test = test.drop(['rating'], axis=1)
y_test = test['rating']

print(f"\n=== APRÈS NETTOYAGE ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Vérifier si les colonnes encodées existent déjà
encoded_cols = ['fit_encoded', 'garment_group_name_encoded',
                'duct_type_name_encoded', 'shape_encoded']

if all(col in X_train.columns for col in encoded_cols):
    print("\n✓ Encodage détecté - utilisation des colonnes encodées")
    # Sélectionner uniquement les colonnes numériques et encodées
    feature_cols = numeric_cols + encoded_cols
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]
else:
    print("\n✗ Encodage non trouvé - vérifiez vos colonnes")

# Gérer les valeurs manquantes
print(f"\n=== VALEURS MANQUANTES ===")
print(f"Train: {X_train.isnull().sum().sum()}")
print(f"Test: {X_test.isnull().sum().sum()}")

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Entraîner le modèle
print("\n=== ENTRAÎNEMENT DU MODÈLE ===")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print("\n=== RÉSULTATS ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\n✓ Matrice de confusion sauvegardée: confusion_matrix.png")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== IMPORTANCE DES FEATURES ===")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Importance des Variables')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\n✓ Feature importance sauvegardée: feature_importance.png")