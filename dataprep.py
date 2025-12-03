import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:\dataware\venv\finaledata_enrichi.xlsx"
OUTPUT_FILE = r"C:\dataware\venv\data_prepared_for_ml.xlsx"

print("=" * 80)
print("PIPELINE DE PRÉPARATION DES DONNÉES POUR ML")
print("Modèles ciblés: Réseaux de Neurones & XGBoost")
print("=" * 80)

# ============================================================
# ÉTAPE 1: CHARGEMENT DES DONNÉES
# ============================================================

print("\n[ÉTAPE 1/10] Chargement des données...")

df = pd.read_excel(INPUT_FILE)
print(f"✓ Données chargées: {len(df):,} lignes, {df.shape[1]} colonnes")
print(f"  Mémoire utilisée: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# Sauvegarde des données originales pour comparaison
df_original = df.copy()
# ============================================================
# ÉTAPE 2: SUPPRESSION DES DOUBLONS
# ============================================================

print("\n[ÉTAPE 2/10] Suppression des doublons...")

nb_avant = len(df)
df = df.drop_duplicates()
nb_apres = len(df)
nb_doublons = nb_avant - nb_apres

print(f"✓ Doublons supprimés: {nb_doublons:,}")
print(f"  Lignes restantes: {nb_apres:,} ({nb_apres / nb_avant * 100:.2f}% des données)")

# ============================================================
# ÉTAPE 2b: SUPPRESSION DES COLONNES NON SIGNIFICATIVES
# ============================================================

print("\n[ÉTAPE 2b/10] Suppression des colonnes non significatives...")

# Liste des colonnes à supprimer (date et autres colonnes non significatives)
colonnes_a_supprimer = []

# Identifier les colonnes de date
colonnes_date = []
for col in df.columns:
    if any(mot in col.lower() for mot in ['date', 'time', 'jour', 'month', 'year', 'timestamp']):
        colonnes_date.append(col)

if colonnes_date:
    print(f"  Colonnes de date identifiées: {colonnes_date}")
    colonnes_a_supprimer.extend(colonnes_date)

# Autres colonnes non significatives (ajuster selon votre cas)
colonnes_non_significatives = ['id', 'index', 'unnamed', 'user_id']  # Exemples
for col in df.columns:
    if any(mot in col.lower() for mot in colonnes_non_significatives) and col not in colonnes_a_supprimer:
        colonnes_a_supprimer.append(col)

# Supprimer les colonnes (si elles existent)
colonnes_supprimees = []
for col in colonnes_a_supprimer:
    if col in df.columns:
        df = df.drop(columns=[col])
        colonnes_supprimees.append(col)

if colonnes_supprimees:
    print(f"✓ Colonnes supprimées: {colonnes_supprimees}")
    print(f"  Colonnes restantes: {len(df.columns)}")
else:
    print("✓ Aucune colonne non significative à supprimer")
# ============================================================
# ÉTAPE 3: CRÉATION DE LA VARIABLE CIBLE BINAIRE (rating_binary)
# ============================================================

print("\n[ÉTAPE 3/10] Création de rating_binary (Positif/Négatif)...")

# Vérifier quelle colonne de rating existe
rating_col = None
for col in ['rating', 'Rating', 'review_rating', 'score']:
    if col in df.columns:
        rating_col = col
        break

if rating_col is None:
    print("⚠ Aucune colonne de rating trouvée. Colonnes disponibles:")
    print(df.columns.tolist())
    raise ValueError("Colonne de rating non trouvée!")

print(f"  Colonne de rating utilisée: '{rating_col}'")

# Analyse de la distribution des ratings
print(f"\n  Distribution des ratings originaux:")
print(df[rating_col].value_counts().sort_index())
print(f"\n  Statistiques: min={df[rating_col].min()}, max={df[rating_col].max()}, "
      f"moyenne={df[rating_col].mean():.2f}, médiane={df[rating_col].median()}")

# UTILISER UN SEUIL FIXE DE 6
SEUIL_FIXE = 6
print(f"\n  SEUIL FIXE APPLIQUÉ: {SEUIL_FIXE}")

# Créer rating_binary avec >= 6 pour Positif
df['rating_binary'] = df[rating_col].apply(
    lambda x: 'Positif' if x >= SEUIL_FIXE else 'Négatif'
)

print(f"\n  Règle appliquée: rating ≥ {SEUIL_FIXE} = Positif, rating < {SEUIL_FIXE} = Négatif")
print(f"\n  Distribution de rating_binary:")
distribution = df['rating_binary'].value_counts()
print(distribution)

if 'Positif' in distribution.index and 'Négatif' in distribution.index:
    ratio = distribution['Positif'] / distribution['Négatif']
    print(f"\n  Ratio Positif/Négatif: {ratio:.2f}")
    if ratio < 0.2 or ratio > 5:
        print(f"  ⚠️  Déséquilibre significatif détecté - TOMEK sera appliqué")
else:
    print("\n  ❌ ERREUR: Une seule classe détectée!")
    print(f"  → Le seuil {SEUIL_FIXE} ne permet pas de séparer les données")

    # Essayer avec un seuil plus bas si nécessaire
    SEUIL_FIXE = 5
    print(f"  → Essai avec nouveau seuil: {SEUIL_FIXE}")

    df['rating_binary'] = df[rating_col].apply(
        lambda x: 'Positif' if x >= SEUIL_FIXE else 'Négatif'
    )

    distribution = df['rating_binary'].value_counts()
    print(f"\n  Nouvelle distribution de rating_binary:")
    print(distribution)

    if 'Positif' in distribution.index and 'Négatif' in distribution.index:
        ratio = distribution['Positif'] / distribution['Négatif']
        print(f"\n  Ratio Positif/Négatif: {ratio:.2f}")
    else:
        raise ValueError("Impossible de créer deux classes avec cette distribution de ratings!")

# ============================================================
# ÉTAPE 4: GESTION DES VALEURS MANQUANTES
# ============================================================

print("\n[ÉTAPE 4/10] Gestion des valeurs manquantes...")

# Afficher les colonnes avec des valeurs manquantes
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Colonne': missing.index,
    'Valeurs_manquantes': missing.values,
    'Pourcentage': missing_pct.values
})
missing_df = missing_df[missing_df['Valeurs_manquantes'] > 0].sort_values(
    'Valeurs_manquantes', ascending=False
)

if len(missing_df) > 0:
    print(f"\n  Colonnes avec valeurs manquantes:")
    print(missing_df.to_string(index=False))

    # Stratégie de gestion des valeurs manquantes
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                # Variables numériques: remplir avec la médiane
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Variables catégorielles: remplir avec le mode ou 'Unknown'
                if df[col].mode().empty:
                    df[col].fillna('Unknown', inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"\n✓ Valeurs manquantes traitées")
else:
    print("✓ Aucune valeur manquante détectée")

# ============================================================
# ÉTAPE 5: IDENTIFICATION DES TYPES DE VARIABLES
# ============================================================

print("\n[ÉTAPE 5/10] Identification des types de variables...")

# Séparer les colonnes numériques et catégorielles
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Exclure la variable cible et l'ID
if 'rating_binary' in categorical_cols:
    categorical_cols.remove('rating_binary')
if 'user_id' in categorical_cols:
    categorical_cols.remove('user_id')
if 'user_id' in numeric_cols:
    numeric_cols.remove('user_id')
if rating_col in numeric_cols:
    numeric_cols.remove(rating_col)

print(f"\n  Variables numériques ({len(numeric_cols)}):")
for col in numeric_cols:
    print(f"    - {col}")

print(f"\n  Variables catégorielles ({len(categorical_cols)}):")
for col in categorical_cols:
    print(f"    - {col}")

# ============================================================
# ÉTAPE 6: ENCODAGE DES VARIABLES CATÉGORIELLES (Label Encoding)
# ============================================================

print("\n[ÉTAPE 6/10] Encodage des variables catégorielles (Label Encoding)...")

# Dictionnaire pour stocker les encodeurs (utile pour l'inférence future)
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

        print(f"  ✓ {col}: {len(le.classes_)} catégories uniques → {col}_encoded")

# Encoder également la variable cible
le_target = LabelEncoder()
df['rating_binary_encoded'] = le_target.fit_transform(df['rating_binary'])
label_encoders['rating_binary'] = le_target

print(f"\n✓ {len(categorical_cols)} variables catégorielles encodées")
print(f"  Classes de rating_binary: {le_target.classes_}")

# ============================================================
# ÉTAPE 7: NORMALISATION DES VARIABLES NUMÉRIQUES
# ============================================================

print("\n[ÉTAPE 7/10] Normalisation des variables numériques (StandardScaler)...")

scaler = StandardScaler()

# Créer des colonnes normalisées
numeric_cols_to_scale = [col for col in numeric_cols if col in df.columns]

if numeric_cols_to_scale:
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols_to_scale]),
        columns=[col + '_scaled' for col in numeric_cols_to_scale],
        index=df.index
    )

    df = pd.concat([df, df_scaled], axis=1)

    print(f"✓ {len(numeric_cols_to_scale)} variables numériques normalisées")
    print(f"  Moyenne ≈ 0, Écart-type ≈ 1")
else:
    print("⚠ Aucune variable numérique à normaliser")

# ============================================================
# ÉTAPE 8: MATRICE DE CORRÉLATION
# ============================================================

print("\n[ÉTAPE 8/10] Analyse de corrélation...")

# Créer un DataFrame avec toutes les variables encodées et normalisées
df_for_corr = pd.DataFrame()

# Ajouter les variables numériques normalisées
for col in numeric_cols_to_scale:
    if col + '_scaled' in df.columns:
        df_for_corr[col] = df[col + '_scaled']

# Ajouter les variables encodées
for col in categorical_cols:
    if col + '_encoded' in df.columns:
        df_for_corr[col + '_enc'] = df[col + '_encoded']

# Ajouter la variable cible encodée
df_for_corr['rating_binary'] = df['rating_binary_encoded']

# Calculer la matrice de corrélation
correlation_matrix = df_for_corr.corr()

print(f"✓ Matrice de corrélation calculée ({correlation_matrix.shape[0]}x{correlation_matrix.shape[1]})")

# Corrélations avec la variable cible
target_correlations = correlation_matrix['rating_binary'].drop('rating_binary').sort_values(
    ascending=False)

print(f"\n📊 Top 10 corrélations POSITIVES avec rating_binary:")
print(target_correlations.head(10))

print(f"\n📊 Top 10 corrélations NÉGATIVES avec rating_binary:")
print(target_correlations.tail(10))

# Identifier les features fortement corrélées entre elles (multicolinéarité)
print(f"\n⚠️  Paires de features fortement corrélées (|r| > 0.8):")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Feature_1': correlation_matrix.columns[i],
                'Feature_2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    df_high_corr = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False, key=abs)
    print(df_high_corr.to_string(index=False))
else:
    print("  Aucune corrélation forte détectée (bon signe!)")

# Sauvegarder la matrice de corrélation
correlation_file = r"C:\dataware\venv\correlation_matrix.xlsx"
with pd.ExcelWriter(correlation_file, engine='openpyxl') as writer:
    correlation_matrix.to_excel(writer, sheet_name='Matrice_Correlation')
    target_correlations.to_frame('Correlation').to_excel(writer, sheet_name='Corr_avec_Target')
    if high_corr_pairs:
        df_high_corr.to_excel(writer, sheet_name='Multicolinearite', index=False)

print(f"\n✓ Matrice de corrélation exportée: {correlation_file}")

# Visualisation optionnelle avec matplotlib/seaborn (si disponible)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Figure 1: Heatmap de corrélation complète
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Matrice de Corrélation Complète', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r"C:\dataware\venv\correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Corrélations avec la target
    plt.figure(figsize=(10, 8))
    top_features = pd.concat([target_correlations.head(15), target_correlations.tail(15)]).sort_values()
    colors = ['red' if x < 0 else 'green' for x in top_features.values]
    plt.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features.index, fontsize=9)
    plt.xlabel('Corrélation avec rating_binary', fontsize=12)
    plt.title('Top 30 Features corrélées avec rating_binary', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(r"C:\dataware\venv\correlation_target.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Graphiques de corrélation sauvegardés:")
    print(f"  - correlation_heatmap.png")
    print(f"  - correlation_target.png")

except ImportError:
    print("\n⚠️  matplotlib/seaborn non disponible - graphiques non générés")
    print("   Installez avec: pip install matplotlib seaborn")

# ============================================================
# ÉTAPE 9: PRÉPARATION POUR TOMEK LINKS (équilibrage)
# ============================================================

print("\n[ÉTAPE 9/10] Application de TOMEK Links pour équilibrage...")

# Préparer les features et la target
# Utiliser les colonnes encodées et normalisées
feature_cols = ([col + '_encoded' for col in categorical_cols if col + '_encoded' in df.columns] +
                [col + '_scaled' for col in numeric_cols_to_scale if col + '_scaled' in df.columns])

print(f"\n  Features sélectionnées pour le modèle ({len(feature_cols)}):")
for i, feat in enumerate(feature_cols, 1):
    print(f"    {i}. {feat}")

X = df[feature_cols].values
y = df['rating_binary_encoded'].values

print(f"\n  Distribution avant TOMEK:")
unique, counts = np.unique(y, return_counts=True)
for val, count in zip(unique, counts):
    label = le_target.inverse_transform([val])[0]
    print(f"    {label}: {count:,} ({count / len(y) * 100:.2f}%)")

# Appliquer TOMEK Links
tomek = TomekLinks(sampling_strategy='auto')
X_tomek, y_tomek = tomek.fit_resample(X, y)

print(f"\n  Distribution après TOMEK:")
unique, counts = np.unique(y_tomek, return_counts=True)
for val, count in zip(unique, counts):
    label = le_target.inverse_transform([val])[0]
    print(f"    {label}: {count:,} ({count / len(y_tomek) * 100:.2f}%)")

print(f"\n✓ Échantillons supprimés par TOMEK: {len(X) - len(X_tomek):,}")

# Créer un DataFrame avec les données équilibrées
df_balanced = pd.DataFrame(X_tomek, columns=feature_cols)
df_balanced['rating_binary_encoded'] = y_tomek
df_balanced['rating_binary'] = le_target.inverse_transform(y_tomek)

# ============================================================
# ÉTAPE 10: SPLIT TRAIN/TEST
# ============================================================

print("\n[ÉTAPE 10/10] Split Train/Test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_tomek, y_tomek,
    test_size=0.2,
    random_state=42,
    stratify=y_tomek
)

print(f"✓ Train set: {len(X_train):,} échantillons")
print(f"✓ Test set:  {len(X_test):,} échantillons")

# Distribution dans les sets
print(f"\n  Distribution Train:")
unique, counts = np.unique(y_train, return_counts=True)
for val, count in zip(unique, counts):
    label = le_target.inverse_transform([val])[0]
    print(f"    {label}: {count:,} ({count / len(y_train) * 100:.2f}%)")

print(f"\n  Distribution Test:")
unique, counts = np.unique(y_test, return_counts=True)
for val, count in zip(unique, counts):
    label = le_target.inverse_transform([val])[0]
    print(f"    {label}: {count:,} ({count / len(y_test) * 100:.2f}%)")

# ============================================================
# EXPORTATION DES DONNÉES PRÉPARÉES
# ============================================================

print("\n" + "=" * 80)
print("EXPORTATION DES DONNÉES")
print("=" * 80)

# Créer les DataFrames pour l'export
df_train = pd.DataFrame(X_train, columns=feature_cols)
df_train['rating_binary_encoded'] = y_train
df_train['rating_binary'] = le_target.inverse_transform(y_train)

df_test = pd.DataFrame(X_test, columns=feature_cols)
df_test['rating_binary_encoded'] = y_test
df_test['rating_binary'] = le_target.inverse_transform(y_test)

# Export vers Excel avec plusieurs onglets
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    # Onglet 1: Données complètes équilibrées
    df_balanced.to_excel(writer, sheet_name='Data_Balanced', index=False)

    # Onglet 2: Train set
    df_train.to_excel(writer, sheet_name='Train_Set', index=False)

    # Onglet 3: Test set
    df_test.to_excel(writer, sheet_name='Test_Set', index=False)

    # Onglet 4: Mapping des encodeurs
    encoding_info = []
    for col, encoder in label_encoders.items():
        for idx, classe in enumerate(encoder.classes_):
            encoding_info.append({
                'Colonne': col,
                'Valeur_originale': classe,
                'Valeur_encodée': idx
            })

    df_encoding = pd.DataFrame(encoding_info)
    df_encoding.to_excel(writer, sheet_name='Encodage_Mapping', index=False)

    # Onglet 5: Liste des features
    features_info = pd.DataFrame({
        'Feature': feature_cols,
        'Type': ['Catégorielle encodée' if '_encoded' in f else 'Numérique normalisée'
                 for f in feature_cols]
    })
    features_info.to_excel(writer, sheet_name='Features_Liste', index=False)

    # Onglet 6: Rapport de préparation
    rapport = pd.DataFrame({
        'Étape': [
        'Données originales',
        'Après suppression doublons',
        'Après TOMEK Links',
        'Train set (80%)',
        'Test set (20%)',
        'Nombre de features',
        'Variables catégorielles encodées',
        'Variables numériques normalisées',
        'Seuil rating_binary',
        'Classe positive',
        'Classe négative'
    ],
    'Valeur': [
        f"{len(df_original):,}",
        f"{len(df):,}",
        f"{len(df_balanced):,}",
        f"{len(df_train):,}",
        f"{len(df_test):,}",
        len(feature_cols),
        len(categorical_cols),
        len(numeric_cols_to_scale),
        f">={SEUIL_FIXE}",  # ← CORRECTION ICI
        le_target.classes_[1] if len(le_target.classes_) > 1 else 'N/A',
        le_target.classes_[0]
    ]
})
    rapport.to_excel(writer, sheet_name='Rapport', index=False)

    # Onglet 7: Matrice de corrélation
    correlation_matrix.to_excel(writer, sheet_name='Correlation_Matrix')

    # Onglet 8: Corrélations avec target
    target_correlations.to_frame('Correlation').to_excel(writer, sheet_name='Corr_Target')

print(f"\n✓ Fichier Excel créé: {OUTPUT_FILE}")
print(f"  Contient 8 onglets:")
print(f"    1. Data_Balanced       - Données complètes équilibrées")
print(f"    2. Train_Set           - Ensemble d'entraînement (80%)")
print(f"    3. Test_Set            - Ensemble de test (20%)")
print(f"    4. Encodage_Mapping    - Correspondance valeurs encodées")
print(f"    5. Features_Liste      - Liste des features pour les modèles")
print(f"    6. Rapport             - Résumé de la préparation")
print(f"    7. Correlation_Matrix  - Matrice de corrélation complète")
print(f"    8. Corr_Target         - Corrélations avec rating_binary")

# Sauvegarder également les objets Python pour utilisation directe
import pickle

objects_to_save = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'feature_cols': feature_cols,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'le_target': le_target
}

pickle_file = r"C:\dataware\venv\ml_objects.pkl"
with open(pickle_file, 'wb') as f:
    pickle.dump(objects_to_save, f)

print(f"\n✓ Objets Python sauvegardés: {pickle_file}")

# ============================================================
# RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 80)
print("RÉSUMÉ DE LA PRÉPARATION")
print("=" * 80)
print(f"📊 Données originales:              {len(df_original):,} lignes")
print(f"📊 Après nettoyage:                 {len(df):,} lignes")
print(f"📊 Après équilibrage (TOMEK):       {len(df_balanced):,} lignes")
print(f"📊 Train set:                       {len(df_train):,} lignes")
print(f"📊 Test set:                        {len(df_test):,} lignes")
print(f"\n🔧 Features préparées:              {len(feature_cols)}")
print(f"   - Catégorielles encodées:        {len(categorical_cols)}")
print(f"   - Numériques normalisées:        {len(numeric_cols_to_scale)}")
print(f"\n🎯 Variable cible:                  rating_binary")
print(f"   - Classes: {le_target.classes_}")
print(f"   - Seuil utilisé: >={SEUIL_FIXE}")
print(f"\n✅ Données prêtes pour:")
print(f"   - Réseaux de Neurones (features normalisées)")
print(f"   - XGBoost (features encodées)")
print("=" * 80)

print("\n🚀 Données prêtes pour l'entraînement des modèles!")
print("\n💡 Prochaines étapes:")
print("   1. Charger les données depuis 'Train_Set' et 'Test_Set'")
print("   2. Entraîner vos modèles (NN, XGBoost)")
print("   3. Évaluer les performances sur Test_Set")
print("   4. Utiliser les encodeurs sauvegardés pour de nouvelles prédictions")