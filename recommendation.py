import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import warnings
import time
from collections import Counter

warnings.filterwarnings('ignore')

print("=" * 90)
print("SYSTÈME DE RECOMMANDATION DE PRODUITS BASÉ SUR BODY TYPE")
print("Recommande des articles selon l'historique d'achat des clients similaires")
print("=" * 90)

start_time = time.time()

# ==================== CHARGEMENT DU MODÈLE XGBOOST EXTREME ====================
print("\n[ÉTAPE 1] Chargement du modèle XGBoost Extreme...")

model_path = r'C:\dataware\venv\model_xgboost_extreme.pkl'

if os.path.exists(model_path):
    print(f"   ✓ Modèle trouvé: {model_path}")
    with open(model_path, 'rb') as f:
        xgb_model = pickle.load(f)
else:
    print("   ⚠ Modèle non trouvé. Entraînement en cours...")
    print("   Ce processus peut prendre 5-10 minutes...")

    train_data = pd.read_excel(r"C:\dataware\venv\train_data.xlsx")
    test_data = pd.read_excel(r"C:\dataware\venv\test_data.xlsx")

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    print(f"   Train: {X_train.shape[0]:,} échantillons | {X_train.shape[1]} features")
    print(f"   Test:  {X_test.shape[0]:,} échantillons")

    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n   Distribution des classes:")
    for cls, cnt in zip(unique, counts):
        print(f"      Classe {cls}: {cnt:,} ({cnt / len(y_train) * 100:.1f}%)")

    print("\n   Entraînement XGBoost EXTREME (3000 estimators, depth=30)...")
    t0 = time.time()

    xgb_model = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=30,
        learning_rate=0.015,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.3,
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective='multi:softmax',
        num_class=6,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n   ✓ Entraînement terminé!")
    print(f"   Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Temps: {time.time() - t0:.1f}s")

    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"   ✓ Modèle sauvegardé: {model_path}")

# Mapping des classes body type
BODY_TYPES = {
    0: 'Athletic',
    1: 'Full Bust',
    2: 'Hourglass',
    3: 'Pear',
    4: 'Petite',
    5: 'Straight & Narrow'
}

# ==================== CRÉATION BASE DE DONNÉES D'ACHATS ====================
print("\n[ÉTAPE 2] Création de la base de données d'achats...")

train_data = pd.read_excel(r"C:\dataware\venv\train_data.xlsx")
test_data = pd.read_excel(r"C:\dataware\venv\test_data.xlsx")

all_data = pd.concat([train_data, test_data], ignore_index=True)
X_all = all_data.iloc[:, :-1].values
y_all = all_data.iloc[:, -1].values

print(f"   Base de données: {len(X_all):,} achats")

# Prédire les body types
print("   Prédiction des body types...")
y_pred_all = xgb_model.predict(X_all)

acc_all = accuracy_score(y_all, y_pred_all)
print(f"   Accuracy globale: {acc_all:.4f} ({acc_all * 100:.2f}%)")

# Créer DataFrame avec historique d'achats
feature_names = train_data.columns[:-1].tolist()
purchases_db = pd.DataFrame(X_all, columns=feature_names)
purchases_db['body_type_real'] = y_all
purchases_db['body_type_predicted'] = y_pred_all
purchases_db['body_type_name'] = purchases_db['body_type_predicted'].map(BODY_TYPES)
purchases_db['purchase_id'] = range(len(purchases_db))

# Sauvegarder
purchases_db.to_excel(r'C:\dataware\venv\purchases_database.xlsx', index=False)
print("   ✓ Base de données d'achats sauvegardée: purchases_database.xlsx")

# ==================== ANALYSE DES PRODUITS PAR BODY TYPE ====================
print("\n[ÉTAPE 3] Analyse des produits populaires par body type...")

# Créer un dictionnaire : body_type -> liste des produits achetés
body_type_products = {}

for bt_id, bt_name in BODY_TYPES.items():
    # Filtrer les achats pour ce body type
    bt_purchases = purchases_db[purchases_db['body_type_predicted'] == bt_id]

    # Extraire les produits (product_type_name_encoded)
    products = bt_purchases['product_type_name_encoded'].values

    # Compter la fréquence de chaque produit
    product_counts = Counter(products)

    body_type_products[bt_id] = {
        'name': bt_name,
        'total_purchases': len(bt_purchases),
        'product_counts': product_counts,
        'top_products': product_counts.most_common(20)  # Top 20 produits
    }

    print(f"   {bt_name:20s}: {len(bt_purchases):7,} achats | "
          f"{len(product_counts):3} produits différents")

# Sauvegarder l'analyse
with open(r'C:\dataware\venv\body_type_products_analysis.pkl', 'wb') as f:
    pickle.dump(body_type_products, f)
print("   ✓ Analyse sauvegardée: body_type_products_analysis.pkl")

# ==================== CRÉATION DU SYSTÈME KNN POUR CLIENTS SIMILAIRES ====================
print("\n[ÉTAPE 4] Création du système KNN pour trouver clients similaires...")

# Normaliser les features des clients (sans les produits)
client_features = ['rating', 'weight_kg', 'height_cm', 'bust_size_num',
                   'size_num', 'age', 'fit_encoded']
client_features_available = [f for f in client_features if f in feature_names]

X_clients = purchases_db[client_features_available].values

scaler = StandardScaler()
X_clients_normalized = scaler.fit_transform(X_clients)

# Ajouter le body type encodé
body_type_onehot = np.zeros((len(y_pred_all), 6))
for i, bt in enumerate(y_pred_all):
    body_type_onehot[i, bt] = 1

# Combiner: features client + body type
X_embeddings = np.concatenate([X_clients_normalized, body_type_onehot * 2.0], axis=1)
print(f"   Dimensions des embeddings: {X_embeddings.shape}")

# Entraîner KNN
knn_model = NearestNeighbors(
    n_neighbors=min(100, len(X_embeddings)),
    metric='cosine',
    algorithm='brute',
    n_jobs=-1
)
knn_model.fit(X_embeddings)

# Sauvegarder
with open(r'C:\dataware\venv\scaler_recommendation.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(r'C:\dataware\venv\knn_recommendation_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
np.save(r'C:\dataware\venv\client_embeddings.npy', X_embeddings)

print("   ✓ Modèle KNN sauvegardé")

# ==================== FONCTIONS DE RECOMMANDATION ====================
print("\n[ÉTAPE 5] Création des fonctions de recommandation...")


def predict_body_type(client_features_dict):
    """
    Prédire le body type d'une nouvelle cliente

    Args:
        client_features_dict: dict avec les features de la cliente
        Exemple: {'weight_kg': 65, 'height_cm': 170, 'age': 28, ...}

    Returns:
        dict avec body_type_id, body_type_name
    """
    # Créer un vecteur de features dans le bon ordre
    feature_vector = []
    for fname in feature_names:
        if fname in client_features_dict:
            feature_vector.append(client_features_dict[fname])
        else:
            feature_vector.append(0)  # Valeur par défaut

    feature_vector = np.array(feature_vector).reshape(1, -1)
    body_type_id = xgb_model.predict(feature_vector)[0]
    body_type_name = BODY_TYPES[body_type_id]

    return {
        'body_type_id': int(body_type_id),
        'body_type_name': body_type_name
    }


def recommend_products_by_body_type(body_type_id, top_n=10):
    """
    Recommander les produits les plus populaires pour un body type

    Args:
        body_type_id: ID du body type (0-5)
        top_n: nombre de produits à recommander

    Returns:
        Liste de (product_id, purchase_count, percentage)
    """
    bt_data = body_type_products[body_type_id]
    top_products = bt_data['top_products'][:top_n]
    total_purchases = bt_data['total_purchases']

    recommendations = []
    for product_id, count in top_products:
        percentage = (count / total_purchases) * 100
        recommendations.append({
            'product_id': int(product_id),
            'purchase_count': count,
            'percentage': round(percentage, 2)
        })

    return recommendations


def recommend_products_for_new_client(client_features_dict, top_n=10, n_similar=100):
    """
    Recommander des produits pour une nouvelle cliente
    Basé sur les achats des clientes similaires (MÉTHODE 1)

    Args:
        client_features_dict: features de la cliente
        top_n: nombre de produits à recommander
        n_similar: nombre de clientes similaires à considérer (par défaut 100)

    Returns:
        dict avec body_type, recommandations, similar_clients_count
    """
    # Prédire le body type
    prediction = predict_body_type(client_features_dict)
    body_type_id = prediction['body_type_id']
    body_type_name = prediction['body_type_name']

    print(f"\n   Body Type prédit: {body_type_name}")

    # Trouver les clientes similaires
    client_vector = []
    for fname in client_features_available:
        if fname in client_features_dict:
            client_vector.append(client_features_dict[fname])
        else:
            client_vector.append(0)

    client_vector = np.array(client_vector).reshape(1, -1)
    client_normalized = scaler.transform(client_vector)

    # Ajouter body type
    bt_onehot = np.zeros((1, 6))
    bt_onehot[0, body_type_id] = 1
    client_embedding = np.concatenate([client_normalized, bt_onehot * 2.0], axis=1)

    # Trouver clientes similaires avec le même body type
    same_bt_indices = np.where(y_pred_all == body_type_id)[0]
    filtered_embeddings = X_embeddings[same_bt_indices]

    knn_temp = NearestNeighbors(n_neighbors=min(n_similar, len(same_bt_indices)),
                                metric='cosine', algorithm='brute')
    knn_temp.fit(filtered_embeddings)
    distances, indices = knn_temp.kneighbors(client_embedding)

    similar_client_indices = same_bt_indices[indices[0]]

    print(f"   Trouvé {len(similar_client_indices)} clientes similaires avec le même body type")

    # Compter les produits achetés par ces clientes
    similar_purchases = purchases_db.iloc[similar_client_indices]
    products = similar_purchases['product_type_name_encoded'].values
    product_counts = Counter(products)
    top_products = product_counts.most_common(top_n)

    recommendations = []
    for product_id, count in top_products:
        percentage = (count / len(similar_client_indices)) * 100
        recommendations.append({
            'product_id': int(product_id),
            'purchase_count': count,
            'percentage': round(percentage, 2)
        })

    return {
        'body_type_id': body_type_id,
        'body_type_name': body_type_name,
        'similar_clients_count': len(similar_client_indices),
        'recommendations': recommendations
    }


print("   ✓ Fonctions créées")

# ==================== EXEMPLE D'UTILISATION ====================
print("\n" + "=" * 90)
print("EXEMPLE DE RECOMMANDATION (MÉTHODE 1: Clientes Similaires)")
print("=" * 90)

# Prendre une cliente test
test_idx = 0
test_row = purchases_db.iloc[test_idx]

# Extraire ses caractéristiques
test_client = {}
for fname in client_features_available:
    test_client[fname] = test_row[fname]

print(f"\nCliente Test #{test_idx}:")
for key, val in test_client.items():
    print(f"   {key}: {val}")

print(f"\n   Body Type RÉEL: {test_row['body_type_name']}")
print(f"   Produit ACHETÉ: {test_row['product_type_name_encoded']}")

# Recommandations basées sur clientes similaires
print("\n" + "-" * 90)
print("Recommandations basées sur 100 clientes les plus similaires")
print("-" * 90)

result = recommend_products_for_new_client(test_client, top_n=10, n_similar=100)

print(f"\nTop 10 Produits Recommandés pour {result['body_type_name']}:")
print(f"(Basé sur {result['similar_clients_count']} clientes similaires)")
print(f"\n{'Rang':<6} {'Product ID':<12} {'Achats':<10} {'%':<8}")
print("-" * 50)
for i, rec in enumerate(result['recommendations'], 1):
    print(f"{i:<6} {rec['product_id']:<12} {rec['purchase_count']:<10} "
          f"{rec['percentage']:<7.2f}%")

# Sauvegarder les recommandations
recommendations_df = pd.DataFrame(result['recommendations'])
recommendations_df.to_excel(r'C:\dataware\venv\example_product_recommendations.xlsx',
                            index=False)
print("\n✓ Recommandations sauvegardées: example_product_recommendations.xlsx")

# ==================== STATISTIQUES PAR BODY TYPE ====================
print("\n" + "=" * 90)
print("TOP 5 PRODUITS PAR BODY TYPE")
print("=" * 90)

for bt_id, bt_name in BODY_TYPES.items():
    bt_data = body_type_products[bt_id]
    print(f"\n{bt_name} ({bt_data['total_purchases']:,} achats):")
    for i, (product_id, count) in enumerate(bt_data['top_products'][:5], 1):
        percentage = (count / bt_data['total_purchases']) * 100
        print(f"   {i}. Produit {int(product_id):3}: {count:6,} achats ({percentage:5.2f}%)")

# ==================== RÉSUMÉ FINAL ====================
total_time = time.time() - start_time

print("\n" + "=" * 90)
print("SYSTÈME DE RECOMMANDATION PRÊT")
print("=" * 90)
print(f"\nTemps total: {total_time:.1f}s")

print("\nFICHIERS CRÉÉS:")
print("   ✓ purchases_database.xlsx - Base des achats avec body types")
print("   ✓ body_type_products_analysis.pkl - Analyse produits par body type")
print("   ✓ client_embeddings.npy - Vecteurs clients pour KNN")
print("   ✓ knn_recommendation_model.pkl - Modèle KNN")
print("   ✓ scaler_recommendation.pkl - Scaler pour normalisation")
print("   ✓ example_product_recommendations.xlsx - Exemple recommandations")

print("\nFONCTIONS DISPONIBLES:")
print("   1. predict_body_type(client_features_dict)")
print("      → Prédire le body type d'une cliente")
print("\n   2. recommend_products_by_body_type(body_type_id, top_n=10)")
print("      → Produits populaires pour un body type")
print("\n   3. recommend_products_for_new_client(client_dict, top_n=10)")
print("      → Recommandations personnalisées basées sur clientes similaires")

print("\nEXEMPLE D'USAGE:")
print("   client = {'weight_kg': 1.51, 'height_cm': 1.67, 'age': -0.29, ...}")
print("   result = recommend_products_for_new_client(client, top_n=10)")
print("   print(result['recommendations'])")

print("\n" + "=" * 90)