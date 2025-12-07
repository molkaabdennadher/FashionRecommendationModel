import pandas as pd
import numpy as np
import pickle
import os
import traceback
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
from collections import Counter

# ==================== CONFIGURATION ====================
BODY_TYPES = {
    0: 'Athletic',
    1: 'Full Bust',
    2: 'Hourglass',
    3: 'Pear',
    4: 'Petite',
    5: 'Straight & Narrow'
}

# Chemins des fichiers
MODEL_PATH = r'C:\dataware\venv\model_xgboost_extreme.pkl'
SCALER_PATH = r'C:\dataware\venv\scaler_recommendation.pkl'
KNN_PATH = r'C:\dataware\venv\knn_recommendation_model.pkl'
EMBEDDINGS_PATH = r'C:\dataware\venv\train_embeddings.npy'
TRAIN_DB_PATH = r'C:\dataware\venv\train_purchases_database.xlsx'
MAPPING_PATH = r'C:\dataware\venv\product_names_mapping.pkl'
SCALER_PARAMS_PATH = r'C:\dataware\venv\scaler_params.pkl'

print("=" * 90)
print("SYSTÈME DE RECOMMANDATION MODE - VERSION CORRIGÉE")
print("Encodage/Normalisation Fixé")
print("=" * 90)

# ==================== CHARGEMENT DES MODÈLES ====================
print("\n[ÉTAPE 1/4] Chargement des modèles ML...")

try:
    with open(MODEL_PATH, 'rb') as f:
        xgb_model = pickle.load(f)
    print("   ✓ XGBoost chargé")

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("   ✓ StandardScaler chargé")

    with open(KNN_PATH, 'rb') as f:
        knn_model = pickle.load(f)
    print("   ✓ KNN chargé")

    train_embeddings = np.load(EMBEDDINGS_PATH)
    print(f"   ✓ Embeddings chargés ({train_embeddings.shape})")

    train_db = pd.read_excel(TRAIN_DB_PATH)
    print(f"   ✓ Base TRAIN chargée ({len(train_db):,} clientes)")

except Exception as e:
    print(f"   ERREUR: {str(e)}")
    exit(1)

# Features
CLIENT_FEATURES = ['rating', 'weight_kg', 'height_cm', 'bust_size_num',
                   'size_num', 'age', 'fit_encoded']
ALL_FEATURES = train_db.columns[:-4].tolist()

print(f"   ✓ {len(CLIENT_FEATURES)} features client")
print(f"   ✓ {len(ALL_FEATURES)} features totales")

# ==================== CHARGEMENT PARAMÈTRES SCALER ====================
print("\n[ÉTAPE 2/4] Chargement des paramètres du StandardScaler...")

try:
    with open(SCALER_PARAMS_PATH, 'rb') as f:
        SCALER_PARAMS = pickle.load(f)

    print("    Paramètres du scaler chargés:")
    for col, params in SCALER_PARAMS.items():
        print(f"      {col:20s}: mean={params['mean']:6.1f}, std={params['std']:5.1f}")

except Exception as e:
    print(f"   Paramètres non trouvés, calcul depuis train_db...")
    SCALER_PARAMS = {
        'weight_kg': {
            'mean': train_db['weight_kg'].mean(),
            'std': train_db['weight_kg'].std()
        },
        'height_cm': {
            'mean': train_db['height_cm'].mean(),
            'std': train_db['height_cm'].std()
        },
        'bust_size_num': {
            'mean': train_db['bust_size_num'].mean(),
            'std': train_db['bust_size_num'].std()
        },
        'size_num': {
            'mean': train_db['size_num'].mean(),
            'std': train_db['size_num'].std()
        },
        'age': {
            'mean': train_db['age'].mean(),
            'std': train_db['age'].std()
        }
    }

    print("   Paramètres calculés:")
    for col, params in SCALER_PARAMS.items():
        print(f"      {col:20s}: mean={params['mean']:6.1f}, std={params['std']:5.1f}")

# ==================== CHARGEMENT MAPPING PRODUITS ====================
print("\n[ÉTAPE 3/4] Chargement du mapping des noms de produits...")

PRODUCT_NAMES = {}

try:
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, 'rb') as f:
            PRODUCT_NAMES = pickle.load(f)
        print(f"    {len(PRODUCT_NAMES)} produits chargés")

        print(f"\n    Exemples de produits:")
        for i, (key, value) in enumerate(list(PRODUCT_NAMES.items())[:7]):
            print(f"      {key:3d} → {value}")
    else:
        print(f"    Mapping non trouvé, utilisation d'IDs génériques")
        if 'product_type_name_encoded' in train_db.columns:
            unique_products = train_db['product_type_name_encoded'].unique()
            PRODUCT_NAMES = {int(pid): f"Produit #{int(pid)}" for pid in unique_products}

except Exception as e:
    print(f"    ERREUR: {str(e)}")
    PRODUCT_NAMES = {}

if not PRODUCT_NAMES and 'product_type_name_encoded' in train_db.columns:
    unique_products = train_db['product_type_name_encoded'].unique()
    PRODUCT_NAMES = {int(pid): f"Produit #{int(pid)}" for pid in unique_products}


# ==================== FONCTION D'ENCODAGE CORRIGÉE ====================
def encode_client_data(client_data_real):
    """
    FONCTION CORRIGÉE - Normalise les données selon les paramètres d'entraînement

    Args:
        client_data_real: dict avec valeurs réelles {
            'weight_kg': float (ex: 70.0),
            'height_cm': float (ex: 150.0),
            'bust_cm': float (ex: 90.0),
            'clothing_size': float (ex: 46.0),
            'age': int (ex: 44),
            'fit_preference': int (0-3)
        }

    Returns:
        dict avec valeurs NORMALISÉES pour XGBoost
    """

    #  NORMALISATION AVEC Z-SCORE : (valeur - mean) / std
    def normalize(value, mean, std):
        """Normalise une valeur avec StandardScaler"""
        if std == 0:
            return 0.0
        return (value - mean) / std

    # 1. Poids normalisé
    weight_normalized = normalize(
        client_data_real['weight_kg'],
        SCALER_PARAMS['weight_kg']['mean'],
        SCALER_PARAMS['weight_kg']['std']
    )

    # 2. Taille normalisée
    height_normalized = normalize(
        client_data_real['height_cm'],
        SCALER_PARAMS['height_cm']['mean'],
        SCALER_PARAMS['height_cm']['std']
    )

    # 3. Tour de poitrine normalisé
    bust_normalized = normalize(
        client_data_real['bust_cm'],
        SCALER_PARAMS['bust_size_num']['mean'],
        SCALER_PARAMS['bust_size_num']['std']
    )

    # 4. Taille de vêtement normalisée
    size_normalized = normalize(
        client_data_real['clothing_size'],
        SCALER_PARAMS['size_num']['mean'],
        SCALER_PARAMS['size_num']['std']
    )

    # 5. Âge normalisé
    age_normalized = normalize(
        client_data_real['age'],
        SCALER_PARAMS['age']['mean'],
        SCALER_PARAMS['age']['std']
    )

    # 6. Fit déjà encodé (0-3)
    fit_encoded = client_data_real['fit_preference']

    #  VÉRIFICATION DES VALEURS NORMALISÉES
    encoded_data = {
        'weight_kg': weight_normalized,
        'height_cm': height_normalized,
        'bust_size_num': bust_normalized,
        'size_num': size_normalized,
        'age': age_normalized,
        'fit_encoded': fit_encoded
    }

    return encoded_data


def validate_encoded_data(client_data_real, encoded_data):
    """
     NOUVELLE FONCTION - Valide que l'encodage est correct
    """
    print("\n" + "=" * 90)
    print(" VALIDATION DE L'ENCODAGE")
    print("=" * 90)

    print(f"\n{'Feature':<20} {'Valeur Réelle':<20} {'Valeur Normalisée':<20} {'Plage Attendue'}")
    print("-" * 90)

    validations = [
        ('weight_kg', client_data_real['weight_kg'], encoded_data['weight_kg'],
         f"~{SCALER_PARAMS['weight_kg']['mean']:.0f} kg", "[-3, +3]"),
        ('height_cm', client_data_real['height_cm'], encoded_data['height_cm'],
         f"~{SCALER_PARAMS['height_cm']['mean']:.0f} cm", "[-3, +3]"),
        ('bust_size_num', client_data_real['bust_cm'], encoded_data['bust_size_num'],
         f"~{SCALER_PARAMS['bust_size_num']['mean']:.0f} cm", "[-3, +3]"),
        ('size_num', client_data_real['clothing_size'], encoded_data['size_num'],
         f"~{SCALER_PARAMS['size_num']['mean']:.0f}", "[-3, +3]"),
        ('age', client_data_real['age'], encoded_data['age'],
         f"~{SCALER_PARAMS['age']['mean']:.0f} ans", "[-3, +3]"),
        ('fit_encoded', client_data_real['fit_preference'], encoded_data['fit_encoded'],
         "0-3", "[0, 3]")
    ]

    all_valid = True

    for feature, real_val, norm_val, mean_info, expected_range in validations:
        status = "✅" if -5 < norm_val < 5 else "⚠️"
        if feature == 'fit_encoded':
            status = "✅" if 0 <= norm_val <= 3 else "❌"
            if not (0 <= norm_val <= 3):
                all_valid = False
        else:
            if abs(norm_val) > 5:
                all_valid = False

        print(f"{feature:<20} {str(real_val):<20} {norm_val:< 20.4f} {expected_range:<15} {status}")

    print("-" * 90)

    if all_valid:
        print(" Encodage VALIDE - Données prêtes pour XGBoost")
    else:
        print("  ATTENTION - Certaines valeurs sont hors normes")
        print("   → Vérifiez que les valeurs saisies sont réalistes")

    return all_valid


# ==================== FONCTION DE RECOMMANDATION CORRIGÉE ====================
print("\n[ÉTAPE 4/4] Initialisation du système de recommandation...")


def recommend_products(client_data_real, purchased_products=None):
    """
     FONCTION CORRIGÉE - Recommander des produits avec encodage correct
    """
    if purchased_products is None:
        purchased_products = set()
    else:
        purchased_products = set(purchased_products)

    # ENCODAGE CORRECT des données
    client_dict = encode_client_data(client_data_real)

    #  VALIDATION de l'encodage
    is_valid = validate_encoded_data(client_data_real, client_dict)

    if not is_valid:
        print("\n  WARNING: Données encodées hors normes - résultats peuvent être incorrects")

    # 1. Créer le vecteur de features complet
    feature_vector = []
    for fname in ALL_FEATURES:
        feature_vector.append(client_dict.get(fname, 0))
    feature_vector = np.array(feature_vector).reshape(1, -1)

    # 2. Prédire le body type avec XGBoost
    body_type_id = int(xgb_model.predict(feature_vector)[0])
    body_type_name = BODY_TYPES[body_type_id]

    # 3. Extraire les features client pour KNN
    client_vector = []
    for fname in CLIENT_FEATURES:
        if fname == 'rating':
            client_vector.append(0)
        else:
            client_vector.append(client_dict.get(fname, 0))
    client_vector = np.array(client_vector).reshape(1, -1)
    client_normalized = scaler.transform(client_vector)

    # 4. Ajouter le body type encodé (one-hot)
    bt_onehot = np.zeros((1, 6))
    bt_onehot[0, body_type_id] = 1
    client_embedding = np.concatenate([client_normalized, bt_onehot * 2.0], axis=1)

    # 5. Trouver les clientes similaires
    y_train_pred = train_db['body_type_predicted'].values
    same_bt_indices = np.where(y_train_pred == body_type_id)[0]

    if len(same_bt_indices) == 0:
        return {
            'body_type_id': body_type_id,
            'body_type_name': body_type_name,
            'similar_clients_count': 0,
            'products_excluded': len(purchased_products),
            'client_data_real': client_data_real,
            'client_data_encoded': client_dict,
            'recommendations': []
        }

    filtered_embeddings = train_embeddings[same_bt_indices]

    # 6. KNN
    n_similar = min(100, len(same_bt_indices))
    knn_temp = NearestNeighbors(n_neighbors=n_similar, metric='cosine', algorithm='brute')
    knn_temp.fit(filtered_embeddings)
    distances, indices = knn_temp.kneighbors(client_embedding)

    similar_indices = same_bt_indices[indices[0]]

    # 7. Compter les produits
    similar_purchases = train_db.iloc[similar_indices]
    products = similar_purchases['product_type_name_encoded'].values
    product_counts = Counter(products)

    # 8. Filtrer et retourner TOP 5
    recommendations = []
    for product_id, count in product_counts.most_common():
        if product_id in purchased_products:
            continue

        percentage = (count / len(similar_indices)) * 100
        product_name = PRODUCT_NAMES.get(int(product_id), f"Produit #{int(product_id)}")

        recommendations.append({
            'product_id': int(product_id),
            'product_name': product_name,
            'purchase_count': count,
            'percentage': round(percentage, 2)
        })

        if len(recommendations) >= 5:
            break

    return {
        'body_type_id': body_type_id,
        'body_type_name': body_type_name,
        'similar_clients_count': len(similar_indices),
        'products_excluded': len(purchased_products),
        'client_data_real': client_data_real,
        'client_data_encoded': client_dict,
        'recommendations': recommendations
    }


print("   ✅ Système de recommandation prêt")


# ==================== FONCTIONS D'INTERACTION AMÉLIORÉES ====================
def get_float_input(prompt, min_val, max_val, default):
    """ Validation renforcée pour float"""
    while True:
        try:
            value = input(f"{prompt} [{default}]: ").strip()
            if value == "":
                return default

            value = float(value)

            if min_val <= value <= max_val:
                return value
            else:
                print(f"     Valeur hors limites ({min_val}-{max_val}). Réessayez.")

                # Suggestions intelligentes
                if value < min_val:
                    print(f"   Valeur trop basse. Minimum accepté: {min_val}")
                else:
                    print(f"    Valeur trop élevée. Maximum accepté: {max_val}")

        except ValueError:
            print("     Valeur invalide. Entrez un nombre (ex: 65.5). Réessayez.")


def get_int_input(prompt, min_val, max_val, default):
    """ Validation renforcée pour int"""
    while True:
        try:
            value = input(f"{prompt} [{default}]: ").strip()
            if value == "":
                return default

            value = int(value)

            if min_val <= value <= max_val:
                return value
            else:
                print(f"     Valeur hors limites ({min_val}-{max_val}). Réessayez.")

        except ValueError:
            print("    Valeur invalide. Entrez un nombre entier. Réessayez.")


def display_recommendations(result):
    """Affichage amélioré avec données encodées"""
    print("\n" + "=" * 90)
    print(" RÉSULTATS DE L'ANALYSE")
    print("=" * 90)

    # Profil réel
    client_data = result['client_data_real']
    print(f"\n PROFIL DE LA CLIENTE (Valeurs Réelles):")
    print(f"   • Poids: {client_data['weight_kg']:.1f} kg")
    print(f"   • Taille: {client_data['height_cm']:.0f} cm")
    print(f"   • Tour de poitrine: {client_data['bust_cm']:.0f} cm")
    print(f"   • Taille de vêtement: {client_data['clothing_size']:.0f}")
    print(f"   • Âge: {client_data['age']} ans")

    fit_names = {0: "Trop petit", 1: "Ajusté", 2: "Large", 3: "Petit"}
    print(f"   • Préférence de coupe: {fit_names.get(client_data['fit_preference'], 'Inconnu')}")

    # Données encodées
    client_encoded = result['client_data_encoded']
    print(f"\n DONNÉES ENCODÉES (Pour le Modèle):")
    print(f"   • weight_kg:      {client_encoded['weight_kg']:>8.4f}")
    print(f"   • height_cm:      {client_encoded['height_cm']:>8.4f}")
    print(f"   • bust_size_num:  {client_encoded['bust_size_num']:>8.4f}")
    print(f"   • size_num:       {client_encoded['size_num']:>8.4f}")
    print(f"   • age:            {client_encoded['age']:>8.4f}")
    print(f"   • fit_encoded:    {client_encoded['fit_encoded']:>8}")

    print(f"\n######################## MORPHOLOGIE DÉTECTÉE: {result['body_type_name']}")
    print(f"   Basé sur {result['similar_clients_count']} clientes similaires dans la base")

    if result['products_excluded'] > 0:
        print(f"\n {result['products_excluded']} produits exclus (historique d'achat)")

    print("\n" + "-" * 90)
    print(" TOP 5 PRODUITS RECOMMANDÉS")
    print("-" * 90)

    if len(result['recommendations']) == 0:
        print("\n Aucune recommandation disponible pour cette morphologie")
        return

    # Tableau
    print(f"\n{'Rang':<6} {'ID':<8} {'Nom du Produit':<45} {'Achats':<10} {'Popularité'}")
    print("-" * 90)

    for i, rec in enumerate(result['recommendations'], 1):
        product_name = rec['product_name'][:43]
        popularity_bar = "█" * int(rec['percentage'] / 5) + "░" * (20 - int(rec['percentage'] / 5))
        print(
            f"{i:<6} {rec['product_id']:<8} {product_name:<45} {rec['purchase_count']:<10} {rec['percentage']:.1f}% {popularity_bar}")

    # Détails
    print("\n" + "-" * 90)
    print(" DÉTAILS DES RECOMMANDATIONS")
    print("-" * 90)

    for i, rec in enumerate(result['recommendations'], 1):
        print(f"\n[{i}]   {rec['product_name']}")
        print(f"    • ID Produit: {rec['product_id']}")
        print(
            f"    • Popularité: {rec['purchase_count']} achats parmi {result['similar_clients_count']} clientes similaires")
        print(f"    • Taux d'adoption: {rec['percentage']}%")

        if rec['percentage'] >= 50:
            print(f"    •Très recommandé")
        elif rec['percentage'] >= 30:
            print(f"    • Recommandé")
        else:
            print(f"    • Suggéré")


def main():
    """✅ Boucle principale améliorée"""

    print("\n" + "=" * 90)
    print(" SYSTÈME PRÊT - VERSION CORRIGÉE")
    print("=" * 90)

    while True:
        print("\n" + "=" * 90)
        print(" NOUVELLE CLIENTE")
        print("=" * 90)

        print("\n SAISIE DES CARACTÉRISTIQUES PHYSIQUES")
        print("(Appuyez sur Entrée pour utiliser la valeur par défaut)")
        print("-" * 90)

        weight_kg = get_float_input(" Poids (kg)", 30.0, 150.0, 65.0)
        height_cm = get_float_input(" Taille (cm)", 140.0, 200.0, 165.0)
        bust_cm = get_float_input(" Tour de poitrine (cm)", 70.0, 130.0, 90.0)
        clothing_size = get_float_input(" Taille de vêtement (EU)", 32.0, 52.0, 38.0)
        age = get_int_input(" Âge", 18, 100, 30)

        print("\n PRÉFÉRENCE DE COUPE:")
        print("  0 = Trop petit (Too Small)")
        print("  1 = Ajusté (Fit) ! Recommandé")
        print("  2 = Large")
        print("  3 = Petit (Small)")
        fit_preference = get_int_input("Choix de coupe", 0, 3, 1)

        # Historique d'achat
        print("\n" + "-" * 90)
        print(" HISTORIQUE D'ACHAT (optionnel)")
        purchased_input = input("IDs des produits séparés par des virgules (ou Entrée): ").strip()

        purchased_products = []
        if purchased_input:
            try:
                purchased_products = [int(p.strip()) for p in purchased_input.split(',') if p.strip().isdigit()]
                print(f"   ✓ {len(purchased_products)} produits seront exclus")
            except:
                print("    Format invalide")

        # Préparer les données
        client_data_real = {
            'weight_kg': weight_kg,
            'height_cm': height_cm,
            'bust_cm': bust_cm,
            'clothing_size': clothing_size,
            'age': age,
            'fit_preference': fit_preference
        }

        # Recommandations
        print("\n" + "-" * 90)
        print(" ANALYSE EN COURS...")
        print("    Normalisation des données...")
        print("    Prédiction de la morphologie avec XGBoost...")
        print("    Recherche de clientes similaires...")
        print("    Analyse des tendances d'achat...")

        try:
            result = recommend_products(client_data_real, purchased_products=purchased_products)
            display_recommendations(result)

        except Exception as e:
            print(f"\n ERREUR: {str(e)}")
            traceback.print_exc()

        # Continuer
        print("\n" + "=" * 90)
        continuer = input("\n Analyser une nouvelle cliente ? (o/n) [o]: ").strip().lower()
        if continuer in ['n', 'non', 'no']:
            print("\n" + "=" * 90)
            print(" MERCI D'AVOIR UTILISÉ LE SYSTÈME")
            print("=" * 90)
            break


# ==================== POINT D'ENTRÉE ====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interruption par l'utilisateur (Ctrl+C)")
    except Exception as e:
        print(f"\n\n ERREUR FATALE: {str(e)}")
        traceback.print_exc()