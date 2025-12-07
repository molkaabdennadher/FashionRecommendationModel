import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==================== CHARGEMENT DES MODÈLES ET DONNÉES ====================
print("=" * 80)
print("SYSTÈME DYNAMIQUE DE RECOMMANDATION PAR CLUSTER")
print("=" * 80)

try:
    # Charger les modèles
    with open('morpho_fit_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)

    with open('morpho_fit_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('cluster_profiles.pkl', 'rb') as f:
        cluster_profiles = pickle.load(f)

    with open('cluster_names.pkl', 'rb') as f:
        cluster_names = pickle.load(f)

    # Charger le dataset avec clusters pour analyser les patterns réels
    df_clustered = pd.read_excel('modcloth_avec_clusters.xlsx')
    df_clustered = df_clustered[df_clustered['cluster'].notna()].copy()
    df_clustered['cluster'] = df_clustered['cluster'].astype(int)

    print("Système chargé avec succès!")
    print(f"Dataset: {len(df_clustered)} clientes analysées\n")
except FileNotFoundError as e:
    print(f"Erreur: Fichier manquant - {e}")
    exit()

# ==================== ANALYSE DYNAMIQUE DES PATTERNS PAR CLUSTER ====================
print("=" * 80)
print("ANALYSE DES COMPORTEMENTS RÉELS PAR CLUSTER")
print("=" * 80)

# Analyser les tailles réellement portées par cluster
cluster_size_analysis = {}
for cluster_id in sorted(df_clustered['cluster'].unique()):
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]

    cluster_size_analysis[cluster_id] = {
        'size_mean': cluster_data['size'].mean(),
        'size_median': cluster_data['size'].median(),
        'size_std': cluster_data['size'].std(),
        'size_min': cluster_data['size'].min(),
        'size_max': cluster_data['size'].max(),
        'size_q25': cluster_data['size'].quantile(0.25),
        'size_q75': cluster_data['size'].quantile(0.75)
    }

# Analyser les préférences de FIT par cluster
cluster_fit_preferences = {}
for cluster_id in sorted(df_clustered['cluster'].unique()):
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
    fit_counts = cluster_data['fit'].value_counts(normalize=True) * 100
    cluster_fit_preferences[cluster_id] = fit_counts.to_dict()

# Analyser les catégories préférées par cluster
cluster_category_preferences = {}
for cluster_id in sorted(df_clustered['cluster'].unique()):
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
    cat_counts = cluster_data['category'].value_counts().head(5)
    cluster_category_preferences[cluster_id] = cat_counts.to_dict()

print("\nResultats de l'analyse:")
for cluster_id in sorted(cluster_size_analysis.keys()):
    print(f"\nCluster {cluster_id} - {cluster_names[cluster_id]['name']}:")
    print(f"   Tailles portées: {cluster_size_analysis[cluster_id]['size_min']:.0f} à "
          f"{cluster_size_analysis[cluster_id]['size_max']:.0f}")
    print(f"   Taille moyenne: {cluster_size_analysis[cluster_id]['size_mean']:.1f}")
    print(f"   Fit préféré: {max(cluster_fit_preferences[cluster_id], key=cluster_fit_preferences[cluster_id].get)}")

# ==================== FONCTIONS DE CONVERSION TAILLES ====================
SIZE_MAPPING = {
    range(2, 5): "XS",
    range(5, 8): "S",
    range(8, 12): "M",
    range(12, 16): "L",
    range(16, 25): "XL",
    range(25, 35): "XXL"
}


def numeric_to_letter_size(numeric_size):
    """Convertit une taille numérique en lettre"""
    numeric_size = int(round(numeric_size))
    for size_range, letter in SIZE_MAPPING.items():
        if numeric_size in size_range:
            return letter
    return "XXL"


def letter_to_numeric_size(letter_size):
    """Convertit une taille lettre en numérique"""
    mapping = {"XS": 4, "S": 6, "M": 10, "L": 14, "XL": 18, "XXL": 26}
    return mapping.get(letter_size.upper(), 10)


# ==================== PRÉDICTION CLUSTER ====================
def predict_cluster_dynamic(weight_kg, height_cm, bust_size_num, usual_size_letter):
    """Prédit le cluster de la cliente"""
    usual_size_numeric = letter_to_numeric_size(usual_size_letter)
    new_data = np.array([[weight_kg, height_cm, bust_size_num, usual_size_numeric]])
    new_data_scaled = scaler.transform(new_data)
    cluster_pred = kmeans_model.predict(new_data_scaled)[0]

    distances = kmeans_model.transform(new_data_scaled)[0]
    confidence = (1 - distances[cluster_pred] / distances.max()) * 100

    return {
        'cluster_id': int(cluster_pred),
        'confidence': round(confidence, 1),
        'morphology': cluster_names[cluster_pred]['name'],
        'usual_size_letter': usual_size_letter,
        'usual_size_numeric': usual_size_numeric,
        'weight_kg': weight_kg,
        'height_cm': height_cm,
        'bust_size': bust_size_num
    }


# ==================== RECOMMANDATION DYNAMIQUE PAR CLUSTER ====================
def recommend_size_dynamic(user_profile, product_fit='fit'):
    """
    Recommandation DYNAMIQUE basée sur les données RÉELLES du cluster
    """
    cluster_id = user_profile['cluster_id']
    usual_size_numeric = user_profile['usual_size_numeric']

    # Récupérer l'analyse du cluster
    cluster_analysis = cluster_size_analysis[cluster_id]
    cluster_mean_size = cluster_analysis['size_mean']
    cluster_median_size = cluster_analysis['size_median']

    # LOGIQUE DYNAMIQUE: Comparer la taille habituelle avec la moyenne du cluster
    size_difference = usual_size_numeric - cluster_mean_size

    print(f"\nAnalyse pour Cluster {cluster_id} ({user_profile['morphology']}):")
    print(f"   Votre taille habituelle: {usual_size_numeric} ({user_profile['usual_size_letter']})")
    print(f"   Taille moyenne du cluster: {cluster_mean_size:.1f}")
    print(f"   Différence: {size_difference:+.1f}")

    # RECOMMANDATION BASÉE SUR LE CLUSTER
    if size_difference > 5:
        # Cliente porte une taille BEAUCOUP plus grande que la moyenne du cluster
        adjustment = -2
        reason = f"Vous portez une taille plus grande que {cluster_mean_size:.0f}% des clientes {user_profile['morphology']}"
    elif size_difference > 2:
        adjustment = -1
        reason = f"Vous êtes légèrement au-dessus de la moyenne {user_profile['morphology']}"
    elif size_difference < -5:
        # Cliente porte une taille BEAUCOUP plus petite
        adjustment = +2
        reason = f"Vous portez une taille plus petite que {cluster_mean_size:.0f}% des clientes {user_profile['morphology']}"
    elif size_difference < -2:
        adjustment = +1
        reason = f"Vous êtes légèrement en dessous de la moyenne {user_profile['morphology']}"
    else:
        # Dans la moyenne
        adjustment = 0
        reason = f"Vous êtes dans la moyenne des clientes {user_profile['morphology']}"

    # Ajustement selon le FIT du produit (basé sur les préférences réelles du cluster)
    fit_preferences = cluster_fit_preferences[cluster_id]
    preferred_fit = max(fit_preferences, key=fit_preferences.get)

    if product_fit != preferred_fit:
        if product_fit == 'large' and preferred_fit == 'fit':
            adjustment -= 1  # Produit taille large, cluster préfère fit → prendre plus petit
        elif product_fit == 'small' and preferred_fit == 'fit':
            adjustment += 1  # Produit taille petit, cluster préfère fit → prendre plus grand

    # Calculer la taille recommandée
    recommended_size_numeric = usual_size_numeric + adjustment
    recommended_size_numeric = max(2, min(30, recommended_size_numeric))
    recommended_size_letter = numeric_to_letter_size(recommended_size_numeric)

    # Alternatives
    alt_sizes = [recommended_size_numeric - 2, recommended_size_numeric + 2]
    alt_sizes = [s for s in alt_sizes if 2 <= s <= 30]
    alt_sizes_letter = [numeric_to_letter_size(s) for s in alt_sizes]

    return {
        'recommended_size_letter': recommended_size_letter,
        'recommended_size_numeric': int(recommended_size_numeric),
        'adjustment': adjustment,
        'reason': reason,
        'alternative_sizes': alt_sizes_letter,
        'cluster_mean_size': cluster_mean_size,
        'confidence': 'Haute' if abs(size_difference) < 2 else 'Moyenne'
    }


# ==================== CATALOGUE DYNAMIQUE ====================
def get_popular_products_for_cluster(cluster_id, top_n=5):
    """Retourne les produits les plus populaires pour un cluster spécifique"""
    popular_categories = cluster_category_preferences[cluster_id]

    products = []
    for idx, (category, count) in enumerate(popular_categories.items(), 1):
        percentage = (count / df_clustered[df_clustered['cluster'] == cluster_id].shape[0]) * 100
        products.append({
            'rank': idx,
            'category': category,
            'popularity': f"{percentage:.1f}%",
            'count': int(count)
        })

    return products


# ==================== SYSTÈME INTERACTIF ====================
def interactive_system():
    """Système interactif avec recommandations dynamiques"""

    while True:
        print("\n" + "=" * 80)
        print("NOUVEAU CLIENT - Saisissez vos informations")
        print("=" * 80)

        try:
            # Saisie
            print("\nVos mensurations:")
            weight_kg = float(input("   Poids (kg): "))
            height_cm = float(input("   Taille (cm): "))
            bust_size = float(input("   Tour de poitrine (ex: 32, 34, 36...): "))

            print("\nTaille habituelle:")
            print("   Choix: XS, S, M, L, XL, XXL")
            usual_size = input("   Votre taille: ").strip().upper()

            if usual_size not in ["XS", "S", "M", "L", "XL", "XXL"]:
                print("Erreur: Taille invalide!")
                continue

            # Prédiction
            print("\n" + "=" * 80)
            print("ANALYSE DE VOTRE PROFIL")
            print("=" * 80)

            user_profile = predict_cluster_dynamic(weight_kg, height_cm, bust_size, usual_size)

            print(f"\nProfil morphologique: {user_profile['morphology']}")
            print(f"   Cluster: {user_profile['cluster_id']}")
            print(f"   Confiance: {user_profile['confidence']}%")

            # Statistiques du cluster
            cluster_id = user_profile['cluster_id']
            cluster_stats = cluster_size_analysis[cluster_id]

            print(f"\nStatistiques du cluster {cluster_id}:")
            print(f"   - Taille moyenne: {numeric_to_letter_size(cluster_stats['size_mean'])} "
                  f"(taille {cluster_stats['size_mean']:.1f})")
            print(f"   - Gamme: {numeric_to_letter_size(cluster_stats['size_min'])} a "
                  f"{numeric_to_letter_size(cluster_stats['size_max'])}")
            print(f"   - 50% des clientes portent: {numeric_to_letter_size(cluster_stats['size_q25'])} a "
                  f"{numeric_to_letter_size(cluster_stats['size_q75'])}")

            # Produits populaires pour ce cluster
            print(f"\nTOP PRODUITS pour les clientes {user_profile['morphology']}:")
            popular_products = get_popular_products_for_cluster(cluster_id)

            for product in popular_products:
                print(f"   {product['rank']}. {product['category'].capitalize()} "
                      f"- Popularite: {product['popularity']} ({product['count']} clientes)")

            # UNE SEULE Recommandation de taille
            print("\n" + "=" * 80)
            print("VOTRE RECOMMANDATION DE TAILLE")
            print("=" * 80)

            # Utiliser le fit le plus courant du cluster comme référence
            rec = recommend_size_dynamic(user_profile, 'fit')

            print(f"\nTAILLE RECOMMANDEE POUR VOUS: {rec['recommended_size_letter']}")
            print(f"   (Taille numerique: {rec['recommended_size_numeric']})")
            print(f"\nExplication:")
            print(f"   {rec['reason']}")
            print(f"\nNiveau de confiance: {rec['confidence']}")

            if rec['adjustment'] != 0:
                direction = "en dessous" if rec['adjustment'] < 0 else "au-dessus"
                print(f"\nConseil:")
                print(f"   Pour la plupart des articles, prenez {abs(rec['adjustment'])} taille(s) {direction}")
                print(f"   de votre taille habituelle ({usual_size})")
            else:
                print(f"\nConseil:")
                print(f"   Votre taille habituelle ({usual_size}) devrait parfaitement convenir!")

            if rec['alternative_sizes']:
                print(f"\nTailles alternatives a essayer: {', '.join(rec['alternative_sizes'])}")

            recommendations = {'main': rec}

            # Résumé personnalisé
            print("\n" + "=" * 80)
            print("RESUME PERSONNALISE")
            print("=" * 80)

            print(f"\nVotre profil: {user_profile['morphology']}")
            print(f"Taille habituelle: {usual_size}")
            print(f"Taille recommandee: {recommendations['main']['recommended_size_letter']}")

            if recommendations['main']['adjustment'] != 0:
                adj = recommendations['main']['adjustment']
                direction = "plus petit" if adj < 0 else "plus grand"
                print(f"\nConseil principal:")
                print(f"   Commandez generalement {abs(adj)} taille(s) {direction} que votre taille habituelle")
            else:
                print(f"\nConseil principal:")
                print(f"   Commandez votre taille habituelle ({usual_size})")

            # Conseils personnalisés basés sur le cluster
            print(f"\nConseils specifiques pour votre morphologie:")

            cluster_advice = {
                0: [  # Curvy/Plus
                    "- Les clientes comme vous preferent les coupes 'fit' et 'large'",
                    "- Les robes empire et wrap valorisent vos courbes",
                    "- Pour le confort, n'hesitez pas a prendre une taille au-dessus"
                ],
                1: [  # Petite/Slim
                    "- Les coupes 'fit' et ajustees vous reussissent tres bien",
                    "- Les crop tops et mini robes sont parfaits pour vous",
                    "- Attention aux tailles 'large' qui peuvent etre trop amples"
                ],
                2: [  # Standard
                    "- Vous avez une morphologie polyvalente",
                    "- La plupart des styles vous iront bien",
                    "- Suivez nos recommandations pour un fit parfait"
                ]
            }

            for advice in cluster_advice.get(cluster_id, []):
                print(f"   {advice}")

            # Menu
            print("\n" + "=" * 80)
            print("MENU")
            print("=" * 80)
            print("1. Nouveau client")
            print("2. Voir les détails du cluster")
            print("3. Quitter")

            choice = input("\nVotre choix: ").strip()

            if choice == "2":
                print(f"\nDETAILS COMPLETS - Cluster {cluster_id}")
                print("=" * 80)

                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                print(f"Nombre de clientes: {len(cluster_data)}")
                print(f"Poids moyen: {cluster_data['weight_kg'].mean():.1f} kg")
                print(f"Taille moyenne: {cluster_data['height_cm'].mean():.1f} cm")
                print(f"Taille vetement moyenne: {cluster_stats['size_mean']:.1f}")

                print(f"\nDistribution des tailles:")
                size_dist = cluster_data['size'].value_counts().head(10)
                for size, count in size_dist.items():
                    pct = (count / len(cluster_data)) * 100
                    print(f"   Taille {numeric_to_letter_size(size)}: {pct:.1f}%")

                input("\nAppuyez sur Entree pour continuer...")

            elif choice == "3":
                print("\nMerci et a bientot chez ModCloth!")
                break

        except ValueError:
            print("Erreur: Veuillez entrer des valeurs valides!")
        except Exception as e:
            print(f"Erreur: {e}")


# ==================== LANCEMENT ====================
if __name__ == "__main__":
    interactive_system()