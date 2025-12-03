import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION DES CHEMINS
# ============================================================

# Chemins des fichiers
path_finaledata = r"C:\dataware\venv\finaledata.xlsx"
path_modcloth = r"C:\Users\pc msi\Downloads\archive\modcloth_dataaaaaaaaa_clean.xlsx"

# Chemin de sortie
output_path = r"C:\dataware\venv\finaledata_enrichi.xlsx"

print("=" * 70)
print("FUSION DE DATASETS PAR USER_ID")
print("=" * 70)

try:
    # ============================================================
    # ÉTAPE 1: CHARGEMENT DES DONNÉES
    # ============================================================

    print("\n[1/5] Chargement des données...")

    # Charger finaledata
    if os.path.exists(path_finaledata):
        df_finale = pd.read_excel(path_finaledata)
        print(f"✓ finaledata chargé: {len(df_finale):,} lignes, {len(df_finale.columns)} colonnes")
    else:
        raise FileNotFoundError(f"Fichier non trouvé: {path_finaledata}")

    # Charger modcloth
    if os.path.exists(path_modcloth):
        df_modcloth = pd.read_excel(path_modcloth)
        print(f"✓ modcloth chargé: {len(df_modcloth):,} lignes, {len(df_modcloth.columns)} colonnes")
    else:
        raise FileNotFoundError(f"Fichier non trouvé: {path_modcloth}")

    # ============================================================
    # ÉTAPE 2: VÉRIFICATION DES COLONNES
    # ============================================================

    print("\n[2/5] Vérification des colonnes...")

    print(f"\nColonnes de finaledata:")
    print(df_finale.columns.tolist())

    print(f"\nColonnes de modcloth:")
    print(df_modcloth.columns.tolist())

    # Vérifier la présence de user_id
    if 'user_id' not in df_finale.columns:
        raise ValueError("La colonne 'user_id' n'existe pas dans finaledata!")

    if 'user_id' not in df_modcloth.columns:
        raise ValueError("La colonne 'user_id' n'existe pas dans modcloth!")

    print(f"\n✓ Colonne 'user_id' trouvée dans les deux datasets")

    # Colonnes à fusionner depuis modcloth
    colonnes_a_ajouter = ['fit', 'bust size', 'weight', 'height', 'size', 'age']
    colonnes_disponibles = [col for col in colonnes_a_ajouter if col in df_modcloth.columns]
    colonnes_manquantes = [col for col in colonnes_a_ajouter if col not in df_modcloth.columns]

    print(f"\nColonnes disponibles à ajouter: {colonnes_disponibles}")
    if colonnes_manquantes:
        print(f"⚠ Colonnes manquantes dans modcloth: {colonnes_manquantes}")

    # ============================================================
    # ÉTAPE 3: PRÉPARATION DES DONNÉES
    # ============================================================

    print("\n[3/5] Préparation des données...")

    # Vérifier les doublons de user_id dans modcloth
    nb_doublons_modcloth = df_modcloth['user_id'].duplicated().sum()

    if nb_doublons_modcloth > 0:
        print(f"⚠ Attention: {nb_doublons_modcloth:,} doublons de user_id dans modcloth")
        print("  → Conservation de la première occurrence pour chaque user_id")
        df_modcloth_unique = df_modcloth.drop_duplicates(subset='user_id', keep='first')
    else:
        df_modcloth_unique = df_modcloth.copy()

    print(f"✓ Users uniques dans modcloth: {len(df_modcloth_unique):,}")
    print(f"✓ Lignes dans finaledata: {len(df_finale):,}")

    # Sélectionner uniquement les colonnes nécessaires de modcloth
    colonnes_fusion = ['user_id'] + colonnes_disponibles
    df_modcloth_selection = df_modcloth_unique[colonnes_fusion].copy()

    # ============================================================
    # ÉTAPE 4: FUSION DES DATASETS
    # ============================================================

    print("\n[4/5] Fusion des datasets...")

    # Effectuer la jointure LEFT (garder toutes les lignes de finaledata)
    df_enrichi = df_finale.merge(
        df_modcloth_selection,
        on='user_id',
        how='left',
        suffixes=('', '_modcloth')
    )

    print(f"✓ Fusion réussie: {len(df_enrichi):,} lignes, {len(df_enrichi.columns)} colonnes")

    # Statistiques de fusion
    nb_matches = df_enrichi[colonnes_disponibles[0]].notna().sum() if colonnes_disponibles else 0
    nb_non_matches = len(df_enrichi) - nb_matches

    print(f"\n  → Lignes avec correspondance: {nb_matches:,} ({nb_matches / len(df_enrichi) * 100:.1f}%)")
    print(f"  → Lignes sans correspondance: {nb_non_matches:,} ({nb_non_matches / len(df_enrichi) * 100:.1f}%)")

    # ============================================================
    # ÉTAPE 5: EXPORTATION
    # ============================================================

    print("\n[5/5] Exportation vers Excel...")

    # Exporter le fichier enrichi
    df_enrichi.to_excel(output_path, index=False, sheet_name='Données Enrichies')
    print(f"✓ Fichier exporté: {output_path}")

    # Créer un rapport détaillé
    rapport_path = r"C:\dataware\venv\rapport_fusion.xlsx"

    with pd.ExcelWriter(rapport_path, engine='openpyxl') as writer:
        # Onglet 1: Données enrichies
        df_enrichi.to_excel(writer, sheet_name='Données Enrichies', index=False)

        # Onglet 2: Statistiques
        stats = pd.DataFrame({
            'Métrique': [
                'Total lignes finaledata',
                'Total lignes modcloth',
                'Users uniques modcloth',
                'Lignes après fusion',
                'Lignes avec correspondance',
                'Lignes sans correspondance',
                'Taux de correspondance (%)',
                'Colonnes ajoutées'
            ],
            'Valeur': [
                len(df_finale),
                len(df_modcloth),
                len(df_modcloth_unique),
                len(df_enrichi),
                nb_matches,
                nb_non_matches,
                f"{nb_matches / len(df_enrichi) * 100:.2f}",
                ', '.join(colonnes_disponibles)
            ]
        })
        stats.to_excel(writer, sheet_name='Statistiques', index=False)

        # Onglet 3: Aperçu des nouvelles colonnes
        if colonnes_disponibles:
            apercu = df_enrichi[['user_id'] + colonnes_disponibles].head(100)
            apercu.to_excel(writer, sheet_name='Aperçu Nouvelles Colonnes', index=False)

    print(f"✓ Rapport détaillé créé: {rapport_path}")

    # ============================================================
    # RÉSUMÉ FINAL
    # ============================================================

    print("\n" + "=" * 70)
    print("RÉSUMÉ DE LA FUSION")
    print("=" * 70)
    print(f"Dataset original (finaledata):     {len(df_finale):,} lignes")
    print(f"Dataset source (modcloth):         {len(df_modcloth):,} lignes")
    print(f"Dataset enrichi:                   {len(df_enrichi):,} lignes")
    print(f"Colonnes ajoutées:                 {len(colonnes_disponibles)}")
    print(f"Colonnes disponibles:              {', '.join(colonnes_disponibles)}")
    if colonnes_manquantes:
        print(f"Colonnes non trouvées:             {', '.join(colonnes_manquantes)}")
    print(f"\nTaux de correspondance:            {nb_matches / len(df_enrichi) * 100:.2f}%")
    print(f"\nFichier final:                     {output_path}")
    print(f"Rapport détaillé:                  {rapport_path}")
    print("=" * 70)

    # Afficher un échantillon des données
    print("\n📊 APERÇU DES DONNÉES ENRICHIES (5 premières lignes):")
    print(df_enrichi[['user_id'] + colonnes_disponibles].head())

except FileNotFoundError as e:
    print(f"\n❌ Erreur: {e}")
    print("\n💡 Vérifiez que les chemins des fichiers sont corrects:")
    print(f"   - finaledata: {path_finaledata}")
    print(f"   - modcloth:   {path_modcloth}")

except ValueError as e:
    print(f"\n❌ Erreur: {e}")

except Exception as e:
    print(f"\n❌ Erreur inattendue: {e}")
    import traceback

    traceback.print_exc()

print("\n✓ Script terminé")