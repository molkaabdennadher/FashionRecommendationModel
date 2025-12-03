import pandas as pd
from sqlalchemy import create_engine
import urllib
import warnings

# Supprimer l'avertissement pandas (cosmétique)
warnings.filterwarnings('ignore', message='.*SQLAlchemy connectable.*')

# Configuration de la connexion
server = 'DESKTOP-D7M33RA\\MSSQLSERVER01'
database = 'DW_Fashion'

# Créer la connexion SQLAlchemy avec urllib
params = urllib.parse.quote_plus(
    f"DRIVER={{SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"Trusted_Connection=yes;"
)

connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
engine = create_engine(connection_string, fast_executemany=True)

try:
    print("=" * 60)
    print("JOINTURES SQL SERVER - DW_Fashion")
    print("=" * 60)

    # Test de connexion
    with engine.connect() as conn:
        print("✓ Connexion réussie à SQL Server\n")

    # Jointure 1: FaitVente + DimProduit
    print("=== Jointure 1: FaitVente + DimProduit ===")
    query1 = """
    SELECT 
        fv.product_PK,
        fv.Date_PK,
        fv.Client_PK,
        fv.Canal_PK,
        fv.price,
        dp.product_code,
        dp.prod_name,
        dp.graphical_appearance_name,
        dp.index_name,
        dp.garment_group_name
    FROM dbo.DimFaitVente fv
    INNER JOIN dbo.DimProduit dp 
        ON fv.product_PK = dp.product_PK
    """

    df1 = pd.read_sql(query1, engine)
    print(f"Nombre de lignes: {len(df1):,}")
    print(df1.head())

    # Jointure 2: FaitVente + DimProduit + DimType
    print("\n=== Jointure 2: FaitVente + DimProduit + DimType ===")
    query2 = """
    SELECT 
        fv.product_PK,
        fv.Date_PK,
        fv.Client_PK,
        fv.price,
        dp.prod_name,
        dp.garment_group_name,
        dt.product_type_name,
        dt.product_type_no
    FROM dbo.DimFaitVente fv
    INNER JOIN dbo.DimProduit dp 
        ON fv.product_PK = dp.product_PK
    INNER JOIN dbo.DimType dt 
        ON dp.type_FK = dt.type_PK
    """

    df2 = pd.read_sql(query2, engine)
    print(f"Nombre de lignes: {len(df2):,}")
    print(df2.head())

    # Jointure 3: FaitVente + Produit + Type + Couleur
    print("\n=== Jointure 3: Ventes Complètes (avec Couleur) ===")
    query3 = """
    SELECT 
        fv.product_PK,
        fv.Date_PK,
        fv.Client_PK,
        fv.Canal_PK,
        fv.price,
        dp.prod_name,
        dp.garment_group_name,
        dt.product_type_name,
        dc.colour_group_name
    FROM dbo.DimFaitVente fv
    INNER JOIN dbo.DimProduit dp ON fv.product_PK = dp.product_PK
    INNER JOIN dbo.DimType dt ON dp.type_FK = dt.type_PK
    INNER JOIN dbo.DimCouleur dc ON dp.couleur_FK = dc.couleur_PK
    """

    df3 = pd.read_sql(query3, engine)
    print(f"Nombre de lignes: {len(df3):,}")
    print(df3.head())

    # Jointure 4: Statistiques par Type de Produit
    print("\n=== Jointure 4: Statistiques par Type ===")
    query4 = """
    SELECT 
        dt.product_type_name,
        COUNT(*) AS nombre_ventes,
        SUM(fv.price) AS chiffre_affaires_total,
        AVG(fv.price) AS prix_moyen,
        MIN(fv.price) AS prix_min,
        MAX(fv.price) AS prix_max
    FROM dbo.DimFaitVente fv
    INNER JOIN dbo.DimProduit dp ON fv.product_PK = dp.product_PK
    INNER JOIN dbo.DimType dt ON dp.type_FK = dt.type_PK
    GROUP BY dt.product_type_name
    ORDER BY chiffre_affaires_total DESC
    """

    df4 = pd.read_sql(query4, engine)
    print(df4.head(10))

    # Sauvegarder les résultats en Excel
    print("\n" + "=" * 60)
    print("EXPORTATION DES DONNÉES EN EXCEL")
    print("=" * 60)

    # Option 1: Fichiers Excel séparés
    df2.to_excel('ventes_type_produit.xlsx', index=False, sheet_name='Ventes')
    df3.to_excel('ventes_completes.xlsx', index=False, sheet_name='Ventes Complètes')
    df4.to_excel('stats_par_type.xlsx', index=False, sheet_name='Statistiques')
    df5.to_excel('top10_produits.xlsx', index=False, sheet_name='Top 10')
    df6.to_excel('ventes_par_canal.xlsx', index=False, sheet_name='Par Canal')
    df7.to_excel('top10_couleurs.xlsx', index=False, sheet_name='Top Couleurs')

    print("✓ 6 fichiers Excel exportés avec succès")

    # Option 2: UN SEUL fichier Excel avec plusieurs onglets
    with pd.ExcelWriter('DW_Fashion_Analyse_Complete.xlsx', engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Ventes_Produits', index=False)
        df2.to_excel(writer, sheet_name='Ventes_Type', index=False)
        df3.to_excel(writer, sheet_name='Ventes_Complètes', index=False)
        df4.to_excel(writer, sheet_name='Stats_Type', index=False)
        df5.to_excel(writer, sheet_name='Top10_Produits', index=False)
        df6.to_excel(writer, sheet_name='Ventes_Canal', index=False)
        df7.to_excel(writer, sheet_name='Top10_Couleurs', index=False)

    print("✓ Fichier Excel consolidé créé: DW_Fashion_Analyse_Complete.xlsx")

    # Résumé global
    print("\n" + "=" * 60)
    print("RÉSUMÉ GLOBAL")
    print("=" * 60)
    print(f"Total des ventes: {len(df1):,}")
    print(f"Nombre de types de produits: {len(df4)}")
    print(f"Nombre de couleurs: {len(df7)}")
    print(f"Chiffre d'affaires total: {df4['chiffre_affaires_total'].sum():.2f}")
    print(f"Prix moyen par vente: {df1['price'].mean():.4f}")
    print(f"Prix min: {df1['price'].min():.4f}")
    print(f"Prix max: {df1['price'].max():.4f}")

except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback

    traceback.print_exc()

finally:
    engine.dispose()
    print("\n✓ Connexion fermée")


# ============================================================
# FONCTIONS RÉUTILISABLES
# ============================================================

def executer_requete(query, server='DESKTOP-D7M33RA\\MSSQLSERVER01',
                     database='DW_Fashion'):
    """
    Fonction pour exécuter une requête SQL personnalisée

    Args:
        query (str): Requête SQL à exécuter
        server (str): Nom du serveur
        database (str): Nom de la base de données

    Returns:
        DataFrame: Résultats de la requête
    """
    try:
        params = urllib.parse.quote_plus(
            f"DRIVER={{SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
        )
        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
        engine = create_engine(connection_string)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            df = pd.read_sql(query, engine)

        engine.dispose()
        return df

    except Exception as e:
        print(f"Erreur: {e}")
        return None


# EXEMPLES D'UTILISATION
"""
# Exemple 1: Ventes d'un produit spécifique
requete = '''
SELECT * 
FROM dbo.DimFaitVente fv
INNER JOIN dbo.DimProduit dp ON fv.product_PK = dp.product_PK
WHERE dp.prod_name LIKE '%Skinny%'
'''
resultat = executer_requete(requete)
print(resultat)

# Exemple 2: Ventes par mois (si vous avez une table DimTemps)
requete = '''
SELECT 
    dt.year_month,
    COUNT(*) as nb_ventes,
    SUM(fv.price) as ca
FROM dbo.DimFaitVente fv
INNER JOIN dbo.DimTemps dt ON fv.Date_PK = dt.date_PK
GROUP BY dt.year_month
ORDER BY dt.year_month
'''
resultat = executer_requete(requete)
print(resultat)
"""