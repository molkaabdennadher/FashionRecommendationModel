    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, \
        recall_score
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import lightgbm as lgb
    import pickle
    import warnings
    import time
    
    warnings.filterwarnings('ignore')
    
    print("=" * 90)
    print("VERSION OPTIMISEE - LOGISTIC REGRESSION, MLP + MODELES ULTRA")
    print("=" * 90)
    
    start_time = time.time()
    
    # ==================== CHARGEMENT ====================
    print("\nCHARGEMENT DES DONNEES...")
    train_data = pd.read_excel(r"C:\dataware\venv\train_data.xlsx")
    test_data = pd.read_excel(r"C:\dataware\venv\test_data.xlsx")
    
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    print(f"Train: {X_train.shape[0]:,} echantillons | {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]:,} echantillons")
    
    # Distribution des classes
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nDistribution des classes:")
    for cls, cnt in zip(unique, counts):
        print(f"   Classe {cls}: {cnt:,} ({cnt / len(y_train) * 100:.1f}%)")
    
    # ==================== NORMALISATION POUR LOGISTIC REGRESSION, MLP ====================
    print("\nNORMALISATION DES DONNEES (pour Logistic Regression, MLP)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ==================== HYPERPARAMETRES ====================
    print("\n" + "=" * 90)
    print("CONFIGURATION DES MODELES")
    print("=" * 90)
    
    rf_params = {
        'n_estimators': 500,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    knn_params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'multi_class': 'multinomial',
        'max_iter': 1000,
        'random_state': 42,
        'n_jobs': -1
    }
    
    svm_params = {
        'C': 10.0,
        'penalty': 'l2',
        'solver': 'saga',
        'multi_class': 'multinomial',
        'max_iter': 2000,
        'random_state': 42,
        'n_jobs': -1
    }
    
    mlp_params = {
        'hidden_layer_sizes': (256, 128, 64),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 256,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'max_iter': 500,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20
    }
    
    # ==================== ENTRAÎNEMENT DES MODÈLES ====================
    print("\nENTRAINEMENT DES MODELES...")
    
    models_results = {}
    
    # 1. Random Forest
    print("\n[1/8] Random Forest (500 trees)...")
    t0 = time.time()
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
    recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
    models_results['RandomForest'] = {
        'model': rf_model,
        'pred': y_pred_rf,
        'acc': acc_rf,
        'f1': f1_rf,
        'precision': precision_rf,
        'recall': recall_rf
    }
    print(f"   Accuracy: {acc_rf:.4f} ({acc_rf * 100:.2f}%) | F1: {f1_rf:.4f} | Temps: {time.time() - t0:.1f}s")
    
    # 2. Logistic Regression (L2 regularization)
    print("\n[2/8] Logistic Regression (L2, multinomial)...")
    t0 = time.time()
    lr1_model = LogisticRegression(**knn_params)
    lr1_model.fit(X_train_scaled, y_train)
    y_pred_lr1 = lr1_model.predict(X_test_scaled)
    acc_lr1 = accuracy_score(y_test, y_pred_lr1)
    f1_lr1 = f1_score(y_test, y_pred_lr1, average='weighted')
    precision_lr1 = precision_score(y_test, y_pred_lr1, average='weighted')
    recall_lr1 = recall_score(y_test, y_pred_lr1, average='weighted')
    models_results['LogisticRegression_L2'] = {
        'model': lr1_model,
        'pred': y_pred_lr1,
        'acc': acc_lr1,
        'f1': f1_lr1,
        'precision': precision_lr1,
        'recall': recall_lr1
    }
    print(f"   Accuracy: {acc_lr1:.4f} ({acc_lr1 * 100:.2f}%) | F1: {f1_lr1:.4f} | Temps: {time.time() - t0:.1f}s")
    
    # 3. Logistic Regression SAGA (C=10)
    print("\n[3/8] Logistic Regression SAGA (C=10)...")
    t0 = time.time()
    lr2_model = LogisticRegression(**svm_params)
    lr2_model.fit(X_train_scaled, y_train)
    y_pred_lr2 = lr2_model.predict(X_test_scaled)
    acc_lr2 = accuracy_score(y_test, y_pred_lr2)
    f1_lr2 = f1_score(y_test, y_pred_lr2, average='weighted')
    precision_lr2 = precision_score(y_test, y_pred_lr2, average='weighted')
    recall_lr2 = recall_score(y_test, y_pred_lr2, average='weighted')
    models_results['LogisticRegression_SAGA'] = {
        'model': lr2_model,
        'pred': y_pred_lr2,
        'acc': acc_lr2,
        'f1': f1_lr2,
        'precision': precision_lr2,
        'recall': recall_lr2
    }
    print(f"   Accuracy: {acc_lr2:.4f} ({acc_lr2 * 100:.2f}%) | F1: {f1_lr2:.4f} | Temps: {time.time() - t0:.1f}s")
    
    # 4. MLP (Neural Network)
    print("\n[4/8] Multi-Layer Perceptron (256-128-64)...")
    t0 = time.time()
    mlp_model = MLPClassifier(**mlp_params)
    mlp_model.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')
    precision_mlp = precision_score(y_test, y_pred_mlp, average='weighted')
    recall_mlp = recall_score(y_test, y_pred_mlp, average='weighted')
    models_results['MLP'] = {
        'model': mlp_model,
        'pred': y_pred_mlp,
        'acc': acc_mlp,
        'f1': f1_mlp,
        'precision': precision_mlp,
        'recall': recall_mlp
    }
    print(f"   Accuracy: {acc_mlp:.4f} ({acc_mlp * 100:.2f}%) | F1: {f1_mlp:.4f} | Temps: {time.time() - t0:.1f}s")
    
    # 5. XGBoost ULTRA-AGRESSIF
    print("\n[5/8] XGBoost ULTRA (2000 estimators, depth=25)...")
    t0 = time.time()
    xgb_ultra = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=25,
        learning_rate=0.02,
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
    xgb_ultra.fit(X_train, y_train)
    y_pred_xgb_ultra = xgb_ultra.predict(X_test)
    acc_xgb_ultra = accuracy_score(y_test, y_pred_xgb_ultra)
    f1_xgb_ultra = f1_score(y_test, y_pred_xgb_ultra, average='weighted')
    precision_xgb_ultra = precision_score(y_test, y_pred_xgb_ultra, average='weighted')
    recall_xgb_ultra = recall_score(y_test, y_pred_xgb_ultra, average='weighted')
    models_results['XGBoost_Ultra'] = {
        'model': xgb_ultra,
        'pred': y_pred_xgb_ultra,
        'acc': acc_xgb_ultra,
        'f1': f1_xgb_ultra,
        'precision': precision_xgb_ultra,
        'recall': recall_xgb_ultra
    }
    print(
        f"   Accuracy: {acc_xgb_ultra:.4f} ({acc_xgb_ultra * 100:.2f}%) | F1: {f1_xgb_ultra:.4f} | Temps: {time.time() - t0:.1f}s")
    
    # 6. LightGBM ULTRA-AGRESSIF
    print("\n[6/8] LightGBM ULTRA (2000 estimators, leaves=200)...")
    t0 = time.time()
    lgb_ultra = lgb.LGBMClassifier(
        n_estimators=2000,
        max_depth=25,
        learning_rate=0.02,
        num_leaves=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.5,
        min_child_samples=10,
        objective='multiclass',
        num_class=6,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True
    )
    lgb_ultra.fit(X_train, y_train)
    y_pred_lgb_ultra = lgb_ultra.predict(X_test)
    acc_lgb_ultra = accuracy_score(y_test, y_pred_lgb_ultra)
    f1_lgb_ultra = f1_score(y_test, y_pred_lgb_ultra, average='weighted')
    precision_lgb_ultra = precision_score(y_test, y_pred_lgb_ultra, average='weighted')
    recall_lgb_ultra = recall_score(y_test, y_pred_lgb_ultra, average='weighted')
    models_results['LightGBM_Ultra'] = {
        'model': lgb_ultra,
        'pred': y_pred_lgb_ultra,
        'acc': acc_lgb_ultra,
        'f1': f1_lgb_ultra,
        'precision': precision_lgb_ultra,
        'recall': recall_lgb_ultra
    }
    print(
        f"   Accuracy: {acc_lgb_ultra:.4f} ({acc_lgb_ultra * 100:.2f}%) | F1: {f1_lgb_ultra:.4f} | Temps: {time.time() - t0:.1f}s")
    
    # 7. XGBoost EXTREME (3000 estimators)
    print("\n[7/8] XGBoost EXTREME (3000 estimators, depth=30)...")
    t0 = time.time()
    xgb_extreme = xgb.XGBClassifier(
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
    xgb_extreme.fit(X_train, y_train)
    y_pred_xgb_extreme = xgb_extreme.predict(X_test)
    acc_xgb_extreme = accuracy_score(y_test, y_pred_xgb_extreme)
    f1_xgb_extreme = f1_score(y_test, y_pred_xgb_extreme, average='weighted')
    precision_xgb_extreme = precision_score(y_test, y_pred_xgb_extreme, average='weighted')
    recall_xgb_extreme = recall_score(y_test, y_pred_xgb_extreme, average='weighted')
    models_results['XGBoost_Extreme'] = {
        'model': xgb_extreme,
        'pred': y_pred_xgb_extreme,
        'acc': acc_xgb_extreme,
        'f1': f1_xgb_extreme,
        'precision': precision_xgb_extreme,
        'recall': recall_xgb_extreme
    }
    print(
        f"   Accuracy: {acc_xgb_extreme:.4f} ({acc_xgb_extreme * 100:.2f}%) | F1: {f1_xgb_extreme:.4f} | Temps: {time.time() - t0:.1f}s")
    
    # 8. STACKING ENSEMBLE
    print("\n[8/8] Stacking Ensemble (Tous les modeles)...")
    t0 = time.time()
    
    # Créer des modèles wrapper pour utiliser les données normalisées si nécessaire
    from sklearn.base import BaseEstimator, ClassifierMixin
    
    
    class ScaledModelWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
    
        def fit(self, X, y):
            X_scaled = self.scaler.transform(X) if hasattr(self.scaler, 'transform') else X
            self.model.fit(X_scaled, y)
            return self
    
        def predict(self, X):
            X_scaled = self.scaler.transform(X) if hasattr(self.scaler, 'transform') else X
            return self.model.predict(X_scaled)
    
        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X) if hasattr(self.scaler, 'transform') else X
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_scaled)
            else:
                # Pour SVM sans probability=True
                return self.model.decision_function(X_scaled)
    
    
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', rf_model),
            ('lr1', ScaledModelWrapper(lr1_model, scaler)),
            ('lr2', ScaledModelWrapper(lr2_model, scaler)),
            ('mlp', ScaledModelWrapper(mlp_model, scaler)),
            ('xgb_ultra', xgb_ultra),
            ('lgb_ultra', lgb_ultra)
        ],
        final_estimator=xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        ),
        cv=3,
        n_jobs=-1
    )
    stacking_clf.fit(X_train, y_train)
    y_pred_stack = stacking_clf.predict(X_test)
    acc_stack = accuracy_score(y_test, y_pred_stack)
    f1_stack = f1_score(y_test, y_pred_stack, average='weighted')
    precision_stack = precision_score(y_test, y_pred_stack, average='weighted')
    recall_stack = recall_score(y_test, y_pred_stack, average='weighted')
    models_results['Stacking_Ensemble'] = {
        'model': stacking_clf,
        'pred': y_pred_stack,
        'acc': acc_stack,
        'f1': f1_stack,
        'precision': precision_stack,
        'recall': recall_stack
    }
    print(f"   Accuracy: {acc_stack:.4f} ({acc_stack * 100:.2f}%) | F1: {f1_stack:.4f} | Temps: {time.time() - t0:.1f}s")
    
    # ==================== RESULTATS FINAUX ====================
    print("\n" + "=" * 90)
    print("RESULTATS FINAUX - TOUS LES MODELES")
    print("=" * 90)
    
    results_df = pd.DataFrame({
        'Modele': list(models_results.keys()),
        'Accuracy': [v['acc'] for v in models_results.values()],
        'F1-Score': [v['f1'] for v in models_results.values()],
        'Precision': [v['precision'] for v in models_results.values()],
        'Recall': [v['recall'] for v in models_results.values()]
    })
    
    results_df = results_df.sort_values('Accuracy', ascending=False)
    print("\n" + results_df.to_string(index=False))
    
    # Meilleur modèle
    best_model_name = results_df.iloc[0]['Modele']
    best_acc = results_df.iloc[0]['Accuracy']
    best_pred = models_results[best_model_name]['pred']
    
    print("\n" + "=" * 90)
    print(f"MEILLEUR MODELE: {best_model_name}")
    print(f"   Accuracy:  {best_acc:.4f} ({best_acc * 100:.2f}%)")
    print(f"   F1-Score:  {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"   Precision: {results_df.iloc[0]['Precision']:.4f}")
    print(f"   Recall:    {results_df.iloc[0]['Recall']:.4f}")
    print("=" * 90)
    
    if best_acc >= 0.80:
        print(f"\nOBJECTIF ATTEINT! Accuracy >= 80%")
        print(f"Depassement: +{(best_acc - 0.80) * 100:.2f} points")
    else:
        gap = (0.80 - best_acc) * 100
        print(f"\nGap restant: {gap:.2f} points vers 80%")
    
    # ==================== CLASSIFICATION REPORT ====================
    print("\n" + "=" * 90)
    print("CLASSIFICATION REPORT - MEILLEUR MODELE")
    print("=" * 90)
    
    target_names = ['athletic', 'full bust', 'hourglass', 'pear', 'petite', 'straight & narrow']
    print(classification_report(y_test, best_pred, target_names=target_names))
    
    # ==================== VISUALISATIONS ====================
    print("\nGENERATION DES VISUALISATIONS...")
    
    # 1. Graphique de comparaison des modèles
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    bars = ax.barh(results_df['Modele'], results_df['Accuracy'], color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Accuracy', fontweight='bold', fontsize=14)
    ax.set_title('Performance de Tous les Modeles', fontweight='bold', fontsize=16)
    ax.set_xlim([0, 1])
    ax.axvline(x=0.8, color='red', linestyle='--', linewidth=3, label='Objectif 80%', alpha=0.7)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=12)
    
    for bar, acc in zip(bars, results_df['Accuracy']):
        width = bar.get_width()
        color = 'green' if acc >= 0.80 else 'red'
        weight = 'bold' if acc >= 0.80 else 'normal'
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{acc * 100:.2f}%', va='center', fontweight=weight, color=color, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(r'C:\dataware\venv\01_comparison_all_models.png', dpi=300, bbox_inches='tight')
    print("   [OK] 01_comparison_all_models.png")
    plt.close()
    
    # 2. Matrice de confusion pour CHAQUE modèle
    target_names_short = ['athletic', 'full bust', 'hourglass', 'pear', 'petite', 'straight']
    
    print("\nGeneration des matrices de confusion pour chaque modele...")
    for idx, (model_name, data) in enumerate(models_results.items(), 1):
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_test, data['pred'])
    
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax, square=True,
                    cbar_kws={'label': 'Nombre de predictions'},
                    xticklabels=target_names_short, yticklabels=target_names_short,
                    annot_kws={'fontsize': 9})
    
        ax.set_xlabel('Prediction', fontweight='bold', fontsize=12)
        ax.set_ylabel('Reel', fontweight='bold', fontsize=12)
    
        title = f'Matrice de Confusion - {model_name}\n'
        title += f'Accuracy: {data["acc"] * 100:.2f}% | F1: {data["f1"]:.4f} | '
        title += f'Precision: {data["precision"]:.4f} | Recall: {data["recall"]:.4f}'
    
        ax.set_title(title, fontweight='bold', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
    
        filename = f'C:\\dataware\\venv\\{idx + 1:02d}_confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   [OK] {idx + 1:02d}_confusion_matrix_{model_name}.png")
        plt.close()
    
    # 3. Graphique comparatif de toutes les métriques
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        bars = ax.barh(results_df['Modele'], results_df[metric], color=colors_palette, alpha=0.8, edgecolor='black')
        ax.set_xlabel(metric, fontweight='bold', fontsize=12)
        ax.set_title(f'{metric} - Tous les Modeles', fontweight='bold', fontsize=14)
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
    
        for bar, val in zip(bars, results_df[metric]):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val * 100:.2f}%' if metric == 'Accuracy' else f'{val:.3f}',
                    va='center', fontweight='bold', fontsize=10)
    
    plt.suptitle('Comparaison de Toutes les Metriques', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(r'C:\dataware\venv\10_all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("   [OK] 10_all_metrics_comparison.png")
    plt.close()
    
    # 4. Feature Importance (meilleur modèle si applicable)
    if best_model_name in ['XGBoost_Ultra', 'LightGBM_Ultra', 'XGBoost_Extreme', 'RandomForest']:
        fig, ax = plt.subplots(figsize=(12, 10))
    
        best_model_obj = models_results[best_model_name]['model']
    
        if hasattr(best_model_obj, 'feature_importances_'):
            importances = best_model_obj.feature_importances_
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
            indices = np.argsort(importances)[-30:]
    
            ax.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance', fontweight='bold', fontsize=12)
            ax.set_title(f'Top 30 Features - {best_model_name}', fontweight='bold', fontsize=14)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(r'C:\dataware\venv\11_feature_importance_best_model.png', dpi=300, bbox_inches='tight')
            print("   [OK] 11_feature_importance_best_model.png")
            plt.close()
    
    # ==================== SAUVEGARDE ====================
    print("\nSAUVEGARDE DES MODELES ET RESULTATS...")
    
    # Sauvegarder aussi le scaler
    with open(r'C:\dataware\venv\scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("   [OK] scaler.pkl")
    
    for name, data in models_results.items():
        filename = f"C:\\dataware\\venv\\model_{name.lower().replace(' ', '_')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data['model'], f)
        print(f"   [OK] model_{name}.pkl")
    
    results_df.to_excel(r'C:\dataware\venv\results_all_models.xlsx', index=False)
    print("   [OK] results_all_models.xlsx")
    
    results_df.to_csv(r'C:\dataware\venv\results_all_models.csv', index=False)
    print("   [OK] results_all_models.csv")
    
    # ==================== RESUME FINAL ====================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 90)
    print("RESUME FINAL")
    print("=" * 90)
    
    print(f"\nPERFORMANCE:")
    print(f"   Meilleur modele: {best_model_name}")
    print(f"   Accuracy:  {best_acc * 100:.2f}%")
    print(f"   F1-Score:  {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"   Precision: {results_df.iloc[0]['Precision']:.4f}")
    print(f"   Recall:    {results_df.iloc[0]['Recall']:.4f}")
    
    print(f"\nTEMPS TOTAL: {total_time / 60:.1f} minutes")
    
    print(f"\nCONFIGURATION UTILISEE:")
    print(f"   - 9 modeles entraines")
    print(f"   - Logistic Regression (L2 + SAGA)")
    print(f"   - MLP (normalise)")
    print(f"   - RandomForest (500 trees)")
    print(f"   - XGBoost Ultra + Extreme")
    print(f"   - LightGBM Ultra")
    print(f"   - Stacking Ensemble")
    
    print(f"\nTOP 3 MODELES:")
    for i in range(min(3, len(results_df))):
        model = results_df.iloc[i]
        print(f"   {i + 1}. {model['Modele']}: {model['Accuracy'] * 100:.2f}% (F1: {model['F1-Score']:.4f})")
    
    print(f"\nFICHIERS GENERES:")
    print(f"   - {len(models_results) + 1} modeles .pkl (+ scaler)")
    print(f"   - {len(models_results) + 3} fichiers PNG")
    print(f"   - 2 fichiers resultats (Excel + CSV)")
    
    print("\n" + "=" * 90)
    print("TERMINE!")
    print("=" * 90)