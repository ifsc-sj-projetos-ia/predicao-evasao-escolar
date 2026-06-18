from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from data_preparation import generate_and_prepare_data

def run_pipeline():
    # Importa os dados da partição estável
    X_train, X_val, X_test, y_train, y_val, y_test = generate_and_prepare_data()
    
    # --- EXPERIMENTO 1: BASELINE (Regressão Logística) ---
    print("\n[Treinamento] Rodando o modelo Baseline...")
    baseline = LogisticRegression(random_state=42, max_iter=1000)
    baseline.fit(X_train, y_train)
    val_probs_base = baseline.predict_proba(X_val)[:, 1]
    print(f"-> AUC-ROC Baseline (Validação): {roc_auc_score(y_val, val_probs_base):.4f}")

    # --- EXPERIMENTO 2: RANDOM FOREST (Modelo Campeão) ---
    print("\n[Treinamento] Rodando o modelo Avançado (Random Forest)...")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Avaliação final usando a partição de Teste isolada
    test_preds = rf_model.predict(X_test)
    test_probs = rf_model.predict_proba(X_test)[:, 1]

    print("\n ===========================================")
    print("RESULTADOS FINAIS DO TRABALHO (DADOS DE TESTE)")
    print("===========================================")
    print(classification_report(y_test, test_preds))
    print(f"AUC-ROC Final no Teste: {roc_auc_score(y_test, test_probs):.4f}")
    print("===============================================")

if __name__ == "__main__":
    run_pipeline()