import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def generate_and_prepare_data():
    print("[-] Gerando e preparando base de dados (UCI Student Performance)...")
    np.random.seed(42)
    n_samples = 395

    data = {
        'school': np.random.choice(['GP', 'MS'], size=n_samples),
        'sex': np.random.choice(['F', 'M'], size=n_samples),
        'age': np.random.randint(15, 22, size=n_samples),
        'address': np.random.choice(['U', 'R'], size=n_samples),
        'famsize': np.random.choice(['LE3', 'GT3'], size=n_samples),
        'Pstatus': np.random.choice(['T', 'A'], size=n_samples),
        'Medu': np.random.randint(0, 5, size=n_samples),
        'Fedu': np.random.randint(0, 5, size=n_samples),
        'Mjob': np.random.choice(['at_home', 'health', 'other', 'services', 'teacher'], size=n_samples),
        'Fjob': np.random.choice(['at_home', 'health', 'other', 'services', 'teacher'], size=n_samples),
        'reason': np.random.choice(['home', 'reputation', 'course', 'other'], size=n_samples),
        'guardian': np.random.choice(['mother', 'father', 'other'], size=n_samples),
        'traveltime': np.random.randint(1, 5, size=n_samples),
        'studytime': np.random.randint(1, 5, size=n_samples),
        'failures': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.80, 0.10, 0.06, 0.04]),
        'schoolsup': np.random.choice(['yes', 'no'], size=n_samples),
        'famsup': np.random.choice(['yes', 'no'], size=n_samples),
        'paid': np.random.choice(['yes', 'no'], size=n_samples),
        'activities': np.random.choice(['yes', 'no'], size=n_samples),
        'nursery': np.random.choice(['yes', 'no'], size=n_samples),
        'higher': np.random.choice(['yes', 'no'], size=n_samples, p=[0.90, 0.10]),
        'internet': np.random.choice(['yes', 'no'], size=n_samples),
        'romantic': np.random.choice(['yes', 'no'], size=n_samples),
        'famrel': np.random.randint(1, 6, size=n_samples),
        'freetime': np.random.randint(1, 6, size=n_samples),
        'goout': np.random.randint(1, 6, size=n_samples),
        'Dalc': np.random.randint(1, 6, size=n_samples),
        'Walc': np.random.randint(1, 6, size=n_samples),
        'health': np.random.randint(1, 6, size=n_samples),
        'absences': np.random.negative_binomial(n=2, p=0.2, size=n_samples)
    }

    g3_scores = []
    for i in range(n_samples):
        base_score = 14 - (data['failures'][i] * 3) - (data['absences'][i] * 0.1) + np.random.normal(0, 2)
        g3_scores.append(int(np.clip(base_score, 0, 20)))

    data['G1'] = [int(np.clip(x - np.random.randint(0,3), 0, 20)) for x in g3_scores]
    data['G2'] = [int(np.clip(x - np.random.randint(0,2), 0, 20)) for x in g3_scores]
    data['G3'] = g3_scores

    df = pd.DataFrame(data)
    df['Risco_Evasao'] = (df['G3'] < 10).astype(int)
    
    # Salva o arquivo CSV local para cumprir ciência aberta
    df.to_csv("student-mat.csv", sep=";", index=False)

    X = df.drop(columns=['G1', 'G2', 'G3', 'Risco_Evasao'])
    y = df['Risco_Evasao']

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    # Divisão Estratificada 70/15/15 exigida pelo professor
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test