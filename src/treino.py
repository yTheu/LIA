import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

try:
    df_train = pd.read_csv('alfabeto_landmarks_treino.csv')
    df_test = pd.read_csv('alfabeto_landmarks_teste.csv')
except FileNotFoundError:
    print("Erro: 'alfabeto_landmarks_treino.csv' ou 'alfabeto_landmarks_teste.csv' não encontrado.")
    exit()

print(f"Dataset de TREINO carregado com {len(df_train)} amostras.")
print(f"Dataset de TESTE carregado com {len(df_test)} amostras.")

MIN_AMOSTRAS_POR_CLASSE = 5 
contagem_classes = df_train['class'].value_counts()
classes_para_manter = contagem_classes[contagem_classes >= MIN_AMOSTRAS_POR_CLASSE].index

df_train = df_train[df_train['class'].isin(classes_para_manter)]
df_test = df_test[df_test['class'].isin(classes_para_manter)]

print(f"Classes com menos de {MIN_AMOSTRAS_POR_CLASSE} amostras removidas.")
print("Classes que serão treinadas:")
print(df_train['class'].value_counts().sort_index())

labels_train = df_train['class']
features_train = df_train.drop(columns=['class'])

labels_test = df_test['class']
features_test = df_test.drop(columns=['class'])

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Iniciando Validação Cruzada (Cross-Validation)...")
scores = cross_val_score(rf, features_train, labels_train, cv=5, scoring='accuracy')

print("Validação concluída!")
print(f"Acurácias de cada 'fold': {scores}")
print(f"Acurácia média da validação: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

print("\nTreinando o modelo...")
rf.fit(features_train, labels_train)
print("Treinamento concluído!")

y_pred = rf.predict(features_test)
acc = accuracy_score(labels_test, y_pred)
print(f'\nAcurácia REAL no conjunto de teste: {acc:.2f}\n')

print('Relatório de Classificação:')
print(classification_report(labels_test, y_pred, zero_division=0))

joblib.dump(rf, 'modelo_alfabeto.pkl')
print('\nModelo salvo como "modelo_alfabeto.pkl"')