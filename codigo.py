import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Carregamento e Pré-processamento dos Dados ---
# ATENÇÃO: Esta é a seção que você MAIS precisará adaptar!
# Você mencionou ter 3 códigos de extratores de características.
# Aqui, você precisará carregar suas imagens/dados brutos dos grãos,
# aplicar os extratores para obter as características numéricas
# e associar cada conjunto de características à sua respectiva classe (tipo de grão/defeito).

# Exemplo de como seus dados processados podem parecer (substitua por seus dados reais):
# Se seus extratores geram características em arrays, você pode combiná-los.
# Para fins de demonstração, vou criar dados dummy.
# Imagine que você tem uma função 'extrair_caracteristicas(imagem_grao)'
# e um dataset de imagens de grãos e suas classes.

# Exemplo de dados dummy (REMOVA E SUBSTITUA PELOS SEUS DADOS REAIS)
# Criando 150 amostras com 4 características para 3 classes (ex: Grão Bom, Defeito A, Defeito B)
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=150, n_features=4, n_informative=3, n_redundant=0,
                           n_classes=3, n_clusters_per_class=1, random_state=42)

# Convertendo para DataFrame para facilitar a manipulação (opcional, mas bom para visualização)
df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
df['target'] = y

# Mostra as primeiras linhas do DataFrame dummy
print("Exemplo de dados gerados:")
print(df.head())
print("\n")

# Seus dados reais provavelmente virão de uma lista de características extraídas
# e uma lista de rótulos correspondentes.
# Ex:
# caracteristicas = [] # lista de arrays ou listas de características para cada grão
# rotulos = []         # lista de classes (inteiros ou strings) para cada grão
#
# for imagem in lista_de_imagens_dos_graos:
#     feat_extrator1 = extrator1.extrair(imagem)
#     feat_extrator2 = extrator2.extrair(imagem)
#     feat_extrator3 = extrator3.extrair(imagem)
#     # Combine as características dos 3 extratores como você achar melhor
#     caracteristicas.append(np.concatenate([feat_extrator1, feat_extrator2, feat_extrator3]))
#     rotulos.append(obter_rotulo_da_imagem(imagem)) # Função para obter o rótulo da imagem
#
# X = np.array(caracteristicas)
# y = np.array(rotulos)

# --- 2. Divisão dos Dados (Treino e Teste) ---
# Usamos stratify=y para garantir que a proporção das classes seja mantida nos conjuntos de treino e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Tamanho do conjunto de treino: {len(X_train)} amostras")
print(f"Tamanho do conjunto de teste: {len(X_test)} amostras\n")

# --- 3. e 4. Treinamento e Avaliação dos Modelos com Validação Cruzada ---

# Configuração da Validação Cruzada (k=5 é o mínimo, mas você pode aumentar)
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Dicionários para armazenar as métricas de cada classificador
results_knn = {'accuracy': [], 'f1_score': [], 'recall': [], 'precision': [], 'confusion_matrices': []}
results_dtree = {'accuracy': [], 'f1_score': [], 'recall': [], 'precision': [], 'confusion_matrices': []}

print(f"Iniciando validação cruzada com {k_folds} folds...\n")

fold_idx = 1
for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    print(f"--- Fold {fold_idx}/{k_folds} ---")

    # --- KNN ---
    print("Classificador: K-Nearest Neighbors (KNN)")
    knn = KNeighborsClassifier(n_neighbors=5) # Você pode ajustar 'n_neighbors'
    knn.fit(X_train_fold, y_train_fold)
    y_pred_knn = knn.predict(X_val_fold)

    # Métricas para KNN
    results_knn['accuracy'].append(accuracy_score(y_val_fold, y_pred_knn))
    results_knn['f1_score'].append(f1_score(y_val_fold, y_pred_knn, average='weighted')) # 'weighted' para lidar com classes desbalanceadas
    results_knn['recall'].append(recall_score(y_val_fold, y_pred_knn, average='weighted'))
    results_knn['precision'].append(precision_score(y_val_fold, y_pred_knn, average='weighted'))
    results_knn['confusion_matrices'].append(confusion_matrix(y_val_fold, y_pred_knn))

    print(f"  Acurácia: {results_knn['accuracy'][-1]:.4f}")
    print(f"  F1-Score: {results_knn['f1_score'][-1]:.4f}")
    print(f"  Recall: {results_knn['recall'][-1]:.4f}")
    print(f"  Precision: {results_knn['precision'][-1]:.4f}\n")

    # --- Árvore de Decisão ---
    print("Classificador: Árvore de Decisão")
    dtree = DecisionTreeClassifier(random_state=42) # Você pode ajustar parâmetros como max_depth, min_samples_leaf
    dtree.fit(X_train_fold, y_train_fold)
    y_pred_dtree = dtree.predict(X_val_fold)

    # Métricas para Árvore de Decisão
    results_dtree['accuracy'].append(accuracy_score(y_val_fold, y_pred_dtree))
    results_dtree['f1_score'].append(f1_score(y_val_fold, y_pred_dtree, average='weighted'))
    results_dtree['recall'].append(recall_score(y_val_fold, y_pred_dtree, average='weighted'))
    results_dtree['precision'].append(precision_score(y_val_fold, y_pred_dtree, average='weighted'))
    results_dtree['confusion_matrices'].append(confusion_matrix(y_val_fold, y_pred_dtree))

    print(f"  Acurácia: {results_dtree['accuracy'][-1]:.4f}")
    print(f"  F1-Score: {results_dtree['f1_score'][-1]:.4f}")
    print(f"  Recall: {results_dtree['recall'][-1]:.4f}")
    print(f"  Precision: {results_dtree['precision'][-1]:.4f}\n")

    fold_idx += 1

print("--- Resumo dos Resultados da Validação Cruzada ---")

# Médias das métricas para KNN
print("\nResultados Médios (KNN):")
print(f"Acurácia Média: {np.mean(results_knn['accuracy']):.4f} (+/- {np.std(results_knn['accuracy']):.4f})")
print(f"F1-Score Médio: {np.mean(results_knn['f1_score']):.4f} (+/- {np.std(results_knn['f1_score']):.4f})")
print(f"Recall Médio: {np.mean(results_knn['recall']):.4f} (+/- {np.std(results_knn['recall']):.4f})")
print(f"Precision Média: {np.mean(results_knn['precision']):.4f} (+/- {np.std(results_knn['precision']):.4f})")

# Médias das métricas para Árvore de Decisão
print("\nResultados Médios (Árvore de Decisão):")
print(f"Acurácia Média: {np.mean(results_dtree['accuracy']):.4f} (+/- {np.std(results_dtree['accuracy']):.4f})")
print(f"F1-Score Médio: {np.mean(results_dtree['f1_score']):.4f} (+/- {np.std(results_dtree['f1_score']):.4f})")
print(f"Recall Médio: {np.mean(results_dtree['recall']):.4f} (+/- {np.std(results_dtree['recall']):.4f})")
print(f"Precision Média: {np.mean(results_dtree['precision']):.4f} (+/- {np.std(results_dtree['precision']):.4f})")

# --- Avaliação Final no Conjunto de Testes ---
# Treinar os modelos nos dados completos de treino e avaliar no conjunto de teste.
# Isso simula o desempenho do modelo final em dados "não vistos".

print("\n--- Avaliação Final nos Dados de Teste ---")

# Treinar KNN no X_train completo
final_knn = KNeighborsClassifier(n_neighbors=5)
final_knn.fit(X_train, y_train)
y_pred_final_knn = final_knn.predict(X_test)

print("\nKNN - Relatório de Classificação no Conjunto de Teste:")
print(classification_report(y_test, y_pred_final_knn, zero_division=0))

# Matriz de Confusão para KNN
cm_knn = confusion_matrix(y_test, y_pred_final_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Matriz de Confusão - KNN (Conjunto de Teste)')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()

# Treinar Árvore de Decisão no X_train completo
final_dtree = DecisionTreeClassifier(random_state=42)
final_dtree.fit(X_train, y_train)
y_pred_final_dtree = final_dtree.predict(X_test)

print("\nÁrvore de Decisão - Relatório de Classificação no Conjunto de Teste:")
print(classification_report(y_test, y_pred_final_dtree, zero_division=0))

# Matriz de Confusão para Árvore de Decisão
cm_dtree = confusion_matrix(y_test, y_pred_final_dtree)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dtree, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Matriz de Confusão - Árvore de Decisão (Conjunto de Teste)')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()