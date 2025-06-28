import numpy as np
import pandas as pd
import os
from glob import glob

from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_ubyte

# Certifique-se de que o backend Agg está configurado para evitar o erro Tkinter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Importar suas funções de extratores dos arquivos separados
from extractors.glcm import glcm
from extractors.lbp import lbp
from extractors.lpq import lpq

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Usado para capturar a saída do classification_report
import io
import sys

# Configurações Globais
DATA_DIR = 'soybean-images' # Caminho para a pasta que contém as subpastas das classes de imagens
CLASS_NAMES = ['intact', 'spotted', 'immature', 'broken', 'skin-damage'] # Mapeamento de nomes de pastas para rótulos numéricos
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)} # ordem das classes define índices na matriz de confusão

# Funções Axiliares
def load_and_extract_features(data_directory): #Função de Carregamento e Extração de Características
    """
    -> Carrega imagens de diretórios, extrai características GLCM, LBP e LPQ usando as funções dos arquivos separados, 
    e retorna o conjunto de dados (características e rótulos).
    -> Args: data_directory (str) == Caminho para o diretório raiz contendo as pastas de classes.
    -> Returns: tuple == (numpy.ndarray, numpy.ndarray) contendo as características (X) e os rótulos (y).
    """
    all_features = []
    all_labels = []

    # Lista todas as subpastas (classes) dentro do diretório de dados
    class_folders = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]
    class_folders.sort() # Garante uma ordem consistente (opcional, mas boa prática)

    print("Iniciando a extração de características das imagens...")

    for class_folder in class_folders: # Itera sobre cada pasta de classe
        class_name = os.path.basename(class_folder) # Pega o nome da pasta
        if class_name not in LABEL_MAP:
            print(f"Aviso: Pasta '{class_name}' encontrada, mas não mapeada para um rótulo. Ignorando.")
            continue
        
        label = LABEL_MAP[class_name] # Obtém o rótulo numérico correspondente

        # Lista todos os arquivos de imagem dentro da pasta da classe
        image_paths = glob(os.path.join(class_folder, '*.jpg')) + \
                      glob(os.path.join(class_folder, '*.jpeg')) + \
                      glob(os.path.join(class_folder, '*.png'))

        print(f"  Processando {len(image_paths)} imagens da classe '{class_name}'...")

        for img_path in image_paths:
            try:
                img = imread(img_path) # Carrega a imagem

                # Converta para escala de cinza se a imagem for colorida (3 dimensões)
                if img.ndim == 3:
                    img_gray = rgb2gray(img)
                else:
                    img_gray = img # Já está em escala de cinza
                
                # Para GLCM e LBP, o skimage espera imagens uint8 (0-255) ou float (0.0-1.0).
                # Converter para uint8 (0-255) é uma boa prática para GLCM.
                img_for_glcm_lbp = img_as_ubyte(img_gray)
                
                # Para LPQ, o código espera float64.
                img_for_lpq = np.float64(img_gray)

                # Extração de Características
                glcm_feats = glcm(img_for_glcm_lbp)
                lbp_feats = lbp(img_for_glcm_lbp)
                lpq_feats = lpq(img_for_lpq)

                # Combinação de Características (Concatenar todos os vetores de características em um único vetor para a imagem)
                combined_features = np.concatenate([glcm_feats, lbp_feats, lpq_feats])
                
                all_features.append(combined_features)
                all_labels.append(label)

            except Exception as e:
                print(f"    Erro ao processar a imagem {img_path}: {e}. Pulando.")
                continue # Pula para a próxima imagem em caso de erro

    print("Extração de características concluída.")
    return np.array(all_features), np.array(all_labels)

def train_and_evaluate_models(X_train, y_train, X_test, y_test, k_folds=5):
    """
    Treina e avalia classificadores KNN e Árvore de Decisão usando GridSearchCV
    e validação cruzada. Em seguida, avalia os melhores modelos no conjunto de teste.
    Salva os relatórios de classificação em arquivos .txt e as matrizes de confusão em .png.

    Args:
        X_train (numpy.ndarray): Características do conjunto de treino.
        y_train (numpy.ndarray): Rótulos do conjunto de treino.
        X_test (numpy.ndarray): Características do conjunto de teste.
        y_test (numpy.ndarray): Rótulos do conjunto de teste.
        k_folds (int): Número de folds para a validação cruzada.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    print(f"\nIniciando otimização de hiperparâmetros com GridSearchCV ({k_folds} folds)...\n")

    # --- Otimização de Hiperparâmetros para KNN com GridSearchCV ---
    print("Otimizando K-Nearest Neighbors (KNN)...")
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search_knn.fit(X_train, y_train)

    print(f"\nMelhores parâmetros para KNN: {grid_search_knn.best_params_}")
    print(f"Melhor acurácia (média da validação cruzada) para KNN: {grid_search_knn.best_score_:.4f}")
    best_knn_model = grid_search_knn.best_estimator_

    # --- Otimização de Hiperparâmetros para Árvore de Decisão com GridSearchCV ---
    print("\nOtimizando Árvore de Decisão...")
    param_grid_dtree = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_leaf': [1, 5, 10, 20],
        'criterion': ['gini', 'entropy']
    }
    grid_search_dtree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dtree, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search_dtree.fit(X_train, y_train)

    print(f"\nMelhores parâmetros para Árvore de Decisão: {grid_search_dtree.best_params_}")
    print(f"Melhor acurácia (média da validação cruzada) para Árvore de Decisão: {grid_search_dtree.best_score_:.4f}")
    best_dtree_model = grid_search_dtree.best_estimator_

    print("\n--- Avaliação Final nos Dados de Teste ---")

    # --- Avaliação KNN ---
    print("\n--- Avaliando o Melhor Modelo KNN no Conjunto de Teste ---")
    y_pred_knn_final = best_knn_model.predict(X_test)

    # Capturar e salvar o relatório de classificação do KNN
    knn_report_str = io.StringIO() # Cria um objeto "arquivo" em memória
    sys.stdout = knn_report_str # Redireciona a saída padrão para o objeto em memória
    print(classification_report(y_test, y_pred_knn_final, target_names=CLASS_NAMES, zero_division=0))
    sys.stdout = sys.__stdout__ # Restaura a saída padrão para o terminal
    
    knn_report_text = knn_report_str.getvalue() # Obtém o conteúdo capturado
    print("\nKNN - Relatório de Classificação (também salvo em 'relatorio_knn.txt'):")
    print(knn_report_text) # Imprime no terminal
    with open('relatorio_knn.txt', 'w') as f: # Salva em arquivo
        f.write("Relatório de Classificação - K-Nearest Neighbors (KNN)\n")
        f.write(f"Melhores Parâmetros: {grid_search_knn.best_params_}\n")
        f.write(f"Acurácia Média da Validação Cruzada: {grid_search_knn.best_score_:.4f}\n\n")
        f.write(knn_report_text)
    print("Relatório KNN salvo em 'relatorio_knn.txt'\n")


    # Matriz de Confusão para KNN
    cm_knn_final = confusion_matrix(y_test, y_pred_knn_final)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_knn_final, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matriz de Confusão - KNN (Conjunto de Teste)')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.savefig('matriz_confusao_knn.png') # Salva a figura em PNG
    plt.close() # Fecha a figura para liberar recursos


    # --- Avaliação Árvore de Decisão ---
    print("\n--- Avaliando o Melhor Modelo de Árvore de Decisão no Conjunto de Teste ---")
    y_pred_dtree_final = best_dtree_model.predict(X_test)

    # Capturar e salvar o relatório de classificação da Árvore de Decisão
    dtree_report_str = io.StringIO()
    sys.stdout = dtree_report_str
    print(classification_report(y_test, y_pred_dtree_final, target_names=CLASS_NAMES, zero_division=0))
    sys.stdout = sys.__stdout__
    
    dtree_report_text = dtree_report_str.getvalue()
    print("\nÁrvore de Decisão - Relatório de Classificação (também salvo em 'relatorio_arvore_decisao.txt'):")
    print(dtree_report_text)
    with open('relatorio_arvore_decisao.txt', 'w') as f:
        f.write("Relatório de Classificação - Árvore de Decisão\n")
        f.write(f"Melhores Parâmetros: {grid_search_dtree.best_params_}\n")
        f.write(f"Acurácia Média da Validação Cruzada: {grid_search_dtree.best_score_:.4f}\n\n")
        f.write(dtree_report_text)
    print("Relatório Árvore de Decisão salvo em 'relatorio_arvore_decisao.txt'\n")


    # Matriz de Confusão para Árvore de Decisão
    cm_dtree_final = confusion_matrix(y_test, y_pred_dtree_final)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_dtree_final, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matriz de Confusão - Árvore de Decisão (Conjunto de Teste)')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.savefig('matriz_confusao_arvore_decisao.png') # Salva a figura em PNG
    plt.close() # Fecha a figura para liberar recursos

# Main do programa
if __name__ == "__main__":
    # Carregamento e Extração de Características
    X_raw, y_raw = load_and_extract_features(DATA_DIR)

    # Verifica se há dados suficientes
    if X_raw.shape[0] == 0:
        print("\nNenhuma imagem processada. Verifique o caminho DATA_DIR e a estrutura das pastas/arquivos.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=0.3, random_state=42, stratify=y_raw)

        print(f"\nTamanho do conjunto de treino: {len(X_train)} amostras")
        print(f"Tamanho do conjunto de teste: {len(X_test)} amostras\n")

        train_and_evaluate_models(X_train, y_train, X_test, y_test)

    print("\n--- Execução do Sistema de Classificação de Soja Concluída ---")