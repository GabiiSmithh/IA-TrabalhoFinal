# Trabalho Final - Inteligência Computacional

- Autores:
    - Gabriela Smith Ferreira
    - Luiz Felipe Corumba
    - Rafael Machado Wannera

# Instalando as Dependências

- Para executar o código, é necessário ter o Python 3 instalado e as bibliotecas necessárias. Você pode instalar as dependências usando o seguinte comando:

```bash
pip install numpy pandas scikit-learn scikit-image matplotlib seaborn scipy
```

- Se necessário criar um ambiente virtual, você pode usar os seguintes comandos (para Windows):

```bash
py -m venv .venv
.venv\Scripts\activate
```

e depois processeguir com a instalação das dependências.

# Arquivos do Projeto

- O projeto possui os arquivos:
    - gclm.py -> Extrator de caracteristicas (disponibilizados pelo professor)
    - lbp.py -> Extrator de caracteristicas (disponibilizados pelo professor)
    - lpq.py -> Extrator de caracteristicas (disponibilizados pelo professor)
    - system.py -> Implementação do sistema

# Executando o Sistema

- Para executar o programa, basta rodar o seguinte comando (Windows):

```bash
py system.py
```
- Isso irá retornar, no terminal:
    - Informações sobre a extração das características
    - Informações sobre a normalização dos dados
    - Informações sobre o tamanho do conjunto de dados e suas amostras utilizadas
    - Informações para a otimização dos hiperparâmetros
    - Melhores resultados obtidos com cada modelo (KNN e Árvore de Decisão)