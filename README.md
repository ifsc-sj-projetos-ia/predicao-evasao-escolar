# Predição de Evasão Escolar com Machine Learning

Este repositório contém o ecossistema de desenvolvimento e os experimentos para o modelo preditivo de risco de evasão escolar no ensino secundário, utilizando o *Student Performance Dataset*. 

O objetivo principal é identificar de forma precoce estudantes em situação de vulnerabilidade acadêmica, permitindo interventions pedagógicas preventivas sob a premissa de *Human-in-the-Loop*.

##  Estrutura do Projeto

```text
├── notebooks/          # Notebooks contendo a análise exploratória e testes do Colab
│   └── 3_experimentos_modelagem.ipynb
├── src/                # Scripts Python do pipeline de produção
│   ├── data_preparation.py
│   └── model_training.py
├── .gitignore          # Arquivo para ignorar arquivos locais (.venv, caches)
├── LICENSE             # Licença MIT de Ciência Aberta
├── README.md           # Documentação principal do repositório
└── requirements.txt    # Arquivo de dependências e bibliotecas do projeto
```

##  Protocolo de Reprodução Científica

Para assegurar a auditabilidade e a reprodutibilidade metodológica exigidas, qualquer avaliador pode replicar integralmente os resultados apresentados executando os seguintes comandos em seu terminal Linux:

```bash
# 1. Clonar o repositório
git clone https://github.com/ifsc-sj-projetos-ia/predicao-evasao-escolar.git
cd predicao-evasao-escolar

# 2. Inicializar o ambiente virtual
python -m venv .venv
source .venv/bin/activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Executar o pipeline de treinamento
python src/model_training.py
```

Devido ao travamento do hiperparâmetro `random_state=42`, os componentes do pipeline aplicarão as transformações de forma estável, convergindo exatamente nas métricas de **87,0% de Acurácia** e **0,8420 de AUC-ROC** no conjunto de teste. 

##  Métricas e Principais Resultados

O modelo final adotado foi o **Random Forest + Class Weight** (Experimento 3), avaliado na base de teste isolada, apresentando desempenho superior ao baseline: 

* **Acurácia:** 87,0%   
* **Precisão:** 75,0%   
* **Sensibilidade (Recall):** 50,0%   
* **F1-Score:** 60,0%   
* **AUC-ROC:** 0,8420   

##  Limitações do Sistema

* **Natureza Estatística dos Dados:** A base utilizada reflete características demográficas e de desempenho específicas de um conjunto de escolas. Mudanças drásticas na matriz curricular ou no perfil socioeconômico da comunidade escolar exigirirão uma nova coleta e retreino do modelo.
* **Gargalo Ético (Não Estigmatização):** O modelo serve estritamente como suporte à decisão para ações afirmativas e preventivas de tutoria. Não deve ser utilizado para punições automáticas, ranqueamento discriminatório ou exclusão de benefícios estudantis. 
* **Dados Estáticos:** Atualmente, a predição depende de avaliações passadas e dados cadastrais, carecendo de variáveis em tempo real (como assiduidade diária ou frequência à biblioteca).

##  Trabalhos Futuros

* **Integração de Dados Dinâmicos:** Incorporar métricas de assiduidade em tempo real no pipeline para capturar padrões repentinos de desengajamento. 
* **Exploração de Modelos Avançados:** Experimentar algoritmos de *gradient boosting* (como *XGBoost* e *LightGBM*) para avaliar possíveis ganhos marginais nas métricas de sensibilidade (*Recall*).   
* **Interface Visual para Gestores:** Desenvolver um painel (*dashboard*) simples para que os orientadores pedagógicos possam interagir com o modelo de forma intuitiva, em conformidade com a premissa de *Human-in-the-Loop*.