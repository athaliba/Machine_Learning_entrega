# Machine Learning — projetos e experimentos

Repositório de entregas acadêmicas de **Rafael Athaliba**, reunindo projetos de aprendizado de máquina, redes neurais, visão computacional e IA generativa desenvolvidos durante os estudos no **SENAI CIMATEC**.

Os projetos percorrem exploração e preparação dos dados, treinamento, validação, inferência, interpretação e análise dos erros. Cada pasta possui seu próprio ambiente, dependências e artefatos.

## Projetos

| Projeto | Problema e implementação | Tecnologias principais |
|---|---|---|
| [Olist](olist/README.md) | Previsão de atrasos em pedidos, validação cruzada, inferência, explicações SHAP e simulação de active learning. | pandas, scikit-learn, XGBoost, LightGBM, SHAP |
| [Deep Learning](deep_learning/README.md) | MLP para previsão de atrasos na Olist e notebooks de estudo sobre redes neurais e CNNs. | TensorFlow, Keras, scikit-learn |
| [Visão Computacional](visao_computacional/README.md) | Classificação do CIFAR-10 com duas CNNs, análise dos erros e visualização de filtros e ativações. | TensorFlow, Keras, Matplotlib |
| [Reconhecimento Facial](reconhecimento_facial/README.md) | Detecção de rostos e identificação por comparação de embeddings com um cadastro local. | DeepFace, RetinaFace, Facenet512, OpenCV |
| [Diffusion Models](diffusion_models/README.md) | Projeto em grupo: DDPM condicional no MNIST, amostragem DDPM/DDIM e demonstração com SD-Turbo pré-treinado. | PyTorch, Diffusers, Gradio |

O projeto de difusão foi realizado por **Bianca Andrade, Rafael Athaliba e Pedro Lima**.

## Resultados dos experimentos

Os resultados abaixo correspondem aos relatórios disponíveis no repositório. As métricas pertencem a tarefas e protocolos diferentes e não formam um ranking entre os cinco projetos.

| Experimento | Resultado registrado | Relatório |
|---|---|---|
| Olist — XGBoost, limiar 0,14 | Average Precision: **0,2662**; recall: **45,69%**; F1: **0,3309**. | [Métricas de teste](olist/reports/cv_test_metrics.json) |
| Deep Learning — MLP, limiar 0,50 | Average Precision: **0,2398**; recall: **69,84%**; F1: **0,2691**. | [Métricas de teste](deep_learning/reports/mlp_test_metrics.json) |
| Visão Computacional — CIFAR-10 | Acurácia de teste: **64,46%** na CNN inicial e **85,67%** na segunda CNN. | [Métricas de teste](visao_computacional/reports/cifar10_metrics.json) |
| Reconhecimento Facial | Nas três fotos de teste: uma identificação correta, uma rejeição correta e uma falsa rejeição, com limiar 0,30. | [Resultados por imagem](reconhecimento_facial/outputs/recognition_results.csv) |
| Difusão — DDIM, 50 passos, guidance 3 | Acurácia condicional: **97,2%**; FID: **68,63**, na avaliação do grupo com 500 imagens por configuração. | [Métricas do experimento](diffusion_models/reports/metricas.csv) |

### Interpretação dos resultados

- **Olist:** os atrasos são a classe minoritária. A acurácia isolada não descreve a capacidade de identificar esses pedidos; os relatórios também incluem precisão, recall, F1, ROC-AUC e Average Precision.
- **Olist e MLP:** o conjunto de teste já havia sido consultado durante o desenvolvimento. Os resultados são descritivos e não representam um teste final inédito. Os modelos também utilizam tamanhos de treino e limiares diferentes.
- **CIFAR-10:** as redes usam a mesma divisão dos dados, mas diferem em arquitetura, regularização e quantidade de épocas. O ganho não pode ser atribuído a uma única técnica.
- **Reconhecimento facial:** o teste é demonstrativo e contém poucas imagens. A confiança da detecção não representa certeza sobre a identidade.
- **Difusão:** a acurácia condicional depende de um classificador auxiliar. O FID foi calculado com uma amostra pequena e deve ser interpretado no contexto das configurações avaliadas pelo grupo.

## Organização

| Pasta | Conteúdo |
|---|---|
| `olist/` | Pipeline de classificação de atrasos em pedidos. |
| `deep_learning/` | MLP aplicada à Olist e notebooks de redes neurais. |
| `visao_computacional/` | Notebook de classificação de imagens e CNN salva. |
| `reconhecimento_facial/` | Aplicação modular, fotos de cadastro e teste e resultados. |
| `diffusion_models/` | Código, notebook, modelo treinado, métricas e apresentação do trabalho em grupo. |

Os projetos seguem a estrutura do **Cookiecutter Data Science (CCDS)**. Em cada pasta, `module_*` contém o código Python, `notebooks/` reúne os estudos interativos, `models/` armazena os modelos e `reports/` reúne relatórios e figuras. No reconhecimento facial, as imagens marcadas e o CSV são gravados em `outputs/`.

## Preparação do ambiente

Instale **Git** e **[uv](https://docs.astral.sh/uv/)**. Para executar notebooks, utilize um editor com suporte a Jupyter e selecione o interpretador do projeto correspondente.

```powershell
git clone https://github.com/athaliba/Machine_Learning_entrega.git
cd Machine_Learning_entrega
```

Cada projeto possui seu próprio `pyproject.toml`, `uv.lock` e ambiente virtual. Execute `uv sync` dentro da pasta desejada; a raiz do repositório não possui um ambiente único para todos os projetos.

| Projeto | Python declarado no projeto |
|---|---|
| Olist | 3.13.x |
| Deep Learning | 3.12.x |
| Visão Computacional | 3.12.x |
| Reconhecimento Facial | 3.12.x |
| Diffusion Models | Exatamente 3.12.10 |

Por exemplo, para preparar visão computacional, partindo da raiz:

```powershell
cd visao_computacional
uv sync
```

Os comandos com `uv run` usam o ambiente do projeto sem exigir ativação manual. Nos notebooks, selecione o Python da `.venv` da pasta correspondente.

## Execução dos projetos

Execute os comandos abaixo **dentro da pasta de cada projeto**, após `uv sync`.

| Projeto | Entrada principal |
|---|---|
| Olist — validação cruzada, treino e avaliação | `uv run python -m module_olist.main --mode train` |
| Olist — previsão para pedidos abertos | `uv run python -m module_olist.main --mode infer` |
| Olist — explicações SHAP | `uv run python -m module_olist.main --mode explain` |
| Olist — simulação de active learning | `uv run python -m module_olist.main --mode active-learning` |
| Deep Learning — treino e avaliação da MLP | `uv run python -m module_deep_learning.main` |
| Visão Computacional | Executar [Classificador_Imagens.ipynb](visao_computacional/notebooks/Classificador_Imagens.ipynb) com o ambiente do projeto. |
| Reconhecimento Facial | `uv run python -m module_reconhecimento_facial.main` |
| Diffusion Models — testes | `uv run pytest tests -q` |
| Diffusion Models — gerar duas imagens do dígito 7 | `uv run python -m module_diffusion_models.modeling.predict --digito 7 --quantidade 2 --passos 50 --guidance 3 --seed 42` |

### Dados e modelos necessários

- **Olist:** os CSVs não estão versionados. Para treinar, coloque `olist_orders_dataset.csv`, `olist_order_items_dataset.csv` e `olist_customers_dataset.csv` em `olist/data/raw/`. Inferência e explicações também dependem dos dados e dos artefatos usados por seus módulos.
- **Deep Learning:** a MLP lê `olist/data/interim/orders_dataset_refined.csv`, gerado pelo projeto Olist. Mantenha as pastas irmãs ou informe outro caminho com `--data`.
- **Visão Computacional:** o notebook baixa o CIFAR-10 na primeira carga. Executar suas células de treinamento refaz o treino; o modelo salvo está em `visao_computacional/models/classificador_cifar10.keras`.
- **Reconhecimento Facial:** as fotos de exemplo e suas saídas estão incluídas. Os pesos de RetinaFace e Facenet512 são baixados na primeira execução. Os modelos são pré-treinados, sem treinamento de uma rede própria.
- **Diffusion Models:** o checkpoint `models/ddpm_mnist_ema.pt` permite gerar imagens sem refazer o treinamento. A demonstração com SD-Turbo no notebook requer GPU e downloads adicionais.

Novas execuções podem atualizar modelos, relatórios e figuras. Os arquivos publicados documentam os experimentos realizados e não garantem resultados numericamente idênticos em qualquer equipamento.

## Tecnologias e práticas

- Preparação de dados e engenharia de atributos com pandas e NumPy.
- Pipelines, validação cruzada e tratamento de classes desbalanceadas.
- Interpretação com SHAP e seleção de exemplos por incerteza.
- Redes densas e convolucionais com TensorFlow/Keras.
- Comparação de embeddings faciais com modelos pré-treinados.
- Difusão, U-Net condicional e Classifier-Free Guidance com PyTorch.
- Ambientes separados com uv, organização CCDS e histórico incremental em Git.

## Autoria e licença

**Rafael Athaliba — SENAI CIMATEC.** A autoria coletiva do projeto de difusão está identificada na documentação e nos metadados daquele projeto.

As pastas dos projetos contêm seus arquivos `LICENSE`. Bases de dados e modelos de terceiros possuem condições de uso próprias, que devem ser consultadas nas respectivas distribuições.
