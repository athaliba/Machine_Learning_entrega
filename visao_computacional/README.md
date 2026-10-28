# Visão Computacional — classificação de imagens com CNNs

Classificação das 10 classes do **CIFAR-10** com redes neurais convolucionais (TensorFlow/Keras):
uma CNN inicial e uma segunda CNN, com análise dos erros por classe e visualização de filtros e
mapas de ativação.

[Voltar ao repositório principal](../README.md)

## Conteúdo

| Arquivo | O que tem |
|---|---|
| [`notebooks/Classificador_Imagens.ipynb`](notebooks/Classificador_Imagens.ipynb) | Exploração das imagens, preparação dos dados, CNN inicial × segunda CNN, curvas de treino, erros por classe, filtros, mapas de ativação e classificação de um arquivo |
| [`models/classificador_cifar10.keras`](models/classificador_cifar10.keras) | Modelo treinado |
| [`reports/cifar10_metrics.json`](reports/cifar10_metrics.json) | Métricas de teste das duas CNNs |
| [`reports/cifar10_per_class.csv`](reports/cifar10_per_class.csv) | Acertos e recall por classe |
| [`reports/figures/`](reports/figures/) | Matriz de confusão, erros de maior escore, filtros e mapas de ativação |

## Resultados

Base: 45 mil imagens de treino, 5 mil de validação e 10 mil de teste.

| Modelo | Acurácia no teste | Loss |
|---|---:|---:|
| CNN inicial | 64,46% | 1,053 |
| Segunda CNN | **85,67%** | 0,433 |

As classes com mais erros são **gato** (recall 65,1%), **cachorro** (75,8%) e **pássaro** (77,6%);
as com mais acertos são **sapo** (95,9%) e **automóvel** (95,1%).

## Como executar

Pré-requisito: [uv](https://docs.astral.sh/uv/) instalado.

```bash
uv sync
uv run pytest tests
```

Abra o notebook com o kernel do ambiente do projeto (`.venv`). Na primeira execução, o CIFAR-10 é
baixado pelo Keras.
