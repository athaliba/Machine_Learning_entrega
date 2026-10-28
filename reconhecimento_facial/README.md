# Reconhecimento facial — identificação por embeddings

Aplicação em Python para detectar rostos em imagens e compará-los com um cadastro local. O sistema utiliza **RetinaFace** para detecção e alinhamento e **Facenet512**, por meio do DeepFace, para gerar representações numéricas dos rostos.

O resultado inclui o nome da pessoa quando a comparação é aceita, a indicação de desconhecido quando é rejeitada, imagens com marcações e um relatório CSV.

[Voltar ao repositório principal](../README.md)

## Objetivos

- Detectar rostos e extrair embeddings com modelos pré-treinados.
- Organizar um cadastro com uma ou mais fotos por pessoa.
- Comparar cada rosto detectado com as fotos cadastradas.
- Registrar decisões, distâncias, ausência de rosto e falhas de processamento.
- Observar casos de identificação correta, rejeição correta e falsa rejeição.

## Como funciona

1. **Cadastro:** lê as imagens em `data/authorized/`, identifica a pessoa pelo nome da pasta e gera um embedding para cada foto válida.
2. **Detecção:** localiza os rostos das imagens em `data/test/`.
3. **Representação:** gera um vetor de 512 valores para cada rosto elegível.
4. **Comparação:** calcula a distância cosseno entre o rosto de teste e cada foto cadastrada.
5. **Decisão:** seleciona a menor distância e compara com o limiar do modelo.
6. **Saída:** salva uma imagem marcada e registra o resultado em CSV.

Não há treinamento de uma rede própria. Os embeddings do cadastro são recalculados a cada execução e mantidos em memória; não constituem um novo modelo treinado.

## Configuração

| Componente | Configuração |
|---|---|
| Backend de execução | TensorFlow, definido no código antes de importar DeepFace. |
| Detector | RetinaFace. |
| Representação facial | Facenet512. |
| Dimensão do embedding | 512 valores. |
| Métrica de comparação | Distância cosseno. |
| Limiar de identidade | 0,30, retornado pelo DeepFace para Facenet512 com distância cosseno. |
| Confiança mínima de detecção | 0,50. |
| Cadastro | Exatamente um rosto detectado por foto. |
| Imagens aceitas | JPG, JPEG, PNG, BMP e WEBP, incluindo subpastas. |

As configurações estão em `module_reconhecimento_facial/config.py`. O limiar de identidade é obtido em `recognizer.py`, com `find_threshold`, e não foi ajustado com as fotos de teste.

### Distância e decisão

Quanto menor a distância cosseno, mais semelhantes são os embeddings. A comparação é aceita quando a menor distância é **menor ou igual a 0,30**.

Cada foto do cadastro é comparada individualmente. Se uma pessoa tiver várias fotos, não é calculado um embedding médio: vale a foto com menor distância entre todas as cadastradas.

**Distância não é porcentagem de certeza.** A confiança retornada pelo detector também não mede a certeza sobre a identidade: ela se refere à detecção do rosto.

O campo `authorized` usado internamente representa a aceitação dessa comparação. A implementação não constitui um sistema completo de autorização ou controle de acesso.

## Organização das imagens

| Caminho de exemplo | Uso |
|---|---|
| `data/authorized/rafael/foto1.jpg` | Primeira imagem de cadastro de Rafael. |
| `data/authorized/rafael/foto2.jpg` | Segunda imagem de cadastro da mesma pessoa. |
| `data/test/pessoa_nao_cadastrada.jpg` | Teste com uma pessoa fora do cadastro. |
| `data/test/rafael_frontal.jpg` | Teste com uma pessoa cadastrada. |
| `data/test/rafael_outra_foto.jpg` | Outra condição de imagem da pessoa cadastrada. |

Para cadastrar outra pessoa, crie uma pasta diretamente em `data/authorized/` com o nome desejado e coloque suas fotos dentro dela. Imagens soltas na raiz de `authorized/` são ignoradas.

O cadastro descarta fotos com múltiplos rostos, baixa confiança de detecção ou embedding inválido. As imagens de teste devem ser diferentes das usadas no cadastro.

As fotos de exemplo e as saídas desta execução estão incluídas no repositório. Ao utilizar outras imagens, verifique a autorização das pessoas para o uso e a publicação.

## Resultados observados

O cadastro do experimento contém **duas fotos de uma pessoa**. Foram processadas três imagens de teste:

| Imagem | Distância | Decisão do modelo | Interpretação com a identidade conhecida |
|---|---:|---|---|
| `pessoa_nao_cadastrada.jpg` | 0,9356 | Desconhecido | Rejeição correta. |
| `rafael_frontal.jpg` | 0,1683 | Rafael | Identificação correta. |
| `rafael_outra_foto.jpg` | 0,3201 | Desconhecido | Falsa rejeição. |

A foto frontal foi adicionada após a observação da rejeição da outra imagem. As duas foram mantidas, sem alterar o limiar. Assim, esse teste adicional é exploratório, não uma avaliação independente planejada previamente.

O resultado variou entre as fotos da mesma pessoa. Isso pode envolver pose, iluminação, expressão, qualidade ou outras diferenças, mas o experimento não isola a causa.

O CSV registra a decisão do modelo, sem um campo de identidade verdadeira. A classificação de falsa rejeição na tabela acima resulta da comparação manual com a identidade conhecida.

Consulte o [relatório completo](outputs/recognition_results.csv) e as imagens marcadas:

- [Pessoa não cadastrada](outputs/pessoa_nao_cadastrada.jpg.resultado.jpg).
- [Rafael — foto frontal](outputs/rafael_frontal.jpg.resultado.jpg).
- [Rafael — outra foto](outputs/rafael_outra_foto.jpg.resultado.jpg).

## Instalação e execução

Pré-requisitos: **Python 3.12.x** e **uv**. Partindo da raiz do repositório:

```powershell
cd reconhecimento_facial
uv sync
uv run python -m module_reconhecimento_facial.main
```

O `uv sync` instala as dependências definidas no projeto e no lockfile, incluindo DeepFace, TensorFlow, tf-keras e Werkzeug. Não é necessário registrar um kernel para executar a aplicação pelo terminal.

Na primeira execução com as fotos, os pesos de Facenet512 e RetinaFace são baixados para o cache do DeepFace. No Windows da execução registrada, o caminho foi `%USERPROFILE%/.deepface/weights/`. As próximas execuções reutilizam esses arquivos.

O código seleciona o backend TensorFlow automaticamente; não é necessário definir a variável de ambiente manualmente antes do comando principal.

A implementação usa leitura e gravação de arquivos com OpenCV, sem abrir janelas de interface gráfica. A dependência instalada é a variante headless; o projeto não oferece modo webcam.

### Conferência da instalação

```powershell
uv run python -c "import module_reconhecimento_facial.config; from deepface import DeepFace; import cv2; print('DeepFace importado'); print('OpenCV:', cv2.__version__)"
```

A execução registrada utilizou Python **3.12.16**, DeepFace **0.0.103**, TensorFlow **2.21.0**, tf-keras **2.21.0** e OpenCV **5.0.0**.

## Saídas

As imagens são gravadas em `outputs/`, com caixa **verde** e nome para uma comparação aceita ou caixa **vermelha** e texto `DESCONHECIDO` para uma rejeitada.

O nome preserva a extensão original antes do sufixo, por exemplo: `rafael_frontal.jpg.resultado.jpg`. Subpastas do teste também são preservadas para evitar colisões entre arquivos de diretórios diferentes.

O arquivo `outputs/recognition_results.csv` contém:

| Coluna | Significado |
|---|---|
| `image` | Caminho relativo da imagem de teste. |
| `face_index` | Número do rosto na imagem. |
| `status` | `cadastrado`, `desconhecido`, `sem_rosto` ou `erro`. |
| `name` | Nome aceito ou `desconhecido`. |
| `distance` | Menor distância cosseno ao cadastro. |
| `threshold` | Limiar utilizado na comparação. |
| `face_confidence` | Confiança da detecção do rosto. |
| `error` | Mensagem de erro, quando registrada. |

Em imagens sem rosto elegível, é registrada uma linha `sem_rosto`, com os campos de comparação vazios. Quando há vários rostos elegíveis, cada um recebe sua própria linha. Falhas de leitura, valores inválidos e erros do OpenCV tratados pelo programa são registrados com status `erro`.

Novas execuções regravam o CSV e as imagens correspondentes. Arquivos antigos de imagens removidas do conjunto de teste não são apagados automaticamente de `outputs/`.

## Código principal

| Arquivo em `module_reconhecimento_facial/` | Responsabilidade |
|---|---|
| `config.py` | Backend, caminhos, modelos e parâmetros de detecção. |
| `image_io.py` | Busca, leitura e gravação das imagens, incluindo caminhos com acentos. |
| `database.py` | Validação das fotos e construção do cadastro de embeddings. |
| `recognizer.py` | Distância cosseno, busca do rosto mais próximo e decisão pelo limiar. |
| `main.py` | Execução em lote, marcações, logs e relatório CSV. |

As demais pastas e arquivos de scaffold foram gerados pelo Cookiecutter Data Science. A execução demonstrada é a entrada `module_reconhecimento_facial.main`.

## Limitações

O experimento possui uma pessoa cadastrada e três imagens de teste. Ele não permite estimar taxas gerais de falso aceite ou falsa rejeição, nem avaliar desempenho em populações e condições variadas.

O limiar vem da biblioteca e não foi calibrado para um cenário operacional específico. Aumentá-lo somente para aceitar uma imagem conhecida poderia também aumentar aceitações indevidas de outras pessoas.

Não há verificação de vivacidade, proteção contra apresentação de fotos ou vídeo, processamento de webcam ou persistência de embeddings. O projeto é uma **demonstração acadêmica**, não uma solução pronta para controle de acesso.

## Autoria e licença

**Rafael Athaliba — SENAI CIMATEC.** Código sob a licença indicada em [LICENSE](LICENSE). DeepFace e os modelos utilizados possuem suas próprias licenças e condições de uso.
