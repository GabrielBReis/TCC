# Esteira de experimentos YOLO11n

Este documento descreve como os parâmetros da revisão de literatura e as
evidências dos treinamentos anteriores foram transformados em 12 experimentos
executáveis no YOLO11n.

## Limite da comparação

Os artigos usam YOLOv5s, YOLOv7, YOLOv8n, YOLOv8s e Mamba-YOLO. Portanto, os
experimentos deste repositório não são reproduções integrais dos artigos. Eles
aplicam ao YOLO11n os hiperparâmetros numéricos reportados para estudar o efeito
dessas escolhas no mesmo modelo e dataset.

Parâmetros ausentes não foram atribuídos aos artigos. Quando a implementação
exige um valor, é usado o valor comum declarado em models.yolo e a diferença
fica registrada em implementation_notes no YAML e nos parâmetros do MLflow.

## Limites de hardware e protocolo

Nesta fase, todas as execuções usam no máximo 100 épocas, resolução 640 e batch
20. Esses limites substituem 200/300 épocas, resolução 1280 e batch 32 citados
em alguns artigos. O objetivo é encontrar uma configuração promissora no
notebook antes de aumentar o orçamento computacional.

As variantes são comparadas na validação por `coco_map5095`. O teste permanece
isolado e é executado somente para a melhor variante ao final da esteira.

## Matriz executada

| Run | Papel | Batch | Otimizador | LR | Decay | Augmentation |
|---|---|---:|---|---:|---:|---|
| lit01 | ASD-YOLO adaptado, sem pré-treino | 16 | SGD | 0,01 | 0,0005 | não |
| lit02 | YOLOv8n otimizado adaptado | 4 | AdamW | 0,01 | 0,0002 | não |
| lit03 | YOLOv7-AFF/Mamba-YOLO adaptados | 20 | SGD | 0,01 | 0,0005 | não |
| lit04 | YOLOv8s UAS adaptado | 7 | Adam | 0,001 | 0,0005* | não |
| abl01 | Contraparte pré-treinada da lit01 | 16 | SGD | 0,01 | 0,0005 | não |
| abl02 | Referência AdamW conservadora | 16 | AdamW | 0,001 | 0,0005 | não |
| abl03 | Ablation de LR da abl02 | 16 | AdamW | 0,0005 | 0,0005 | não |
| abl04 | Ablation de LR da abl01 | 16 | SGD | 0,005 | 0,0005 | não |
| abl05 | Ablation de batch da abl02 | 8 | AdamW | 0,001 | 0,0005 | não |
| abl06 | Ablation de decay da abl02 | 16 | AdamW | 0,001 | 0,0002 | não |
| aug01 | Augmentation moderado para crack | 16 | AdamW | 0,001 | 0,0005 | moderado |
| aug02 | Ablation de augmentation expandido | 16 | AdamW | 0,001 | 0,0005 | expandido |

`*` Valor comum da implementação porque não foi reportado no artigo.

`lit01` usa `yolo11n.yaml`, sem pesos pré-treinados. `abl01` muda somente a
inicialização para `yolo11n.pt`, permitindo medir diretamente o efeito do
transfer learning. Os artigos YOLOv7-AFF e Mamba-YOLO viram a mesma configuração
executável após os limites de hardware; uma execução compartilhada evita um
resultado duplicado com a mesma seed.

## Augmentation

Os artigos citam HSV, multi-scale, translação, flips, rotação e Mosaic, mas não
informam todos os valores necessários para reproduzir as transformações.

Nas dez primeiras execuções, augmentation online permanece desligado para:

1. não inventar intensidades ausentes;
2. isolar os hiperparâmetros de otimização;
3. considerar que o split de treino já contém variantes pré-aumentadas.

`aug01` aplica rotação de 5 graus, translação de 3%, escala de 10%, flips e
alterações moderadas de brilho/saturação. `aug02` mantém essa base e adiciona
Mosaic, MixUp, perspectiva, cisalhamento e uma pequena alteração de matiz.
Copy-paste permanece desativado porque o dataset contém bounding boxes, não
máscaras de segmentação que preservem a geometria fina das rachaduras.

## Protocolo comum

Todas as tentativas usam o mesmo dataset, splits, seed e protocolo de avaliação.
Predições para AP são geradas com confiança 0,001. Precision, recall, F1 e
matriz de confusão usam confiança 0,25 e IoU 0,50.

Os limiares 0,5 e 0,7 reportados no estudo Mamba-YOLO ficam registrados como
metadados, mas não substituem o protocolo comum.

## Retenção de arquivos

Cada tentativa mantém somente weights/best.pt. last.pt e checkpoints periódicos
são removidos antes de enviar os artefatos de treinamento ao MLflow.
Durante um treinamento ainda em execução, last.pt pode existir para recuperação;
ele é removido somente após a conclusão bem-sucedida, quando também é criado
training_complete.json.

Métricas, configurações, tabelas, gráficos, predições e matrizes continuam
preservados porque são necessários para comparar e documentar os experimentos.

## Execução no Linux

Em um terminal:

~~~bash
source .venv/bin/activate
python scripts/start_mlflow.py --config configs/yolo_literature.yaml
~~~

Em outro:

~~~bash
source .venv/bin/activate
python scripts/train_with_retries.py --model yolo --config configs/yolo_literature.yaml
~~~

Para conferir os 12 experimentos sem treinar:

~~~bash
python scripts/train_with_retries.py --model yolo --config configs/yolo_literature.yaml --dry-run
~~~

As 12 tentativas são executadas mesmo que uma delas ultrapasse a meta. Se a
execução for interrompida, tentativas que já possuem best.pt e metrics_val.json
são reutilizadas no próximo comando.

Após cada validação, a esteira atualiza `attempts_comparison.json` e
`attempts_comparison.csv`. Cada run também registra no MLflow o grupo do
experimento, a hipótese, os parâmetros, o histórico por época, as métricas e os
artefatos. Apenas `weights/best.pt` é mantido.

## Depois desta busca

Esta matriz usa uma única seed para comparar configurações com custo viável. A
configuração escolhida pela validação deverá então ser repetida com as seeds 42,
43 e 44. Só nessa fase serão calculados média e desvio-padrão; se a convergência
justificar, o limite de épocas também poderá ser aumentado. O split de teste
continuará reservado para a configuração já congelada.
