# Pipeline do TCC — Detecção de Defeitos Superficiais em Aeronaves

Este pacote cobre o fluxo **do download dos modelos até a comparação final**, com rastreabilidade opcional por MLflow.

> Antes de publicar ou aceitar contribuições, escolha uma licença para o código e adicione um arquivo `LICENSE`. As licenças dos datasets e pesos não são automaticamente transferidas para este repositório.

Modelos baseline incluídos:

- YOLO11n (Ultralytics)
- Faster R-CNN ResNet-50-FPN (Torchvision)
- RT-DETR-R18 (checkpoint `PekingU/rtdetr_r18vd`, Transformers)

O baseline atual trabalha somente com a classe **`crack`**. A preparação é configurável e combina fontes COCO, incluindo exports do Roboflow como Innovation Hangar v2 e Aircraft Surface Defect Detection.

## 1. Estrutura esperada do dataset

Todas as entradas configuradas são relativas à raiz do repositório. A saída normalizada é:

```text
data/processed/datasets/
└── aircraft_crack/
    ├── train/images/...
    ├── val/images/...
    ├── test/images/...
    ├── annotations/
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    └── preparation_report.json
```

Os JSONs seguem COCO Detection (`images`, `annotations`, `categories`).

Configure fontes, classe-alvo e aliases em `configs/dataset.yaml`. Depois execute:

```bash
python scripts/prepare_single_class_dataset.py --config configs/dataset.yaml
```

O script reconhece os layouts COCO normalizado e Roboflow, mantém apenas a classe configurada, remapeia seu ID para `1`, remove imagens sem a classe-alvo, recorta boxes inválidas e detecta imagens idênticas em splits diferentes. Ele nunca modifica a fonte.

Para usar o Aircraft Surface Defect Detection, descompacte-o em `data/raw/aircraft_surface_defect_detection` e habilite a fonte de exemplo no YAML. Registre no TCC a versão e a licença exatas da fonte escolhida.

## 2. Instalação

Crie o ambiente e instale todas as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e . --no-deps
cp configs/project.example.yaml configs/project.yaml
python scripts/check_environment.py
```

No Windows PowerShell, a ativação é:

```powershell
.\.venv\Scripts\Activate.ps1
```

O `requirements.txt` instala PyTorch e Torchvision pelo índice padrão. Se precisar de uma build CUDA específica, instale primeiro os dois pacotes usando o comando indicado pelo [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/) e depois execute o `requirements.txt`.

Para verificar a instalação:

```bash
python -m pip check
python -m pytest -q
python scripts/check_environment.py
```

## 3. Baixar os modelos

```bash
python scripts/download_models.py --out models/pretrained
```

Isso materializa/cacha:

```text
models/pretrained/
├── yolo11n.pt
├── fasterrcnn_resnet50_fpn_coco.pth
└── rtdetr_r18vd/
```

## 4. Validar e analisar o dataset

```bash
python scripts/validate_dataset.py \
  --images data/processed/datasets/aircraft_crack/train/images \
  --annotations data/processed/datasets/aircraft_crack/annotations/train.json --strict

python scripts/analyze_dataset.py \
  --annotations data/processed/datasets/aircraft_crack/annotations/train.json \
  --out reports/dataset_analysis_train
```

A análise gera CSVs, contagem por classe, distribuição de área relativa e grupos pequeno/médio/grande.

## 5. Preparar o YOLO

```bash
python scripts/convert_coco_to_yolo.py \
  --train-images data/processed/datasets/aircraft_crack/train/images --train-annotations data/processed/datasets/aircraft_crack/annotations/train.json \
  --val-images data/processed/datasets/aircraft_crack/val/images --val-annotations data/processed/datasets/aircraft_crack/annotations/val.json \
  --test-images data/processed/datasets/aircraft_crack/test/images --test-annotations data/processed/datasets/aircraft_crack/annotations/test.json \
  --out data/processed/datasets/aircraft_crack/yolo
```

O `category_mapping.json` gerado é usado para converter as classes do YOLO de volta aos IDs COCO durante a avaliação.

## 6. Treinar os três baselines

```bash
python scripts/train_yolo.py --config configs/project.yaml
python scripts/train_faster_rcnn.py --config configs/project.yaml
python scripts/train_rtdetr.py --config configs/project.yaml
```

Execute um treinamento por vez quando houver apenas uma GPU. Ajuste `batch`, `workers` e `device` em `configs/project.yaml` conforme a memória disponível.

## 7. Fazer predição no TEST

### YOLO11n

```bash
python scripts/predict_yolo.py \
  --weights runs/aircraft_crack/yolo/yolo11n_baseline_640/weights/best.pt \
  --images data/processed/datasets/aircraft_crack/test/images --annotations data/processed/datasets/aircraft_crack/annotations/test.json \
  --mapping data/processed/datasets/aircraft_crack/yolo/category_mapping.json \
  --out runs/aircraft_crack/yolo/yolo11n_baseline_640/predictions.json --imgsz 640
```

### Faster R-CNN

```bash
python scripts/predict_faster_rcnn.py \
  --checkpoint runs/aircraft_crack/faster_rcnn/fasterrcnn_r50_fpn_baseline_640/best.pth \
  --images data/processed/datasets/aircraft_crack/test/images --annotations data/processed/datasets/aircraft_crack/annotations/test.json \
  --out runs/aircraft_crack/faster_rcnn/fasterrcnn_r50_fpn_baseline_640/predictions.json
```

### RT-DETR

```bash
python scripts/predict_rtdetr.py \
  --model runs/aircraft_crack/rtdetr/rtdetr_r18vd_baseline_640/best_model \
  --mapping runs/aircraft_crack/rtdetr/rtdetr_r18vd_baseline_640/class_mapping.json \
  --images data/processed/datasets/aircraft_crack/test/images --annotations data/processed/datasets/aircraft_crack/annotations/test.json \
  --out runs/aircraft_crack/rtdetr/rtdetr_r18vd_baseline_640/predictions.json
```

## 8. Avaliar

Exemplo:

```bash
python scripts/evaluate.py \
  --gt data/processed/datasets/aircraft_crack/annotations/test.json \
  --pred runs/aircraft_crack/yolo/yolo11n_baseline_640/predictions.json \
  --out runs/aircraft_crack/yolo/yolo11n_baseline_640/metrics.json \
  --conf 0.25 --iou 0.50 --small-max 0.01 --medium-max 0.05
```

O mesmo `evaluate.py` deve ser usado para os três detectores. Ele calcula:

- mAP@0.50:0.95, AP50, AP75;
- AP_small / AP_medium / AP_large do COCO;
- Precision, Recall e F1 a um limiar declarado;
- métricas customizadas por **área relativa** para small/medium/large;
- resultados por classe.

Importante: a predição deve ser gerada com `--conf` baixo (ex.: 0.001) para não truncar a curva de AP. O `--conf 0.25` da avaliação é usado apenas para P/R/F1 operacionais.

## 9. Comparar os modelos

```bash
python scripts/compare_experiments.py --metrics \
  runs/aircraft_crack/yolo/yolo11n_baseline_640/metrics.json \
  runs/aircraft_crack/faster_rcnn/fasterrcnn_r50_fpn_baseline_640/metrics.json \
  runs/aircraft_crack/rtdetr/rtdetr_r18vd_baseline_640/metrics.json \
  --out reports/comparison
```

Gera CSV, Markdown e gráficos das principais métricas.

## 10. Pipeline automático

Depois que `configs/project.yaml` estiver correto:

```bash
python scripts/run_baselines.py --config configs/project.yaml
```

Antes de executar horas de treino, use:

```bash
python scripts/run_baselines.py --config configs/project.yaml --dry-run
```

## 11. Patches / tiling

```bash
python scripts/generate_patches.py \
  --images data/processed/datasets/aircraft_crack/train/images \
  --annotations data/processed/datasets/aircraft_crack/annotations/train.json \
  --out data/processed/train_patches_640 \
  --patch 640 --overlap 0.20 --min-visible 0.30 --keep-all-negative
```

Repita para validação/teste **sem misturar patches oriundos da mesma imagem original entre splits**. A maneira segura é dividir por imagem original primeiro e somente depois gerar os patches.

O script preserva `source_image_id` e `source_annotation_id` para auditoria.

Depois da inferência nos patches, reprojete e reconcilie as detecções antes de avaliar contra as imagens originais:

```bash
python scripts/merge_patch_predictions.py \
  --patch-annotations data/processed/test_patches_640/annotations.json \
  --predictions runs/patches/predictions.json \
  --out runs/patches/predictions_merged.json --iou 0.5
```

Os scripts de predição também geram `inference_metrics.json`, com latência, FPS e número de parâmetros. O tempo exclui leitura das imagens e inclui pós-processamento do framework.

## 12. MLflow

Para abrir a interface local com backend e artefatos na raiz do projeto:

```bash
python scripts/start_mlflow.py --config configs/project.yaml
```

A interface ficará disponível em `http://127.0.0.1:5000`. Para compartilhar na rede local, informe o IP atual do servidor nas opções de segurança:

```bash
python scripts/start_mlflow.py --host 0.0.0.0 \
  --allowed-hosts "127.0.0.1,localhost,IP_DO_SERVIDOR" \
  --cors-allowed-origins "http://127.0.0.1:5000,http://localhost:5000,http://IP_DO_SERVIDOR:5000"
```

No outro dispositivo, defina somente a variável de ambiente, sem editar caminhos no YAML:

```bash
export TCC_MLFLOW_TRACKING_URI=http://IP_DO_SERVIDOR:5000
```

No PowerShell:

```powershell
$env:TCC_MLFLOW_TRACKING_URI="http://IP_DO_SERVIDOR:5000"
```

Os experimentos são organizados como `aircraft_defects_tcc/aircraft_crack/<modelo>` e os artefatos usam o proxy `mlflow-artifacts:`, evitando caminhos Windows em clientes Linux.

Os scripts de treino criam uma execução quando `tracking.enabled=true`. Após gerar `metrics.json`, você também pode anexar a avaliação ao mesmo run:

```bash
python scripts/log_evaluation_to_mlflow.py \
  --config configs/project.yaml \
  --run-dir runs/aircraft_crack/yolo/yolo11n_baseline_640 \
  --metrics runs/aircraft_crack/yolo/yolo11n_baseline_640/metrics.json
```

## 13. Retreinamento condicionado por métricas

Os conjuntos de parâmetros e o limite mínimo ficam em `retraining` no YAML. A seleção usa validação; o teste é executado apenas para o candidato escolhido:

A tentativa 1 é o baseline sem augmentation. A tentativa 2 mantém os mesmos
hiperparâmetros e aplica um preset conservador para trincas: rotação de até 10°,
translação de até 5%, escala de até 10%, flips e pequenas variações fotométricas.
A tentativa 3 é experimental e acrescenta matiz, perspectiva e cisalhamento. No
YOLO ela também habilita `mosaic` e `mixup`. As três tentativas sempre são
executadas para permitir comparação direta; ajustes de taxa de aprendizado só
entram se nenhuma delas alcançar a meta.

O parâmetro `copy_paste` também fica registrado na tentativa experimental do
YOLO, mas o Ultralytics só o aplica em tarefas de segmentação. Como este dataset
possui caixas, e não máscaras, ele não altera as imagens do treino Detect. Não
são criadas máscaras artificiais a partir das caixas.

```bash
python scripts/train_with_retries.py --model yolo --config configs/project.yaml
python scripts/train_with_retries.py --model faster_rcnn --config configs/project.yaml
python scripts/train_with_retries.py --model rtdetr --config configs/project.yaml
```

Para executar YOLO, Faster R-CNN e RT-DETR sequencialmente com um único comando:

```bash
python scripts/train_with_retries.py --model all --config configs/project.yaml
```

Quando uma tentativa termina, sua predição e avaliação são executadas e a
próxima começa automaticamente. Se uma etapa falhar, a esteira para para não
ocultar o erro nem iniciar experimentos sobre um estado incompleto.

Use `--dry-run` para inspecionar a esteira sem iniciar treinos. O relatório final fica em `runs/aircraft_crack/pipelines/<modelo>/pipeline_report.json`.

## 14. Ordem recomendada para o TCC

1. Validar o dataset.
2. Analisar classes e escala das caixas.
3. Congelar os splits.
4. Treinar 3 baselines.
5. Comparar desempenho geral e por escala.
6. Selecionar os 2 modelos mais promissores.
7. Testar resolução maior.
8. Testar patches/tiling.
9. Testar augmentation/hiperparâmetros apenas se houver hipótese clara.
10. Comparar ganho em pequenos defeitos versus custo computacional.

## 15. Observações importantes

- Não use o conjunto de teste para escolher hiperparâmetros.
- Registre `pip freeze`, seed, GPU, versão do dataset e commit Git de cada bateria final.
- Não force a mesma estratégia de optimizer em arquiteturas diferentes só para “padronizar”; preserve uma configuração defensável para cada família e documente-a.
- Os limites 1% e 5% para área relativa são valores iniciais. Devem ser revisados após a análise da base.
- `coco_ap_small` e `relative_small_map50` são métricas diferentes; veja `docs/METHODOLOGY.md`.
- A reimplementação de ASD-YOLO/INN-YOLO/YOLO-FDD não foi incluída automaticamente porque a disponibilidade oficial de código/checkpoints especializados não está suficientemente confiável para ser dependência do baseline. Eles podem entrar depois como extensão.

Consulte `docs/SOURCES.md` para as fontes técnicas oficiais usadas para montar o pipeline.
