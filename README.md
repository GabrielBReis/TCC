# Aircraft Crack Detection Pipeline

Pipeline de visão computacional para treinamento e avaliação de detectores de trincas em superfícies de aeronaves.

Modelos disponíveis:

- YOLO11n;
- Faster R-CNN ResNet-50-FPN;
- RT-DETR-R18.

O projeto inclui preparação do dataset, conversão COCO–YOLO, treinamento, avaliação, matriz de confusão, MLflow e uma esteira de experimentos.

## Requisitos

- Python 3.10 ou 3.11;
- Git;
- Git LFS, caso o dataset seja distribuído pelo repositório;
- GPU NVIDIA recomendada para treinamento;
- drivers compatíveis com a versão CUDA do PyTorch.

## Estrutura principal

~~~text
configs/                 Configurações do projeto e datasets
data/raw/                Datasets originais
data/processed/          Datasets tratados
models/pretrained/       Pesos iniciais
reports/                 Relatórios e comparações
runs/                    Resultados dos treinamentos
scripts/                 Scripts executáveis
src/tcc_pipeline/        Código reutilizável
tests/                   Testes automatizados
~~~

Todos os comandos devem ser executados na raiz do repositório.

## 1. Clonar

~~~bash
git clone https://github.com/GabrielBReis/TCC.git
cd TCC
~~~

Se houver arquivos controlados pelo Git LFS:

~~~bash
git lfs install
git lfs pull
~~~

## 2. Criar o ambiente virtual

### Windows PowerShell

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
~~~

Se o PowerShell bloquear a ativação:

~~~powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
~~~

### Linux

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
~~~

Para uma build específica de CUDA, instale Torch e Torchvision usando o comando indicado pelo seletor oficial do PyTorch antes de instalar as demais dependências.

## 3. Verificar o ambiente

~~~bash
python -m pip check
python scripts/check_environment.py
~~~

Verificação rápida da GPU:

~~~bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

## 4. Preparar a configuração

O arquivo padrão é:

~~~text
configs/project.yaml
~~~

Para criar uma configuração nova:

### Windows

~~~powershell
Copy-Item .\configs\project.example.yaml .\configs\project.yaml
~~~

### Linux

~~~bash
cp configs/project.example.yaml configs/project.yaml
~~~

Os caminhos são relativos à raiz do projeto. Evite caminhos absolutos específicos de uma máquina.

## 5. Preparar o dataset

O dataset bruto deve estar em:

~~~text
data/raw/aircraft_surface_damage/
├── train/
├── valid/
└── test/
~~~

Cada split deve conter seu arquivo _annotations.coco.json e as imagens correspondentes.

Execute:

~~~bash
python scripts/prepare_crack_dataset_v2.py --config configs/dataset_aircraft_surface_damage_v2.yaml
~~~

Saída:

~~~text
data/processed/datasets/aircraft_surface_damage_crack_v2/
├── annotations/
├── train/images/
├── val/images/
├── test/images/
├── yolo/
├── preparation_report.json
└── split_manifest.csv
~~~

O script também gera a auditoria em:

~~~text
reports/dataset_audit/aircraft_surface_damage_crack_v2/
~~~

O diretório de saída deve estar vazio ou não existir.

## 6. Validar o dataset

### Treino

~~~bash
python scripts/validate_dataset.py --images data/processed/datasets/aircraft_surface_damage_crack_v2/train/images --annotations data/processed/datasets/aircraft_surface_damage_crack_v2/annotations/train.json --strict
~~~

### Validação

~~~bash
python scripts/validate_dataset.py --images data/processed/datasets/aircraft_surface_damage_crack_v2/val/images --annotations data/processed/datasets/aircraft_surface_damage_crack_v2/annotations/val.json --strict
~~~

### Teste

~~~bash
python scripts/validate_dataset.py --images data/processed/datasets/aircraft_surface_damage_crack_v2/test/images --annotations data/processed/datasets/aircraft_surface_damage_crack_v2/annotations/test.json --strict
~~~

## 7. Baixar os modelos

~~~bash
python scripts/download_models.py --out models/pretrained
~~~

Pesos esperados:

~~~text
models/pretrained/
├── yolo11n.pt
├── fasterrcnn_resnet50_fpn_coco.pth
└── rtdetr_r18vd/
~~~

Para selecionar somente alguns:

~~~bash
python scripts/download_models.py --out models/pretrained --models yolo,faster
~~~

## 8. Iniciar o MLflow

~~~bash
python scripts/start_mlflow.py --config configs/project.yaml
~~~

A interface local fica disponível em:

~~~text
http://127.0.0.1:5000
~~~

Mantenha o MLflow em um terminal e execute os treinamentos em outro.

### Acesso pela rede local

No servidor:

~~~bash
python scripts/start_mlflow.py --config configs/project.yaml --host 0.0.0.0 --port 5000 --allowed-hosts "127.0.0.1,localhost,IP_DO_SERVIDOR" --cors-allowed-origins "http://127.0.0.1:5000,http://localhost:5000,http://IP_DO_SERVIDOR:5000"
~~~

No cliente Linux:

~~~bash
export TCC_MLFLOW_TRACKING_URI=http://IP_DO_SERVIDOR:5000
~~~

No cliente PowerShell:

~~~powershell
$env:TCC_MLFLOW_TRACKING_URI="http://IP_DO_SERVIDOR:5000"
~~~

Teste de conexão:

~~~bash
curl http://IP_DO_SERVIDOR:5000/health
~~~

~~~powershell
Invoke-WebRequest http://IP_DO_SERVIDOR:5000/health
~~~

## 9. Conferir a execução sem treinar

~~~bash
python scripts/run_baselines.py --config configs/project.yaml --dry-run --skip-download --skip-prepare-yolo
~~~

O dry-run mostra os comandos e caminhos sem iniciar os treinamentos.

## 10. Treinar os modelos

### YOLO11n

~~~bash
python scripts/train_yolo.py --config configs/project.yaml
~~~

### Faster R-CNN

~~~bash
python scripts/train_faster_rcnn.py --config configs/project.yaml
~~~

### RT-DETR

~~~bash
python scripts/train_rtdetr.py --config configs/project.yaml
~~~

Em uma máquina com apenas uma GPU, execute um modelo por vez.

Os resultados são organizados em:

~~~text
runs/aircraft_surface_damage_crack_v2/
├── yolo/
├── faster_rcnn/
└── rtdetr/
~~~

## 11. Executar o benchmark completo

~~~bash
python scripts/run_baselines.py --config configs/project.yaml
~~~

Para evitar download ou reconversão já realizados:

~~~bash
python scripts/run_baselines.py --config configs/project.yaml --skip-download --skip-prepare-yolo
~~~

O pipeline executa treinamento, predição, avaliação e comparação.

## 12. Esteira de tentativas

### Um modelo

~~~bash
python scripts/train_with_retries.py --model yolo --config configs/project.yaml
python scripts/train_with_retries.py --model faster_rcnn --config configs/project.yaml
python scripts/train_with_retries.py --model rtdetr --config configs/project.yaml
~~~

### Todos em sequência

~~~bash
python scripts/train_with_retries.py --model all --config configs/project.yaml
~~~

### Conferir sem executar

~~~bash
python scripts/train_with_retries.py --model all --config configs/project.yaml --dry-run
~~~

Quando uma tentativa termina, a esteira avalia o resultado e inicia a próxima configuração.

### Esteira de experimentos YOLO11n

Para executar 12 configurações de YOLO11n baseadas na literatura, em ablações
controladas e em dois cenários de augmentation:

~~~bash
python scripts/train_with_retries.py --model yolo --config configs/yolo_literature.yaml
~~~

Conferir antes de treinar:

~~~bash
python scripts/train_with_retries.py --model yolo --config configs/yolo_literature.yaml --dry-run
~~~

Cada conjunto gera um run separado no MLflow. A esteira executa as 12
configurações em sequência, reutiliza tentativas concluídas após uma interrupção
e mantém somente `weights/best.pt` de cada treinamento. Todas as execuções usam
YOLO11n, resolução 640, no máximo 100 épocas e batch de até 20.

Os resumos acumulados ficam em `attempts_comparison.json` e
`attempts_comparison.csv` dentro do diretório da esteira.

A correspondência entre artigos, parâmetros aplicados e adaptações ao YOLO11n
está documentada em docs/YOLO_LITERATURE_EXPERIMENTS.md.

## 13. Avaliação manual

As predições dos modelos são salvas no formato COCO. Para avaliar:

~~~bash
python scripts/evaluate.py --gt data/processed/datasets/aircraft_surface_damage_crack_v2/annotations/test.json --pred runs/DATASET/MODELO/RUN/predictions.json --out runs/DATASET/MODELO/RUN/metrics.json --conf 0.25 --iou 0.50 --small-max 0.01 --medium-max 0.05
~~~

O script gera:

- métricas COCO;
- precision, recall e F1;
- métricas por escala;
- matriz de confusão em CSV;
- matriz de confusão em PNG.

Para comparar:

~~~bash
python scripts/compare_experiments.py --metrics CAMINHO_YOLO/metrics.json CAMINHO_FASTER/metrics.json CAMINHO_RTDETR/metrics.json --out reports/comparison
~~~

## 14. Resultados do HPC

Depois de baixar o pacote produzido no HPC:

~~~bash
python scripts/import_hpc_results.py --bundle hpc_results/RESULTADOS.zip --config configs/project.yaml
~~~

Inicie o MLflow antes da importação.

## 15. Testes

~~~bash
python -m pytest -q
~~~

Testes relacionados ao dataset:

~~~bash
python -m pytest -p no:cacheprovider tests/test_dataset_v2.py tests/test_single_class_dataset.py -q
~~~

Lint:

~~~bash
python -m ruff check --no-cache .
~~~

## 16. Problemas comuns

### CUDA out of memory

- encerre processos antigos;
- reduza o batch do modelo;
- mantenha AMP quando suportado;
- confira nvidia-smi;
- reduza resolução somente se necessário.

### MLflow não mostra runs

- selecione State: Active;
- selecione All time;
- remova filtros antigos;
- confira o experimento;
- confira TCC_MLFLOW_TRACKING_URI.

### Falha ao baixar modelos

- verifique acesso ao GitHub e Hugging Face;
- remova somente arquivos incompletos;
- execute novamente;
- copie models/pretrained de outra máquina se necessário.

### Dataset já existe

O preparador não sobrescreve uma saída não vazia. Arquive a versão anterior ou configure outro diretório. Não remova data/raw.

## Observações

- não publique tokens ou credenciais;
- não adicione mlflow.db, runs ou checkpoints diretamente ao Git;
- use Git LFS somente quando decidir versionar arquivos grandes;
- mantenha configs/project.example.yaml sem caminhos pessoais;
- registre as licenças dos datasets e checkpoints antes da publicação.
