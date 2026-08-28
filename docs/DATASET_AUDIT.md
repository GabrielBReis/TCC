# Innovation Hangar v2 — dataset audit

## Recorte de classe única (`aircraft_crack`)

A preparação de `configs/dataset.yaml` mantém apenas `crack` e remapeia a classe para `category_id=1`:

| Split | Imagens | Anotações de crack |
|---|---:|---:|
| train | 1.775 | 3.151 |
| val | 359 | 627 |
| test | 242 | 417 |

- Imagens sem `crack` foram removidas.
- Nenhuma caixa precisou ser recortada.
- Não foram encontradas imagens idênticas entre splits.
- O relatório fica em `data/processed/datasets/aircraft_crack/preparation_report.json`.

Audit date: 2026-08-21  
Source: Roboflow export `Innovation Hangar v2.v2i.coco`  
Declared license: CC BY 4.0  
Declared preprocessing: resize to 640 × 640; no image augmentation.

## Original export

| Split | Images | Annotations | Empty images | Invalid/out-of-bounds boxes |
|---|---:|---:|---:|---:|
| train | 3,217 | 6,642 | 16 | 13 |
| valid | 642 | 1,265 | 5 | 2 |
| test | 429 | 861 | 3 | 0 |

The export contains five annotated classes (`crack`, `dent`, `missing-head`, `paint-off`, and `scratch`) plus an unused category named `fbdf`. Class support is strongly imbalanced: `scratch` has only 83 annotations in the complete dataset and `paint-off` has 294.

The Roboflow-generated filename suffix was removed to recover source-image groups. The original split contained 36 groups in more than one split, involving 107 images. There were no byte-identical files across splits, but several same-source pairs were visually close. The original split must therefore not be used for the final benchmark.

Relative bounding-box area in the complete export:

- area below 1% of the image: 3,646 annotations (41.6%);
- area from 1% to below 5%: 3,295 annotations (37.6%);
- area of 5% or more: 1,827 annotations (20.8%).

This confirms that small-object performance is a material part of the dataset, although the final `small` threshold must remain justified in the methodology.

## Clean benchmark version

The command below creates the canonical local version:

```bash
python scripts/prepare_innovation_hangar.py \
  --source "Innovation Hangar v2.v2i.coco" \
  --out data/raw/innovation_hangar_v2 --seed 42
```

Processing decisions:

- keep all records that belong to the same recovered source group in one split;
- use deterministic 75%/15%/10% grouped and class-aware splitting;
- clip the 15 out-of-bounds boxes to image bounds;
- remove the unused `fbdf` category;
- retain empty images as valid negative examples;
- preserve source filename, source split, and source group in the COCO image metadata;
- record the operation in `preparation_report.json`.

| Clean split | Images | Annotations | crack | dent | missing-head | paint-off | scratch |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3,211 | 6,587 | 3,151 | 2,874 | 278 | 221 | 63 |
| val | 645 | 1,310 | 627 | 572 | 55 | 44 | 12 |
| test | 432 | 871 | 417 | 380 | 37 | 29 | 8 |

Post-preparation validation found zero invalid files, zero annotation warnings, zero source groups crossing splits, and zero exact file hashes crossing splits.

## Consequences for experiments

- Use the clean split under `data/raw/innovation_hangar_v2`; never train directly from the extracted Roboflow folders.
- Report per-class support with every per-class metric. Results for `scratch`, especially the eight test instances, will have high uncertainty.
- Keep `paint-off` and `scratch` in the benchmark, but avoid strong conclusions from small metric differences.
- Do not use the test split for threshold selection, augmentation choices, or hyperparameter tuning.
- Consider repeated seeds or confidence intervals for the final selected configurations.
