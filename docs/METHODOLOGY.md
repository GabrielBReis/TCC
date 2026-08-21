# Notas metodológicas para o TCC

## Princípio do benchmark

Use o mesmo split de treino/validação/teste, as mesmas classes e a mesma política de avaliação para todos os modelos. O conjunto de teste não deve ser usado para ajuste de hiperparâmetros.

## Métricas

O script `evaluate.py` calcula dois grupos de métricas:

1. **COCO padrão:** mAP@[0.50:0.95], AP50, AP75, AP_small, AP_medium, AP_large e AR@100 via `pycocotools`.
2. **Métricas do TCC por escala relativa:** Precision, Recall, F1, AP50 e mAP50:95 customizados para `small`, `medium` e `large`, usando a razão `área_bbox / área_imagem`.

Não confunda `coco_ap_small` com `relative_small_map50`. O primeiro usa faixas de área absolutas do COCO; o segundo usa uma definição relativa configurável, útil para imagens aeronáuticas de resoluções distintas.

## Definição de pequeno/médio/grande

Os valores `small_max=0.01` e `medium_max=0.05` no exemplo são **iniciais**, não uma verdade científica. Execute `analyze_dataset.py`, examine a distribuição e escolha limites justificáveis. Uma alternativa é manter as métricas COCO como referência e usar a área relativa como análise complementar.

Nas métricas relativas, a escala é definida pela caixa de referência (ground truth). Objetos e detecções fora da faixa recebem tratamento de `ignore` semelhante ao COCO; uma predição não é descartada antes do matching apenas porque seu tamanho estimado atravessou um limiar.

## Augmentation controlado

As augmentations ficam desativadas por padrão nos três baselines. Qualquer experimento com augmentation deve habilitar `models.<modelo>.augmentation.enabled` em uma configuração separada e registrar essa configuração como artefato.

## Experimentos sugeridos

1. Baseline dos três modelos em 640.
2. Análise por escala do defeito.
3. Para os dois melhores modelos, testar 1024/1280 quando suportado e computacionalmente viável.
4. Gerar dataset por patches/tiling e repetir os dois melhores modelos.
5. Só então testar augmentations/hiperparâmetros adicionais.

## Fairness experimental

YOLO, Faster R-CNN e RT-DETR não possuem pipelines internos idênticos. Não force hiperparâmetros incompatíveis apenas para dizer que são “iguais”. Mantenha constantes as variáveis que podem ser controladas (dataset, split, seed, protocolo de teste) e documente as diferenças arquiteturais/preprocessamento de cada framework.
