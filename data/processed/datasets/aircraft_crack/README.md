# Aircraft Crack

Dataset processado para detecção de uma única classe: `crack`.

## Origem e licença

- Dataset original: Innovation Hangar v2
- Fonte: https://universe.roboflow.com/innovation-hangar/innovation-hangar-v2
- Licença do dataset original: CC BY 4.0

Esta versão foi derivada do dataset original e mantém apenas as anotações da
classe `crack`, identificada como categoria `1` nos arquivos COCO.

## Conteúdo

| Divisão | Imagens | Anotações |
| --- | ---: | ---: |
| Treino | 1.775 | 3.151 |
| Validação | 359 | 627 |
| Teste | 242 | 417 |

Imagens sem instâncias da classe-alvo foram removidas. As divisões não possuem
imagens duplicadas entre si. Os detalhes do processamento ficam registrados em
`preparation_report.json`.

As imagens são versionadas com Git LFS. O diretório `yolo/` não faz parte do
versionamento, pois é uma representação derivada que pode ser recriada por
`scripts/convert_coco_to_yolo.py`.
