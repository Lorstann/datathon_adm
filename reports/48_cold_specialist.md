# Cold uzman modeli
Uretim: 2026-08-24 23:49:50.074478

## Fold A_mevsim

| variant | warm | cold | blend |
| --- | --- | --- | --- |
| tek | 0.8723 | 2.3627 | 1.3525 |
| ayrik | 0.8712 | 2.3659 | 1.3532 |
| ayrik_karma | 0.8723 | 2.3639 | 1.3530 |
| warm_uzman_only | 0.8712 | 2.3627 | 1.3519 |

## Fold B_guncel

| variant | warm | cold | blend |
| --- | --- | --- | --- |
| tek | 0.6530 | 1.7863 | 1.0193 |
| ayrik | 0.6587 | 1.7886 | 1.0231 |
| ayrik_karma | 0.6530 | 1.7869 | 1.0195 |
| warm_uzman_only | 0.6587 | 1.7863 | 1.0221 |

## Fold C_ara

| variant | warm | cold | blend |
| --- | --- | --- | --- |
| tek | 0.8310 | 2.0378 | 1.2073 |
| ayrik | 0.8344 | 2.0394 | 1.2098 |
| ayrik_karma | 0.8310 | 2.0378 | 1.2074 |
| warm_uzman_only | 0.8344 | 2.0378 | 1.2092 |

## Ozet

| variant | mean | std | warm | cold |
| --- | --- | --- | --- | --- |
| ayrik | 1.1953 | 0.1352 | 0.7881 | 2.0646 |
| ayrik_karma | 1.1933 | 0.1365 | 0.7854 | 2.0629 |
| tek | 1.1930 | 0.1364 | 0.7854 | 2.0622 |
| warm_uzman_only | 1.1944 | 0.1350 | 0.7881 | 2.0622 |

- ayrik blend: {'A_mevsim': 0.0007, 'B_guncel': 0.0038, 'C_ara': 0.0024}  cold: {'A_mevsim': 0.0032, 'B_guncel': 0.0024, 'C_ara': 0.0017}
- ayrik_karma blend: {'A_mevsim': 0.0005, 'B_guncel': 0.0002, 'C_ara': 0.0}  cold: {'A_mevsim': 0.0012, 'B_guncel': 0.0006, 'C_ara': 0.0001}
- tek blend: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}  cold: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}
- warm_uzman_only blend: {'A_mevsim': -0.0005, 'B_guncel': 0.0028, 'C_ara': 0.0018}  cold: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}