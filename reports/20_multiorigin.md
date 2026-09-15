# Cok-origin egitim cercevesi
Uretim: 2026-08-21 19:59:24.958728

`is_cold` duzeltmesi sonrasi (cold = origin'de gecmisi olmayan satir). Eski raporlardaki sayilarla dogrudan karsilastirilamaz.

## Fold A_mevsim  cold_row_rate=0.113  egitim satiri=87,907  (22s)

| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| level_shape_C_eski | 1.1468 | 0.8638 | 2.4043 | 1.3645 |
| multiorigin_C | 1.1675 | 0.8871 | 2.4254 | 1.3843 |

## Fold B_guncel  cold_row_rate=0.157  egitim satiri=1,200,000  (90s)

| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| level_shape_C_eski | 0.9843 | 0.7441 | 1.7878 | 1.0674 |
| multiorigin_C | 0.9319 | 0.6593 | 1.7875 | 1.0229 |

## Fold C_ara  cold_row_rate=0.228  egitim satiri=1,167,286  (78s)

| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| level_shape_C_eski | 1.2656 | 0.8964 | 2.0731 | 1.2561 |
| multiorigin_C | 1.2367 | 0.8436 | 2.0715 | 1.2268 |

## Ozet (rmsle_blend)

| Model | mean | std |
| --- | --- | --- |
| level_shape_C_eski | 1.2293 | 0.1228 |
| multiorigin_C | 1.2113 | 0.1479 |

Delta (multiorigin - eski), blend ortalamasi: **-0.0180**
