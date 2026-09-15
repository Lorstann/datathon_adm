# CDD / HDD carpanlari ayri ayri
Uretim: 2026-08-23 05:30:20.036294

## Fold A_mevsim  (ufuk ortalama CDD 3.36)

| variant | warm | cold | blend |
| --- | --- | --- | --- |
| v2_base | 0.8771 | 2.3721 | 1.3586 |
| cdd_only | 0.8771 | 2.3721 | 1.3586 |
| hdd_only | 0.8711 | 2.3622 | 1.3518 |
| cdd_hdd | 0.8711 | 2.3622 | 1.3518 |

## Fold B_guncel  (ufuk ortalama CDD 0.00)

| variant | warm | cold | blend |
| --- | --- | --- | --- |
| v2_base | 0.6515 | 1.7855 | 1.0183 |
| cdd_only | 0.6506 | 1.7844 | 1.0174 |
| hdd_only | 0.6585 | 1.7855 | 1.0218 |
| cdd_hdd | 0.6626 | 1.7862 | 1.0241 |

## Fold C_ara  (ufuk ortalama CDD 0.00)

| variant | warm | cold | blend |
| --- | --- | --- | --- |
| v2_base | 0.8316 | 2.0361 | 1.2070 |
| cdd_only | 0.8303 | 2.0410 | 1.2082 |
| hdd_only | 0.8422 | 2.0378 | 1.2134 |
| cdd_hdd | 0.8387 | 2.0383 | 1.2117 |

## Ozet (rmsle_blend)

| variant | mean | std | warm |
| --- | --- | --- | --- |
| cdd_hdd | 1.1958 | 0.1342 | 0.7908 |
| cdd_only | 1.1947 | 0.1396 | 0.7860 |
| hdd_only | 1.1957 | 0.1353 | 0.7906 |
| v2_base | 1.1946 | 0.1392 | 0.7867 |

### Fold bazinda v2_base farki (negatif = iyi)

- cdd_hdd: {'A_mevsim': -0.0068, 'B_guncel': 0.0058, 'C_ara': 0.0046}
- cdd_only: {'A_mevsim': 0.0, 'B_guncel': -0.0009, 'C_ara': 0.0012}
- hdd_only: {'A_mevsim': -0.0068, 'B_guncel': 0.0035, 'C_ara': 0.0064}
- v2_base: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}
