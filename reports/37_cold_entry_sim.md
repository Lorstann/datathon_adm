# Cold giris zamanlamasi taklidi
Uretim: 2026-08-23 05:59:31.661685

## Fold A_mevsim

| variant | egitim satiri | warm | cold | blend |
| --- | --- | --- | --- | --- |
| no_entry_sim | 87,907 | 0.8771 | 2.3767 | 1.3604 |
| entry_sim | 66,243 | 0.8732 | 2.3626 | 1.3529 |

## Fold B_guncel

| variant | egitim satiri | warm | cold | blend |
| --- | --- | --- | --- | --- |
| no_entry_sim | 1,200,000 | 0.6515 | 1.7864 | 1.0186 |
| entry_sim | 1,063,414 | 0.6569 | 1.7844 | 1.0206 |

## Fold C_ara

| variant | egitim satiri | warm | cold | blend |
| --- | --- | --- | --- | --- |
| no_entry_sim | 1,167,286 | 0.8316 | 2.0361 | 1.2070 |
| entry_sim | 1,032,227 | 0.8334 | 2.0376 | 1.2086 |

## Ozet (rmsle_blend)

| variant | mean | std | warm | cold |
| --- | --- | --- | --- | --- |
| entry_sim | 1.1940 | 0.1361 | 0.7878 | 2.0615 |
| no_entry_sim | 1.1953 | 0.1398 | 0.7867 | 2.0664 |

blend farki (entry_sim - no): {'A_mevsim': -0.0074, 'B_guncel': 0.002, 'C_ara': 0.0016}

cold farki (entry_sim - no): {'A_mevsim': -0.0141, 'B_guncel': -0.0019, 'C_ara': 0.0015}
