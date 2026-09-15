# CV with weather (confirmed cache)
Uretim: 2026-08-21 16:49:17.226116
Hava cache: `C:\Users\musta\OneDrive\Masaüstü\datathon-adm\data\external\weather\era5_daily.parquet`

## Fold A_mevsim
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| lgbm_A | 1.4384 | 1.0753 | 2.5130 | 1.5164 |
| level_shape_C | 1.3435 | 1.0040 | 2.3480 | 1.4165 |
| level_shape_C_coldpatch | 1.4274 | 1.1326 | 2.3555 | 1.4927 |

## Fold B_guncel
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| lgbm_A | 1.2692 | 1.0525 | 2.0227 | 1.3300 |
| level_shape_C | 1.1605 | 1.0175 | 1.6976 | 1.2019 |
| level_shape_C_coldpatch | 1.2936 | 1.1992 | 1.6832 | 1.3219 |

## Fold C_ara
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| lgbm_A | 1.4809 | 1.3809 | 1.9496 | 1.5253 |
| level_shape_C | 1.3711 | 1.3309 | 1.5781 | 1.3895 |
| level_shape_C_coldpatch | 1.5327 | 1.5251 | 1.5752 | 1.5363 |

## Ozet (rmsle_blend)
| Model | mean | std |
| --- | --- | --- |
| level_shape_C | 1.3359 | 0.0954 |
| level_shape_C_coldpatch | 1.4503 | 0.0925 |
| lgbm_A | 1.4573 | 0.0900 |

Cold patch: blend 1.3359 -> 1.4503 (delta +0.1143); cold 1.8746 -> 1.8713 (delta -0.0033)
Cold patch kabul: **False**

## Cold patch re-eval (milder; see `06b_cold_patch.md`)

Aggressive soft-zero on level hurt warm via shared shape residuals.
Re-test with `peer_guc_z` only and soft-zero as cold-only post-gate:

| variant | blend | cold |
| --- | --- | --- |
| baseline (global_z + log_guc) | 1.3359 | 1.8746 |
| peer_guc_z + log_guc | 1.3583 | 1.9179 |
| peer + soft post-gate | 1.3583 | 1.9178 |

**Sonuc:** peer seviyesi cold'u kotulestiriyor; baseline korunuyor. `submission_level_shape_v2` gonderilmedi (plan esigi: blend ≥0.01 iyilesme + cold kotulesmesin).
