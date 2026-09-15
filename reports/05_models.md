# Model validasyonu
Uretim: 2026-08-21 16:16:19.201287

## Fold A_mevsim  gate_thr=1.01  checks={'grouped_sorted': 'OK', 'no_future_target': 'OK', 'masked_absent': 'OK (637 trafo)'}
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| lgbm_A | 1.4384 | 1.0753 | 2.5130 | 1.5164 |
| lgbm_z | 1.4100 | 1.0457 | 2.4797 | 1.4879 |
| lgbm_A_gate | 1.4384 | 1.0753 | 2.5130 | 1.5164 |
| lgbm_Bcoldwarm | 1.4080 | 1.0753 | 2.4138 | 1.4802 |
| level_shape_C | 1.3436 | 1.0038 | 2.3485 | 1.4165 |
| catboost | 1.3801 | 1.0623 | 2.3483 | 1.4493 |
| xgboost | 1.3807 | 1.0490 | 2.3781 | 1.4525 |
| ensemble | 1.3710 | 1.0342 | 2.3765 | 1.4436 |
| ensemble_shift | 1.3755 | 1.0114 | 2.4363 | 1.4531 |

## Fold B_guncel  gate_thr=0.99  checks={'grouped_sorted': 'OK', 'no_future_target': 'OK', 'masked_absent': 'OK (983 trafo)'}
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| lgbm_A | 1.2692 | 1.0525 | 2.0227 | 1.3300 |
| lgbm_z | 1.2760 | 1.0609 | 2.0263 | 1.3364 |
| lgbm_A_gate | 1.2683 | 1.0512 | 2.0227 | 1.3292 |
| lgbm_Bcoldwarm | 1.2128 | 1.0512 | 1.8096 | 1.2593 |
| level_shape_C | 1.1602 | 1.0173 | 1.6973 | 1.2016 |
| catboost | 1.2088 | 1.0550 | 1.7826 | 1.2532 |
| xgboost | 1.2159 | 1.0394 | 1.8550 | 1.2663 |
| ensemble | 1.1759 | 1.0165 | 1.7623 | 1.2217 |
| ensemble_shift | 1.1748 | 1.0078 | 1.7825 | 1.2226 |

## Fold C_ara  gate_thr=1.01  checks={'grouped_sorted': 'OK', 'no_future_target': 'OK', 'masked_absent': 'OK (741 trafo)'}
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| lgbm_A | 1.4809 | 1.3809 | 1.9496 | 1.5253 |
| lgbm_z | 1.4712 | 1.3702 | 1.9440 | 1.5162 |
| lgbm_A_gate | 1.4809 | 1.3809 | 1.9496 | 1.5253 |
| lgbm_Bcoldwarm | 1.4211 | 1.3809 | 1.6288 | 1.4395 |
| level_shape_C | 1.3713 | 1.3311 | 1.5784 | 1.3897 |
| catboost | 1.4445 | 1.4033 | 1.6571 | 1.4633 |
| xgboost | 1.3763 | 1.3239 | 1.6400 | 1.4001 |
| ensemble | 1.3711 | 1.3301 | 1.5820 | 1.3898 |
| ensemble_shift | 1.3717 | 1.3300 | 1.5856 | 1.3907 |

## Fold ozeti (rmsle_blend mean / std)
| Model | mean | std |
| --- | --- | --- |
| catboost | 1.3886 | 0.0959 |
| ensemble | 1.3517 | 0.0945 |
| ensemble_shift | 1.3555 | 0.0974 |
| level_shape_C | 1.3360 | 0.0956 |
| lgbm_A | 1.4573 | 0.0900 |
| lgbm_A_gate | 1.4570 | 0.0904 |
| lgbm_Bcoldwarm | 1.3930 | 0.0960 |
| lgbm_z | 1.4468 | 0.0789 |
| xgboost | 1.3730 | 0.0784 |

Post-process kaydirma: 4/4 grup isareti fold'lar arasi sabit. Karar kurali geregi yalnizca sabit isaretli gruplara uygulanir; tamami sabit degilse gonderimde kaydirma KAPALI kalir.
Sifir kapisi esik medyani: 1.01

## H1 karari

En iyi naif taban `entity_mean_28` blend ortalamasi 1.4439. `level_shape_C` 1.3360 (std 0.0956). Uc fold'un tamaminda C naifi geciyor (A: 1.416 vs 1.491; B: 1.202 vs 1.308; C: 1.390 vs 1.533). Iyilesme fold A ve C'de std'den buyuk, B'de de ayni yonde. H1 dogrulandi.

Post-process kaydirma ensemble'i 1.3517 -> 1.3555 kotulestirdi; gonderimde KAPALI.

Sifir kapisi esigi 1.01 (kapali): maskelenen cold trafoda `hist_zero_rate` yok, kapi tetiklenmiyor. Warm tarafta EDA'daki kalici sifirlar zaten hist_mean ile yaklaniyor.

