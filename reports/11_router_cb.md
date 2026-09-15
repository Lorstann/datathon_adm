# Warm C + cold B router
Uretim: 2026-08-21 17:01:36.689564
Warm = level_shape_C; cold = hist-dusurulmus z-LGBM; router = is_cold.

## Fold A_mevsim
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| level_shape_C | 1.3435 | 1.0040 | 2.3480 | 1.4165 |
| cold_B_alone | 1.6620 | 1.4484 | 2.4192 | 1.7117 |
| router_CB | 1.3657 | 1.0040 | 2.4192 | 1.4428 |

## Fold B_guncel
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| level_shape_C | 1.1605 | 1.0175 | 1.6976 | 1.2019 |
| cold_B_alone | 1.7939 | 1.7895 | 1.8158 | 1.7953 |
| router_CB | 1.1903 | 1.0175 | 1.8158 | 1.2396 |

## Fold C_ara
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| level_shape_C | 1.3712 | 1.3310 | 1.5781 | 1.3896 |
| cold_B_alone | 1.6690 | 1.6757 | 1.6306 | 1.6658 |
| router_CB | 1.3804 | 1.3310 | 1.6306 | 1.4029 |

## Ozet (rmsle_blend / rmsle_cold)
| Model | blend | cold |
| --- | --- | --- |
| cold_B_alone | 1.7243 | 1.9552 |
| level_shape_C | 1.3360 | 1.8746 |
| router_CB | 1.3618 | 1.9552 |

Delta vs C: blend +0.0258, cold +0.0806
submit_ok: **False**
