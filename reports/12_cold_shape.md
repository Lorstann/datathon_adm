# Cold-only shape router
Uretim: 2026-08-21 17:03:49.093633
Cold = (global_z+log_guc) + shape(no hist). Warm = C.

## Fold A_mevsim
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| level_shape_C | 1.3435 | 1.0040 | 2.3480 | 1.4165 |
| cold_shape | 1.8709 | 1.7401 | 2.3868 | 1.9025 |
| router_C_cs | 1.3556 | 1.0040 | 2.3868 | 1.4307 |
| router_C_cs_monthbias_oracle | 1.3454 | 1.0040 | 2.3541 | 1.4187 |

## Fold B_guncel
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| level_shape_C | 1.1605 | 1.0175 | 1.6976 | 1.2019 |
| cold_shape | 1.8566 | 1.8834 | 1.7185 | 1.8481 |
| router_C_cs | 1.1656 | 1.0175 | 1.7185 | 1.2084 |
| router_C_cs_monthbias_oracle | 1.1706 | 1.0175 | 1.7385 | 1.2148 |

## Fold C_ara
| Model | all | warm | cold | blend |
| --- | --- | --- | --- | --- |
| level_shape_C | 1.3711 | 1.3309 | 1.5781 | 1.3895 |
| cold_shape | 1.7943 | 1.8293 | 1.5837 | 1.7778 |
| router_C_cs | 1.3721 | 1.3309 | 1.5837 | 1.3909 |
| router_C_cs_monthbias_oracle | 1.3747 | 1.3309 | 1.5987 | 1.3947 |

## Ozet
| Model | blend | cold |
| --- | --- | --- |
| cold_shape | 1.8428 | 1.8963 |
| level_shape_C | 1.3359 | 1.8746 |
| router_C_cs | 1.3434 | 1.8963 |
| router_C_cs_monthbias_oracle | 1.3427 | 1.8971 |

Delta router_C_cs vs C: blend +0.0074, cold +0.0218
submit_ok: **False**
Not: `monthbias_oracle` ayni fold artik medyani — tavan; secim kurali degil.
