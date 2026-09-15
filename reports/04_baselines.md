# Naif tabanlar
Uretim: 2026-08-21 16:14:01.902196

## Fold A_mevsim (sizinti {'grouped_sorted': 'OK', 'no_future_target': 'OK', 'masked_absent': 'OK (637 trafo)'})
train 134,676  valid 249,588  cold 44160
| Model | rmsle_all | rmsle_warm | rmsle_cold | rmsle_blend |
| --- | --- | --- | --- | --- |
| global_median | 2.1668 | 2.1025 | 2.4437 | 2.1827 |
| global_z | 1.9098 | 1.7947 | 2.3728 | 1.9378 |
| entity_mean | 1.4245 | 1.0925 | 2.4324 | 1.4967 |
| entity_mean_28 | 1.4186 | 1.0840 | 2.4307 | 1.4913 |
| entity_dow | 1.4233 | 1.0905 | 2.4324 | 1.4956 |
| seasonal_naive | 1.4245 | 1.0925 | 2.4324 | 1.4967 |
| seasonal_level_adj | 1.4245 | 1.0925 | 2.4324 | 1.4967 |
| peer_guc_z | 1.9100 | 1.7799 | 2.4250 | 1.9415 |

## Fold B_guncel (sizinti {'grouped_sorted': 'OK', 'no_future_target': 'OK', 'masked_absent': 'OK (983 trafo)'})
train 558,983  valid 406,124  cold 68484
| Model | rmsle_all | rmsle_warm | rmsle_cold | rmsle_blend |
| --- | --- | --- | --- | --- |
| global_median | 2.1501 | 2.1888 | 1.9478 | 2.1378 |
| global_z | 1.8882 | 1.9218 | 1.7130 | 1.8775 |
| entity_mean | 1.3236 | 1.1544 | 1.9543 | 1.3725 |
| entity_mean_28 | 1.2504 | 1.0478 | 1.9645 | 1.3076 |
| entity_dow | 1.3243 | 1.1553 | 1.9543 | 1.3731 |
| seasonal_naive | 1.3802 | 1.2316 | 1.9543 | 1.4237 |
| seasonal_level_adj | 1.3749 | 1.2244 | 1.9543 | 1.4189 |
| peer_guc_z | 1.8683 | 1.8970 | 1.7200 | 1.8593 |

## Fold C_ara (sizinti {'grouped_sorted': 'OK', 'no_future_target': 'OK', 'masked_absent': 'OK (741 trafo)'})
train 459,870  valid 356,648  cold 53862
| Model | rmsle_all | rmsle_warm | rmsle_cold | rmsle_blend |
| --- | --- | --- | --- | --- |
| global_median | 2.1179 | 2.1619 | 1.8513 | 2.0971 |
| global_z | 1.8447 | 1.8848 | 1.6006 | 1.8256 |
| entity_mean | 1.5238 | 1.4572 | 1.8544 | 1.5540 |
| entity_mean_28 | 1.5005 | 1.4288 | 1.8525 | 1.5329 |
| entity_dow | 1.5235 | 1.4568 | 1.8544 | 1.5537 |
| seasonal_naive | 1.5387 | 1.4755 | 1.8544 | 1.5674 |
| seasonal_level_adj | 1.6197 | 1.5743 | 1.8544 | 1.6405 |
| peer_guc_z | 1.8206 | 1.8576 | 1.5967 | 1.8031 |

## Fold'lar arasi ozet (rmsle_blend)
| Model | mean | std |
| --- | --- | --- |
| entity_dow | 1.4741 | 0.0753 |
| entity_mean | 1.4744 | 0.0758 |
| entity_mean_28 | 1.4439 | 0.0979 |
| global_median | 2.1392 | 0.0350 |
| global_z | 1.8803 | 0.0458 |
| peer_guc_z | 1.8679 | 0.0568 |
| seasonal_level_adj | 1.5187 | 0.0918 |
| seasonal_naive | 1.4959 | 0.0586 |
