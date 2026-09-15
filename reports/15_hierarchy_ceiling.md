# Hierarchy / profile ceiling
Uretim: 2026-08-21 17:24:11.351880
Referans C: blend=1.3359, cold=1.8746
Go: blend <= C-0.05 VEYA (cold <= C-0.08 ve blend kotulesmesin).

## Fold A_mevsim
| method | warm | cold | blend | coverage |
| --- | --- | --- | --- | --- |
| td_guc_dow | 1.2042 | 2.3098 | 1.5202 | 1.000 |
| td_ilce_guc_dow | 1.3678 | 2.5656 | 1.7073 | 0.976 |
| td_guc_dow_yoy | 1.2042 | 2.3098 | 1.5202 | 1.000 |
| td_ilce_guc_dow_yoy | 1.3678 | 2.5656 | 1.7073 | 0.976 |
| guc_month_z | 1.7811 | 2.3572 | 1.9237 | 0.000 |
| guc_month_z_yoy | 1.7811 | 2.3572 | 1.9237 | 0.000 |
| profile_kmeans | 1.0231 | 2.3567 | 1.4303 | 1.000 |

## Fold B_guncel
| method | warm | cold | blend | coverage |
| --- | --- | --- | --- | --- |
| td_guc_dow | 1.9530 | 1.8552 | 1.9318 | 1.000 |
| td_ilce_guc_dow | 2.1487 | 2.4927 | 2.2295 | 0.992 |
| td_guc_dow_yoy | 1.9530 | 1.8552 | 1.9318 | 1.000 |
| td_ilce_guc_dow_yoy | 2.1487 | 2.4927 | 2.2295 | 0.992 |
| guc_month_z | 1.8840 | 1.7074 | 1.8463 | 0.808 |
| guc_month_z_yoy | 1.8840 | 1.7074 | 1.8463 | 0.808 |
| profile_kmeans | 1.1076 | 1.7007 | 1.2632 | 1.000 |

## Fold C_ara
| method | warm | cold | blend | coverage |
| --- | --- | --- | --- | --- |
| td_guc_dow | 1.6263 | 1.6951 | 1.6418 | 1.000 |
| td_ilce_guc_dow | 1.8385 | 2.3502 | 1.9634 | 0.994 |
| td_guc_dow_yoy | 1.6263 | 1.6951 | 1.6418 | 1.000 |
| td_ilce_guc_dow_yoy | 1.8385 | 2.3502 | 1.9634 | 0.994 |
| guc_month_z | 1.8537 | 1.5808 | 1.7968 | 0.315 |
| guc_month_z_yoy | 1.8537 | 1.5808 | 1.7968 | 0.315 |
| profile_kmeans | 1.3731 | 1.5864 | 1.4231 | 1.000 |

## Ozet
| method | blend | cold | vs C blend | vs C cold |
| --- | --- | --- | --- | --- |
| guc_month_z | 1.8556 | 1.8818 | +0.5197 | +0.0072 |
| guc_month_z_yoy | 1.8556 | 1.8818 | +0.5197 | +0.0072 |
| profile_kmeans | 1.3722 | 1.8813 | +0.0363 | +0.0067 |
| td_guc_dow | 1.6979 | 1.9534 | +0.3620 | +0.0788 |
| td_guc_dow_yoy | 1.6979 | 1.9534 | +0.3620 | +0.0788 |
| td_ilce_guc_dow | 1.9667 | 2.4695 | +0.6308 | +0.5949 |
| td_ilce_guc_dow_yoy | 1.9667 | 2.4695 | +0.6308 | +0.5949 |

**Go/No-go: NO-GO** → `narrow_C`
