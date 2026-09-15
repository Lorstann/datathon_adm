# Seviye cesitliligi + XGBoost rovansi
Uretim: 2026-08-23 05:19:53.757411

## Fold A_mevsim

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm_m7 | 0.8738 | 2.3611 | 1.3526 |
| lgbm_m28 | 0.8636 | 2.3619 | 1.3478 |
| xgb_cat | 1.1712 | 2.7852 | 1.6693 |
| avg_m7_m28 | 0.8644 | 2.3614 | 1.3481 |
| avg_m7_xgb | 0.9220 | 2.3872 | 1.3873 |
| avg_all3 | 0.9006 | 2.3764 | 1.3721 |

## Fold B_guncel

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm_m7 | 0.6556 | 1.7869 | 1.0209 |
| lgbm_m28 | 0.6595 | 1.7854 | 1.0222 |
| xgb_cat | 0.7638 | 2.8887 | 1.5176 |
| avg_m7_m28 | 0.6495 | 1.7861 | 1.0175 |
| avg_m7_xgb | 0.6670 | 1.9344 | 1.0842 |
| avg_all3 | 0.6597 | 1.8945 | 1.0650 |

## Fold C_ara

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm_m7 | 0.8485 | 2.0377 | 1.2168 |
| lgbm_m28 | 0.8825 | 2.0378 | 1.2355 |
| xgb_cat | 1.0004 | 2.2885 | 1.3927 |
| avg_m7_m28 | 0.8584 | 2.0377 | 1.2222 |
| avg_m7_xgb | 0.8479 | 2.0267 | 1.2124 |
| avg_all3 | 0.8458 | 2.0231 | 1.2099 |

## Ozet (rmsle_blend)

| model | mean | std | warm | cold |
| --- | --- | --- | --- | --- |
| avg_all3 | 1.2157 | 0.1255 | 0.8020 | 2.0980 |
| avg_m7_m28 | 1.1959 | 0.1362 | 0.7907 | 2.0617 |
| avg_m7_xgb | 1.2280 | 0.1242 | 0.8123 | 2.1161 |
| lgbm_m28 | 1.2019 | 0.1350 | 0.8018 | 2.0617 |
| lgbm_m7 | 1.1968 | 0.1362 | 0.7926 | 2.0619 |
| xgb_cat | 1.5266 | 0.1131 | 0.9785 | 2.6541 |

### Fold bazinda lgbm_m7 farki (negatif = iyi)

- avg_all3: {'A_mevsim': 0.0195, 'B_guncel': 0.0441, 'C_ara': -0.0069}
- avg_m7_m28: {'A_mevsim': -0.0046, 'B_guncel': -0.0034, 'C_ara': 0.0054}
- avg_m7_xgb: {'A_mevsim': 0.0346, 'B_guncel': 0.0633, 'C_ara': -0.0044}
- lgbm_m28: {'A_mevsim': -0.0048, 'B_guncel': 0.0014, 'C_ara': 0.0187}
- lgbm_m7: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}
- xgb_cat: {'A_mevsim': 0.3167, 'B_guncel': 0.4968, 'C_ara': 0.176}

En iyi: **avg_m7_m28**
