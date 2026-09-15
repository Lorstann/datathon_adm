# Model cesitliligi (cok-origin cercevesi)
Uretim: 2026-08-22 11:36:55.027070

## Fold A_mevsim  (egitim satiri 87,907)

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm | 0.8771 | 2.3721 | 1.3586 |
| catboost | 0.9842 | 2.5138 | 1.4678 |
| xgboost | 1.1984 | 2.7626 | 1.6760 |
| lgb_cat | 0.9012 | 2.3563 | 1.3647 |
| lgb_cat_xgb | 0.9272 | 2.3848 | 1.3890 |

## Fold B_guncel  (egitim satiri 900,000)

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm | 0.6566 | 1.7772 | 1.0176 |
| catboost | 0.6766 | 1.9324 | 1.0880 |
| xgboost | 0.7108 | 2.0907 | 1.1670 |
| lgb_cat | 0.6525 | 1.7955 | 1.0227 |
| lgb_cat_xgb | 0.6541 | 1.8240 | 1.0346 |

## Fold C_ara  (egitim satiri 900,000)

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm | 0.8279 | 2.0388 | 1.2061 |
| catboost | 0.8526 | 2.0961 | 1.2408 |
| xgboost | 0.9220 | 2.2240 | 1.3258 |
| lgb_cat | 0.8230 | 2.0240 | 1.1979 |
| lgb_cat_xgb | 0.8234 | 2.0286 | 1.1999 |

## Ozet (rmsle_blend)

| model | mean | std | warm | cold |
| --- | --- | --- | --- | --- |
| catboost | 1.2655 | 0.1560 | 0.8378 | 2.1808 |
| lgb_cat | 1.1951 | 0.1397 | 0.7922 | 2.0586 |
| lgb_cat_xgb | 1.2078 | 0.1448 | 0.8016 | 2.0792 |
| lgbm | 1.1941 | 0.1394 | 0.7872 | 2.0627 |
| xgboost | 1.3896 | 0.2126 | 0.9437 | 2.3591 |

### Fold bazinda lgbm farki (negatif = iyi)

- catboost: {'A_mevsim': 0.1092, 'B_guncel': 0.0704, 'C_ara': 0.0347}
- lgb_cat: {'A_mevsim': 0.0062, 'B_guncel': 0.0051, 'C_ara': -0.0082}
- lgb_cat_xgb: {'A_mevsim': 0.0305, 'B_guncel': 0.017, 'C_ara': -0.0062}
- lgbm: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}
- xgboost: {'A_mevsim': 0.3175, 'B_guncel': 0.1494, 'C_ara': 0.1197}

En iyi: **lgbm**
