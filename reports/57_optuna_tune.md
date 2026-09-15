# Optuna hiperparametre taramasi
Uretim: 2026-08-28 12:55:40.032453

Arama: fold A, 25 deneme, 350 agac, 500,000 satir. Dogrulama: 3 fold, 600 agac, 1,200,000 satir.

## Ozet (rmsle_blend, 3 fold ortalama/std)

| config | mean | std |
| --- | --- | --- |
| baseline_default | 1.1940 | 0.1356 |
| optuna_top1 | 1.1917 | 0.1333 |
| optuna_top2 | 1.1935 | 0.1324 |
| optuna_top3 | 1.1923 | 0.1323 |

## En iyi 3 adayin parametreleri

- optuna_top1 (arama blend=1.3481): {'num_leaves': 145, 'min_child_samples': 63, 'learning_rate': 0.07686932565906376, 'subsample': 0.8400055837408262, 'colsample_bytree': 0.5402386648520059, 'reg_lambda': 0.020901689794526224, 'reg_alpha': 0.003402265507666337, 'max_depth': 4}
- optuna_top2 (arama blend=1.3483): {'num_leaves': 97, 'min_child_samples': 10, 'learning_rate': 0.06467688894067607, 'subsample': 0.7320832480224757, 'colsample_bytree': 0.5050431295870819, 'reg_lambda': 0.0026566771008530725, 'reg_alpha': 0.6664310287751155, 'max_depth': 4}
- optuna_top3 (arama blend=1.3486): {'num_leaves': 165, 'min_child_samples': 81, 'learning_rate': 0.05441000474584713, 'subsample': 0.7438601957680462, 'colsample_bytree': 0.523848543815336, 'reg_lambda': 0.026855650086142296, 'reg_alpha': 0.004449119983202131, 'max_depth': 4}
