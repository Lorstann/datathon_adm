# Adversarial validation: train vs gercek test
Uretim: 2026-08-28 11:37:29.454306

tr_f=356,522 satir (max_rows=400,000), te_f=714,688 satir. 5-fold OOF AUC.

## AUC: **1.0000**

0.5 = train/test ayni dagilimdan, ayirt edilemiyor. 1.0 = mukemmel ayirt ediliyor, train/test dagilim farki buyuk.

## En cok ayirt eden 25 ozellik (gain)

| feature | gain |
| --- | --- |
| peer_global_z | 48080342.2 |
| cos_month | 3470524.7 |
| year | 3082500.5 |
| month | 507005.0 |
| peer_zero_guc | 215327.8 |
| sunshine_duration | 170224.0 |
| bayrama_kalan_gun | 77580.3 |
| doy | 74853.7 |
| cos_doy | 65522.9 |
| guc_band | 64500.3 |
| peer_guc_wknd_z | 59150.2 |
| peer_guc_z | 22023.1 |
| guc | 4741.5 |
| cdd_sum_14 | 2260.7 |
| hist_span | 1904.7 |
| horizon_day | 963.0 |
| day | 774.6 |
| peer_guc_log | 588.9 |
| hist_cdd_slope | 456.9 |
| devreye_alinma_yasi | 353.3 |
| test_gun_sayisi | 251.3 |
| sin_month | 232.3 |
| temperature_2m_max | 222.9 |
| cdd_sum_7 | 198.6 |
| hist_mom_7_91 | 85.8 |
