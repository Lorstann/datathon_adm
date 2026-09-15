# Feature gain and cold error slices (baseline C, with weather)
Uretim: 2026-08-21 16:57:43.768847

## Top-30 LightGBM gain (Fold A)
| feature | gain |
| --- | --- |
| hist_mean | 22497.9 |
| hist_mean_7 | 19738.2 |
| lokasyon_cat | 18551.1 |
| hist_mean_z | 12773.2 |
| guc | 9363.8 |
| bayrama_kalan_gun | 9325.5 |
| hist_zero_rate | 6759.2 |
| peer_lok_z | 5704.8 |
| test_gun_sayisi | 4907.5 |
| hist_n | 4699.0 |
| hist_mean_28 | 4398.3 |
| temperature_2m_max | 3968.6 |
| devreye_alinma_yasi | 3430.2 |
| hdd_sum_14 | 3184.1 |
| hist_mean_91 | 3155.1 |
| hdd_x_logguc | 2645.8 |
| log_guc | 2316.4 |
| cos_doy | 2047.4 |
| peer_guc_wknd_z | 1839.6 |
| wind_speed_10m_max | 1811.8 |
| hdd_sum_7 | 1721.4 |
| temp_savgol | 1614.1 |
| temp_lag1 | 1418.1 |
| day | 1398.6 |
| weekend_x_logguc | 1371.6 |
| hdd_sum_3 | 1332.5 |
| temp_lag2 | 1273.5 |
| temperature_2m_min | 1256.6 |
| hdd | 1074.9 |
| holiday_x_logguc | 1009.4 |

## Cold residual by month (Fold A)
| month | n_cold | sqrtMSLE |
| --- | --- | --- |
| 4 | 312 | 3.6158 |
| 5 | 12245 | 3.0062 |
| 6 | 15922 | 2.4691 |
| 7 | 15681 | 1.4211 |

## Cold residual holiday vs not
- holiday=0: n=40280 sqrtMSLE=2.2940
- holiday=1: n=3880 sqrtMSLE=2.8489

## Cold residual by CDD tercile
- CDD (-0.001, 2.2]: n=14758 sqrtMSLE=2.9027
- CDD (2.2, 6.4]: n=15021 sqrtMSLE=2.2979
- CDD (6.4, 14.1]: n=14381 sqrtMSLE=1.6637
