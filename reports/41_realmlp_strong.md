# RealMLP vs LightGBM + topluluk
Uretim: 2026-08-23 13:18:39.424342

## Fold A_mevsim  (hata korelasyonu lgbm-realmlp: **0.9618**)

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm | 0.8756 | 2.3622 | 1.3540 |
| realmlp | 0.9950 | 2.4081 | 1.4338 |
| blend_0.1 | 0.8832 | 2.3642 | 1.3586 |
| blend_0.2 | 0.8866 | 2.3653 | 1.3607 |
| blend_0.3 | 0.8946 | 2.3680 | 1.3659 |

## Fold B_guncel  (hata korelasyonu lgbm-realmlp: **0.8298**)

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm | 0.6491 | 1.7857 | 1.0171 |
| realmlp | 0.7058 | 2.2468 | 1.2273 |
| blend_0.1 | 0.6486 | 1.8003 | 1.0226 |
| blend_0.2 | 0.6491 | 1.8100 | 1.0266 |
| blend_0.3 | 0.6514 | 1.8366 | 1.0382 |

## Fold C_ara  (hata korelasyonu lgbm-realmlp: **0.9108**)

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm | 0.8314 | 2.0368 | 1.2072 |
| realmlp | 0.8543 | 2.2154 | 1.2868 |
| blend_0.1 | 0.8226 | 2.0347 | 1.2017 |
| blend_0.2 | 0.8206 | 2.0364 | 1.2013 |
| blend_0.3 | 0.8181 | 2.0433 | 1.2025 |

## Ozet (rmsle_blend)

| model | mean | std |
| --- | --- | --- |
| blend_0.1 | 1.1943 | 0.1373 |
| blend_0.2 | 1.1962 | 0.1364 |
| blend_0.3 | 1.2022 | 0.1338 |
| lgbm | 1.1928 | 0.1379 |
| realmlp | 1.3160 | 0.0868 |

### Fold bazinda lgbm farki (negatif = iyi)

- blend_0.1: {'A_mevsim': 0.0046, 'B_guncel': 0.0054, 'C_ara': -0.0055}
- blend_0.2: {'A_mevsim': 0.0067, 'B_guncel': 0.0095, 'C_ara': -0.0059}
- blend_0.3: {'A_mevsim': 0.0119, 'B_guncel': 0.021, 'C_ara': -0.0047}
- lgbm: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}
- realmlp: {'A_mevsim': 0.0798, 'B_guncel': 0.2102, 'C_ara': 0.0796}
