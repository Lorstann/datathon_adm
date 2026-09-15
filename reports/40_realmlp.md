# RealMLP vs LightGBM + topluluk
Uretim: 2026-08-23 12:44:15.805675

## Fold B_guncel  (hata korelasyonu lgbm-realmlp: **0.8635**)

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm | 0.6537 | 1.7854 | 1.0193 |
| realmlp | 0.7056 | 2.0940 | 1.1658 |
| blend_0.1 | 0.6528 | 1.7854 | 1.0189 |
| blend_0.2 | 0.6534 | 1.7928 | 1.0220 |
| blend_0.3 | 0.6553 | 1.8076 | 1.0287 |
| blend_0.4 | 0.6585 | 1.8295 | 1.0389 |

## Fold C_ara  (hata korelasyonu lgbm-realmlp: **0.9277**)

| model | warm | cold | blend |
| --- | --- | --- | --- |
| lgbm | 0.8297 | 2.0368 | 1.2063 |
| realmlp | 0.8482 | 2.1646 | 1.2642 |
| blend_0.1 | 0.8252 | 2.0321 | 1.2022 |
| blend_0.2 | 0.8222 | 2.0313 | 1.2003 |
| blend_0.3 | 0.8206 | 2.0346 | 1.2006 |
| blend_0.4 | 0.8204 | 2.0418 | 1.2032 |

## Ozet (rmsle_blend)

| model | mean | std |
| --- | --- | --- |
| blend_0.1 | 1.1105 | 0.0916 |
| blend_0.2 | 1.1111 | 0.0891 |
| blend_0.3 | 1.1147 | 0.0859 |
| blend_0.4 | 1.1210 | 0.0821 |
| lgbm | 1.1128 | 0.0935 |
| realmlp | 1.2150 | 0.0492 |

### Fold bazinda lgbm farki (negatif = iyi)

- blend_0.1: {'B_guncel': -0.0004, 'C_ara': -0.0041}
- blend_0.2: {'B_guncel': 0.0027, 'C_ara': -0.006}
- blend_0.3: {'B_guncel': 0.0094, 'C_ara': -0.0057}
- blend_0.4: {'B_guncel': 0.0196, 'C_ara': -0.0031}
- lgbm: {'B_guncel': 0.0, 'C_ara': 0.0}
- realmlp: {'B_guncel': 0.1465, 'C_ara': 0.0579}
