# Iki asamali mimari (reports/02 karar 3)
Uretim: 2026-08-25 00:09:44.607305

## Fold A_mevsim

| variant | warm | cold | blend |
| --- | --- | --- | --- |
| mevcut | 0.8695 | 2.3595 | 1.3499 |
| iki_asamali_k0.75 | 0.8848 | 2.3611 | 1.3582 |
| iki_asamali_k1.0 | 0.8492 | 2.3750 | 1.3459 |
| hibrit_k0.75 | 0.8848 | 2.3595 | 1.3576 |
| hibrit_k1.0 | 0.8492 | 2.3595 | 1.3398 |

## Fold B_guncel

| variant | warm | cold | blend |
| --- | --- | --- | --- |
| mevcut | 0.6564 | 1.7864 | 1.0211 |
| iki_asamali_k0.75 | 0.7098 | 1.7849 | 1.0480 |
| iki_asamali_k1.0 | 0.6469 | 1.7898 | 1.0177 |
| hibrit_k0.75 | 0.7098 | 1.7864 | 1.0485 |
| hibrit_k1.0 | 0.6469 | 1.7864 | 1.0163 |

## Fold C_ara

| variant | warm | cold | blend |
| --- | --- | --- | --- |
| mevcut | 0.8350 | 2.0381 | 1.2096 |
| iki_asamali_k0.75 | 0.8311 | 2.0403 | 1.2084 |
| iki_asamali_k1.0 | 0.8392 | 2.0331 | 1.2100 |
| hibrit_k0.75 | 0.8311 | 2.0381 | 1.2076 |
| hibrit_k1.0 | 0.8392 | 2.0381 | 1.2119 |

## Ozet

| variant | mean | std | warm | cold |
| --- | --- | --- | --- | --- |
| hibrit_k0.75 | 1.2046 | 0.1262 | 0.8086 | 2.0614 |
| hibrit_k1.0 | 1.1894 | 0.1330 | 0.7784 | 2.0614 |
| iki_asamali_k0.75 | 1.2049 | 0.1267 | 0.8086 | 2.0621 |
| iki_asamali_k1.0 | 1.1912 | 0.1346 | 0.7784 | 2.0660 |
| mevcut | 1.1935 | 0.1347 | 0.7870 | 2.0614 |

- hibrit_k0.75: {'A_mevsim': 0.0077, 'B_guncel': 0.0274, 'C_ara': -0.0021}
- hibrit_k1.0: {'A_mevsim': -0.0101, 'B_guncel': -0.0047, 'C_ara': 0.0023}
- iki_asamali_k0.75: {'A_mevsim': 0.0083, 'B_guncel': 0.0269, 'C_ara': -0.0012}
- iki_asamali_k1.0: {'A_mevsim': -0.004, 'B_guncel': -0.0034, 'C_ara': 0.0004}
- mevcut: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}