# Ozellik paketi v2 + harmanlanmis seviye
Uretim: 2026-08-22 11:28:58.199660

Hepsine cold olu-trafo kapisi k=0.75 uygulanmistir.

## Fold A_mevsim

| variant | warm | cold | blend | s |
| --- | --- | --- | --- | --- |
| base | 0.8871 | 2.3766 | 1.3654 | 9 |
| blend_level | 0.8662 | 2.3708 | 1.3526 | 8 |
| feat_v2 | 0.8771 | 2.3721 | 1.3586 | 9 |
| feat_v2_blend | 0.8622 | 2.3759 | 1.3526 | 10 |

## Fold B_guncel

| variant | warm | cold | blend | s |
| --- | --- | --- | --- | --- |
| base | 0.6593 | 1.7812 | 1.0205 | 54 |
| blend_level | 0.6548 | 1.7831 | 1.0190 | 59 |
| feat_v2 | 0.6514 | 1.7857 | 1.0183 | 59 |
| feat_v2_blend | 0.6423 | 1.7831 | 1.0128 | 51 |

## Fold C_ara

| variant | warm | cold | blend | s |
| --- | --- | --- | --- | --- |
| base | 0.8436 | 2.0367 | 1.2138 | 39 |
| blend_level | 0.8617 | 2.0405 | 1.2250 | 40 |
| feat_v2 | 0.8316 | 2.0361 | 1.2070 | 44 |
| feat_v2_blend | 0.8594 | 2.0364 | 1.2222 | 43 |

## Ozet (rmsle_blend)

| variant | mean | std | warm | cold |
| --- | --- | --- | --- | --- |
| base | 1.1999 | 0.1411 | 0.7967 | 2.0648 |
| blend_level | 1.1989 | 0.1375 | 0.7943 | 2.0648 |
| feat_v2 | 1.1946 | 0.1392 | 0.7867 | 2.0646 |
| feat_v2_blend | 1.1959 | 0.1400 | 0.7880 | 2.0651 |

### Fold bazinda base farki (negatif = iyi)

- base: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}
- blend_level: {'A_mevsim': -0.0127, 'B_guncel': -0.0015, 'C_ara': 0.0113}
- feat_v2: {'A_mevsim': -0.0068, 'B_guncel': -0.0022, 'C_ara': -0.0067}
- feat_v2_blend: {'A_mevsim': -0.0127, 'B_guncel': -0.0077, 'C_ara': 0.0085}

En iyi: **feat_v2**
