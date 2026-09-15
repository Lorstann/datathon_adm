# Origin yogunlugu + kapasite
Uretim: 2026-08-22 10:55:50.673843

Her varyanta cold olu-trafo kapisi k=0.75 uygulanmistir.

## Fold B_guncel

| variant | egitim satiri | warm | cold | blend | s |
| --- | --- | --- | --- | --- | --- |
| base_45d_l63_t600 | 1,200,000 | 0.6593 | 1.7812 | 1.0205 | 52 |
| dense_30d_l63_t600 | 1,800,000 | 0.6695 | 1.7788 | 1.0248 | 74 |
| dense_30d_l127_t900 | 1,800,000 | 0.6719 | 1.7830 | 1.0276 | 89 |
| dense_30d_l63_t1200_lr03 | 1,800,000 | 0.6648 | 1.7797 | 1.0227 | 103 |
| dense_21d_l63_t600 | 2,400,000 | 0.6676 | 1.7789 | 1.0238 | 101 |

## Fold C_ara

| variant | egitim satiri | warm | cold | blend | s |
| --- | --- | --- | --- | --- | --- |
| base_45d_l63_t600 | 1,167,286 | 0.8436 | 2.0367 | 1.2138 | 35 |
| dense_30d_l63_t600 | 1,800,000 | 0.8374 | 2.0397 | 1.2115 | 57 |
| dense_30d_l127_t900 | 1,800,000 | 0.8450 | 2.0412 | 1.2162 | 84 |
| dense_30d_l63_t1200_lr03 | 1,800,000 | 0.8391 | 2.0391 | 1.2122 | 99 |
| dense_21d_l63_t600 | 2,400,000 | 0.8330 | 2.0364 | 1.2079 | 75 |

## Ozet (rmsle_blend)

| variant | mean | std | warm | cold |
| --- | --- | --- | --- | --- |
| base_45d_l63_t600 | 1.1171 | 0.0966 | 0.7515 | 1.9090 |
| dense_21d_l63_t600 | 1.1159 | 0.0921 | 0.7503 | 1.9076 |
| dense_30d_l127_t900 | 1.1219 | 0.0943 | 0.7584 | 1.9121 |
| dense_30d_l63_t1200_lr03 | 1.1174 | 0.0948 | 0.7519 | 1.9094 |
| dense_30d_l63_t600 | 1.1181 | 0.0934 | 0.7535 | 1.9093 |

### Fold bazinda base farki (negatif = iyi)

- base_45d_l63_t600: {'B_guncel': 0.0, 'C_ara': 0.0}
- dense_21d_l63_t600: {'B_guncel': 0.0033, 'C_ara': -0.0058}
- dense_30d_l127_t900: {'B_guncel': 0.0071, 'C_ara': 0.0024}
- dense_30d_l63_t1200_lr03: {'B_guncel': 0.0022, 'C_ara': -0.0016}
- dense_30d_l63_t600: {'B_guncel': 0.0043, 'C_ara': -0.0022}

En iyi: **dense_21d_l63_t600**
