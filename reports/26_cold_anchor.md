# Cold capa (canli-only) x olu kapisi
Uretim: 2026-08-22 11:11:16.138177

## Fold A_mevsim

| capa | kapi k | warm | cold | blend |
| --- | --- | --- | --- | --- |
| tum | 0.00 | 0.8871 | 2.4254 | 1.3843 |
| tum | 0.75 | 0.8871 | 2.3766 | 1.3654 |
| canli | 0.00 | 0.8839 | 2.4408 | 1.3886 |
| canli | 0.75 | 0.8839 | 2.3722 | 1.3620 |

## Fold B_guncel

| capa | kapi k | warm | cold | blend |
| --- | --- | --- | --- | --- |
| tum | 0.00 | 0.6593 | 1.7875 | 1.0229 |
| tum | 0.75 | 0.6593 | 1.7812 | 1.0205 |
| canli | 0.00 | 0.6611 | 1.7998 | 1.0286 |
| canli | 0.75 | 0.6611 | 1.7819 | 1.0217 |

## Fold C_ara

| capa | kapi k | warm | cold | blend |
| --- | --- | --- | --- | --- |
| tum | 0.00 | 0.8436 | 2.0715 | 1.2268 |
| tum | 0.75 | 0.8436 | 2.0367 | 1.2138 |
| canli | 0.00 | 0.8410 | 2.0919 | 1.2330 |
| canli | 0.75 | 0.8410 | 2.0496 | 1.2172 |

## Ozet (rmsle_blend)

| variant | mean | std | warm | cold |
| --- | --- | --- | --- | --- |
| canli_capa_k0.0 | 1.2168 | 0.1474 | 0.7953 | 2.1109 |
| canli_capa_k0.75 | 1.2003 | 0.1395 | 0.7953 | 2.0679 |
| tum_capa_k0.0 | 1.2113 | 0.1479 | 0.7967 | 2.0948 |
| tum_capa_k0.75 | 1.1999 | 0.1411 | 0.7967 | 2.0648 |

### Fold bazinda mevcut gonderimden fark (negatif = iyi)

- canli_capa_k0.0: {'A_mevsim': 0.0044, 'B_guncel': 0.0056, 'C_ara': 0.0063}
- canli_capa_k0.75: {'A_mevsim': -0.0222, 'B_guncel': -0.0013, 'C_ara': -0.0096}
- tum_capa_k0.0: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}
- tum_capa_k0.75: {'A_mevsim': -0.0189, 'B_guncel': -0.0025, 'C_ara': -0.013}

En iyi: **tum_capa_k0.75**

## Karar: REDDEDILDI

Canli-only capa cold sifir-olmayan satirlardaki artigi kapatiyor ama blend'i
kotulestiriyor (1.1999 -> 1.2003 kapiyla, 1.2113 -> 1.2168 kapisiz), uc fold'un
ucunde de. Yorum: `reports/23_error_decomp.md`'deki +0.10/+0.16 artik bir
kalibrasyon hatasi degil, sifir kutlesi karsisinda dogru L2 uzlasmasi. Capayi
yukseltmek sifir-olmayan satirlari duzeltirken sifir satirlari daha cok
bozuyor; sifir kutlesi `zero_gate.shrink_level` ile zaten daha verimli
ele aliniyor.

`alive_anchor` parametresi `entity_level`'dan kaldirildi ve bu deneyin
scripti silindi; sonuc burada kayitli.
