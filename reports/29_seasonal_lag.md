# Satir duzeyinde gecen-yil lag'i
Uretim: 2026-08-22 11:49:01.961954

Kapsama asimetrisi: gonderimde tam, fold A/C'de nerdeyse yok. Karar fold B'ye gore.

## Fold A_mevsim

| variant | kapsama(valid) | warm | cold | blend |
| --- | --- | --- | --- | --- |
| base | - | 0.8771 | 2.3721 | 1.3586 |
| lag364 | 0.0% (egitim 0.0%) | 0.8771 | 2.3721 | 1.3586 |

## Fold B_guncel

| variant | kapsama(valid) | warm | cold | blend |
| --- | --- | --- | --- | --- |
| base | - | 0.6515 | 1.7855 | 1.0183 |
| lag364 | 33.6% (egitim 0.0%) | 0.6515 | 1.7855 | 1.0183 |

## Fold C_ara

| variant | kapsama(valid) | warm | cold | blend |
| --- | --- | --- | --- | --- |
| base | - | 0.8316 | 2.0361 | 1.2070 |
| lag364 | 13.3% (egitim 0.0%) | 0.8316 | 2.0361 | 1.2070 |

## Ozet (rmsle_blend)

| variant | mean | std | warm |
| --- | --- | --- | --- |
| base | 1.1946 | 0.1392 | 0.7867 |
| lag364 | 1.1946 | 0.1392 | 0.7867 |

Fold bazinda fark (lag364 - base): {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}

## Karar: BU CV TASARIMIYLA OLCULEMEZ -> GONDERILMEDI

Uc fold'un da EGITIM kapsamasi %0. Fold origin'lerinin tamami 2025 icinde,
dolayisiyla ufuklarinin 364 gun oncesi 2024'e, panel baslangicindan (2025-01-01)
oncesine dusuyor. Kolon egitimde bastan sona NaN kaldigi icin LightGBM hic
dallanmadi; uc fold'da da fark tam olarak 0.0000. Sonuc "faydasiz" degil,
**olculemedi**.

Gonderim origin'inde durum farkli olurdu:
- test kapsamasi %95.5
- egitim kapsamasi ~%25-30 (yalnizca 2025-11 ve sonrasi origin'ler)

Gondermemek icin gerekce, bilinmeyen fayda degil **bilinen risk**: model bu
iliskiyi yalnizca kis ufuklu origin'lerden ogrenip yaz ufkuna (Nis-Tem)
uygulardi, ustelik egitimde %28 / testte %95 kapsama farkiyla. Dogrulanamayan
ve dagilimi kayan bir ozellik, sinirli gonderim hakkiyla korunacak bir bahis
degil.

Olculebilir hale gelmesi 2024 verisi gerektirir; elimizde yok.
`seasonal_lag` parametresi varsayilan olarak KAPALI, bu script yeniden
calistirilabilsin diye duruyor.
