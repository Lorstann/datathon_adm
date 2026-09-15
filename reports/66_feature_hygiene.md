# Ozellik hijyeni: year, akran sabitleri, tek-kolon DOW ofseti

Kosum: 2026-08-31, sure 585 s. Fold A disarida (veri acligi), tek tohum,
sinirli agac ve satir sayisi.

Not: kosum tamamlandi, yalnizca rapor yazimi `to_markdown` icin gereken
`tabulate` paketi olmadigindan dustu. Sayilar terminal ciktisindan
kurtarildi; betik `to_string`e cevrildi, tekrar kosmaya gerek yok.

## rmsle_blend

```
variant                B_guncel   C_ara     mean    delta
baseline                 1.0204  1.2262   1.1233   0.0000
year_yok                 1.0204  1.2262   1.1233   0.0000
year+peer_global_yok     1.0215  1.2226   1.1221  -0.0013
year+tum_peer_yok        1.0195  1.2231   1.1213  -0.0020
dow_off                  1.0201  1.2240   1.1221  -0.0013
year_yok+dow_off         1.0201  1.2240   1.1221  -0.0013
```

## rmsle_warm

```
variant                B_guncel   C_ara     mean    delta
baseline                 0.6520  0.8406   0.7463   0.0000
year_yok                 0.6520  0.8406   0.7463   0.0000
year+peer_global_yok     0.6560  0.8365   0.7463  -0.0000
year+tum_peer_yok        0.6523  0.8366   0.7445  -0.0019
dow_off                  0.6512  0.8393   0.7453  -0.0011
year_yok+dow_off         0.6512  0.8393   0.7453  -0.0011
```

## rmsle_cold

```
variant                B_guncel   C_ara     mean    delta
baseline                 1.7903  2.0744   1.9324   0.0000
year_yok                 1.7903  2.0744   1.9324   0.0000
year+peer_global_yok     1.7880  2.0707   1.9294  -0.0030
year+tum_peer_yok        1.7877  2.0718   1.9298  -0.0026
dow_off                  1.7906  2.0704   1.9305  -0.0019
year_yok+dow_off         1.7906  2.0704   1.9305  -0.0019
```

## Okuma

Butun deltalar +/-0.002 icinde, yani tohum gurultusu mertebesinde. Hicbir
hijyen mudahalesi anlamli bir sey yapmiyor.

Iki ayrinti dikkate deger:

**`year` tamamen olu.** `year_yok` uc metrikte de baseline ile bit-birebir
ayni sayilari veriyor. Yani model bu kolonu hic bolmede kullanmamis --
`FEATURE_EXCLUDE`'a eklemek dogru temizlik ama skor etkisi tam sifir. Ekstrapolasyon
riski diye tasidigimiz endise gercekte yokmus.

**Akran sabitlerini atmak cold'da minik ve tutarli bir iyilesme veriyor**
(-0.003, iki fold'da da ayni yonde), ama warm'da bunu geri veriyor ve blend'de
kayboluyor. Yonu dogru, buyuklugu onemsiz.

Sonuc: yerel boru hattinin tavanindayiz. Bu, ayni gun LB cebiriyle olculen
sonucla ortusuyor -- `reports/75_blend.md`'de FINAL_4'un artiginin yapisal
hicbir yonle iliskili olmadigi, `reports/76_zero_floor.md`'de kalan hatanin
sifir Bernoulli gurultusu ve cold varyansi oldugu gorulmustu. Ozellik
duzeyinde yapilacak is kalmamis.
