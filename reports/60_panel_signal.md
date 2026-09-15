# Panel/kod/kohort/YoY sinyal taramasi
Uretim: 2026-08-31 11:35:27.872572

# H1 — Panel uyelik deseni sifiri ele veriyor mu

Test'te her trafonun hangi gunlerde satiri oldugu bilinir. Ayni
ozellikleri train'in iki 122 gunluk penceresinde uretip sifir
oraniyla karsilastiriyorum.

## Pencere 2025-04-01..2025-07-31 (test'in mevsimsel ikizi)
satir 274,929  trafo 2,891  taban sifir orani 0.0542

### Pencere sonuna kalan gun (`gun_cikisa`)
                    n  sifir_orani   ort_log
gun_cikisa                                  
(-0.501, 0.5]    2891     0.074715  6.674787
(0.5, 3.0]       8425     0.064214  6.794897
(3.0, 7.0]      10417     0.063358  6.840662
(7.0, 14.0]     18045     0.061402  6.788623
(14.0, 30.0]    40719     0.059358  6.701778
(30.0, 60.0]    69581     0.066915  6.436660
(60.0, 200.0]  124851     0.042571  6.329860

### Pencere basindan gecen gun (`gun_giristen`)
                    n  sifir_orani   ort_log
gun_giristen                                
(-0.501, 0.5]    2891     0.090626  5.947349
(0.5, 3.0]       8123     0.071772  6.244495
(3.0, 7.0]      10038     0.070532  6.237984
(7.0, 14.0]     17474     0.069818  6.303503
(14.0, 30.0]    39791     0.069011  6.185247
(30.0, 60.0]    70576     0.068791  6.177627
(60.0, 200.0]  126036     0.036014  6.812026

### Sonraki satira kalan gun (`bosluk_sonra`) — bosluk oncesi satir
                    n  sifir_orani   ort_log
bosluk_sonra                                
(-0.501, 1.5]  274353     0.054087  6.483702
(1.5, 2.5]        125     0.072000  4.726335
(2.5, 7.0]        195     0.071795  4.474075
(7.0, 30.0]       192     0.187500  4.094977
(30.0, 200.0]      64     0.234375  4.365971

### Pencere ici kapsam (`kapsam` = satir / gun araligi)
                     n  sifir_orani   ort_log
kapsam                                       
(-0.001, 0.5]     1681     0.054134  6.549654
(0.5, 0.8]        4951     0.064229  5.983028
(0.8, 0.95]       5464     0.016105  6.347355
(0.95, 0.999]    10074     0.050725  5.783861
(0.999, 1.001]  252759     0.055013  6.519141

Pencere bitmeden kaydi biten trafo: 268 / 2,891
  bu trafolarin son 10 gunundeki sifir orani: 0.5575
  pencere sonuna kadar devam edenlerin sifir orani: 0.0267

## Pencere 2025-12-01..2026-03-31 (en guncel 122 gun)
satir 444,076  trafo 4,640  taban sifir orani 0.0441

### Pencere sonuna kalan gun (`gun_cikisa`)
                    n  sifir_orani   ort_log
gun_cikisa                                  
(-0.501, 0.5]    4640     0.048276  6.313788
(0.5, 3.0]      12708     0.046034  6.400460
(3.0, 7.0]      16163     0.046464  6.370746
(7.0, 14.0]     27908     0.045829  6.453388
(14.0, 30.0]    62832     0.045582  6.445016
(30.0, 60.0]   113213     0.044695  6.456354
(60.0, 200.0]  206612     0.042747  6.511430

### Pencere basindan gecen gun (`gun_giristen`)
                    n  sifir_orani   ort_log
gun_giristen                                
(-0.501, 0.5]    4640     0.045043  6.235784
(0.5, 3.0]      12902     0.042862  6.460744
(3.0, 7.0]      16544     0.043399  6.471537
(7.0, 14.0]     28530     0.042306  6.512216
(14.0, 30.0]    64007     0.043011  6.542329
(30.0, 60.0]   113773     0.043657  6.519820
(60.0, 200.0]  203680     0.045110  6.428010

### Sonraki satira kalan gun (`bosluk_sonra`) — bosluk oncesi satir
                    n  sifir_orani   ort_log
bosluk_sonra                                
(-0.501, 1.5]  443201     0.044109  6.476654
(1.5, 2.5]        253     0.047431  5.047269
(2.5, 7.0]        292     0.047945  5.011620
(7.0, 30.0]       193     0.067358  5.033721
(30.0, 200.0]     137     0.051095  5.613442

### Pencere ici kapsam (`kapsam` = satir / gun araligi)
                     n  sifir_orani   ort_log
kapsam                                       
(-0.001, 0.5]     2752     0.035610  6.709292
(0.5, 0.8]        7242     0.033692  6.081216
(0.8, 0.95]      12364     0.065432  5.819024
(0.95, 0.999]    25691     0.042544  6.023043
(0.999, 1.001]  396027     0.043813  6.529232

Pencere bitmeden kaydi biten trafo: 711 / 4,640
  bu trafolarin son 10 gunundeki sifir orani: 0.0472
  pencere sonuna kadar devam edenlerin sifir orani: 0.0435

# H2 — 2026-05-11 toplu girisinin train tarafinda analogu

## Train'de ilk gorulme tarihine gore en yogun 12 gun
tarih
2025-01-01    2059
2026-03-26     329
2025-07-28     177
2025-11-25     167
2025-09-10      99
2025-11-18      96
2025-06-17      46
2025-12-10      45
2025-12-18      45
2025-09-12      35
2025-11-21      34
2025-06-23      33

2026-05-11'de test'e giren trafo: 2,222, bunlarin cold olani: 1,326
guc medyani — 05-11 cold kohortu 630, diger cold 400
IZMIR payi — 05-11 cold 0.785, diger cold 0.751

Train'de sonradan giren trafo: 3,174
### Yeni trafolarin yasa gore sifir orani ve seviyesi
                   n  sifir_orani   ort_log
yas                                        
(-0.501, 7.0]  21691     0.066018  6.275665
(7.0, 14.0]    17624     0.058670  6.431754
(14.0, 30.0]   38297     0.061336  6.406359
(30.0, 60.0]   65051     0.061798  6.406244
(60.0, 90.0]   58542     0.062161  6.352986
(90.0, 121.0]  49153     0.067809  6.224388

# H3 — `tanim` kodu guc+lokasyon otesinde bilgi tasiyor mu

sayisal olmayan tanim: 9
## Trafo seviyesinin (ortalama log1p) aciklanan varyansi
anahtar                            R2      grup
guc (kategorik)                0.2045        41
lokasyon                       0.1151        47
guc + lokasyon                 0.2923       424
tanim onek 2                   0.0285        10
tanim onek 3                   0.0855        52
tanim onek 4                   0.1325       117
tanim onek 5                   0.1612       219
guc + onek 3                   0.2814       383
guc + onek 4                   0.3233       611
guc + lokasyon + onek 3        0.3404       709
guc + lokasyon + onek 4        0.4211      1322

Uyari: cok gruplu anahtarlarda R2 ornek-ici sisli. Asagida ayni
olcum, cold benzeri bir bolmede (trafolarin %30'u dislanarak).

anahtar                        OOS R2
guc (kategorik)                0.1569
lokasyon                       0.0903
guc + lokasyon                 0.1224
tanim onek 2                   0.0272
tanim onek 3                   0.0675
tanim onek 4                   0.0867
tanim onek 5                   0.0832
guc + onek 3                   0.1342
guc + onek 4                   0.1134
guc + lokasyon + onek 3        0.1045
guc + lokasyon + onek 4        0.0808

cold trafolarin onek-3 kapsamasi (train'de ayni onekten trafo var): 1.000
cold trafolarin onek-4 kapsamasi (train'de ayni onekten trafo var): 0.984

# H4 — Trafoya ozgu mevsimsel sekil yildan yila tekrar ediyor mu

## Ayni ay ciftinin iki yildaki trafo-bazli sapmasinin korelasyonu
cift                        n   pearson  spearman
02-01 25 vs 26             1732    0.1295   -0.1745
03-01 25 vs 26             1736    0.1734    0.3424
03-02 25 vs 26             1745    0.0900    0.2535

## Trafo seviyesinin yildan yila kalıcılığı (ayni ay)
ay              n   pearson
01->01       1827    0.8249
02->02       1768    0.8408
03->03       1857    0.8199

## Nis-Tem 2025 capasinin gucu: Oca-Mar seviyesinden ne kadar sapiyor
trafo sayisi (her uc pencerede de veri var): 1,945
Nis-Tem 2025 eksi Oca-Mar 2025 farki: ortalama 0.1201, std 0.7112, medyan 0.0145
farkin %10-%90 araligi: -0.344 .. 0.632

Bu farkin trafoya ozgu kismi gercekse, gonderimde Oca-Mar 2026
seviyesine eklenerek Nis-Tem 2026 seviyesi kestirilebilir. Ustteki
kis-ici korelasyonlar bu terimin ne kadar guvenilir oldugunu soyler.
