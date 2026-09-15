# Cold seviye kalibrasyonu
Uretim: 2026-08-31 11:45:01.110927

## A 2025-03-31  pencere 2025-04-01..2025-07-31
dogal cold trafo 678, satir 20,633
gercek ortalama log1p 6.6429, sifir orani 0.0400
referans havuzu: 2,270 trafo, z medyan 0.7501, z ortalama 0.1247

cold capasi                    kapi yok  kapi k=0.75  + en iyi sabit   sabit
medyan_z (MEVCUT)                1.7975       1.8606          1.8080   0.439
ortalama_z                       1.8828       2.0750          1.8081   1.018
band ortalama z                  1.8742       2.0574          1.7918   1.011
lokasyon ortalama z              1.8896       2.0797          1.7989   1.044
band x lokasyon ortalama z       1.8839       2.0707          1.8057   1.014

## B 2025-11-30  pencere 2025-12-01..2026-03-31
dogal cold trafo 1,223, satir 61,918
gercek ortalama log1p 6.5584, sifir orani 0.0523
referans havuzu: 4,121 trafo, z medyan 0.8247, z ortalama 0.2376

cold capasi                    kapi yok  kapi k=0.75  + en iyi sabit   sabit
medyan_z (MEVCUT)                1.8215       1.8182          1.8164   0.080
ortalama_z                       1.8704       1.9282          1.8160   0.648
band ortalama z                  1.8631       1.9160          1.8148   0.614
lokasyon ortalama z              1.8707       1.9390          1.8248   0.656
band x lokasyon ortalama z       1.8598       1.9225          1.8123   0.642

## C 2025-09-30  pencere 2025-10-01..2026-01-31
dogal cold trafo 1,455, satir 87,952
gercek ortalama log1p 6.2577, sifir orani 0.0765
referans havuzu: 3,271 trafo, z medyan 0.8728, z ortalama 0.2701

cold capasi                    kapi yok  kapi k=0.75  + en iyi sabit   sabit
medyan_z (MEVCUT)                2.0686       2.0382          2.0297  -0.186
ortalama_z                       2.0489       2.0691          2.0301   0.400
band ortalama z                  2.0526       2.0671          2.0361   0.356
lokasyon ortalama z              2.0526       2.0711          2.0340   0.390
band x lokasyon ortalama z       2.0417       2.0579          2.0250   0.366

## D 2025-05-31  pencere 2025-06-01..2025-09-30
dogal cold trafo 818, satir 34,313
gercek ortalama log1p 6.8013, sifir orani 0.0376
referans havuzu: 2,453 trafo, z medyan 0.6239, z ortalama -0.0466

cold capasi                    kapi yok  kapi k=0.75  + en iyi sabit   sabit
medyan_z (MEVCUT)                1.7566       1.8418          1.7626   0.534
ortalama_z                       1.9171       2.1107          1.7622   1.162
band ortalama z                  1.9040       2.0851          1.7627   1.114
lokasyon ortalama z              1.9221       2.1126          1.7588   1.170
band x lokasyon ortalama z       1.9157       2.1057          1.7602   1.156

# Ozet: dort pencere ortalamasi (kapi k=0.75 uygulanmis)

cold capasi                      RMSLE  en iyi sabitle  ort sabit
medyan_z (MEVCUT)               1.8897          1.8542      0.217
ortalama_z                      2.0458          1.8541      0.807
band ortalama z                 2.0314          1.8514      0.774
lokasyon ortalama z             2.0506          1.8541      0.815
band x lokasyon ortalama z      2.0392          1.8508      0.794

mevcut cold capasi referansi: 1.8897

## Kazanc aritmetigi (cold satirlar test'in %22,16'si)
MSE_toplam degisimi = 0.2216 * (cold_RMSLE^2 farki)
aday                              dMSE  LB tahmini
medyan_z (MEVCUT)              -0.0295      1.0532
ortalama_z                     -0.0295      1.0532
band ortalama z                -0.0318      1.0521
lokasyon ortalama z            -0.0296      1.0532
band x lokasyon ortalama z     -0.0323      1.0519

Uyari: buradaki cold capalari SAF capa; gercek boru hattinda uzerine
sekil modeli biniyor. Yine de capalar arasi FARK sekil modelinden
bagimsiz olarak tasinir, cunku sekil ayni artik uzerinde egitiliyor.

# Test tarafi: gonderim cold satirlarda ne diyor

cold trafo 2,024, satir 158,369 (22.16%)
cold guc medyani 630; train havuzu guc medyani 400
referans z: medyan 0.8656, ortalama 0.3118, fark 0.5538

gonderim cold ortalama log1p tahmini: 6.9518
  saf medyan_z capasi (kapili) ortalama: 6.7971
  saf ortalama_z capasi (kapili) ortalama: 6.2645

Gonderim tahmini capadan yuksekse sekil modeli cold satirlari
yukari itiyor demektir; asagidaki fold sonuclari bunun dogru olup
olmadigini soyler.
