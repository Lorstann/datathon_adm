# `tanim` sayisal komsulugu cold seviyeyi tasiyor mu
Uretim: 2026-08-31 12:37:59.795139

train trafosu (>=30 gun): 4,521, sayisal kimlikli: 4,513
kimlik araligi 61,740,209 .. 700,850,780

## Ardisik kimlikler arasi fark
medyan 4, %25 1, %75 24, fark==1 orani 0.292

## Kimlik farki kucuk olan ciftler ayni lokasyonda mi
|fark|            cift  ayni ilce  ayni band  |z farki| ort
1-1              1,319      0.880      0.418         1.4030
2-3              2,158      0.836      0.353         1.5441
4-10             5,825      0.816      0.315         1.6057
11-50           24,030      0.802      0.279         1.6470
51-500          73,758      0.808      0.247         1.6648
rastgele       200,000      0.090      0.173         1.7882

Kimlik farki kucuk ciftlerde |z farki| rastgele ciftlerden belirgin
kucukse, kimlik komsulugu gercek bir sinyaldir.

## Cold taklidi: trafolarin %30'u disarida, OOS R2

referans havuzu 3,181, degerlendirme 1,332
yontem                               kapsam    OOS R2
global                                1.000   -0.0008
band                                  1.000    0.0079
ilce                                  1.000    0.0052
band+ilce                             1.000    0.0144
kimlik-kNN K=2 gap=None               1.000   -0.3883
  + band+ilce (yari yariya)           1.000   -0.0630
kimlik-kNN K=2 gap=50                 0.891   -0.3867
  + band+ilce (yari yariya)           0.891   -0.0578
kimlik-kNN K=2 gap=500                0.977   -0.3870
  + band+ilce (yari yariya)           0.977   -0.0582
kimlik-kNN K=4 gap=None               1.000   -0.2061
  + band+ilce (yari yariya)           1.000   -0.0219
kimlik-kNN K=4 gap=50                 0.891   -0.2595
  + band+ilce (yari yariya)           0.891   -0.0303
kimlik-kNN K=4 gap=500                0.977   -0.2139
  + band+ilce (yari yariya)           0.977   -0.0190
kimlik-kNN K=8 gap=None               1.000   -0.1123
  + band+ilce (yari yariya)           1.000   -0.0051
kimlik-kNN K=8 gap=50                 0.891   -0.2088
  + band+ilce (yari yariya)           0.891   -0.0232
kimlik-kNN K=8 gap=500                0.977   -0.1299
  + band+ilce (yari yariya)           0.977   -0.0039
kimlik-kNN K=16 gap=None              1.000   -0.0293
  + band+ilce (yari yariya)           1.000    0.0186
kimlik-kNN K=16 gap=50                0.891   -0.1900
  + band+ilce (yari yariya)           0.891   -0.0200
kimlik-kNN K=16 gap=500               0.977   -0.0654
  + band+ilce (yari yariya)           0.977    0.0143

En iyi kimlik tabanli: kimlik-kNN K=16 gap=None -> OOS R2 0.0186
band+ilce referansi:                  OOS R2 0.0144

## Test'teki cold trafolarin kimlik komsulugu
cold trafo 2,024, sayisal kimlikli 2,021
en yakin train kimligine uzaklik: medyan 4, %25 1, %75 12
  uzaklik <= 1   : 0.260
  uzaklik <= 2   : 0.402
  uzaklik <= 5   : 0.612
  uzaklik <= 10  : 0.730
  uzaklik <= 50  : 0.880
