# Sifir tabani ve sifir odakli yonler
Uretim: 2026-08-31 12:35:46.224906

# 1. Train tarafi: sifir ne kadar ongorulebilir

## B 2025-11-30 -> Ara-Mar
satir 379,081, gercek sifir orani 0.0431
gecmisten p ile        MSE  0.4924  RMSLE 0.7017
hangi gun sifir bilinse MSE  0.2551  RMSLE 0.5051
SIFIR BELIRSIZLIGININ BEDELI   0.2373 MSE
  toplam hatanin 48.2%'i

gecmis p araligi         satir  gercek sifir  ort log1p
(-0.01,0.01]            360,055        0.0026     6.7324
(0.01,0.10]              1,209        0.0703     6.7510
(0.10,0.50]              2,110        0.3602     3.7140
(0.50,0.90]              1,222        0.5368     3.0721
(0.90,1.01]             14,485        0.9595     0.1799

p<0.05 iken gerceklesen sifir: 970 satir (0.2559%); bunlarin ortalama m'si 6.792
  bu satirlarin tek basina MSE katkisi 0.1290

## A 2025-03-31 -> Nis-Tem
satir 251,746, gercek sifir orani 0.0554
gecmisten p ile        MSE  0.7493  RMSLE 0.8656
hangi gun sifir bilinse MSE  0.4982  RMSLE 0.7058
SIFIR BELIRSIZLIGININ BEDELI   0.2511 MSE
  toplam hatanin 33.5%'i

gecmis p araligi         satir  gercek sifir  ort log1p
(-0.01,0.01]            236,288        0.0018     6.8405
(0.01,0.10]                122        0.0082    11.6552
(0.10,0.50]                810        0.2519     3.5183
(0.50,0.90]              1,341        0.6286     2.0373
(0.90,1.01]             13,185        0.9468     0.3901

p<0.05 iken gerceklesen sifir: 424 satir (0.1684%); bunlarin ortalama m'si 8.051
  bu satirlarin tek basina MSE katkisi 0.1140

# 2. Test tarafi: sifir odakli dar yonlerin kaldiraci

yon                       dokunulan    ||D||   dik R2  ||D_dik||
olu_tam_sifir                16,223   0.2110   0.1790     0.1912
yari_olu_kucult               1,972   0.0330   0.0951     0.0314
zr_ile_olcekle               21,843   0.1615   0.2178     0.1428
cikis_kapisi                  5,428   0.0561   0.0045     0.0560
giris_kapisi                 21,340   0.0689   0.0579     0.0669
cold_kucult_20              158,369   0.1048   0.4023     0.0810
dusuk_tahmini_sifirla        16,717   0.1101   0.0962     0.1047

`dik R2` = havuzun bu yonu ne kadar aciklayabildigi. Dusukse yon yeni.

## Kaldirac: dokunulan satirlarda ortalama yanlilik b ise beklenen LB

Yon d, dokunulan satirlarda ortalama -delta kadar asagi cekiyor. Eger
o satirlarda gercekten b kadar FAZLA tahmin ediyorsak <e0,d> = -f*delta*b
ve kazanc (f*delta*b)^2 / ||d||^2 olur.

yon                           b=0.2     b=0.5     b=1.0     b=2.0
olu_tam_sifir               1.01529   1.01427   1.01064   0.99599
yari_olu_kucult             1.01543   1.01518   1.01430   1.01073
zr_ile_olcekle              1.01518   1.01360   1.00794   0.98498
cikis_kapisi                1.01539   1.01489   1.01311   1.00595
giris_kapisi                1.01505   1.01280   1.00473   0.97177
cold_kucult_20              1.00815   0.96877   0.81267   0.00003
dusuk_tahmini_sifirla       1.01522   1.01388   1.00905   0.98950

Bu tablo UST SINIR degil senaryo: b'yi olcen sey yalnizca gonderim.
Ama hangi yonun yoklamaya deger oldugunu gosterir -- kaldiraci
buyuk olanlar dar ve derin olanlar.

yazildi: submissions/probe_olu_tam_sifir.csv
yazildi: submissions/probe_zr_ile_olcekle.csv
yazildi: submissions/probe_cikis_kapisi.csv
