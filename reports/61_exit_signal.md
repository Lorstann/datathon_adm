# Cikis/giris deseni: buyukluk olcumu
Uretim: 2026-08-31 11:37:23.661149

# Train tarafi

## Pencere 2025-04-01..2025-07-31
pencere trafosu 2,891; pencere bitmeden kaydi biten ve
pencereden sonra hic donmeyen: 167
### Temiz cikanlarda cikisa kalan gune gore
                  n  sifir_orani   ort_log
gun_cikisa                                
(-0.5, 0.5]     167     0.670659  2.146956
(0.5, 2.0]      294     0.704082  2.266341
(2.0, 6.0]      552     0.708333  2.222458
(6.0, 13.0]     936     0.708333  2.177199
(13.0, 29.0]   2013     0.738202  1.921785
(29.0, 59.0]   3597     0.766194  1.696347
(59.0, 400.0]  2452     0.657830  2.652978
karsilastirma — pencere sonuna kadar devam edenler: n=264,918, sifir orani 0.0290, ort_log 6.6458

## Pencere 2025-01-01..2025-10-31
pencere trafosu 3,526; pencere bitmeden kaydi biten ve
pencereden sonra hic donmeyen: 671
### Temiz cikanlarda cikisa kalan gune gore
                   n  sifir_orani   ort_log
gun_cikisa                                 
(-0.5, 0.5]      671     0.213115  4.655647
(0.5, 2.0]      1227     0.198859  5.754991
(2.0, 6.0]      2335     0.196146  5.776938
(6.0, 13.0]     4010     0.193766  5.852074
(13.0, 29.0]    7971     0.212144  5.768835
(29.0, 59.0]   11312     0.262288  5.571166
(59.0, 400.0]  37581     0.258136  5.374677
karsilastirma — pencere sonuna kadar devam edenler: n=629,900, sifir orani 0.0300, ort_log 6.5762

## Pencere 2025-04-01..2025-07-31 — giris tarafi
pencere icinde ilk kez gorulen (once hic yok): 678
                  n  sifir_orani   ort_log
gun_giristen                              
(-0.5, 0.5]     678     0.146018  5.498326
(0.5, 2.0]     1116     0.073477  6.438270
(2.0, 6.0]     1604     0.055486  6.505678
(6.0, 13.0]    2398     0.044621  6.615485
(13.0, 29.0]   5337     0.037287  6.757981
(29.0, 59.0]   6269     0.032382  6.673797
(59.0, 400.0]  3231     0.014547  6.792388

# Test tarafinda kac satiri kapsiyor

test trafosu 7,036; 2026-07-31'den once kaydi biten: 241 (3.4%)

## Erken cikan trafolarin satirlari, cikisa kalan gune gore
                  n  cold_pay  tum_test_payi
gun_cikisa                                  
(-0.5, 0.5]     241  0.257261       0.000337
(0.5, 2.0]      420  0.261905       0.000588
(2.0, 6.0]      788  0.267766       0.001103
(6.0, 13.0]    1271  0.216365       0.001778
(13.0, 29.0]   2789  0.204733       0.003902
(29.0, 59.0]   4400  0.200682       0.006157
(59.0, 400.0]  3542  0.076510       0.004956

erken cikan trafolarin toplam satiri: 13,451 (1.88% tum test)
bunlarin son 14 gunu: 2,720 satir (0.38% tum test)
bunlarin son 30 gunu: 5,509 satir (0.77% tum test)

## Erken cikis tarihlerinin dagilimi (en yogun 15)
tarih
2026-07-07    15
2026-07-30    14
2026-06-18    10
2026-07-28     9
2026-07-24     8
2026-07-08     7
2026-07-01     6
2026-07-22     6
2026-06-25     6
2026-07-21     6
2026-07-23     5
2026-07-25     5
2026-06-30     5
2026-05-14     5
2026-07-06     4

## Test penceresinde gec giren trafolar
2026-04-01'den sonra baslayan: 3,108 (44.2%)
tarih
2026-05-11    2222
2026-05-03     141
2026-04-30     119
2026-05-07     102
2026-05-13      54
2026-05-30      30
2026-07-01      20
2026-05-05      20
2026-06-29      16
2026-06-08      15

# Mevcut en iyi gonderim bu satirlarda ne diyor

## Erken cikanlarda ortalama log1p tahmini
                  n  ort_log_tahmin
gun_cikisa                         
(-0.5, 0.5]     241        6.156240
(0.5, 2.0]      420        6.211295
(2.0, 6.0]      788        6.252168
(6.0, 13.0]    1271        6.198328
(13.0, 29.0]   2789        6.157987
(29.0, 59.0]   4400        6.086517
(59.0, 400.0]  3542        5.846802
pencere sonuna kadar devam edenlerde: 6.7229

## Kazanc aritmetigi
Bir satirda gercek sifir olasiligi p, sifir olmayan hali m ise
RMSLE-optimal tahmin (1-p)*m ve mevcut m'ye gore MSE kazanci p^2*m^2.
Toplam test satiri = 714.688; kazanc paylari buna gore.

dilim                     n       m      p  dMSE_toplam
cikisa 0-0               241   6.156   0.56      0.00401
cikisa 1-2               420   6.211   0.56      0.00711
cikisa 3-6               788   6.252   0.56      0.01352
cikisa 7-13            1,271   6.198   0.56      0.02143
cikisa 14-29            2,789   6.158   0.40      0.02368
cikisa 30-59            4,400   6.087   0.25      0.01425

Referans: LB 1.06713 -> MSE 1.13877. Hedef ilk 10 = 0.99971 -> MSE 0.99942.
Gereken toplam MSE dususu: 0.139.
