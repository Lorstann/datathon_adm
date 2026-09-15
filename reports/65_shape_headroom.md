# Sekil terimlerinin bos alani
Uretim: 2026-08-31 11:47:00.953903

Olcum: warm, sifir-olmayan satirlar. Tahmin = origin son-7-gun seviyesi
+ ofset. Her varyant kendi optimal sabitiyle degerlendiriliyor, yani
mevsimsel kayma disariida; olculen yalnizca GUN-ICI sekil.

## A 2025-03-31  warm sifir-olmayan satir 236,572, trafo 2,010
varyant                   RMSE(log)    delta      dMSE
yalniz seviye                0.7628   0.0000    0.0000
+ global DOW                 0.7623  -0.0005   -0.0008
+ trafo DOW                  0.7570  -0.0058   -0.0089
+ trafo hava                 0.9145   0.1517    0.2544
+ trafo DOW + hava           0.9093   0.1465    0.2450

## B 2025-11-30  warm sifir-olmayan satir 361,281, trafo 3,154
varyant                   RMSE(log)    delta      dMSE
yalniz seviye                0.5173   0.0000    0.0000
+ global DOW                 0.5172  -0.0001   -0.0001
+ trafo DOW                  0.5102  -0.0071   -0.0073
+ trafo hava                 0.6286   0.1113    0.1275
+ trafo DOW + hava           0.6227   0.1055    0.1202

## C 2025-09-30  warm sifir-olmayan satir 281,041, trafo 2,412
varyant                   RMSE(log)    delta      dMSE
yalniz seviye                0.7823   0.0000    0.0000
+ global DOW                 0.7823   0.0000    0.0000
+ trafo DOW                  0.7769  -0.0054   -0.0084
+ trafo hava                 0.7871   0.0049    0.0077
+ trafo DOW + hava           0.7818  -0.0005   -0.0007

## D 2025-05-31  warm sifir-olmayan satir 244,164, trafo 2,114
varyant                   RMSE(log)    delta      dMSE
yalniz seviye                0.8761   0.0000    0.0000
+ global DOW                 0.8753  -0.0007   -0.0012
+ trafo DOW                  0.8700  -0.0061   -0.0106
+ trafo hava                 0.9356   0.0595    0.1079
+ trafo DOW + hava           0.9303   0.0542    0.0979

# Dort pencere ortalamasi

varyant                        MSE      dMSE  warm payi  LB tahmini
yalniz seviye               0.5572    0.0000     0.0000      1.0671
+ global DOW                0.5567   -0.0005    -0.0004      1.0669
+ trafo DOW                 0.5484   -0.0088    -0.0066      1.0640
+ trafo hava                0.6816    0.1244     0.0933      1.1100
+ trafo DOW + hava          0.6728    0.1156     0.0867      1.1070

Not: bu ust sinir degil alt sinir tarafinda -- ofsetler ham, model
bunlari ozellik olarak alirsa daha iyisini yapabilir. Ama isaret ve
buyukluk mertebesi buradan okunur.