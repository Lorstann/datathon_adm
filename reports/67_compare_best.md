(yok: C:\Users\musta\Downloads\FINAL_1.csv)
# FINAL_4 vs depo: gonderimlerin anatomisi
Uretim: 2026-08-31 11:57:50.249966

## Genel
gonderim                  ort log1p   medyan     std  p<0.5 orani    tam 0
FINAL_4 (1.01548)            6.5955   6.9550  1.7004       0.0195     4724
SUB_A_blend (1.04876)        6.6747   7.0333  1.7408       0.0191     1841
SUB_D_cold (1.08311)         6.7855   7.1532  1.7646       0.0191     1841
P_YOY (1.28964)              6.6600   7.0467  1.9244       0.0311    12569
P_EMP (1.26800)              6.6561   7.0472  1.8997       0.0290    11107
repo BEST (1.06713)          6.7105   7.0777  1.7498       0.0193     4116
repo optuna (1.06666)        6.7158   7.0784  1.7531       0.0179     6301

## Dilim bazinda ortalama log1p tahmini
gonderim                          cold   warm_canli     warm_olu
FINAL_4 (1.01548)               6.7748       6.7755       0.6484
SUB_A_blend (1.04876)           6.8563       6.8560       0.6769
SUB_D_cold (1.08311)            7.3563       6.8560       0.6769
P_YOY (1.28964)                 6.8761       6.8443       0.3214
P_EMP (1.26800)                 6.8761       6.8372       0.3700
repo BEST (1.06713)             6.9518       6.8749       0.6932
repo optuna (1.06666)           6.9454       6.8841       0.6849

## Ay bazinda ortalama log1p tahmini (mevsimsel rampa)
gonderim                         4        5        6        7  Nis->Tem
FINAL_4 (1.01548)           6.3320   6.4065   6.6895   6.8154    0.4834
SUB_A_blend (1.04876)       6.4152   6.4609   6.7285   6.9525    0.5373
SUB_D_cold (1.08311)        6.4191   6.5720   6.8672   7.0958    0.6767
P_YOY (1.28964)             6.3975   6.3617   6.6993   7.0264    0.6289
P_EMP (1.26800)             6.3981   6.3591   6.6933   7.0210    0.6229
repo BEST (1.06713)         6.4441   6.5486   6.7856   6.9267    0.4826
repo optuna (1.06666)       6.4449   6.5478   6.7868   6.9435    0.4986

## Warm canli satirlarda: tahmin eksi Mart son-28g seviyesi
FINAL_4 (1.01548)        ortalama +0.0978  medyan +0.0138
SUB_A_blend (1.04876)    ortalama +0.1715  medyan +0.0733
SUB_D_cold (1.08311)     ortalama +0.1715  medyan +0.0733
P_YOY (1.28964)          ortalama +0.2295  medyan +0.0921
P_EMP (1.26800)          ortalama +0.2218  medyan +0.1132
repo BEST (1.06713)      ortalama +0.1933  medyan +0.0950
repo optuna (1.06666)    ortalama +0.1843  medyan +0.0902

## Cold satirlarda: tahmin eksi log(guc)  (yani ima edilen z)
FINAL_4 (1.01548)        ortalama z +0.5746  medyan +0.5888
SUB_A_blend (1.04876)    ortalama z +0.6561  medyan +0.6722
SUB_D_cold (1.08311)     ortalama z +1.1561  medyan +1.1722
P_YOY (1.28964)          ortalama z +0.6759  medyan +0.6495
P_EMP (1.26800)          ortalama z +0.6759  medyan +0.6495
repo BEST (1.06713)      ortalama z +0.7516  medyan +0.7037
repo optuna (1.06666)    ortalama z +0.7452  medyan +0.7028

## FINAL_4 - repo BEST farkinin dagilimi
korelasyon 0.9842, ortalama fark -0.1150, std 0.3106

dilim                n   ort fark  |fark| medyan
cold            158369    -0.1770         0.2707
warm_canli      535347    -0.0994         0.1524
warm_olu         20972    -0.0448         0.0874

band                 n   ort fark
1000-1600       197752    -0.1409
400-630         153245    -0.1007
250-400         117110    -0.0891
<100             21397    -0.1017
100-160          48797    -0.1096
630-1000         83271    -0.1275
2600+             3910    -0.1974
1600-2600         8451    -0.2066
160-250          80755    -0.0963

## Harman aritmetigi (log uzayinda w*FINAL_4 + (1-w)*repo)
Gercek bilinmiyor; yalnizca ne kadar ayrildiklarini gosterir.
w=0.50  ortalama 6.6530  FINAL_4'ten ortalama sapma 0.1154
w=0.70  ortalama 6.6300  FINAL_4'ten ortalama sapma 0.0692
w=0.85  ortalama 6.6128  FINAL_4'ten ortalama sapma 0.0346
