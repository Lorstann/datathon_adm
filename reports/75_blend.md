# Nihai harman
Uretim: 2026-08-31 12:32:53.259417

Referans FINAL_4  LB 1.01548  MSE 1.03120

## Yonler
yon                     LB    ||D||     <e0,D>  tek basina
optuna             1.06666   0.3404   -0.00467     1.01539
T_1.06713          1.06713   0.3233    0.00153     1.01547
seed7              1.06727   0.3277    0.00022     1.01548
multiorigin        1.06766   0.3392   -0.00318     1.01544
T_1.06929          1.06929   0.3303    0.00153     1.01547
T_1.07042          1.07042   0.3399   -0.00046     1.01548
v4_wx              1.10990   0.4448    0.00140     1.01548
catboost_seg       1.11019   0.4434    0.00234     1.01547
v6                 1.11447   0.4777   -0.00868     1.01532
c_tuned            1.14554   0.5275    0.00139     1.01548
level_shape        1.14884   0.5333    0.00212     1.01547
P_EMP              1.26800   0.7583    0.00079     1.01548
P_YOY              1.28964   0.7931    0.00149     1.01548
T_1.29422          1.29422   0.8004    0.00160     1.01548
T_1.33686          1.33686   0.8695   -0.00002     1.01548
T_1.37997          1.37997   0.9307    0.00347     1.01547
Q_COMBO            1.33324   0.8643   -0.00031     1.01548

`tek basina` = yalnizca o yon kullanilirsa ulasilacak skor.

## Q_COMBO: havuza dik bilesenin degeri
<e0,D> toplam           -0.00031
havuzun acikladigi kisim -0.00124
DIK bilesen <e0,D_dik>  +0.00093
||D_dik||^2             0.31393
dik bilesenin tek basina kazanci  0.000003 MSE
  -> LB 1.01548

## Ridge taramasi
 lambda/tr   nominal  gurultulu   max|w|    |w|_1
    0.0003   1.01316    1.01331    0.320    1.894
    0.0010   1.01316    1.01329    0.308    1.822
    0.0030   1.01318    1.01329    0.279    1.658
    0.0100   1.01330    1.01336    0.211    1.314
    0.0300   1.01361    1.01363    0.145    0.887
    0.1000   1.01419    1.01420    0.102    0.461
    0.3000   1.01475    1.01475    0.055    0.213
    1.0000   1.01518    1.01518    0.022    0.079

Secilen lambda/tr = 0.003, beklenen LB 1.01329 (FINAL_4 1.01548, kazanc +0.00219)

## Marjinal katki: bileseni cikarinca beklenen skor
cikarilan         beklenen LB   bozulma
optuna                1.01385  +0.00056
T_1.06713             1.01350  +0.00021
seed7                 1.01327  -0.00002
multiorigin           1.01365  +0.00036
T_1.06929             1.01363  +0.00034
T_1.07042             1.01335  +0.00006
v4_wx                 1.01345  +0.00016
catboost_seg          1.01331  +0.00002
v6                    1.01406  +0.00077
c_tuned               1.01332  +0.00003
level_shape           1.01332  +0.00003
P_EMP                 1.01330  +0.00001
P_YOY                 1.01328  -0.00001
T_1.29422             1.01329  +0.00000
T_1.33686             1.01329  +0.00000
T_1.37997             1.01332  +0.00003
Q_COMBO               1.01329  -0.00000

## Cikti
kirpilan satir 0 (0.0000%)
ortalama log1p 6.5975 (referans 6.5955)
referanstan ortalama mutlak sapma 0.0409

yazildi: submissions/submission_BLEND.csv
