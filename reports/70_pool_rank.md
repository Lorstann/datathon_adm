# Havuzun gercek boyutu
Uretim: 2026-08-31 12:21:35.373074

20 gonderim, 714,688 satir.

## FINAL_4 havuzun geri kalanindan yeniden kurulabiliyor mu

dogrusal yeniden kurulum R2 = 0.99629355
artik RMS = 0.103521

R2 ~ 1 ise FINAL_4 havuzun tam kombinasyonudur; o zaman onu referans
alip bilesenlerini yon olarak kullanmak tekil sistem yaratir ve
`68_lb_algebra`'daki 1.01308 gurultuden ibarettir.

bilesen             katsayi
(sabit)             -0.0782
SUB_A_blend          0.1378
optuna               0.0439
T_1.06713            0.1046
seed7               -0.0828
multiorigin          0.0337
T_1.06929           -0.0474
T_1.07042            0.2060
SUB_B_plus020        0.1222
SUB_D_cold           0.1124
v4_wx                0.2077
catboost_seg         0.3985
v6                   0.0513
c_tuned              0.1890
level_shape         -0.1451
P_EMP               -0.4408
P_YOY                0.1818
T_1.29422           -0.0445
T_1.33686            0.1164
T_1.37997           -0.1421

## Turetilmis harmanlar atildiktan sonra
kalan bilesen: optuna, T_1.06713, seed7, multiorigin, T_1.06929, T_1.07042, v4_wx, catboost_seg, v6, c_tuned, level_shape, P_EMP, P_YOY, T_1.29422, T_1.33686, T_1.37997

Gram ozdegerleri: [2.12847e-03 4.88110e-03 5.75474e-03 8.21187e-03 1.01442e-02 1.62392e-02
 2.42373e-02 4.23334e-02 8.57070e-02 1.37054e-01 2.16858e-01 2.60908e-01
 3.57663e-01 4.41376e-01 6.11369e-01 3.09491e+00]
kosul sayisi = 1.454e+03

## Ridge taramasi ve gurultu altinda beklenen skor

Olcum gurultusu: yon basina sigma_r = 0.0005. Her lambda icin r'ye
gurultu ekleyip agirliklari yeniden cozuyor, sonra GERCEK r ile
degerlendiriyoruz. `gurultulu` sutunu fiilen beklenecek skordur.

 lambda/tr   nominal  gurultulu   max|w|    |w|_1
    0.0001   1.01316    1.01331    0.324    1.914
    0.0003   1.01316    1.01330    0.320    1.893
    0.0010   1.01317    1.01329    0.309    1.825
    0.0030   1.01318    1.01329    0.282    1.670
    0.0100   1.01329    1.01335    0.217    1.336
    0.0300   1.01358    1.01361    0.147    0.915
    0.1000   1.01415    1.01416    0.105    0.482
    0.3000   1.01472    1.01472    0.058    0.223
    1.0000   1.01516    1.01516    0.023    0.083

En iyi (gurultu altinda): lambda/tr = 0.003, beklenen LB 1.01329
FINAL_4 referansi: 1.01548
kazanc: +0.00219

yon                      w
optuna             -0.2028
T_1.06713           0.2257
seed7               0.0392
multiorigin        -0.2232
T_1.06929           0.2820
T_1.07042          -0.1000
v4_wx               0.0674
catboost_seg        0.0372
v6                 -0.1745
c_tuned            -0.1209
level_shape         0.1210
P_EMP               0.0409
T_1.29422          -0.0068
T_1.33686          -0.0068
T_1.37997           0.0166

## Sonuc

Havuz tuketilmis durumda: FINAL_4 bu bilesenlerin optimumu. Yeni
kazanc ancak havuza GERCEKTEN FARKLI bir tahminci eklemekle gelir.
Notun kendi ifadesiyle: kazanc bilesenlerin ne kadar farkli olduguyla
orantili. Bir sonraki gonderim hakki bir yoklama (probe) olmali.
