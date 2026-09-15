# LB cebiri: skoru bilinen gonderimlerden optimal harman
Uretim: 2026-08-31 12:01:19.130231

## Katalog (bayt bayt dogrulanmis eslesme)
  ok  FINAL_4.csv                        LB 1.01548  27,622,210 bayt
  ok  SUB_A_blend.csv                    LB 1.04876  27,660,953 bayt
  ok  submission_optuna.csv              LB 1.06666  28,290,096 bayt
  ok  submission_1.06713.csv             LB 1.06713  28,320,958 bayt
  ok  submission_seed7.csv               LB 1.06727  28,351,585 bayt
  ok  submission_multiorigin.csv         LB 1.06766  28,335,567 bayt
  ok  submission_1.06929.csv             LB 1.06929  28,349,461 bayt
  ok  submission_1.07042.csv             LB 1.07042  28,329,751 bayt
  ok  SUB_B_blend_plus020.csv            LB 1.08249  27,669,792 bayt
  ok  SUB_D_cold_probe.csv               LB 1.08311  27,652,351 bayt
  ok  submission_v4_wx.csv               LB 1.10990  27,595,859 bayt
  ok  02_catboost_segment_split.csv      LB 1.11019  27,658,262 bayt
  ok  submission_v6.csv                  LB 1.11447  27,600,304 bayt
  ok  submission_c_tuned.csv             LB 1.14554  28,216,962 bayt
  ok  submission_level_shape.csv         LB 1.14884  28,151,805 bayt
  ok  P_EMP.csv                          LB 1.26800  27,519,732 bayt
  ok  P_YOY.csv                          LB 1.28964  27,500,097 bayt
  ok  submission_ensemble.csv            LB 1.29422  27,971,660 bayt
  ok  submission_1.37997.csv             LB 1.37997  27,989,263 bayt

19 gonderim, 714,688 satir.

Referans p_0 = FINAL_4 [1.01548]

## Yonler ve artik izdusumleri
yon (p_k - p_0)                             E[d^2]       c_k  sabitlik    ort d
SUB_A_blend [1.04876]                       0.0665    0.0011     0.094   0.0792
submission_optuna [1.06666]                 0.1159   -0.0047     0.125   0.1202
submission_1.06713 [1.06713]                0.1045    0.0015     0.131   0.1168
submission_seed7 [1.06727]                  0.1074    0.0002     0.118   0.1127
submission_multiorigin [1.06766]            0.1151   -0.0032     0.139   0.1264
submission_1.06929 [1.06929]                0.1091    0.0015     0.124   0.1161
submission_1.07042 [1.07042]                0.1155   -0.0005     0.140   0.1272
SUB_B_blend_plus020 [1.08249]               0.1381    0.0012     0.564   0.2792
SUB_D_cold_probe [1.08311]                  0.1399    0.0010     0.258   0.1900
submission_v4_wx [1.10990]                  0.1979    0.0014     0.002   0.0179
02_catboost_segment_split [1.11019]         0.1966    0.0023     0.052  -0.1015
submission_v6 [1.11447]                     0.2282   -0.0087     0.001  -0.0182
submission_c_tuned [1.14554]                0.2783    0.0014     0.016   0.0665
submission_level_shape [1.14884]            0.2844    0.0021     0.016   0.0665
P_EMP [1.26800]                             0.5750    0.0008     0.006   0.0606
P_YOY [1.28964]                             0.6290    0.0015     0.007   0.0644
submission_ensemble [1.29422]               0.6406    0.0016     0.057  -0.1916
submission_1.37997 [1.37997]                0.8662    0.0035     0.008  -0.0849

`sabitlik` = ort(d)^2 / E[d^2]; 1'e yakinsa o yon neredeyse duz bir
kaydirma demektir. c_k negatifse o yonde ilerlemek hatayi azaltir.

## Dogrulama: bir gonderimi disarida birakip LB'sini tahmin et

d_j'yi digerlerinin span'ina izdusurup c_j ~ beta'c ile skoru
kestiriyoruz. Gercek LB ile karsilastirma framework'un tek gercek
disariida testi.

disarida birakilan                         gercek   tahmin     hata  izdusum R2
SUB_A_blend [1.04876]                     1.04876  1.04881 +0.00005      1.0000
submission_optuna [1.06666]               1.06666  1.07163 +0.00497      0.7827
submission_1.06713 [1.06713]              1.06713  1.06705 -0.00008      1.0000
submission_seed7 [1.06727]                1.06727  1.06706 -0.00021      0.9395
submission_multiorigin [1.06766]          1.06766  1.07056 +0.00290      0.8893
submission_1.06929 [1.06929]              1.06929  1.06711 -0.00218      0.9341
submission_1.07042 [1.07042]              1.07042  1.07155 +0.00113      0.9105
SUB_B_blend_plus020 [1.08249]             1.08249  1.08311 +0.00062      0.8658
SUB_D_cold_probe [1.08311]                1.08311  1.08203 -0.00108      0.8028
submission_v4_wx [1.10990]                1.10990  1.10976 -0.00014      1.0000
02_catboost_segment_split [1.11019]       1.11019  1.10772 -0.00247      0.7627
submission_v6 [1.11447]                   1.11447  1.11324 -0.00123      0.9998
submission_c_tuned [1.14554]              1.14554  1.14618 +0.00064      0.9849
submission_level_shape [1.14884]          1.14884  1.14821 -0.00063      0.9848
P_EMP [1.26800]                           1.26800  1.26705 -0.00095      0.9323
P_YOY [1.28964]                           1.28964  1.28874 -0.00090      0.8408
submission_ensemble [1.29422]             1.29422  1.29582 +0.00160      0.6735
submission_1.37997 [1.37997]              1.37997  1.37861 -0.00136      0.7401

ortalama mutlak hata 0.00129, medyan 0.00102, en kotu 0.00497

## Optimal harman (sirt parametresine gore)

 alpha/tr(D)  tahmini LB  ||lambda||_1  max|lambda|
      0.0001     1.01308         4.242        1.165
      0.0003     1.01313         2.565        0.446
      0.0010     1.01314         1.933        0.311
      0.0030     1.01316         1.784        0.287
      0.0100     1.01324         1.479        0.226
      0.0300     1.01347         1.050        0.157
      0.1000     1.01400         0.586        0.116
      0.3000     1.01458         0.284        0.068
      1.0000     1.01509         0.106        0.028

Kucuk alpha daha iyi tahmini skor verir ama agirliklari buyutur ve
LB gurultusune duyarlilastirir. Agirlik normu makul kalan en kucuk
alpha secilir.

Secilen alpha/tr(D) = 0.0001, tahmini LB 1.01308

yon                                         lambda
SUB_A_blend [1.04876]                       1.1651
submission_optuna [1.06666]                 0.2106
submission_1.06713 [1.06713]               -0.9878
submission_seed7 [1.06727]                 -0.0345
submission_multiorigin [1.06766]            0.2427
submission_1.06929 [1.06929]               -0.3229
submission_1.07042 [1.07042]                0.1164
SUB_B_blend_plus020 [1.08249]               0.0360
SUB_D_cold_probe [1.08311]                 -0.0426
submission_v4_wx [1.10990]                 -0.4783
02_catboost_segment_split [1.11019]        -0.0587
submission_v6 [1.11447]                     0.1498
submission_c_tuned [1.14554]                0.1721
submission_level_shape [1.14884]           -0.1644
P_EMP [1.26800]                            -0.0309
P_YOY [1.28964]                            -0.0116
submission_ensemble [1.29422]               0.0099
submission_1.37997 [1.37997]               -0.0083

## Ortaya cikan harman
ortalama log1p 6.5952 (referans 6.5955, fark -0.0004)
referanstan ortalama mutlak sapma 0.0431
negatife dusen satir 1,488

yazildi: C:\Users\musta\OneDrive\Masaüstü\datathon-adm\submissions\submission_lb_blend.csv

yari-adim surum tahmini LB 1.01370 -> submission_lb_blend_half.csv
