# Artigin anatomisi: dilim bazinda yanlilik
Uretim: 2026-08-31 12:03:37.361539

Referans: FINAL_4, LB 1.01548, MSE 1.03120. 18 yon, 714,688 satir.

## Dilim yanliliklari

`E[g r]` LB cebirinden geliyor; `yanlilik` = E[g r] / pay, yani o
dilimdeki ortalama (tahmin - gercek). Pozitif = FAZLA tahmin.
`dMSE` o dilimi yanliligi kadar kaydirirsak toplam MSE dususu.
`R2` izdusum kalitesi: 1'e yakin degilse sayi guvenilmez.

dilim                              pay      R2    E[g r]  yanlilik      dMSE
SABIT (tum satirlar)             1.000  1.0000   0.00034    0.0003   0.00000
cold                             0.222  1.0000  -0.00029   -0.0013   0.00000
warm_canli                       0.749  0.9645   0.00066    0.0009   0.00000
warm_olu (Mart'ta olu)           0.029  0.0941  -0.00003   -0.0012   0.00000
ay=4                             0.165  0.2805   0.00079    0.0048   0.00000
ay=5                             0.256  0.3474  -0.00011   -0.0004   0.00000
ay=6                             0.284  0.3156  -0.00082   -0.0029   0.00000
ay=7                             0.295  0.4822   0.00048    0.0016   0.00000
hafta sonu                       0.276  0.2783  -0.00002   -0.0001   0.00000
band=1000-1600                   0.277  0.3440  -0.00043   -0.0015   0.00000
band=400-630                     0.214  0.2259   0.00023    0.0011   0.00000
band=250-400                     0.164  0.1818   0.00002    0.0002   0.00000
band=<100                        0.030  0.0557  -0.00068   -0.0227   0.00002
band=100-160                     0.068  0.0963   0.00012    0.0017   0.00000
band=630-1000                    0.117  0.1291  -0.00002   -0.0002   0.00000
band=2600+                       0.005  0.0208   0.00007    0.0123   0.00000
band=1600-2600                   0.012  0.0296   0.00015    0.0130   0.00000
band=160-250                     0.113  0.1416   0.00087    0.0077   0.00001
en dusuk %10 tahmin              0.100  0.1594  -0.00020   -0.0020   0.00000
en dusuk %25 tahmin              0.250  0.3277   0.00003    0.0001   0.00000
en yuksek %10 tahmin             0.100  0.1349   0.00090    0.0090   0.00001
lok=İZMİR>GÜNEY BÖLGE>ÖDEMİŞ     0.067  0.1429  -0.00093   -0.0138   0.00001
lok=İZMİR>METROPOL>BORNOVA       0.066  0.1198  -0.00135   -0.0203   0.00003
lok=İZMİR>GÜNEY BÖLGE>MENDERES   0.045  0.0673   0.00041    0.0091   0.00000
lok=İZMİR>GÜNEY BÖLGE>URLA       0.041  0.0781   0.00039    0.0094   0.00000
lok=İZMİR>GÜNEY BÖLGE>TİRE       0.038  0.0537  -0.00041   -0.0108   0.00000
lok=İZMİR>METROPOL>BUCA          0.036  0.0465   0.00028    0.0077   0.00000
warm, gecmis<90g                 0.180  0.2745   0.00060    0.0033   0.00000
tahmin seviyesi (merkezli)         nan  0.0810   0.00393    0.0014   0.00001
log(guc) (merkezli)                nan  0.1387   0.00045    0.0006   0.00000
Mart seviyesi (merkezli)           nan  0.1034   0.00573    0.0019   0.00001

Surekli degiskenlerde `pay` yok; `yanlilik` sutunu egimi, `dMSE` o
egimi duzeltmenin kazancini verir.

R2 > 0.90 olan dilim sayisi: 3 / 31

## Olculebilir dilimlerde toplam potansiyel

(kaba ust sinir, dilimler ortusuyor) toplam dMSE ~ 0.00000
MSE 1.03120 -> 1.03120, LB 1.01548 -> 1.01548

## Olculemeyen (R2 dusuk) dilimler -- kalan gonderim hakkiyla yoklanacak

  warm_olu (Mart'ta olu)         R2  0.094
  ay=4                           R2  0.281
  ay=5                           R2  0.347
  ay=6                           R2  0.316
  ay=7                           R2  0.482
  hafta sonu                     R2  0.278
  band=1000-1600                 R2  0.344
  band=400-630                   R2  0.226
  band=250-400                   R2  0.182
  band=<100                      R2  0.056
  band=100-160                   R2  0.096
  band=630-1000                  R2  0.129
  band=2600+                     R2  0.021
  band=1600-2600                 R2  0.030
  band=160-250                   R2  0.142
  en dusuk %10 tahmin            R2  0.159
  en dusuk %25 tahmin            R2  0.328
  en yuksek %10 tahmin           R2  0.135
  lok=İZMİR>GÜNEY BÖLGE>ÖDEMİŞ   R2  0.143
  lok=İZMİR>METROPOL>BORNOVA     R2  0.120
  lok=İZMİR>GÜNEY BÖLGE>MENDERES R2  0.067
  lok=İZMİR>GÜNEY BÖLGE>URLA     R2  0.078
  lok=İZMİR>GÜNEY BÖLGE>TİRE     R2  0.054
  lok=İZMİR>METROPOL>BUCA        R2  0.046
  warm, gecmis<90g               R2  0.274
  tahmin seviyesi (merkezli)     R2  0.081
  log(guc) (merkezli)            R2  0.139
  Mart seviyesi (merkezli)       R2  0.103
