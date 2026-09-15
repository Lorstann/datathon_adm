# Mevsimsel rampa kalibrasyonu
Uretim: 2026-08-31 11:43:05.373830

Lift = (hedef aydaki ortalama log1p) - (origin'de son 28 gun seviyesi),
yalnizca origin'de CANLI olan ayni trafolar uzerinde.

## origin 2025-03-31  (takvim ayi 3)
              n  trafo  gercek  origin_lvl    lift
ay_ofset                                          
1         59138   1987  6.6060      6.6547 -0.0486
2         60764   1984  6.5702      6.6562 -0.0860
3         58236   1972  6.8912      6.6570  0.2342
4         59665   1951  7.1818      6.6490  0.5328

## origin 2025-05-31  (takvim ayi 5)
              n  trafo  gercek  origin_lvl    lift
ay_ofset                                          
1         61817   2125  6.8762      6.5352  0.3411
2         64103   2103  7.1539      6.5035  0.6503
3         63591   2082  7.0590      6.4919  0.5672
4         58757   2078  6.7646      6.4708  0.2938

## origin 2025-07-31  (takvim ayi 7)
              n  trafo  gercek  origin_lvl    lift
ay_ofset                                          
1         77237   2586  7.0640      7.1664 -0.1024
2         66304   2403  6.7596      7.1279 -0.3684
3         65265   2152  6.4879      7.1060 -0.6180
4         62988   2140  6.5600      7.1175 -0.5575

## origin 2025-09-30  (takvim ayi 9)
              n  trafo  gercek  origin_lvl    lift
ay_ofset                                          
1         73590   2424  6.4891      6.7272 -0.2381
2         70778   2413  6.5469      6.7278 -0.1809
3         72218   2372  6.7137      6.7349 -0.0211
4         68700   2346  6.7292      6.7336 -0.0045

## origin 2025-11-30  (takvim ayi 11)
              n  trafo  gercek  origin_lvl    lift
ay_ofset                                          
1         95834   3143  6.7417      6.5735  0.1682
2         93399   3091  6.7491      6.5671  0.1820
3         83133   3025  6.6689      6.5625  0.1063
4         91562   3083  6.6433      6.5658  0.0776

# Gonderimin uyguladigi lift

2026-03-31'de canli trafo: 4,310, ortalama son-28g seviye 6.6429

## Gonderim: origin'de canli trafolar icin uygulanan lift
               n  trafo  tahmin  origin_lvl    lift
ay_ofset                                           
1         111526   3784  6.7281      6.7216  0.0064
2         125752   4278  6.7097      6.6715  0.0382
3         126657   4248  6.9368      6.6524  0.2844
4         129743   4222  7.0696      6.6545  0.4152

## Yan yana: 2025 gerceklesen vs 2026 gonderim (ayni takvim aylari)
          lift_2025_gercek  lift_2026_tahmin    fark
ay_ofset                                            
1                  -0.0486            0.0064  0.0551
2                  -0.0860            0.0382  0.1242
3                   0.2342            0.2844  0.0501
4                   0.5328            0.4152 -0.1176

Fark pozitifse gonderim 2025'ten daha dik, negatifse daha duz.

## 2026 yazinin 2025'e gore hava farki
       cdd        temperature_2m_mean        
yil   2025   2026                2025    2026
ay                                           
4    0.001  0.000              13.610  13.952
5    0.511  0.085              19.675  18.030
6    4.224  3.159              26.051  24.969
7    7.537  5.075              29.533  27.034

## Kazanc aritmetigi
Bir ayda ortalama sapma d ise, o ayin MSE katkisi d^2 kadar fazla.
  ay    n_pay     fark      dMSE
   4    0.156   0.0551   0.00047
   5    0.176   0.1242   0.00272
   6    0.177   0.0501   0.00045
   7    0.182  -0.1176   0.00251
toplam                     0.00615

Not: bu, 2026'nin 2025 ile ayni mevsimsel imzaya sahip oldugunu
varsayarsa gecerli ust sinir. Hava farki bu farkin bir kismini
mesru kiliyor; ustteki CDD tablosu ne kadarini acikladigini gosterir.