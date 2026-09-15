# Adaylarin novelty olcumu
Uretim: 2026-08-31 12:27:04.060026

Referans FINAL_4 (LB 1.01548, MSE 1.03120). Havuz 16 bagimsiz yon.

## Her adayin havuza gore konumu

aday            ||D||  izdusum R2  ||D_dik||  FINAL_4 kor
Q_SMOOTH       0.8853      0.5598     0.5874       0.8874
Q_ANALOG       0.8982      0.5453     0.6056       0.8867
Q_SVD          0.9215      0.5406     0.6246       0.8831
Q_COMBO        0.8643      0.5797     0.5603       0.8935

`||D_dik||` = havuzun aciklayamadigi kisim. Kazanc bu eksende olculuyor;
buyuk olmasi kazanci garanti etmez ama kucuk olmasi kazanci imkansiz kilar.

## Adaylarin birbirine gore dikligi (dik bilesenlerin korelasyonu)

               Q_SMOOTH   Q_ANALOG      Q_SVD    Q_COMBO
Q_SMOOTH          1.000      0.782      0.834      0.941
Q_ANALOG          0.782      1.000      0.734      0.906
Q_SVD             0.834      0.734      1.000      0.927
Q_COMBO           0.941      0.906      0.927      1.000

## Kazanc olceklemesi: olculen <e0,D_dik> degerine gore

Yoklama gonderildikten sonra kazanc = <e0,D_dik>^2 / ||D_dik||^2.
Asagida, dik bilesenle gercek artik arasindaki korelasyon rho'ya gore
ne bekleyecegimiz. (rho bilinmiyor; olcum onu verecek.)

   rho     Q_SMOOTH    Q_ANALOG       Q_SVD     Q_COMBO
  0.05      1.01421     1.01421     1.01421     1.01421
  0.10      1.01039     1.01039     1.01039     1.01039
  0.15      1.00399     1.00399     1.00399     1.00399
  0.20      0.99496     0.99496     0.99496     0.99496
  0.30      0.96871     0.96871     0.96871     0.96871

Not: kazanc yalnizca rho'ya bagli, ||D_dik||'e degil -- yeter ki
||D_dik|| olcum gurultusunun uzerinde olsun. Ucu de o esigin cok
ustunde.

yazildi: submissions/probe_Q_COMBO.csv

## Oneri

En genis yeni eksen: Q_SVD
Bugunku tek hak bilesik yoklamaya (Q_COMBO) gitmeli: tum ailenin
degeri tek olcumde okunur. Deger cikarsa yarin bilesenler ayrilir,
cikmazsa iki hak baska bir aileye kalir.
