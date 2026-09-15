# Panel sinir kapisi: ikinci yon ailesi
Uretim: 2026-08-31 12:30:50.404891

test satiri 714,688
erken cikan trafonun satiri 13,451 (1.88%), trafo 241
gec giren trafonun satiri 244,867 (34.26%), trafo 3,108

## Uygulanan carpanlar
dilim                            satir     pay  carpan   ort log1p tahmin
cikisa 0-2 gun                     661  0.0009    0.25             6.0567
cikisa 3-6 gun                     788  0.0011    0.40             6.1270
cikisa 7-13 gun                  1,271  0.0018    0.60             6.0901
cikisa 14-29 gun                 2,789  0.0039    0.80             6.0405
giristen 0-0 gun                 3,108  0.0043    0.45             6.2567
giristen 1-2 gun                 6,128  0.0086    0.65             6.5965
giristen 3-6 gun                12,111  0.0169    0.85             6.5928

toplam dokunulan satir 26,661 (3.73%)
bunlarin cold payi 0.564, Mart'ta olu payi 0.013

yonun buyuklugu ||D|| = 0.0886
ortalama kayma -0.0136

yazildi: submissions/probe_P_EXIT.csv

Olcum yorumu: bu yoklamanin skoru S ise
  <e0,D> = (S^2 - 1.01548^2 - ||D||^2) / 2
negatif cikarsa panel sinir satirlarinda gercekten fazla tahmin
ediyoruz ve kapinin optimal siddeti w = -<e0,D>/||D||^2 ile bulunur.
