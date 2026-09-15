# Yarinin yoklamalari
Uretim: 2026-08-31 12:40:13.827689

## DISP: dagilim keskinligi
a = 0.15, q = ort + (1+a)(p0 - ort), ort = 6.5955
||D|| = 0.2127, ortalama kayma +0.0186
tahmin std 1.7004 -> 1.8871

Okuma: skor S ise <e0,D> = (S^2 - S0^2 - ||D||^2)/2.
  <e0,D> < 0  ->  daha KESKIN olmaliyiz (tahminler fazla puruzsuz)
  <e0,D> > 0  ->  daha PURUZSUZ olmaliyiz (tahminler fazla keskin)
optimal agirlik w = -<e0,D>/||D||^2, uygulanan keskinlik 0.15*w olur.

yazildi: submissions/probe_DISP.csv

## ZGATE: birlesik sifir/panel kapisi
alt desen                      satir      pay  carpan   ort p0
Mart'ta olu (zr>0.9)          20,777   0.0291    0.30   0.6511
yari olu (0.5<zr<=0.9)         1,681   0.0024    0.55   1.5945
cikisa 0-2 gun                   661   0.0009    0.30   6.0567
cikisa 3-6 gun                   788   0.0011    0.45   6.1270
cikisa 7-13 gun                1,271   0.0018    0.65   6.0901
cikisa 14-29 gun               2,789   0.0039    0.85   6.0405
giristen 0-0 gun               3,108   0.0043    0.45   6.2567
giristen 1-2 gun               6,128   0.0086    0.65   6.5965
giristen 3-6 gun              12,111   0.0169    0.85   6.5928

toplam dokunulan 48,658 (6.81%)
||D|| = 0.1173, ortalama kayma -0.0223

yazildi: submissions/probe_ZGATE.csv

## Kaldirac senaryolari

yon          ||D||^2      b=0.1     b=0.2     b=0.5     b=1.0
DISP         0.04523    1.01544   1.01533   1.01453   1.01169
ZGATE        0.01377    1.01530   1.01477   1.01103   0.99755
