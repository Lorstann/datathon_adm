# Besinci tur: sekiz deneme, sekiz ret — ve CV'nin guvenilirlik sinirlari

Uretim tarihi: 2026-08-25

LB en iyi: **1.06713** (degismedi). Arsivlendi:
`submissions/submission_BEST_1.06713.csv`.

## Denenenler

| # | deneme | CV | LB | karar |
| --- | --- | --- | --- | --- |
| 1 | cold lokasyon z-ofseti (EB, prior=50) | -0.0021 | **1.06766** | RET |
| 2 | guc elastikiyeti (b=1.16 yerine b=1) | +0.0009 | — | RET |
| 3 | origin'den bagimsiz olcekler | +0.0003 | — | RET |
| 4 | tohum 1/3/7 x agac 600/1000 | ±0.001 | — | fark yok |
| 5 | cold uzman modeli (router) | +0.0003..+0.0023 | — | RET |
| 6 | satir kirpmasi kaldirma (1.2M -> 1.8M) | +0.0024 | — | RET |
| 7 | iki asamali mimari (tum satirlar) | +0.0013 | — | RET |
| 8 | **melez iki asamali (warm)** | **-0.0041** | **1.08097** | **RET** |

## En onemli sonuc: CV artik guvenilmez

8 numarali deneme projedeki en buyuk CV/LB sapmasini uretti: CV -0.0041 (fold
A -0.0101, B -0.0047, C +0.0023) derken LB **+0.0138** verdi.

Suphelendigim mekanizmayi test ettim ve suclu cikmadi: `p = hist_zero_28`
kalibrasyonu test ile fold'lar arasinda neredeyse ayni --

| kume | p>0 | ortalama p | gercek sifir orani |
| --- | --- | --- | --- |
| GERCEK TEST warm | 0.0435 | 0.0431 | (bilinmiyor) |
| fold A warm | 0.0468 | 0.0448 | 0.0433 |
| fold B warm | 0.0476 | 0.0425 | 0.0435 |
| fold C warm | 0.0339 | 0.0284 | 0.0271 |

Yani sifir olasiligi hem test'te hem fold'larda ayni dagilimda ve fold'larda
gercekle ortusuyor. Sapmanin kaynagi bulunamadi.

Pratik sonuc: **-0.004 mertebesinde, 2/3 fold'da kazanan bir CV sinyali
gonderim icin yeterli kanit degil.** Fold A'nin yapisal bozuklugu (kis
agirlikli, 45 gunluk ufuklu egitim cercevesi) ortalamayi ele geciriyor;
kazancin buyugu oradan geldiginde sinyal sahte olabiliyor.

## Ikinci sonuc: hangi mudahale turu ise yariyor

Bes turun tamamina bakinca ayrim net:

**Kazandiranlar** — hepsi ya train/serve uyusmazligi ya yeni bilgi:
- cok-origin egitim cercevesi (-0.052)
- olu-trafo kapisi (-0.013)
- ozellik paketi v2 (-0.012)
- cold giris zamanlamasi (-0.002)

**Kaybedenler** — hepsi SEVIYE duzeltmesi ya da TOPLULUK:
- lokasyon ofseti, guc elastikiyeti, canli capa, seviye harmani, seviye
  toplulugu, CatBoost/XGBoost harmani, RealMLP harmani, iki asamali seviye

Aciklama: artik modeli `log_guc`, `guc`, `guc_band`, `hist_*` goruyor ve
seviyedeki sistematik sapmayi zaten soguruyor. Analitik duzeltme yuku yerinden
oynatiyor; elastikiyet denemesinde acikca asiri duzeltip isareti cevirdi
(`<100` bandi artigi -0.13 -> +0.28).

## Ucuncu sonuc: sistematik uyusmazlik taramasi tukendi

gain x dagilim kaymasi taramasi iki aday isaretledi:

| ozellik | gain | test'in egitim araligi disinda kalma orani |
| --- | --- | --- |
| `devreye_alinma_yasi` | %1.76 | %32.6 |
| `hist_n` | %1.24 | %27.5 |

Oran halleri hizalamayi %0.0 ve %4.3'e indirdi ama CV kazandirmadi: agacin son
esikte doymasi zarar vermiyormus. Yani bilinen uyusmazliklarin hepsi ya
duzeltildi ya zararsiz cikti.

## Surec hatasi ve duzeltmesi

En iyi gonderim dosyasi hic saklanmiyordu; her yapi `submission_multiorigin.csv`
uzerine yaziyordu. Bir varyant reddedildiginde en iyi dosyayi yeniden uretmek
gerekiyordu ve bu, kod durumu degistiginde risk demek. Artik
`submissions/submission_BEST_1.06713.csv` olarak arsivli.

## Durum

Kod, LB'deki en iyi yapilandirmayi uretiyor: cok-origin cerceve + cold giris
zamanlamasi + olu-trafo kapisi (k=0.75) + ozellik paketi v2, seviye zinciri m7,
tek model x 3 tohum, 600 agac, 1.8M satir. 30 test geciyor.

Siralama: 1.06713 ile 50/200. Lider 1.02995.

## Kalan

Bu noktada denenip reddedilen liste 20'yi gecti. Bilinen tavan
(`reports/24_zero_shrink.md`) hala cold trafolarin olu olup olmadigini
bilmekte (-0.34) ve varyans ayristirmasi (`reports/43_variance.md`) o bilginin
veride olmadigini gosteriyor: guc+lokasyon trafo seviyesinin %57.7'sini
acikliyor, kalan %42.3 cold'da tanim geregi erisilemez.

Kalan gonderim haklari, CV sinyali -0.004 mertebesinde olan adaylara
harcanmamali; 8 numarali deneme bunun neden yanlis oldugunu gosterdi.
