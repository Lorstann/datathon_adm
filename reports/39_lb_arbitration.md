# LB hakemligi turu + siralama gercegi

Uretim tarihi: 2026-08-23

## Siralama tablosu (200 takim)

| | skor |
| --- | --- |
| lider | **1.02995** |
| 10. | 1.04160 |
| 23. | 1.05000 |
| **biz (50.)** | **1.06713** |
| sonuncu | 1.29722 |

**1.00 altinda takim yok.** Ilk 23 takim 1.030-1.050 araligina sikismis.
Yani "1'in altina inmek" liderden 0.03 daha iyi olmak demek ve bu, 1. ile
23. arasindaki tum farktan buyuk. Bu veri setinde ulasilabilir gorunmuyor.

Gercekci hedef: **1.03-1.05** (ilk 10 / ilk 25), yani 0.02-0.04 kapatmak.

## Karar kurali degisikligi ve gerekcesi

3/3 fold kurali sekiz degisikligi reddetmisti, birkacinin ortalamasi
pozitifken. Ayni donemde LB her seferinde CV'den IYI cikti:

| degisiklik | CV beklentisi | LB gerceklesen |
| --- | --- | --- |
| cok-origin cercevesi | -0.018 | -0.052 |
| olu-trafo kapisi | -0.011 | -0.013 |
| ozellik paketi v2 | -0.005 | -0.012 |

Uc uzerinden uc kez CV kotumser. Fold A yapisal olarak bozuk (kis-agirlikli,
45 gunluk ufuklu egitim cercevesi) ve ortalamayi asagi cekiyor. Bu yuzden
**ilkeli ama CV'de belirsiz** kalan az sayida degisiklik icin hakem olarak LB
kullanildi: 714.688 satir ve gonderim basina tek soru, LB'ye asiri uyum riski
ihmal edilebilir.

## Bugunku uc gonderim, uc net cevap

| # | degisiklik | LB | karar |
| --- | --- | --- | --- |
| 1 | cold giris zamanlamasi taklidi | **1.06713** | KABUL (-0.00186) |
| 2 | seviye toplulugu (m7 + m28) | 1.06929 | RET (+0.00216) |
| 3 | mevsim-esli agirlik (Nis-Tem w=2) | 1.07042 | RET (+0.00329) |

### 1. Cold giris zamanlamasi (KABUL)

Gercek bir egitim/tahmin uyusmazligi. Cold taklidi yalnizca gecmisi siliyordu;
oysa gercek cold trafonun transduktif kolonlari giris zamanlamasini ele
veriyor:

| kolon | gercek test cold | eski egitim taklidi | duzeltilmis |
| --- | --- | --- | --- |
| `devreye_alinma_yasi` | 40 | 202 | 33 |
| `test_gun_sayisi` | 82 | 122 | 82 |
| `horizon_day` | 82 | 51 | 78 |

Model "gecmis yok + yasli" ogrenip "gecmis yok + genc" ile karsilasiyordu;
egitimde genc satirlarin tamaminin gecmisi vardi, yani o bolge agac icin
bostu. Duzeltme ampirik giris ofseti ornekliyor, giristen onceki satirlari
dusuruyor, yas ve gun sayisini yeniden yaziyor. Ayni duzeltme dogrulama
tarafina da uygulandi.

CV: 1.1953 -> 1.1940 (fold farklari -0.0074 / +0.0020 / +0.0016, yani 3/3
kuralini gecmiyordu). LB dogruladi.

### 2. Seviye toplulugu (RET)

CV'de fold C isaret degistiriyordu (+0.0054); LB ayni yonde cikti. Ogrenici
cesitliligi (CatBoost/XGBoost) zaten reddedilmisti, seviye cesitliligi de
oyle.

### 3. Mevsim-esli agirlik (RET) — ve mevsimsel korunun kapanmasi

Olculmustu: gonderimde Nis->Tem sekli +0.33, ayni trafolarin gecen yilki
gerceklesen sekli +0.64. Yani model mevsimsel genligi yariya sikistiriyor
gorunuyordu.

Ama 2026 yazi gercekten daha serin:

| ay | 2026 CDD - 2025 CDD | 2026 tmean - 2025 tmean |
| --- | --- | --- |
| 5 | -0.43 | -1.64 |
| 6 | -1.06 | -1.08 |
| 7 | **-2.46** | **-2.50** |

Temmuz'da 2,46 birim daha az CDD x medyan trafo katsayisi 0,056 ~ -0.14 log
birimi; olculen -0.19'luk aciklanmamis sapmanin buyuk kismi bu. Model bunu
zaten hava degiskenlerinden okuyor.

Agirliklandirma LB'de kotulestirdi (+0.00329), yani modelin daha duz rampasi
DOGRU. Mevsimsel hipotez kapandi.

## CV'nin yapisal koru (onemli)

Fold'larin egitim ufuklari tanim geregi fold origin'inde bitiyor, dolayisiyla
dogrulama aylarini hic kapsamiyor:

| fold | mevsim-ici egitim satiri |
| --- | --- |
| A_mevsim | %0.0 |
| B_guncel | %2.9 |
| C_ara | %0.0 |

Gonderimde ayni oran %19.6. Yani **bu CV tasarimi hicbir mevsimsel ozelligi
olcemez**. Ayni kor nokta `lag364`'u de vurmustu.

`lag364` icin kapsama sayilari bu turda kesinlesti: gonderim egitim
cercevesinde kapsama %16.2 ve **yalnizca Oca/Sub/Mar'da** (%46/%46/%44),
Nis-Ara'da %0. Test tarafinda %95.5. Model iliskiyi sadece kis satirlarindan
ogrenip yaz satirlarina uygulardi. Reddi kesinlesti.

## Durum

Kod, LB'deki en iyi yapilandirmayi uretiyor (1.06713): cok-origin cerceve +
cold giris zamanlamasi + olu-trafo kapisi (k=0.75) + ozellik paketi v2,
seviye zinciri m7, mevsim agirligi 1.0, tek model x 3 tohum.

29 test geciyor.

## Kalan

Bu noktada denenip reddedilenler: kapasite, ozellik paketi v3, seviye harmani,
seviye toplulugu, ogrenici toplulugu, kapi kalibrasyon gridi, dis veri,
mevsim agirligi, satir duzeyinde yil-once lag'i, olu-trafo siniflandiricisi,
trafo yenileme/yeniden adlandirma eslesmesi.

Bilinen tavan (`reports/24_zero_shrink.md`) hala cold trafolarin olu olup
olmadigini bilmekte (-0.34) ve o bilgi veride yok. Lidere olan 0.037'lik
farkin nereden geldigi hala bilinmiyor.
