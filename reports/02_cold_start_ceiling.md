# Cold-Start Tavan Olcumu

Uretim tarihi: 2026-08-21 16:00

Varliklarin %25'i lokasyon x guc bandi icinde tabakali secilip gecmisleri tamamen gizlendi, yani gercek cold-start durumu birebir taklit edildi. Butun sayilar **RMSLE**.

`kapsama` kolonu onemli: tahmincinin grup anahtari dogrulama satirlarinda ne oranda eslesiyor. Kapsama dusukse raporlanan skor aslinda geri dusum degerinin skorudur, o anahtarin skoru degil.

## Fold A_mevsim: origin 2025-03-31, ufuk 2025-04-01 - 2025-07-31

- Holdout (cold taklidi) varlik **524**, fit varlik 1,746
- Cold satir **60,248**, warm satir 214,681
- Cold satirlarda sifir orani **5.65%**, warm satirlarda 5.36%
- Fit'te bulunan aylar: `1,2,3` | dogrulamada gereken aylar: `4,5,6,7`
- **Warm referans** (gecmisten trafo log1p ortalamasi): RMSLE **1.1302**, sifir satirlar cikarilinca 1.0127; kare hatanin %24.0'i sifir satirlardan

| Tahminci | Uzay | RMSLE | RMSLE (sifirsiz) | Sifir hata payi | Kapsama | Not |
| --- | --- | --- | --- | --- | --- | --- |
| global ortalama | log1p | **2.2203** | 1.6911 | 45.3% | 100.0% |  |
| global medyan | log1p | **2.2674** | 1.5997 | 53.0% | 100.0% |  |
| global ortalama | z | **1.9780** | 1.2172 | 64.3% | 100.0% | guc ile olceklenmis |
| global medyan | z | **2.0174** | 1.0681 | 73.6% | 100.0% | guc ile olceklenmis |
| guc_band | z | **1.9762** | 1.2173 | 64.2% | 100.0% | mean, n_grup=9 |
| lokasyon | z | **1.9893** | 1.2427 | 63.2% | 100.0% | mean, n_grup=47 |
| il | z | **1.9769** | 1.2090 | 64.7% | 100.0% | mean, n_grup=2 |
| lokasyon x guc_band | z | **2.0718** | 1.3750 | 58.4% | 100.0% | mean, n_grup=285 |
| lokasyon x guc_band x haftasonu | z | **2.0719** | 1.3747 | 58.5% | 100.0% | mean, n_grup=569 |
| lokasyon x guc_band x ay | z | **1.9780** | 1.2172 | 64.3% | 0.0% | mean, n_grup=847 |
| lokasyon x guc_band x haftasonu | z | **2.0918** | 1.2177 | 68.0% | 100.0% | median, n_grup=569 |
| lokasyon x guc_band x haftasonu | log1p | **2.0667** | 1.3793 | 58.0% | 100.0% | mean, n_grup=569; guc ofseti YOK |
| KAHIN KAPI: sifir bilinir + metadata | log1p | **1.0919** | 1.1241 | 0.0% | 100.0% | sifir kapisinin cold ust siniri |
| YUMUSAK KAPI: grup sifir olasiligi ile agirlikli | log1p | **2.0710** | 1.3731 | 58.5% | 100.0% | kesin bilgi yok, L2-optimal karisim |
| KAHIN SEVIYE: varligin ufuk-ici log1p ortalamasi | log1p | **0.5565** | 0.5114 | 20.3% | 100.0% | ulasilamaz alt sinir |
| KAHIN SEVIYE: varlik x haftagunu | log1p | **0.5447** | 0.4986 | 20.9% | 100.0% | ulasilamaz alt sinir |

## Fold B_guncel: origin 2025-11-30, ufuk 2025-12-01 - 2026-03-31

- Holdout (cold taklidi) varlik **828**, fit varlik 3,293
- Cold satir **92,716**, warm satir 351,360
- Cold satirlarda sifir orani **3.42%**, warm satirlarda 4.67%
- Fit'te bulunan aylar: `1,2,3,4,5,6,7,8,9,10,11` | dogrulamada gereken aylar: `1,2,3,12`
- **Warm referans** (gecmisten trafo log1p ortalamasi): RMSLE **1.1407**, sifir satirlar cikarilinca 0.8995; kare hatanin %40.7'i sifir satirlardan

| Tahminci | Uzay | RMSLE | RMSLE (sifirsiz) | Sifir hata payi | Kapsama | Not |
| --- | --- | --- | --- | --- | --- | --- |
| global ortalama | log1p | **2.0539** | 1.7047 | 33.5% | 100.0% |  |
| global medyan | log1p | **2.0978** | 1.6817 | 37.9% | 100.0% |  |
| global ortalama | z | **1.7427** | 1.2945 | 46.7% | 100.0% | guc ile olceklenmis |
| global medyan | z | **1.7494** | 1.2032 | 54.3% | 100.0% | guc ile olceklenmis |
| guc_band | z | **1.7189** | 1.2584 | 48.2% | 100.0% | mean, n_grup=9 |
| lokasyon | z | **1.7437** | 1.3221 | 44.5% | 100.0% | mean, n_grup=47 |
| il | z | **1.7356** | 1.2804 | 47.4% | 100.0% | mean, n_grup=2 |
| lokasyon x guc_band | z | **1.8107** | 1.3810 | 43.8% | 100.0% | mean, n_grup=304 |
| lokasyon x guc_band x haftasonu | z | **1.8104** | 1.3807 | 43.8% | 100.0% | mean, n_grup=606 |
| lokasyon x guc_band x ay | z | **1.9457** | 1.5757 | 36.7% | 72.4% | mean, n_grup=3138 |
| lokasyon x guc_band x haftasonu | z | **1.7405** | 1.2056 | 53.7% | 100.0% | median, n_grup=606 |
| lokasyon x guc_band x haftasonu | log1p | **1.8074** | 1.3776 | 43.9% | 100.0% | mean, n_grup=606; guc ofseti YOK |
| KAHIN KAPI: sifir bilinir + metadata | log1p | **1.1446** | 1.1647 | 0.0% | 100.0% | sifir kapisinin cold ust siniri |
| YUMUSAK KAPI: grup sifir olasiligi ile agirlikli | log1p | **1.8102** | 1.3804 | 43.8% | 100.0% | kesin bilgi yok, L2-optimal karisim |
| KAHIN SEVIYE: varligin ufuk-ici log1p ortalamasi | log1p | **0.4755** | 0.3764 | 39.5% | 100.0% | ulasilamaz alt sinir |
| KAHIN SEVIYE: varlik x haftagunu | log1p | **0.4619** | 0.3606 | 41.1% | 100.0% | ulasilamaz alt sinir |

## Fold C_ara: origin 2025-09-30, ufuk 2025-10-01 - 2026-01-31

- Holdout (cold taklidi) varlik **620**, fit varlik 2,651
- Cold satir **72,558**, warm satir 312,430
- Cold satirlarda sifir orani **2.09%**, warm satirlarda 4.24%
- Fit'te bulunan aylar: `1,2,3,4,5,6,7,8,9` | dogrulamada gereken aylar: `1,10,11,12`
- **Warm referans** (gecmisten trafo log1p ortalamasi): RMSLE **1.4432**, sifir satirlar cikarilinca 1.0539; kare hatanin %48.9'i sifir satirlardan

| Tahminci | Uzay | RMSLE | RMSLE (sifirsiz) | Sifir hata payi | Kapsama | Not |
| --- | --- | --- | --- | --- | --- | --- |
| global ortalama | log1p | **1.9378** | 1.7193 | 22.9% | 100.0% |  |
| global medyan | log1p | **1.9860** | 1.7252 | 26.1% | 100.0% |  |
| global ortalama | z | **1.5696** | 1.2446 | 38.4% | 100.0% | guc ile olceklenmis |
| global medyan | z | **1.5918** | 1.2036 | 44.0% | 100.0% | guc ile olceklenmis |
| guc_band | z | **1.5699** | 1.2287 | 40.0% | 100.0% | mean, n_grup=9 |
| lokasyon | z | **1.5665** | 1.2513 | 37.5% | 100.0% | mean, n_grup=47 |
| il | z | **1.5632** | 1.2353 | 38.9% | 100.0% | mean, n_grup=2 |
| lokasyon x guc_band | z | **1.6099** | 1.3223 | 33.9% | 100.0% | mean, n_grup=300 |
| lokasyon x guc_band x haftasonu | z | **1.6099** | 1.3223 | 33.9% | 100.0% | mean, n_grup=598 |
| lokasyon x guc_band x ay | z | **1.6323** | 1.3355 | 34.5% | 24.8% | mean, n_grup=2575 |
| lokasyon x guc_band x haftasonu | z | **1.5533** | 1.2008 | 41.5% | 100.0% | median, n_grup=598 |
| lokasyon x guc_band x haftasonu | log1p | **1.6118** | 1.3236 | 34.0% | 100.0% | mean, n_grup=598; guc ofseti YOK |
| KAHIN KAPI: sifir bilinir + metadata | log1p | **1.1377** | 1.1497 | 0.0% | 100.0% | sifir kapisinin cold ust siniri |
| YUMUSAK KAPI: grup sifir olasiligi ile agirlikli | log1p | **1.6081** | 1.3197 | 34.1% | 100.0% | kesin bilgi yok, L2-optimal karisim |
| KAHIN SEVIYE: varligin ufuk-ici log1p ortalamasi | log1p | **0.4990** | 0.4211 | 30.3% | 100.0% | ulasilamaz alt sinir |
| KAHIN SEVIYE: varlik x haftagunu | log1p | **0.4866** | 0.4074 | 31.4% | 100.0% | ulasilamaz alt sinir |

## Sonuclar ve stratejik karar

### 1. Sifir satirlar cold-start hatasinin ana kaynagi

| Fold | Cold sifir orani | Metadata tabani RMSLE | Sifir hata payi | Kahin kapi RMSLE | Kazanc |
| --- | --- | --- | --- | --- | --- |
| A_mevsim | 5.65% | 1.9780 | 64.3% | 1.0919 | **0.8861** |
| B_guncel | 3.42% | 1.7427 | 46.7% | 1.1446 | **0.5981** |
| C_ara | 2.09% | 1.5696 | 38.4% | 1.1377 | **0.4320** |

Cold satirlarin yalnizca %2-6'si sifir, ama kare hatanin %38-74'unu bunlar uretiyor. Satirin sifir olup olmadigini bilmek tek basina RMSLE'yi 0,43-0,89 dusuruyor. **Bu, projedeki en yuksek getirili tek bilgi parcasi.**

### 2. `guc` disindaki statik metadata cold-start icin bilgilendirici degil

| Fold | log1p global | z global (guc ile) | + lokasyon | + lokasyon x guc_band |
| --- | --- | --- | --- | --- |
| A_mevsim | 2.2203 | 1.9780 | 1.9893 | 2.0718 |
| B_guncel | 2.0539 | 1.7427 | 1.7437 | 1.8107 |
| C_ara | 1.9378 | 1.5696 | 1.5665 | 1.6099 |

`guc` ile olcekleme gercek ve tutarli bir kazanc (fold A'da 2,2203 -> 1,9780). Ama `lokasyon` eklemek skoru **kotulestiriyor**, `lokasyon x guc_band` daha da kotulestiriyor. Bu grup ortalamasi asiri uyumu: 285-304 grup, fit varliklarindan tahmin edilip hic gorulmemis varliklara tasindiginda gurultu tasiyor.

> **Cerceve dokumanindaki H2 hipotezi reddedildi.** `guc` bilgilendirici, `lokasyon` degil. Sonuc: metadata-kNN ve DTW-kume atama mimarisine yatirim yapmak gerekcesiz. Zaragoza calismasinin %95,6'lik kume atama dogrulugu bagli musteri tipi ve sayisi gibi niteliklere dayaniyordu; bizim elimizde yalnizca kurulu guc ve idari bolge var, o kadar bilgi tasimiyorlar. Kaynak, cold segmentte metadata retrieval'a degil sifir kapisina ve duzenlilestirilmis global modele ayrilacak.

### 3. Regresor sifir olmayan satirlarda kurulmali

| Fold | z global ORTALAMA (sifirsiz RMSLE) | z global MEDYAN (sifirsiz RMSLE) |
| --- | --- | --- |
| A_mevsim | 1.2172 | 1.0681 |
| B_guncel | 1.2945 | 1.2032 |
| C_ara | 1.2446 | 1.2036 |

Sifir satirlar cikarildiginda medyan ortalamayi geciyor (fold A'da 1,0681'e karsi 1,2172). Nedeni mekanik: sifir satirlarin `z` degeri `-log(guc)` gibi cok negatif bir sayi, dolayisiyla fit setindeki sifirlar ortalamayi asagi cekiyor ve sifir olmayan satirlarda sistematik dusuk tahmin uretiyor. **Karar: regresor yalnizca sifir olmayan satirlarda egitilecek, sifir olasiligi ayri bir siniflandiriciyla modellenecek.** Bu, iki asamali mimarinin ampirik gerekcesi.

### 4. Fold A yilin ayini ogrenemez, bu yapisal bir kisit

| Fold | Fit'te bulunan aylar | Dogrulamada gereken aylar | `ay` anahtarinin kapsamasi |
| --- | --- | --- | --- |
| A_mevsim | 1,2,3 | 4,5,6,7 | **0.0%** |
| B_guncel | 1,2,3,4,5,6,7,8,9,10,11 | 1,2,3,12 | **72.4%** |
| C_ara | 1,2,3,4,5,6,7,8,9 | 1,10,11,12 | **24.8%** |

Fold A'nin egitim penceresi 2025-03-31'de bitiyor ve veri 2025-01-01'de basliyor, yani Nisan-Temmuz aylarina ait **hicbir** gecmis gozlem yok: `ay` anahtarinin kapsamasi tam olarak %0. Bu duzeltilebilir bir hata degil, verinin yapisal kisiti. Panel yalnizca bir yaz iceriyor (Nis-Tem 2025) ve o yaz fold A'nin dogrulama penceresinin kendisi.

Sonuclar, uc ayri baslikta:

1. **Fold A saf mevsim ekstrapolasyonunu test ediyor.** Model Ocak-Mart'tan Temmuz'a gitmek zorunda. Bunu yapabilmesinin tek yolu, ufuk boyunca zaten elimizde olan **hava degiskenleri**. Yani fold A hava ozelliklerinin degerini olcen fold; takvim-mevsim ozelliklerinin degerini olcemez.
2. **Fold B yil-uzeri mevsimsel ozellikleri test edebilen tek fold** (kapsama %72,4): Ara 2025-Mar 2026 ufkunun Oca-Mar kismi icin Oca-Mar 2025 analogu var. `gecen yil ayni ay` tipi ozellikler burada dogrulanacak.
3. **Gercek gonderim, hicbir fold'un dogrulayamadigi bir avantaja sahip:** origin 2026-03-31 oldugunda Nis-Tem 2025 tamamen train icinde. Yani mevsimsel naif ve gecen-yil ozellikleri gonderimde calisacak ama fold A'da olculemez. Bu asimetri, fold A skorlarinin gercek test skorundan **kotumser** olmasi beklendigi anlamina gelir ve fold'lar arasi karsilastirmada akilda tutulmali.

### 5. Warm satirlarda bile tarihsel seviye zayif bir tahminci

| Fold | Warm: gecmis ortalamasi | Cold: kahin ufuk-ici seviye |
| --- | --- | --- |
| A_mevsim | 1.1302 | 0.5565 |
| B_guncel | 1.1407 | 0.4755 |
| C_ara | 1.4432 | 0.4990 |

Gecmisi olan bir trafo icin tarihsel ortalamayi tasimak 1,13-1,44 RMSLE veriyor; oysa ayni trafonun ufuk-ici gercek ortalamasini bilmek 0,46-0,56 veriyor. Aradaki bosluk, seviyenin gecmisten ufka **kaydigini** gosteriyor: buyume, rejim degisimi ve mevsimsel seviye kaymasi. Yani warm segmentte de is bitmiyor; seviye tahminini duzeltmek (trend, buyume orani, mevsimsel olcekleme) gercek bir kazanc alani.
