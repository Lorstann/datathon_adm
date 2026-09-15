# EDA

Uretim tarihi: 2026-08-21 16:04

Bu EDA, cold-start tavan olcumunun (reports/02) birakti somut sorulara odakli; genel bir betimleyici tarama degil.

## 1. Sifir durumu: kalicilik ve gecisler

Gunluk gecis olasiliklari (train, ardisik gun ciftleri):

| Onceki gun | Bugun sifir | Bugun sifir degil |
| --- | --- | --- |
| sifir | **0.9755** | 0.0245 |
| sifir degil | 0.0011 | **0.9989** |

Sifir durumu gunluk olcekte cok kalici: sifirdan sifira gecis olasiligi 0.976. Yani sifirlar tek gunluk olaylar degil, uzun bloklar halinde geliyor.

122 gunluk blok olceginde (origin 2025-11-30), oncesi ve sonrasi sifir oranlarinin korelasyonu **0.938** (3,387 trafo).

| Origin oncesi durum | Trafo | Ufukta sifir orani (ortalama) |
| --- | --- | --- |
| tamamen sifir (>0,9) | 126 | 0.9548 |
| hic sifir yok (<0,1) | 3,222 | 0.0030 |

Durum degistiren trafolar: **6** sifirdan cikiyor (devreye aliniyor), **8** sifira giriyor (de-enerjize oluyor).

> **Tasarim sonucu.** Sifir durumu yuksek oranda kalici, yani warm trafolar icin gecmisten okunabiliyor. Warm hatanin buyuk kisminin sifir satirlardan gelmesi bu yuzden bir celiski degil: hata durumu YANLIS bilmekten degil, durum GECISLERINDEN geliyor. Sifir kapisi bu nedenle yalnizca 'su an sifir mi' degil 'ufuk boyunca sifir kalir mi' sorusunu cevaplamali; girdi olarak son sifir blogunun uzunlugu ve devreye alinma yasi kullanilacak.

## 2. Seviye kaymasi: tarihsel ortalama neden zayif

Sifir olmayan satirlarda, origin oncesi 122 gun ile sonraki 122 gun arasindaki log1p seviye kaymasi (3,260 trafo):

- Ortalama kayma: **+0.0321** log1p birimi (carpansal olarak 1.033x)
- Medyan kayma: **+0.1004**
- Kaymanin standart sapmasi: **0.5304**
- Kaymanin mutlak degeri > 0,5 olan trafo orani: **15.4%**

Ortalama kayma kucuk ama dagilim genis. Yani sistematik bir buyume duzeltmesi degil, trafo bazinda oynaklik hakim. Bu, tarihsel ortalamanin 1,13 RMSLE'de kalmasinin nedeni: seviye tahmininin kendisi gurultulu. Kayma yonu ongorulemiyorsa en iyi strateji kaymayi tahmin etmeye calismamak, sadece seviyeyi olabildigince saglam olcmek.

Kayma, gecmis uzunluguna gore:

| Gecmis | Trafo | Ortalama kayma | Kayma std |
| --- | --- | --- | --- |
| [0.0, 30.0) | 573 | +0.1815 | 0.3087 |
| [30.0, 90.0) | 465 | +0.1417 | 0.3738 |
| [90.0, 180.0) | 298 | -0.1034 | 0.6678 |
| [180.0, 270.0) | 156 | -0.1508 | 0.7577 |
| [270.0, inf) | 1,768 | -0.0061 | 0.5553 |

Kisa gecmisli trafolarda kayma hem daha buyuk hem daha oynak: yeni devreye alinan trafo rampa halinde. Bu, `devreye_alinma_yasi` ozelliginin gerekcesi.

![03_seviye_kaymasi.png](C:/Users/musta/OneDrive/Masaüstü/datathon-adm/reports/figures/03_seviye_kaymasi.png)

## 3. Mevsimsellik

Aylik ortalama `log1p(tuketim)` (sifir olmayan satirlar):

| Ay | Ortalama | Medyan | Ocak 2025'e gore |
| --- | --- | --- | --- |
| 2025-01 | 6.7972 | 7.1576 | +0.0000 |
| 2025-02 | 6.8663 | 7.2538 | +0.0691 |
| 2025-03 | 6.6950 | 7.0242 | -0.1022 |
| 2025-04 | 6.6359 | 6.9620 | -0.1613 |
| 2025-05 | 6.5895 | 6.8894 | -0.2077 |
| 2025-06 | 6.9046 | 7.1854 | +0.1074 |
| 2025-07 | 7.1969 | 7.4720 | +0.3997 |
| 2025-08 | 7.0982 | 7.3625 | +0.3010 |
| 2025-09 | 6.8001 | 7.0490 | +0.0029 |
| 2025-10 | 6.5167 | 6.7809 | -0.2805 |
| 2025-11 | 6.5847 | 6.8563 | -0.2124 |
| 2025-12 | 6.7855 | 7.0832 | -0.0117 |
| 2026-01 | 6.8137 | 7.1076 | +0.0165 |
| 2026-02 | 6.7585 | 7.0518 | -0.0387 |
| 2026-03 | 6.7361 | 7.0212 | -0.0611 |

![03_aylik_seviye.png](C:/Users/musta/OneDrive/Masaüstü/datathon-adm/reports/figures/03_aylik_seviye.png)

Nisan-Temmuz 2025 araligi: 6.5895 - 7.1969, tepe 2025-07. Ufkumuz tam bu pencereye denk geliyor ve icinde **0.6074 log1p birimlik** bir mevsimsel rampa var (carpansal 1.84x). Yani ufuk sabit bir seviye degil, yukselen bir egri; ufuk-agnostik tek bir seviye tahmini bu rampayi kaciracak.

## 4. Hafta gunu etkisi

| Gun | Ortalama log1p | Pazartesi'ye gore |
| --- | --- | --- |
| Pzt | 6.7804 | +0.0000 |
| Sal | 6.7882 | +0.0078 |
| Car | 6.7966 | +0.0162 |
| Per | 6.8016 | +0.0212 |
| Cum | 6.7932 | +0.0128 |
| Cmt | 6.7878 | +0.0074 |
| Paz | 6.7442 | -0.0362 |

Hafta ici-hafta sonu farki toplam 0.0573 log1p birimi, yani mevsimsel rampanin (0.6074) 9%'i kadar. Ikincil ama ihmal edilemez.

Guc bandina gore hafta gunu profili (banttan sapma, log1p):

| Band | Pzt | Sal | Car | Per | Cum | Cmt | Paz | Hafta sonu dususu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100-160 | -0.023 | -0.017 | -0.007 | -0.001 | -0.004 | +0.016 | +0.036 | **-0.036** |
| 1000-1600 | +0.019 | +0.030 | +0.034 | +0.038 | +0.026 | -0.022 | -0.124 | **+0.103** |
| 160-250 | -0.023 | -0.018 | -0.007 | -0.004 | -0.005 | +0.025 | +0.032 | **-0.040** |
| 1600-2600 | +0.021 | +0.033 | +0.033 | +0.036 | +0.039 | -0.015 | -0.148 | **+0.114** |
| 250-400 | -0.019 | -0.012 | -0.006 | +0.001 | -0.007 | +0.020 | +0.023 | **-0.030** |
| 2600+ | -0.034 | +0.030 | -0.002 | +0.009 | +0.029 | +0.014 | -0.046 | **+0.023** |
| 400-630 | -0.009 | -0.002 | +0.007 | +0.011 | +0.002 | +0.012 | -0.021 | **+0.006** |
| 630-1000 | +0.015 | +0.025 | +0.026 | +0.033 | +0.018 | -0.011 | -0.105 | **+0.081** |
| <100 | -0.029 | -0.018 | -0.013 | -0.003 | -0.002 | +0.031 | +0.034 | **-0.046** |

Hafta sonu dususunun buyuklugu guc bandina gore degisiyor: bu, sanayi agirlikli ve konut agirlikli trafolari ayirt eden gozlenebilir bir imza. `guc_band x haftagunu` etkilesimi bu yuzden ozellik setinde.

![03_haftagunu_profili.png](C:/Users/musta/OneDrive/Masaüstü/datathon-adm/reports/figures/03_haftagunu_profili.png)

## 5. Cold ve warm trafolarin statik profili

Guc bandina gore cold/warm dagilimi:

| Band | Warm | Cold | Cold orani |
| --- | --- | --- | --- |
| 100-160 | 371 | 93 | 0.200 |
| 1000-1600 | 1,260 | 759 | 0.376 |
| 160-250 | 591 | 189 | 0.242 |
| 1600-2600 | 62 | 27 | 0.303 |
| 250-400 | 848 | 286 | 0.252 |
| 2600+ | 33 | 3 | 0.083 |
| 400-630 | 1,121 | 354 | 0.240 |
| 630-1000 | 565 | 262 | 0.317 |
| <100 | 161 | 51 | 0.241 |

Genel cold orani: **0.288**

Cold orani bandlar arasinda degisiyorsa cold populasyon rastgele degil, belirli guc siniflarinda kumelenmis. Maskeleme tabakalamasi bu yuzden uniform degil (bkz. src/validation/folds.py).

Il bazinda:

| Il | Warm | Cold | Cold orani |
| --- | --- | --- | --- |
| MANİSA | 1,220 | 459 | 0.273 |
| İZMİR | 3,792 | 1,565 | 0.292 |

Cold orani en yuksek ve en dusuk 5 lokasyon:

| Lokasyon | Trafo | Cold orani |
| --- | --- | --- |
| İZMİR>KUZEY BÖLGE>DİKİLİ | 122 | 0.582 |
| İZMİR>METROPOL>KARŞIYAKA | 141 | 0.532 |
| İZMİR>METROPOL>BAYRAKLI | 232 | 0.478 |
| İZMİR>METROPOL>ÇİĞLİ | 107 | 0.477 |
| İZMİR>METROPOL>BORNOVA | 485 | 0.452 |
| MANİSA>DEMİRCİ | 43 | 0.140 |
| MANİSA>AHMETLİ | 29 | 0.138 |
| İZMİR>GÜNEY BÖLGE>KİRAZ | 193 | 0.135 |
| İZMİR>GÜNEY BÖLGE>ÖDEMİŞ | 445 | 0.128 |
| İZMİR>GÜNEY BÖLGE>BAYINDIR | 204 | 0.108 |

## 6. Trafo davranis kumeleri

En az 60 sifir olmayan gozlemi olan 4,067 trafo uzerinde:

| Olcu | Medyan | 10. yuzdelik | 90. yuzdelik |
| --- | --- | --- | --- |
| log1p seviye | 7.0351 | 4.6903 | 8.4692 |
| gun-ici oynaklik (std) | 0.2536 | 0.1375 | 0.6076 |
| hafta sonu dususu | -0.0203 | -0.0871 | 0.1248 |
| sifir orani | 0.0000 | 0.0000 | 0.0000 |

Hafta sonu dususunun 10.-90. yuzdelik araligi genis: bazi trafolar hafta sonu neredeyse hic dusmuyor (surekli yuk), bazilari belirgin dusuyor (ticari/sanayi). Bu, kume atamasi icin gercek bir gozlenebilir eksen -- ama yalnizca gecmisi olan trafolar icin olculebiliyor, cold olanlar icin degil.

![03_davranis_uzayi.png](C:/Users/musta/OneDrive/Masaüstü/datathon-adm/reports/figures/03_davranis_uzayi.png)

## 7. Tatil etkisi

Tatil gununun, ayni ay ve ayni hafta gunundeki tatil olmayan gunlere gore log1p sapmasi:

| Tarih | Tatil | Satir | Sapma |
| --- | --- | --- | --- |
| 2025-06-06 | Kurban Bayramı | 2,034 | **-0.2888** |
| 2025-06-07 | Kurban Bayramı | 2,037 | **-0.2366** |
| 2025-06-05 | Kurban Bayramı (saat 13.00'ten) | 2,038 | **-0.2342** |
| 2025-08-30 | Zafer Bayramı | 2,418 | **-0.1646** |
| 2025-03-31 | Ramazan Bayramı | 1,973 | **-0.1514** |
| 2025-04-23 | Ulusal Egemenlik ve Çocuk Bayramı | 1,979 | **-0.1277** |
| 2025-03-30 | Ramazan Bayramı | 1,970 | **-0.1120** |
| 2025-03-29 | Ramazan Bayramı (saat 13.00'ten) | 1,973 | **-0.0566** |
| 2025-06-09 | Kurban Bayramı | 2,038 | **-0.0397** |
| 2025-07-15 | Demokrasi ve Millî Birlik Günü | 2,364 | **-0.0333** |
| 2025-10-28 | Cumhuriyet Bayramı (saat 13.00'ten) | 2,585 | **-0.0296** |
| 2026-03-22 | Ramazan Bayramı | 3,692 | **-0.0283** |
| 2025-04-01 | Ramazan Bayramı | 1,966 | **-0.0197** |
| 2025-10-29 | Cumhuriyet Bayramı | 2,585 | **-0.0112** |
| 2026-03-21 | Ramazan Bayramı | 3,688 | **-0.0071** |
| 2025-06-08 | Kurban Bayramı | 2,036 | **-0.0009** |
| 2025-05-01 | Emek ve Dayanışma Günü | 1,987 | **+0.0167** |
| 2025-01-01 | Yılbaşı | 1,912 | **+0.0493** |
| 2025-05-19 | Atatürk'ü Anma, Gençlik ve Spor Bayramı | 2,007 | **+0.0532** |
| 2026-03-20 | Ramazan Bayramı | 3,690 | **+0.0544** |
| 2026-03-19 | Ramazan Bayramı (saat 13.00'ten) | 3,697 | **+0.1142** |
| 2026-01-01 | Yılbaşı | 3,381 | **+0.1185** |

Tatil etkisi ortalama -0.0516, en guclusu -0.2888 (2025-06-06, Kurban Bayramı). Buyukluk hafta sonu etkisiyle ayni mertebede, yani modellenmeye deger.

Bayram tarihlerinin yil-uzeri kaymasi (Kurban 2025-06-06 -> 2026-05-27) dogrulandi; `holidays` paketinin 2025-2026 tarihleri Diyanet kaynakli onaylanmis aralikta.
