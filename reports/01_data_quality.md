# Veri Kalitesi Denetimi

Uretim tarihi: 2026-08-21 15:53

## Boyut ve kapsam

- **train**: 1,226,237 satir, 5,344 trafo, 2025-01-01 - 2026-03-31 (455 tekil gun)
- **test**: 714,688 satir, 7,036 trafo, 2026-04-01 - 2026-07-31 (122 tekil gun)

## Hedef (`tuketim`) degerleri

- NaN: **0** (0.000%)
- Negatif: **0** (0.000%)
- Tam sifir: **57,536** (4.692%)
- 0 < x < 1: **4,478** (0.365%)

- Sifir satirlarin RMSLE kaldiraci: gercek 0'a karsi 500 kWh tahmini tek satirda 38.6 kare-log hata uretir.

## Sifirlarin dagilimi: gurultu mu de-enerjizasyon mu

- Hic sifir icermeyen trafo: **4,792** / 5,344
- En az bir sifir iceren trafo: **552**
- TAMAMI sifir olan trafo: **298**
- Sifir orani > 0,5 olan trafo: **354**
- Tamami sifir trafolarin kapsadigi satir: **38,767** (67.4% tum sifirlarin)

> **Tasarim sonucu.** Sifirlar dagilmis gurultu degil, trafo duzeyinde yogunlasmis: sifirlarin buyuk kismi kalici olarak de-enerjize trafolardan geliyor. Bu, sifir kapisini satir duzeyinde bir olay tahmini yerine **varlik duzeyinde bir durum siniflandirmasi** yapiyor. Warm trafolarda durum gecmisten neredeyse kesin okunur; risk tamamen cold trafolarda, cunku orada durumu yalnizca statik nitelikler ve panel uyeligi desenleri ima edebilir.

- Vekil olcum: train icinde sonradan devreye giren 3,147 trafonun ilk 30 gunundeki sifir orani **6.14%** (genel oran 4.69%). Yeni devreye alinan trafolarda sifir riski farkli; cold segment tahmininde bu oran referans alinacak.

## Tekrarli (trafo, gun) ciftleri

- train: **0**
- test: **0**

## Statik nitelik tutarliligi

- `guc` birden fazla degere sahip trafo: **0**
- `lokasyon` birden fazla degere sahip trafo: **0**
- Ikisi de 0 ise test donemi statik degerini egitim satirinda kullanmak ileri bakis yaratmaz.

## Lokasyon hiyerarsisi

- Tekil `lokasyon` degeri: **47**
  - 2 seviyeli: 17 tekil deger
  - 3 seviyeli: 30 tekil deger
- Trafo sayisi il bazinda:
  - İZMİR: 5,580
  - MANİSA: 1,788
- Tekil bolge: 20, tekil ilce: 30
- `ilce` seviyesi olmayan trafo: **1,788**
- Derinlik il bazinda:
  - MANİSA, 2 seviye: 1,788
  - İZMİR, 3 seviye: 5,580

> **Tasarim sonucu.** Manisa kayitlari `IL>BOLGE` formatinda, yani ilce seviyesi hic yok; Izmir kayitlari `IL>BOLGE>ILCE`. Bu yuzden gruplama ve tabakalama anahtari `ilce` degil **`lokasyon`** (47 tekil deger): her iki ilde de mevcut olan en ince cozunurluk. `ilce` uzerinden tabakalamak 1.788 Manisa trafosunun tamamini tek bir NA tabakasinda cokertirdi.

## `guc` dagilimi

- Tekil deger sayisi: 42
- Aralik: 40 - 35900 kVA
- En yaygin 10 deger (trafo sayisi):
  - 400 kVA: 1,523
  - 1000 kVA: 1,340
  - 250 kVA: 1,177
  - 630 kVA: 824
  - 160 kVA: 810
  - 1250 kVA: 786
  - 100 kVA: 480
  - 50 kVA: 189
  - 1600 kVA: 59
  - 800 kVA: 35
- **>= 2600 kVA: 38 varlik** (10-36 MVA). Bunlar dagitim trafosu degil, dagitim merkezi sinifi; ayri bir guc bandinda tutuluyor.

Kullanilan guc bandlari ve trafo sayilari:

| Band | Trafo | Test satiri |
| --- | --- | --- |
| <100 | 227 | 21,397 |
| 100-160 | 488 | 48,797 |
| 160-250 | 814 | 80,755 |
| 250-400 | 1,177 | 117,110 |
| 400-630 | 1,533 | 153,245 |
| 630-1000 | 864 | 83,271 |
| 1000-1600 | 2,136 | 197,752 |
| 1600-2600 | 91 | 8,451 |
| 2600+ | 38 | 3,910 |

## Panel giris ve cikis desenleri

- Yalnizca train'de: **332** trafo
- Yalnizca test'te (cold start): **2,024** trafo
- Her ikisinde: **5,012** trafo
- Cold-start test satiri: **158,369** / 714,688 = **22.16%**

Gecmis uzunluguna gore test dagilimi:

| Segment | Trafo | Test satiri | Pay |
| --- | --- | --- | --- |
| 0_cold | 2,024 | 158,369 | 22.16% |
| 1_1-29g | 747 | 65,582 | 9.18% |
| 2_30-89g | 590 | 63,244 | 8.85% |
| 3_90-269g | 1,586 | 181,219 | 25.36% |
| 4_270g+ | 2,089 | 246,274 | 34.46% |

## Trafo ici gun bosluklari (train)

- Ortalama bosluk orani: 0.0472
- Bosluksuz trafo: 4,104
- Bosluk orani > 0,5 olan trafo: 159

## 2026-05-11 toplu giris artifakti

Test'te ilk gorulme tarihine gore trafo sayisi (en yogun 8):

| Tarih | Trafo | Bunlarin cold olani |
| --- | --- | --- |
| 2026-04-01 | 3,928 | 1 |
| 2026-05-11 | 2,222 | 1,326 |
| 2026-05-03 | 141 | 105 |
| 2026-04-30 | 119 | 114 |
| 2026-05-07 | 102 | 88 |
| 2026-05-13 | 54 | 36 |
| 2026-05-30 | 30 | 28 |
| 2026-07-01 | 20 | 19 |

Gunluk test satir sayisindaki en buyuk sicramalar:
- 2026-05-11: +2370 satir (toplam 6,630)
- 2026-05-03: +141 satir (toplam 4,190)
- 2026-04-30: +121 satir (toplam 4,052)
- 2026-05-07: +91 satir (toplam 4,278)
- 2026-05-13: +54 satir (toplam 6,691)

Train gunluk satir sayisi: 2,041 - 4,429 (ilk gun 2,059, son gun 3,929). Panel buyuyor.

2026-05-11 grubundaki **896 trafo train'de zaten var**. Train gecmislerinin bittigi tarih:
- 2025-01: 5 trafo
- 2025-02: 2 trafo
- 2025-04: 1 trafo
- 2025-05: 5 trafo
- 2025-06: 30 trafo
- 2025-07: 2 trafo
- 2025-08: 144 trafo
- 2025-09: 131 trafo
- 2025-10: 5 trafo
- 2025-11: 7 trafo
- 2025-12: 11 trafo
- 2026-01: 18 trafo
- 2026-02: 19 trafo
- 2026-03: 516 trafo

Bunlarin **516**'i Mart 2026'da hala aktif, yani test'te 2026-04-01 - 2026-05-10 arasi 40 gunluk bir bosluk var ve sonra geri geliyorlar. Gercek bir devreye alma degil, **veri teslim artifakti**: test paneli iki partide olusturulmus. Modelleme sonucu: bu trafolarin gecmisi tam ve kullanilabilir; 2026-05-11 tarihini bir rejim degisimi sinyali olarak yorumlamak hata olur. Buna karsilik ayni gun giren 1,326 cold trafo icin gecmis gercekten yok.

## `id` format kontrolu

- `id` != `tanim_tarih` olan satir: **0**
- sample_submission satir sayisi: 714,688
- sample_submission id kumesi test id kumesine esit mi: **True**
- sample_submission id sirasi test ile ayni mi: **True**
