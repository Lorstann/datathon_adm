# Boru hatti duzeltmesi: dogrulama metrigi + cok-origin egitim cercevesi

Uretim tarihi: 2026-08-21

Denetim dort yapisal sorun buldu. Hepsi giderildi; asagida her biri icin
olcum ve sonuc var.

## 1. `is_cold` dogrulama penceresindeki dogal cold satirlari kaciriyordu

`make_fold` yalnizca maskelenen trafolari cold sayiyordu. Dogrulama
penceresinde origin sonrasi ilk kez gorunen trafolarin da gecmisi yok, yani
gercek test'teki cold trafolarla ayni durumdalar, ama warm sayiliyorlardi.

Duzeltmeden onceki fold profilleri:

| fold | dogrulama satiri | maskeli cold | gecmissiz ama cold sayilmayan | toplam gecmissiz |
| --- | --- | --- | --- | --- |
| A_mevsim | 249,588 | 17.7% | 8.3% | 26.0% |
| B_guncel | 406,124 | 16.9% | 15.2% | 32.1% |
| C_ara | 356,648 | 15.1% | **24.7%** | **39.8%** |

Yani `rmsle_warm` fold C'de dortte bir oranda cold satir iceriyordu ve
`rmsle_blend` bunun uzerine bir de %22,16 agirlikla maskeli cold ekliyordu.
Dogrudan kanit: fold C'de `mean_all` seviyesi warm satirlarda 1.4667
gorunuyordu, ama tahmincinin gercekten eslesebildigi satirlarda (kapsama
0.71) 0.9516.

**Duzeltme** (`src/validation/split.py`): `is_cold = history_days == 0`.
Maskeleme kotasi da dogal cold trafolarin *ustune* degil *yerine* gelecek
sekilde hesaplaniyor, yoksa fold'un cold orani hedefi asiyordu.

Sonuc: eski model (`level_shape_C`) hic degismeden blend 1.3359 -> **1.2293**.
Model iyilesmedi; metrik duzeldi. Bu yuzden `reports/09`-`19` arasindaki
kapi kararlari (router, cold-shape, mask/augment, hiyerarsi, YoY, dis veri)
kirli bir metrik uzerinde alinmisti ve tekrar edilmeden gecerli sayilmamali.

## 2. Egitim satirlari test satirlariyla ayni sekilde uretilmiyordu

Eski boru hatti egitim satirlarini `add_lagged_history` ile uretiyordu:

- Egitimde `hist_mean_7` = satirdan bir onceki 7 gunun ortalamasi, yani bir
  adim ilerisi. Gunluk seride medyan `|Δlog1p|` sadece 0.057, dolayisiyla bu
  kolon egitimde neredeyse kahin. Gain siralamasinda 2. sirada
  (`reports/08_importance.md`).
- Test'te ayni kolon origin'de donuyor ve 1-122 gun bayatliyor.
- `horizon_day` egitimde **negatif** (-455..0), test'te **pozitif**
  (+1..+122). Agac ufka bagli bozulmayi hic ogrenemiyordu.

**Duzeltme** (`src/models/multiorigin.py`): egitim satirlari da (origin, ufuk)
ciftlerinden uretiliyor. Gonderim icin 45 gun arayla 8 origin, her birinin
sonraki 122 gunu egitim satiri; toplam 1,8M satir. Butun `hist_*` kolonlari
ve `horizon_day` artik egitimde ve tahminde ayni anlami tasiyor.

## 3. Seviye tahmincisi elde olanlarin en kotusuydu

C, varlik seviyesi olarak `hist_mean`'i (genisleyen ortalama) kullaniyordu.
Yalnizca seviye ile tahmin, warm satirlar, 122 gunluk ufuk, RMSLE:

| origin | genisleyen ortalama | son 7 gun | son 14 gun | son 28 gun | kahin sifir orani |
| --- | --- | --- | --- | --- | --- |
| 2025-11-30 | 0.8897 | **0.6723** | 0.6790 | 0.6783 | 0.5532 |
| 2025-09-30 | 0.9887 | 0.8256 | 0.8484 | 0.8843 | 0.7618 |
| 2025-06-30 | 1.0379 | 0.8490 | **0.8287** | 0.8675 | 0.6697 |

Gonderimin kendisi de bunu gosteriyordu: tahminleri seviyelere regresyon
`pred ≈ 0.72 + 0.59·genisleyen + 0.32·son14`, yani agirligin cogu yanlis
pencerede.

**Duzeltme** (`src/models/level_shape.WARM_LEVEL_CHAIN`): seviye artik bir
geri dusum zinciri, `hist_mean_7 -> 28 -> 91 -> hist_mean`. Zincir ablasyonu
(`reports/21_mo_tune.md`, 2 fold blend ortalamasi):

| zincir | blend | warm |
| --- | --- | --- |
| son 7 gun | **1.1249** | **0.7515** |
| son 14 gun | 1.1349 | 0.7696 |
| son 21 gun | 1.1377 | 0.7749 |
| genisleyen ortalama | 1.1637 | 0.8226 |

Ayni ablasyonda iki sey daha olculdu: 1500 agac 600'u gecmiyor (1.1363 vs
1.1349), ve seviye ofsetini kaldirip dogrudan `log1p` regresyonu yapmak acik
farkla kotu (1.2060) — ozellikler artik tutarli olsa bile ofset gerekli.

## 4. Egitim tarafinda olu ozellikler

Eski yolda `assemble(..., include_history=False)` yuzunden `hist_median`,
`hist_std`, `hist_coverage`, `hist_last_gap`, `hist_last_zero_streak`,
`hist_trend`, `hist_weekend_drop`, `hist_dow_0..6`, `hist_mean_365`,
`hist_yoy_mean`, `hist_hijri_mean` egitimde bastan sona NaN'di; `align` ile
kolon olarak duruyor ama agac hic bolmuyordu. Test tarafinda ise gercek
degerleri vardi.

Cok-origin yolunda egitim ve dogrulama ayni `entity_history_features`
ciktisini kullaniyor, yani bu kolonlarin tamami artik canli. Ayrica
`hist_mean_14` ve `hist_mean_21` pencereleri eklendi.

## Sonuc

3 fold, duzeltilmis metrik (`reports/20_multiorigin.md`):

| Model | A_mevsim | B_guncel | C_ara | ortalama blend |
| --- | --- | --- | --- | --- |
| level_shape_C (eski cerceve) | 1.3645 | 1.0674 | 1.2561 | 1.2293 |
| multiorigin_C | 1.3843 | **1.0229** | **1.2268** | **1.2113** |

warm bazinda: B 0.7441 -> 0.6593, C 0.8964 -> 0.8436. cold neredeyse
degismiyor (1.7878 -> 1.7875, 2.0731 -> 2.0715) — beklenen, cunku cold zaten
`reports/02`'deki metadata tavaninda.

**Fold A istisnasi.** A'nin origin'i 2025-03-31, veri 2025-01-01'de basliyor;
45 gun arayla tek bir kullanilabilir origin kaliyor ve egitim cercevesi
87,907 satira dusuyor (B/C'de 1,2M). Cok-origin orada 0.020 kotulesiyor. Bu
A'nin veri acligi, yontemin degil: gonderim origin'inde 8 origin ve 1,8M
satir var, yani A'nin rejimi gonderimi temsil etmiyor.

## Gonderim

`submissions/submission_multiorigin.csv` — `scripts/22_submit_multiorigin.py`,
8 origin, 1,8M egitim satiri, 600 agac, 3 tohum log1p uzayinda ortalanmis.

Mevsimsel rampa da duzeldi. Nisan -> Temmuz medyan `log1p` farki:

| | Nis -> Tem |
| --- | --- |
| eski gonderim | +0.209 |
| yeni gonderim | +0.384 |
| 2025 gercegi (ayni aylar) | +0.510 |

## Kalan acik uclar

1. **Sifir gecisleri.** Warm kare hatasinin %45'i 3,289 trafodan 20'sinden
   geliyor; hepsi sifir gecisi (gecmisi tamamen sifir olup devreye giren ya
   da tersi). Kahin sifir orani seviyeyi 0.67 -> 0.55 tasiyor, yani tavan
   var, ama gecisin kendisi gecmisten okunamiyor. `reports/02` karar 3
   (regresoru yalnizca sifir olmayan satirlarda egit + ayri siniflandirici)
   hala uygulanmadi.
2. **Fold A'nin veri acligi.** Kisa pencerede daha sik origin (20 gun
   araliksa 3 origin) A'yi karsilastirilabilir yapabilir; gonderimi
   etkilemedigi icin yapilmadi.
3. **`reports/09`-`19` kapilari.** Kirli metrikle alinmis kararlar; ozellikle
   dis veri (TÜİK/EPİAŞ) ve hiyerarsi denemeleri duzeltilmis metrikle tekrar
   olculmeli.
