# Ikinci tur: hata kirilimi, sifir kapisi, ozellik paketi v2

Uretim tarihi: 2026-08-22

## LB seyri

| gonderim | public LB | degisim |
| --- | --- | --- |
| `submission_c_tuned` (tur 1 oncesi en iyi) | 1.14554 | — |
| multiorigin C (metrik + egitim cercevesi + seviye) | 1.09347 | -0.05207 |
| + cold olu-trafo kapisi | 1.08074 | -0.01273 |
| + ozellik paketi v2 | **1.06899** | -0.01175 |

Toplam: **-0.07655**.

## Yontem: once olc, sonra kur

Bu turun tamami `reports/23_error_decomp.md`'den cikti. Kare hata (log1p
uzayi) dilimlere bolununce tablo netlesti:

| dilim | satir payi | kare hata payi | RMSLE |
| --- | --- | --- | --- |
| warm / sifir degil | %75-81 | %24-31 | 0.51-0.80 |
| warm / sifir | %2-4 | %5-18 | 1.9-2.1 |
| cold / sifir degil | %15-21 | %13-16 | 0.96-0.99 |
| **cold / sifir** | **%0.8-1.7** | **%42-51** | **6.7-6.9** |

Sifirlar satirlarin %4'u, hatanin %55-60'i. Bundan sonraki her deney bu iki
dilimden birine nisan aldi.

## Kabul edilenler

### 1. Cold olu-trafo kapisi (`src/models/zero_gate.py`)

Sifir, satir duzeyinde olay degil varlik duzeyinde durum: panele sonradan
giren 2.383 trafonun 2.130'unda hic sifir yok, 132'sinde satirlarin >%90'i
sifir, arasi bos. Yas ile degismiyor (her yas kovasinda ~%6), yani rampa
artifakti degil.

RMSLE log1p uzayinda L2 oldugu icin "olu ya da canli" karisiminin optimal
tahmini `(1-p) * log-seviye`. p lokasyon bazinda, ampirik Bayes ile globale
cekilerek kestiriliyor (ham aralik 0.000 Alasehir/Gordes/Kiraz - 0.234 Urla,
genel 0.055).

k=0.75: fold farklari -0.0189 / -0.0025 / -0.0130 (3/3 ayni isaret).

### 2. Ozellik paketi v2

| ozellik | gerekce |
| --- | --- |
| `hist_mom_{7_28,7_91,28_91,7_365}` | ortalamaya donus; corr(kayma, m7-m91) = -0.31, 18.198 trafo-ufuk cifti |
| `hist_yoy_gap` | gecen yil ayni pencere, varligin genel seviyesine gore |
| `hist_q25/q75/iqr_91` | ortalamanin tasiyamadigi saglam olcek |
| `hist_dow_rel_0..6` | hafta gunu profili kendi seviyesine gore (sanayi/konut imzasi) |
| `hist_cdd_slope`, `hist_hdd_slope` | trafo basina sicaklik duyarliligi (OLS, >=30 gozlem). Ufuk sogutma rampasi; ayni lokasyondaki iki trafo ayni havayi gorur, tepkileri farklidir |
| `hist_rel_peer` | akranlara gore konum |

Fold farklari -0.0068 / -0.0022 / -0.0067 (3/3). warm 0.7967 -> 0.7867.

## Reddedilenler (hepsi olculdu)

| deneme | sonuc | neden |
| --- | --- | --- |
| Harmanlanmis seviye 0.7*m7+0.3*m91 | -0.0127 / -0.0015 / **+0.0113** | Izole seviye RMSE'sinde en iyi (0.6093 vs 0.6242) ama fold C'de isaret degistiriyor; sekil modeli donusun bir kismini zaten topluyor |
| Origin yogunlugu + LGBM kapasitesi | en iyi +0.0012, isaretler celisiyor | Kapasite doymus. `reports/25_capacity.md` |
| Canli-only cold capasi | 1.1999 -> 1.2003 | +0.10/+0.16 artik hata degil, sifir kutlesi karsisinda dogru L2 uzlasmasi. `reports/26_cold_anchor.md` |
| CatBoost / XGBoost harmani | lgbm 1.1941 en iyi | Harman yalniz fold C'de kazaniyor. XGBoost sarmalayicisi kategorikleri dusuruyor (lokasyon_cat ilk 3 gain'de), yani adil olmayan bir karsilastirma; harman zaten zarar verdigi icin ayrica ayarlanmadi. `reports/28_diversity.md` |
| Olu-trafo siniflandiricisi | AUC 0.584 | Lokasyon tek basina 0.616. Batch kumelenmesi train'de guclu (LOO 0 -> %3.6, >0.67 -> %33) ama TASINMIYOR: cold varligin es-girisleri de cold |
| Satir duzeyinde `lag364` | uc fold'da tam olarak 0.0000 | **Olculemedi**, faydasiz degil: egitim kapsamasi her fold'da %0. `reports/29_seasonal_lag.md` |

## Kalan tavan ve neden ulasilamiyor

`reports/24_zero_shrink.md`'deki kahin satiri: cold trafolarin hangisinin olu
oldugu bilinseydi blend 1.2113 -> **0.869** olurdu. -0.34'luk bir tavan.
Elimizdeki lokasyon onselinin yakaladigi kisim ~%3. Siniflandirici sonucu
(AUC 0.584 < lokasyonun 0.616) bu tavanin veriden ulasilamaz oldugunu
gosteriyor: cold trafo hakkinda `guc`, `lokasyon` ve panel giris tarihi
disinda hicbir sey bilmiyoruz.

## Tabular foundation model (TabPFN vb.) hakkinda

Denenmedi; gerekce olculmus:

1. **Olcek ters yonde.** Bu ailenin baglami ~10k satir mertebesinde; bizde
   1,8M egitim satiri ve **714.688 tahmin satiri** var. Cikarim maliyeti
   kabaca O(n_egitim x n_test).
2. **Onsel uymuyor.** TabPFN bagimsiz-esdagilimli tablo satiri varsayar; bu
   bir panel tahmini (varlik durumu, origin'de donmus gecmis, 1-122 gunluk
   ufuk). Buraya kadarki butun kazanc tam da bu yapiyi kodlamaktan geldi.
3. **Darbogaz ogrenici degil.** `reports/25_capacity.md`: iki kat satir, iki
   kat yaprak, iki kat agac, dusuk lr -- hepsi gurultu icinde. Kapasite hicbir
   sey satin almiyorsa kisit bilgide, model sinifinda degil.

## Sonraki tur icin

1. `reports/09`-`19` arasindaki kapi kararlari hala kirli metrikle alinmis
   durumda; ozellikle dis veri (TÜİK/EPİAŞ) ve hiyerarsi denemeleri
   duzeltilmis metrikle tekrar olculmeli.
2. XGBoost sarmalayicisinin kategorikleri dusurmesi (`fit_xgboost`,
   `select_dtypes`) gercek bir eksik; adil bir cesitlilik denemesi once bunu
   duzeltmeli.
3. Fold A'nin veri acligi (tek origin, 88k satir) uc fold ortalamasini
   kotumser yapiyor. LB kazanci uc turda da CV kazancindan buyuk cikti
   (-0.052 vs -0.018; -0.013 vs -0.011; -0.012 vs -0.005).
