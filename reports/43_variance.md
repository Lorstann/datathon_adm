# Varyans ayristirmasi: problemin nerede oldugu

Uretim tarihi: 2026-08-23

Bu analiz projede uc gun gecikmeyle yapildi ve yapilir yapilmaz o uc gunun
denemelerinin cogunun neden basarisiz oldugunu acikladi. Onceki EDA
(`reports/03_eda.md`) betimleyiciydi: mevsimsellik VAR, hafta sonu etkisi VAR,
hava etkisi VAR. Dogru ama yetersiz -- soru "var mi" degil, "ne kadar buyuk".

## 1. Hedef varyansi nereye dagiliyor

`log1p(tuketim)`, sifir olmayan satirlar, train 2025-01 - 2026-03.
Toplam varyans **2.6787**. Iki yonlu sabit etki ayristirmasi:

| bilesen | pay | yorum |
| --- | --- | --- |
| trafolar arasi (entity) | **91.9%** | Problem buradan ibaret |
| ortak zaman (gun etkisi) | **1.4%** | Mevsim + hava + takvim, TUM trafolarda ortak |
| trafo x gun etkilesimi | **7.1%** | Trafoya ozgu sapma; kismen ogrenilebilir |

**Sonuc 1.** Butun takvim ve hava muhendisligi, toplam varyansin %1.4'unu
paylasan bir kutuya sikismis durumda. `reports/07_ablation.md`'nin "hava
notr" bulgusu, mevsim-esli agirliklandirmanin LB'de kaybetmesi
(`reports/39`), v3 hava carpanlarinin reddi (`reports/33`) -- ucu de ayni
yapisal gercegin sonucu, ayri ayri talihsizlik degil.

**Sonuc 2.** Model tasariminin dogru onceligi seviyeydi. `hist_mean_7`
zincirine gecmek (LB 1.14554 -> 1.09347) neden bu kadar buyuk kazandirdi:
varyansin %92'sinin oturdugu yere dokunuyordu.

## 2. Trafo seviyesini ne belirliyor

Trafo basina ortalama `log1p` (n=5.046), aciklanan varyans orani:

| anahtar | R2 |
| --- | --- |
| `guc` | 0.490 |
| `lokasyon` | 0.279 |
| `guc` + `lokasyon` | **0.577** |

**Cold trafolar icin ulasilabilir tavan budur.** Cold trafonun gecmisi yok,
yani seviyesini yalnizca statik nitelikler ima edebilir; trafo seviyesi
varyansinin %42.3'u tanim geregi erisilemez.

Bunun sayisal karsiligi: trafo seviyesi varyansi 0.919 x 2.6787 = 2.462.
Erisilemez kisim 0.423 x 2.462 = 1.041 (RMSE 1.02), ustune trafo x gun
gurultusu 0.189 (RMSE 0.435) -> toplam ~1.23, **RMSLE ~1.11**.

`reports/02_cold_start_ceiling.md` bu sayiyi tamamen bagimsiz bir yoldan
(kahin sifir kapisi ile cold RMSLE 1.09-1.14) bulmustu. Iki yontemin ayni
yere cikmasi tavanin gercek oldugunu gosteriyor.

## 3. Hava gercekte ne kadar aciklayabilir

Trafo x gun artigi (varyans 0.1892). Trafo-bazli CDD/HDD egimleri bu artigin
**%53.1'ini** acikliyor (n=2.428 trafo, >=200 gozlem). Bu ORNEKLEM-ICI bir
uyum, yani ust sinir.

Toplam varyanstaki karsiligi: **%3.75**.

Yani mukemmel bir trafo-bazli hava modeli bile toplam varyansin %3.75'inden
fazlasina dokunamaz -- ve gercek disi bir ust sinirdan bahsediyoruz.

### Global hava degiskenleri neden hicbir sey yapmiyor

CDD egimi, trafolar arasi:

| olcu | deger |
| --- | --- |
| ortalama | -0.0018 |
| t / p | -0.9 / **0.36** |
| std | 0.0971 |
| %10 - %90 | -0.0788 ... +0.0900 |

Ortalama trafonun sicaklik tepkisi **istatistiksel olarak sifirdan farksiz**.
Tepki trafo bazinda var ama isaretleri karisik ve toplandiginda birbirini
goturuyor. Global bir CDD kolonu bu yuzden bilgi tasimiyor; yalnizca
trafo-bazli katsayi tasiyabilir (`hist_cdd_slope`, v2'de mevcut).

## 4. Hafta sonu: ayni desen

Trafo bazinda hafta sonu - hafta ici farki (n=4.621):

| olcu | deger |
| --- | --- |
| ortalama | -0.0187 (t=-5.6, p=2.8e-08) |
| **medyan** | **+0.0225** |
| std | 0.2289 |
| dusen trafo | %33 |
| **artan trafo** | **%67** |

Ortalama istatistiksel olarak anlamli ama YANILTICI: trafolarin ucte ikisi
hafta sonu ARTIYOR. Toplam etkiyi asagi ceken, azinliktaki buyuk sanayi
trafolari. `reports/03_eda.md` "hafta sonu dususu -0.0573" derken bu
heterojenligi ortalamanin altina gizlemisti.

Model sonucu: `is_weekend` tek basina neredeyse degersiz; degerli olan
`hist_dow_rel_*` ve `hist_weekend_drop`, yani trafonun KENDI profili.

## 5. Bu analizin degistirdigi karar

Cold seviyemiz `global_z + log_guc`, yani yalnizca `guc` (R2 0.490).
`lokasyon`'un ekledigi 0.087 kullanilmiyor. Cold satirlar test'in %22'si ve
kare hatanin ~%45'i. `reports/02` lokasyon grup ortalamalarini reddetmisti
ama o olcum hem kirli `is_cold` tanimiyla hem de duzenlilestirilmemis grup
ortalamasiyla yapilmisti. `scripts/42_cold_lokasyon_level.py` bunu ampirik
Bayes ofsetiyle ve temiz metrikle yeniden olcuyor.

## 6. Ne yapilmamali

Bu ayristirmadan sonra asagidakiler oncelik degil, cunku hepsi %1.4'luk
kutunun icinde kaliyor:

- daha fazla hava degiskeni (nem/yagis/ruzgar/gunes zaten cache'de ve modelde)
- daha zengin takvim ozellikleri
- mevsimsellik agirliklandirmasi (LB'de olculdu ve kaybetti)
- lokasyon hiyerarsisini daha ince parcalama (`il`/`bolge`/`lokasyon_cat`
  zaten var; `ilce` bilgi olarak fazlalik)

Oncelik olan: trafo seviyesi tahmini (varyansin %92'si) ve cold trafolarda
statik nitelikten cikarilabilecek son damla (`guc` + `lokasyon`, tavan 0.577).
