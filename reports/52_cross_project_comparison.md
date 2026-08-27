# Iki projenin karsilastirmasi ve 1.00 altina inmenin aritmetigi

Uretim tarihi: 2026-08-27

Ayni yarismaya iki bagimsiz kod tabani girdi: bu proje (LB **1.06713**) ve
`best-solution` (LB **1.10672**). Bu rapor ikisini karsilastiriyor, ikisinin
bagimsiz olarak ayni yere cikan olcumlerini yan yana koyuyor ve 1.00 altina
inmek icin ne kadar kazanc gerektigini sayiyla veriyor.

## 1. Neden `best-solution` 0.040 geride

Tek yapisal sebep, ve bu projenin `reports/22_pipeline_fix.md` madde 2'de
zaten duzelttigi sey: **egitim satirlari test satirlariyla ayni sekilde
uretilmiyor.**

`best-solution/data/processed/matrices/final/` uzerinde olculdu:

| kolon | train'de trafo basina farkli deger | test'te |
| --- | --- | --- |
| `last_nonzero_consumption` | ortalama 7.85 (en fazla 15) | **1.00** |
| `transformer_mean_consumption` | ortalama 7.99 | **1.00** |

Egitimde gecmis ozellikleri satir ilerledikce guncelleniyor; testte
2026-03-31'de donuyor ve 1-122 gun bayatliyor. 113 kolonun hicbiri ufuk
tasimiyor, yani `horizon_day` karsiligi da yok. Bu projede ayni duzeltmenin
olculmus karsiligi -0.05207'ydi (1.14554 -> 1.09347). Iki proje arasindaki
fark 0.0396. Buyukluk ortusuyor.

Ikinci fark: `best-solution` fold secimini `seasonal_twin`'e (origin
2025-03-31, gecmis yalnizca 3 ay) dayandirdi. O fold'un cold-start skoru
2.0757; ayni yapilandirma `recent` fold'unda (11 ay gecmis) 1.8656 veriyor.
Yani orada "cold-start kotu" diye okunan sey fold'un gecmis acligiydi. Bu
projenin fold A hakkinda tekrar tekrar yazdigi seyin aynisi.

## 2. Iki projenin bagimsiz olarak ayni yere ciktigi bulgular

| bulgu | bu proje | best-solution |
| --- | --- | --- |
| sifir satirlar kare hatanin cogunlugu | %55-60 (`23_error_decomp`) | %49.0 (`18_error_concentration`) |
| hata bir avuc trafoda yigilmis | en kotu 20/4,637 = %34.4 | en kotu 20/4,640 = %26.6 |
| cold seviyesi statik nitelikle sinirli | guc+lokasyon R2 0.577 (`43_variance`) | guc prior 1.8703 vs model 1.8656, lokasyon eklemek kotulestiriyor |
| trafo-arasi varyans hakim | %91.9 | seen: seviye 0.5716 / gun 0.4128, gun bileseni kahin tabaninda (0.4222) |
| olu cold trafo tahmin edilemiyor | siniflandirici AUC 0.584, lokasyon 0.616 | bu raporda yeniden olculdu, asagida |

Iki bagimsiz boru hatti, farkli fold tasarimlariyla, ayni tavanlara carpti.
Tavanlar gercek.

## 3. Olu cold trafo: ucuncu kez, daha zengin ozelliklerle olculdu

`reports/30`'un AUC 0.584 sonucu bugun bagimsiz olarak tekrarlandi. Panele
sonradan giren 2,347 trafo, olu tanimi ilk 122 gunde sifir orani > 0.9, taban
oran %5.67. Giris tarihine gore %70/%30 bolundu (test 701 trafo, olu orani
%4.99). Ozellikler: `log_guc`, giris ayi, giris gunu, panelde gorulen gun
sayisi, ayni lokasyonda +-30 gun icinde giren trafo sayisi (batch), o
lokasyonda daha once giren trafo sayisi ve onlarin olu orani.

| ozellik | AUC |
| --- | --- |
| lokasyonun onceki giris olu orani | **0.5727** |
| giris ayi | 0.5302 |
| giris gunu (doy) | 0.5277 |
| batch buyuklugu | 0.5201 |
| onceki giris sayisi | 0.5109 |
| `log_guc` | 0.5055 |
| panelde gun sayisi | 0.5055 |
| **hepsi birden, LightGBM** | **0.4588** |

Birlesik model tek ozelligin altina dusuyor, yani sinyal yok, gurultuye uyum
var. `reports/30`'un hukmu dogrulandi: lokasyon onselinden fazlasi veride
yok.

## 4. 1.00 altina inmek icin gereken kazanc

LB 1.06713 -> MSE 1.13877. Test satirlarinin %77.8'i warm, %22.2'si cold.
Fold B (origin 2025-11-30) warm 0.6564 / cold 1.7864 test paylarina
yeniden agirliklandirilinca MSE 1.0437 (RMSLE 1.0216) veriyor; LB ile arasindaki
0.0951'lik fark iki dilime oranli dagitilirsa test tarafinda warm ~0.686,
cold ~1.866 demektir.

Hedef 0.999 -> MSE 0.998, yani **-0.141 MSE**.

| tek basina kapatilacaksa | simdi | gereken | fark |
| --- | --- | --- | --- |
| cold RMSLE | 1.866 | **1.688** | -0.178 |
| warm RMSLE | 0.686 | **0.538** | -0.148 |

Bunlarin kahin tavanlarina gore nerede oldugu:

- **cold:** fold B'de kahin olu-kapisi cold'u 1.7875 -> 0.9710 tasiyor
  (`24_zero_shrink`), yani mevcut kapinin yakalayabilecegi toplam MSE
  kazanci 2.2524. Gereken 0.635. Yani **kahin kapinin %28'i**. Mevcut kapi
  (k=0.75) bunun ~%3'unu yakaliyor ve madde 3'teki AUC bunun neden
  buyumedigini gosteriyor.
- **warm:** trafo seviyesi bilinseydi warm ~0.42 olurdu. Gereken 0.538,
  yani **seviye bosluğunun %44'u**.

Ikisi de olculmus hicbir kaldiraca yakin degil. Bes turda kabul edilen en
buyuk dort degisiklik sirasiyla -0.052, -0.013, -0.012, -0.002 idi ve
toplamlari -0.079. Gereken tek kalemde -0.141.

## 5. Bugun ayrica elenenler

Ucu de ucuz, ucu de negatif; tekrar denenmesin diye yaziliyor.

| kontrol | sonuc |
| --- | --- |
| `tanim` normalizasyon eslesmesi (cold test trafosu train'de baska yazimla var mi) | 2,024 cold trafonun **0**'i eslesiyor |
| satir sirasi sizintisi (gun ici konum hedefi tahmin ediyor mu) | trafo-ici corr **-0.002** |
| bayat gecmisli trafolarin mevsimsel olmasi (sulama hipotezi) | 432 trafo / 34,306 test satiri; gecen yil ayni ay seviyesi ile genisleyen ortalama arasindaki fark **+0.034 log**, yani yok |

Ucuncusu ozellikle onemliydi: gecmisi 150-400 gun bayat 432 trafo test'in
%4.8'i ve neredeyse tamami May-Tem'de panele donuyor. Aylik ortalamalari
Ocak 4.4'ten Temmuz 7.4'e cikiyor gorunuyordu, ama bu **bilesim etkisi** --
her ayda farkli trafolar var. Trafo bazinda bakilinca gecen yil ayni ay
seviyesi ile genel seviye arasinda fark yok. `29_seasonal_lag`'in lag364
reddi bu populasyon icin de gecerli.

## 6. Sonuc

1. Kod tabani olarak bu proje devam etmeli; `best-solution`'in katkisi
   teshis, kodu bir tur geride.
2. Cold hatti kapali. Uc bagimsiz olcum (bu projenin siniflandiricisi,
   varyans ayristirmasi, bugunku zengin ozellikli tekrar) ayni seyi soyluyor:
   olu cold trafo bilgisi veride yok.
3. Kalan tek canli hat warm seviyesi ve orada gereken sey seviye bosluğunun
   %44'u. Bugune kadarki tum warm kazanclari toplami bunun altinda.
4. **Siralamanin kendisi dogrulanmali.** `reports/39_lb_arbitration.md`
   23 Agustos'ta lideri 1.02995 olarak ve "1.00 altinda takim yok" diye
   kaydetti. Bugun 0.99403 rapor ediliyor. Dort gunde 0.036'lik bir lider
   sicramasi ya yeni bir yapisal bulus ya da farkli bir tablo (private/
   degismis metrik) demek. Hedefi 1.00 alti diye sabitlemeden once hangisi
   oldugu bilinmeli.
