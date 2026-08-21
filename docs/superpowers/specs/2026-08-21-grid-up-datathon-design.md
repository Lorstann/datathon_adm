# Grid Up Datathon — Tasarım Dokümanı

Tarih: 2026-08-21. Yarışma bitişi: 2026-09-01 23:59. Metrik: RMSLE (düşük iyi).

## 1. Mimari

Kademeli inşa. Her aşama öncekini taban çizgisi olarak kullanır ve `experiments.csv`'ye kaydedilir.

```
Aşama 0  Naif tabanlar
Aşama A  Tek global LightGBM (log1p hedef, eksik geçmişe dayanıklı)
Aşama B  Cold kolunun ayrılması (statik + takvim + hava)
Aşama C  Cold kolunun çekirdeğinde seviye + şekil ayrıştırması
Aşama D  Bölümleme ensemble'ı + seed ortalaması
Aşama E  Segment bazlı post-processing kaydırması (koşullu)
```

### Aşama A: global model

Tüm satırlar tek modelde. Hedef `log1p(tuketim)`, amaç fonksiyonu L2. Trafo geçmişi özellikleri cold trafolar için NaN kalır; LightGBM NaN'ı yerel olarak dallandırır. Boşluk akran özellikleriyle doldurulur.

Eğitim sırasında **geçmiş maskelemesi**: her eğitim origin'inde trafoların bir kısmının geçmiş özellikleri düşürülür, böylece model lag'lere aşırı yaslanmayı öğrenmez.

**Çok-origin eğitim seti**: tek bir forecast origin yerine eğitim döneminde birden fazla origin örneklenir. Her origin için o origin'e kadarki geçmişten özellikler, sonraki 122 gün hedef.

### Aşama B: yönlendirici

Test satırları trafonun origin'e kadarki geçmiş uzunluğuna göre yönlendirilir. Eşik validasyonla seçilir; başlangıç adayı 0 gün (yalnızca tam cold) ve 30 gün.

- **Cold modeli:** yalnızca statik (`guc`, log(`guc`), güç bandı, il, bölge, ilçe) + takvim + hava + akran özellikleri. Tüm satırlar üzerinde eğitilir, yani 5.344 trafonun tamamından statik→tüketim eşlemesini öğrenir.
- **Warm modeli:** tam geçmiş özellikleri dahil.

### Aşama C: seviye + şekil

`log1p(tuketim) = seviye(trafo) + şekil(takvim, hava, mevsim, küme)`

- Seviye: trafonun log tüketim düzeyi. Warm için geçmişten ölçülür, cold için `guc` + lokasyon + akran regresyonundan tahmin edilir.
- Şekil: tüm panelden paylaşımlı öğrenilen çarpımsal profil.

RMSLE log uzayında hata olduğu için bu ayrıştırma metriğe birebir oturur ve cold-start problemini tek bilinmeyene (seviye) indirger.

## 2. Hedef ve metrik işleme

- `log1p(tuketim)` üzerinde L2 ile eğitim, `expm1` ile geri dönüş. **Duan smearing veya `exp(sigma^2/2)` düzeltmesi uygulanmayacak.** RMSLE zaten log uzayında RMSE olduğu için `E[log1p(y)|x]` Bayes-optimaldir; smearing çarpanı daima 1'den büyük olduğu için skoru kesinlikle kötüleştirir.
- Tahminler `[0, inf)` aralığına kırpılır.
- İkinci parametrizasyon: `log1p(tuketim) - log(guc)`. Metriğin kendi ölçeğinde yeniden parametrizasyon; `guc` aralığında varyansı homojenleştirir. Ensemble üyesi olarak ve ablasyon olarak değerlendirilir.

## 3. Validasyon

Her fold tam 122 günlük ufku sıfır hedef geri beslemesiyle tahmin eder.

| Fold | Train sonu (origin) | Doğrulama penceresi | Rol |
| --- | --- | --- | --- |
| A | 2025-03-31 | 2025-04-01 – 2025-07-31 | Mevsim-hizalı, en ağırlıklı. Kurban Bayramı içeride. |
| B | 2025-11-30 | 2025-12-01 – 2026-03-31 | En güncel popülasyon. Ramazan + Ramazan Bayramı 2026. |
| C | 2025-09-30 | 2025-10-01 – 2026-01-31 | Üçüncü origin, sağlamlık. |

**Model seçim kuralı:** fold'lar arası ortalama RMSLE **ve** standart sapma birlikte raporlanır. İki konfigürasyon arasındaki ortalama farkı, fold'lar arası standart sapmadan küçükse fark anlamlı sayılmaz ve daha basit / daha düşük std olan seçilir.

**Cold-start maskelemesi** (üniform rastgele değil):

1. Gerçek 2.024 cold trafonun (ilçe, güç bandı, satır sayısı bandı) ampirik ortak dağılımı çıkarılır.
2. Doğrulama trafoları bu ortak dağılıma göre tabakalı örnekle seçilir. Hedef oran: entity bazında ~%28,8, satır bazında ~%22,2.
3. Seçilen trafoların origin öncesi geçmişi eğitim çerçevesinden **tamamen silinir**.
4. Bir alt küme için doğrulama penceresinin erken satırları düşürülür (ufuk ortası devreye girme deseni).
5. Raporlanan üç sayı: `RMSLE_warm`, `RMSLE_cold`, gerçek test oranlarıyla satır-ağırlıklı harman.

**Segment kırılımı** (her deneyde raporlanır): geçmiş uzunluğu 0 / 1-29 / 30-89 / 90-269 / 270+ gün.

## 4. Özellik aileleri

1. **Takvim:** yıl, ay, ayın günü, haftanın günü, yılın günü, ISO hafta, hafta sonu, ay başı/sonu, yıllık ve haftalık sin/cos harmonikleri.
2. **Tatil:** resmi tatil bayrağı, arife yarım gün bayrağı, köprü günü, işaretli `bayrama_kalan_gun`, `bayram_gun_indeksi`, Ramazan ayı bayrağı, hicri hizalı yıl-üzeri lag.
3. **Hava:** günlük ortalama/max/min sıcaklık, hissedilen sıcaklık, nem, yağış, rüzgâr, kısa dalga ışınım, güneşlenme süresi. Türevler: `CDD = max(0, T-22)`, `HDD = max(0, 18-T)`, 3/7/14 günlük hareketli CDD/HDD toplamları, Savitzky-Golay yumuşatılmış sıcaklık, sıcaklık karesi.
4. **Statik:** `guc`, `log(guc)`, güç bandı (kuantil), il, bölge, ilçe, lokasyon derinliği.
5. **Trafo geçmişi (origin'e kadar):** son 7/28/91/365 gün log ortalaması ve medyanı, std, doğrusal trend eğimi, hafta günü profil sapmaları, ay profili, geçen yıl aynı ay ortalaması, sıfır-gün oranı, veri kapsama oranı, toplam geçmiş uzunluğu, son gözlemden origin'e geçen gün.
6. **Akran / hiyerarşik:** `ilce × guc bandı` out-of-fold hedef kodlaması (medyan), bölge ve il ortalamaları, statik uzayda kNN komşu profili, normalize profil üzerinde küme ataması.
7. **Transdüktif panel:** trafonun panelde ilk/son görülme tarihi, devreye alınma yaşı (gün), test'te kaç gün var, aynı ilçede o gün aktif trafo sayısı.
8. **Etkileşimler:** sıcaklık × güç bandı, CDD × ilçe, hafta sonu × yaz, tatil × ilçe.

## 5. Sıfır kapısı

RMSLE'de sıfır satırlar en yüksek kaldıraca sahip: gerçek 0'a karşı 500 kWh tahmini tek satırda `log(501)^2 ≈ 38,6` hata, tipik satırın ~40 katı.

İki aşamalı yapı: `tuketim` yaklaşık 0 mı (de-enerjize / uzun sabit seri / yeni devreye alınma rampası) sınıflandırıcısı, regresörü kapılar. Sınıflandırıcı olasılığı bir özellik olarak da regresöre verilir. Eşik validasyonda RMSLE'yi minimize edecek şekilde seçilir.

## 6. Ensemble

Bölümleme ekseni: global, il, güç bandı, ilçe × güç bandı, cold/warm. Öğrenici ekseni: LightGBM, CatBoost (yüksek kardinaliteli `tanim`/ilçe için), XGBoost. Hedef ekseni: `log1p` ve `log1p - log(guc)`. Ufuk ekseni: ufuk-agnostik model ve test ayı başına model.

Harmanlama ağırlıkları validasyonda, segment bazında optimize edilir. Seed ortalaması: en az 3 seed.

## 7. Post-processing (koşullu)

`{cold, warm} × il` bazında log uzayında tek bir kesişim kaydırması tahmin edilir. **Yalnızca** işareti üç fold'da da aynıysa ve büyüklüğü fold'lar arası standart sapmaya göre anlamlıysa uygulanır. Aksi halde uygulanmaz.

İklim normalleri ablasyonu: gerçek ERA5 ile 2015-2024 normalleri arasındaki RMSLE farkı raporlanır.

## 8. Deney takibi

`experiments.csv` kolonları: `exp_id`, `timestamp`, `model`, `target`, `feature_set`, `fold`, `rmsle_all`, `rmsle_warm`, `rmsle_cold`, `rmsle_blend`, `seed`, `lb_public`, `notes`.

Her satır tek bir (konfigürasyon, fold) çiftini temsil eder. Ortalama ve std ayrı bir raporlayıcı ile hesaplanır.

## 9. Tekrarüretilebilirlik

- Tek global seed sabiti (`src/config.py`), tüm rastgelelik oradan türer.
- `data/raw/` asla değiştirilmez. Türevler `data/interim/` ve `data/processed/`.
- Dış veri `data/external/` altına cache'lenir; çekme betiği repoda, kaynak/tarih/lisans `docs/external-data.md`'de.
- `requirements.txt` sürüm sabitli.
