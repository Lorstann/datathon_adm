# Dış veri kaynakları

Çekim tarihi: 2026-08-21. Ham yarışma verisi `data/raw/` içinde durur ve değiştirilmez.

## 1. Open-Meteo Historical Weather API (ERA5-Land)

- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
- **Betik:** `scripts/04_fetch_weather.py` → `data/external/weather/era5_daily.parquet`
- **Pencere:** 2025-01-01 – 2026-07-31
- **Değişkenler:** `temperature_2m_mean/max/min`, `apparent_temperature_mean`, `relative_humidity_2m_mean`, `precipitation_sum`, `wind_speed_10m_max`, `shortwave_radiation_sum`, `sunshine_duration`
- **Zaman dilimi:** `Europe/Istanbul`
- **Lisans:** [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Atıf: Copernicus Climate Change Service (ERA5-Land) via Open-Meteo.
- **Kota:** ücretsiz katman 10.000 çağrı/gün; bu çekim ~47 istek.
- **Izgaraya oturma:** ERA5-Land ~0.1° / ~11 km. Metropol İzmir ilçelerinin bir kısmı aynı hücreye düşer; bu beklenen bir çözünürlük kaybıdır, mikro-klima iddiası yoktur.
- **Retrospektif öngörü notu:** arşiv gerçekleşmiş reanaliz döndürür. Nisan–Temmuz 2026 ufku geçmişte olduğu için erişilebilir; gerçek bir 4 ay ileri operasyonel tahminde bu bilgi olmaz. Yarışma dış veriyi açıkça teşvik ediyor. İklim normalleri ablasyonu (`weather_features(..., normals=True)` veya 2015–2024 arşivi) jüri sunumunda havanın katkısını ayırır.

## 2. İlçe koordinatları

- **Dosya:** `src/external/coords.py` (`LOCATION_COORDS`)
- **Kapsam:** veri setindeki 47 `lokasyon` dizesinin tamamı, 2013–14 büyükşehir reorganizasyonu sonrası Yunusemre, Şehzadeler, Menderes, Karaburun dahil.
- **Kaynak:** OpenStreetMap / Wikipedia ilçe merkezleri. Topluluk gist'indeki Buca ve Karşıyaka il-merkezi kopyaları **kullanılmadı**.
- **Lisans:** OSM ODbL (koordinat), kod BSD/MIT projeye ait.

## 3. Türkiye resmi tatilleri

- **Kütüphane:** `holidays==0.103`, `country="TR"`, `categories=(PUBLIC, HALF_DAY)`, `language="tr"`
- **Bayram tarihleri:** 1936–2032 Diyanet kaynaklı doğrulanmış aralıkta, tahmini değil.
- **Elle çapraz kontrol (test ufku):**
  - 2026-04-23 Ulusal Egemenlik ve Çocuk Bayramı
  - 2026-05-01 Emek ve Dayanışma
  - 2026-05-19 Atatürk'ü Anma, Gençlik ve Spor
  - 2026-05-26 Kurban arife; 2026-05-27–30 Kurban 1–4
  - 2026-07-15 Demokrasi ve Millî Birlik
- **Ramazan ayları (Diyanet):** 2025-03-01–29; 2026-02-19–03-19. Test ufkunda Ramazan yok.
- **Yıl-üzeri kayma:** Kurban 2025-06-06 → 2026-05-27 (~10 gün). `bayrama_kalan_gun` ve `hist_hijri_mean` bu kaymayı taşır.
- **1 Nisan 2025** Ramazan Bayramı 3. günüdür; mevsimsel naif referans penceresinin ilk günü maskelenmelidir.

## 4. EPİAŞ / kamuya açık gerçekleşen tüketim — sızıntı taraması

ASHRAE GEPIII'de public test setinin bir kısmı internetten kazınabiliyordu. Bu yüzden GDZ/ADM ve EPİAŞ'ın Nisan–Temmuz 2026 için **trafo-gün** gerçekleşen tüketim yayımlayıp yayımlamadığı kontrol edildi (2026-08-21).

- EPİAŞ Şeffaflık Platformu tüketimi **il bazında, aylık** yayımlar (`/v1/consumption/data/consumption-quantity`). Dağıtım bölgesi / abone grubu kırılımı var; **ilçe ve trafo kırılımı yok**.
- Gerçek zamanlı tüketim sistem yükü (ulusal, saatlik) yarışma birimiyle (trafo-gün) eşleşmez.
- **Sonuç:** skorlanan birimde kamuya açık gerçekleşen hedef bulunamadı. İl-aylık EPİAŞ serisi ensemble üyesi olarak denenebilir ama sızıntı değildir; bu sürümde kullanılmadı.

## 5. P0 dış veri (yüklendi)

| Kaynak | Dosya | Durum |
| --- | --- | --- |
| EPİAŞ il-aylık tüketim | `data/external/epias/consumption_quantity_izmir_manisa.csv` | 2024-01–2026-07, İzmir+Manisa, 6 profil grubu |
| TÜİK ADNKS ilçe nüfus | `data/external/tuik/ilce_nufus.csv` | 2024–2025, 47 lokasyon eşlemesi %100 |

Modüller: `src/external/epias.py` (lag-1 + YoY; aynı ay gerçekleşen kullanılmaz), `src/external/tuik.py` (`log_nufus`, `log_guc_per_nufus`, `is_coastal_tourism`). `assemble()` otomatik ekler.

## 6. Kullanılmayan alternatifler

- NASA POWER: 0.5° ızgara, kimlik doğrulamasız. ERA5-Land daha ince; çapraz kontrol için saklandı.
- Meteostat: istasyon boşlukları. GDZ 2023'te kullanılmıştı.
- MGM: açık API yok, toplu veri ücretli — tekrarüretilemez.
