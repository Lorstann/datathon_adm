# Trafo Bazlı Günlük Elektrik Tüketimi Tahmini (Grid Up Datathon)

**Araştırma sorusu:** GDZ/ADM dağıtım şebekesindeki 7.036 trafonun 2026-04-01 – 2026-07-31 arasındaki günlük aktif tüketimi (kWh), 2026-03-31'e kadarki geçmiş tüketim ile trafo statik nitelikleri (kurulu güç, lokasyon) ve dışsal takvim/hava değişkenlerinden, RMSLE metriğinde naif taban çizgilerini anlamlı ölçüde geçecek şekilde tahmin edilebilir mi?

**Arka plan / motivasyon:** Dağıtım şebekesi planlaması trafo seviyesinde yük öngörüsü gerektirir: aşırı yüklenme riski, yatırım planlaması ve şebeke yenileme takvimi bu tahmine dayanır. Yarışma bağlamında ise soru, private leaderboard sıralaması ve jüri değerlendirmesiyle ölçülüyor. Bu doküman kararların hangi gerekçeyle alındığını sabitler, böylece sonradan anlatı uydurmak yerine önceden bağlanmış olalım.

## Hipotezler

- **H0 (null):** Trafo geçmişi, statik nitelikler ve dışsal değişkenlerden kurulan öğrenilmiş model, RMSLE'de en iyi naif taban çizgisini (trafo bazlı mevsimsel naif / akran medyanı) geçmez.
- **H1 (alternatif, yönlü):** Öğrenilmiş global model, en iyi naif taban çizgisinden daha düşük RMSLE üretir; ve bu iyileşme hem geçmişi olan (warm) hem de geçmişi olmayan (cold) segmentte ayrı ayrı gözlenir.
- **H2 (cold-start alt hipotezi, yönlü):** Geçmişi olmayan trafolar için `guc` + `lokasyon` statik nitelikleri, tüketim seviyesini global sabitten anlamlı ölçüde daha iyi açıklar. Yani statik metadata bilgilendirici.
  - H2 yanlışlanırsa cold segment için indirgenemez gürültü kabul edilir ve strateji "yansız olmak"a (log uzayında medyanı doğru tutmak) kayar.

## Popülasyon ve analiz birimi

- **Popülasyon:** GDZ/ADM lisans bölgesindeki (İzmir ve Manisa) dağıtım trafoları.
- **Analiz birimi:** trafo-gün (`tanim` × `tarih`). Test setinde 714.688 birim.
- **Örneklem:** Yarışma tarafından verilen panel. Train 1.226.237 satır / 5.344 trafo / 2025-01-01 – 2026-03-31. Test 714.688 satır / 7.036 trafo / 2026-04-01 – 2026-07-31.
- **Panel dengesiz:** trafolar dönem boyunca sisteme girip çıkıyor. Train'de medyan geçmiş 170 gün; yalnızca 1.253 trafo 455 günün tamamına sahip.

## Anahtar değişkenler (operasyonelleştirilmiş)

| Rol | Kavram | Ölçü / kolon |
| --- | --- | --- |
| Hedef | Günlük aktif tüketim | `tuketim` (kWh), modelde `log1p(tuketim)` |
| Hedef (alt.) | kVA başına log yoğunluk | `log1p(tuketim) - log(guc)` |
| Statik | Kurulu güç | `guc` (kVA), `tanim` başına sabit olduğu doğrulandı |
| Statik | Konum | `lokasyon`, `IL>BOLGE>ILCE` hiyerarşisi, 47 tekil değer |
| Zaman | Gün | `tarih`, günlük çözünürlük |
| Geçmiş | Trafo seviyesi | Forecast origin'e kadarki pencere özetleri (son 28/91/365 gün ortalaması, hafta günü profili, geçen yıl aynı ay) |
| Dışsal | Hava | Open-Meteo ERA5-Land günlük sıcaklık/nem/yağış/rüzgâr/ışınım, HDD/CDD türevleri |
| Dışsal | Takvim | Türkiye resmi tatilleri, arife yarım günleri, Ramazan/Kurban Bayramı, hafta sonu |
| Transdüktif | Panel üyeliği | Trafonun panelde ilk/son görülme tarihi, devreye alınma yaşı, test'teki satır sayısı |

## Neyin cevap sayılacağı

Üç fold'un (A: Nis–Tem 2025, B: Ara 2025 – Mar 2026, C: Eki 2025 – Oca 2026) her birinde, tabakalı cold-start maskelemesi uygulanmış halde:

1. **H1 doğrulanır** eğer öğrenilmiş model, en iyi naif taban çizgisine göre satır-ağırlıklı RMSLE'de üç fold'un tamamında daha iyiyse ve iyileşme fold'lar arası standart sapmadan büyükse.
2. **H1 kısmen doğrulanır** eğer iyileşme yalnızca warm segmentte varsa. Bu durumda cold kolu ayrı ele alınır.
3. **H2 doğrulanır** eğer `ilce × guc bandı × ay × hafta günü` gruplaması, `log1p(tuketim) - log(guc)` artık varyansını global sabite göre anlamlı ölçüde düşürüyorsa. Ölçüt: açıklanan varyans oranı ve trafo kimliği eklendiğindeki tavana kıyasla konumu.

Kesin karar kuralları (eşikler, seed sayısı, model seçim prosedürü) tasarım dokümanında sabitlenir.

## Kapsam ve dışlamalar

- **Kapsam dışı:** saatlik/anlık tahmin, arıza veya kesinti tahmini, şebeke topolojisi modellemesi, nedensel çıkarım. Bu bir tahmin (forecasting) problemidir, etki tahmini değil.
- **Kapsam dışı:** foundation time-series modelleri (Chronos, TimesFM, Moirai). Gerekçe `docs/prior-work.md` içinde.
- **Kapsam dışı:** hiyerarşik uzlaştırmanın (MinT) ana yöntem olarak kullanılması. Metrik yalnızca alt seviyede satır bazlı RMSLE ve MinT ham ölçekte kareli hatayı optimize ediyor. Yalnızca ensemble üyesi olarak denenebilir.
- **Bilinçli tercih:** Nisan–Temmuz 2026 gerçekleşmiş (retrospektif) ERA5 hava verisi kullanılacak. Bu, gerçek bir 4 ay ileri operasyonel tahminde mevcut olmayacak bir öngörüdür. Yarışma dış veriyi açıkça teşvik ettiği için meşru, ama iklim normalleri ablasyonuyla katkısı ayrıca ölçülecek ve raporlanacak.
- **Bilinçli tercih:** transdüktif panel özellikleri (`test.csv`'den türetilen ilk görülme tarihi vb.) kullanılacak. Verilen veriden türediği için sızıntı değil, ama operasyonel bir yöntem olarak sunulursa bu not düşülmeli.

## Sızıntı denetimi kuralları

1. Hiçbir özellik, forecast origin'den sonraki bir zaman damgasının hedef bilgisini kullanamaz.
2. Trafo geçmişi özellikleri yalnızca origin'e kadarki veriden hesaplanır ve 122 günlük ufuk boyunca sabittir.
3. Her lag/rolling penceresi ya as-of-origin hesaplanır ya da en az ufuk uzunluğu (122 gün) kadar kaydırılır.
4. `shift`/`rolling` işlemleri daima `groupby('tanim')` sonrası ve tarihe göre sıralı yapılır.
5. Hedef kodlamaları out-of-fold üretilir.
6. Hava özellikleri yalnızca t anına veya öncesine ait olabilir; t+1 yasak.
7. Cold-start maskelemesinde seçilen trafonun geçmişi eğitim çerçevesinden tamamen silinir, yalnızca lag'leri NaN yapmak yeterli değildir.

## Prior-work taraması için açık sorular

- RMSLE altında geri dönüşüm biası: smearing uygulanmalı mı? (Cevap: hayır, gerekçe tasarım dokümanında.)
- Cold-start için metadata tabanlı küme atama yaklaşımlarının yayınlanmış başarı oranları.
- ASHRAE GEPIII gibi RMSLE tabanlı enerji yarışmalarında kazanan validasyon şemaları.
- Türkiye'de dini bayramların ve Ramazan'ın elektrik tüketimine etkisi; 2025 ve 2026 kesin tarihleri.
- Sayaç arkası güneş PV'sinin ölçülen net tüketim üzerindeki bastırıcı etkisi.
