# Literatür ve Kazanan Çözüm Taraması

Tarama tarihi: 2026-08-21. Kapsam: trafo/fider seviyesi yük tahmini, büyük panel için global forecasting, cold-start, RMSLE optimizasyonu, benzer Kaggle yarışmalarının kazanan çözümleri.

## 1. Mimari: global model tercihi

**Montero-Manso, P. & Hyndman, R.J. (2021). "Principles and algorithms for forecasting groups of time series: Locality and globality." *International Journal of Forecasting* 37(4), 1632–1653.** [DOI](https://doi.org/10.1016/j.ijforecast.2021.03.004) · [arXiv:2008.00444](https://arxiv.org/abs/2008.00444)

Üç sonuç doğrudan bizi ilgilendiriyor:

1. Global yöntemler (tüm serilere tek fonksiyon) lokal yöntemlerden **daha kısıtlayıcı değil**; seriler arası benzerlik varsayımı olmadan herhangi bir lokal tahmini üretebilirler.
2. Lokal yöntem karmaşıklığı seri sayısıyla büyür; global yöntem karmaşıklığı **sabittir**. 5.344–7.036 seri ile global model çok daha karmaşık olup yine de daha iyi genelleştirebilir.
3. Global modeller lokal olarak etkili olacaktan **çok daha uzun otoregresif hafıza** kullanmalı; bölümleme granülaritesi (tüm seriler → kümeler → tek seri) bir felsefe değil bir **karmaşıklık ayarı**.

**Karar:** tek global LightGBM/CatBoost, bölümleme yalnızca ensemble aracı olarak.

## 2. Gradient boosting mi, derin öğrenme mi

**Elsayed, S. ve ark. (2021). "Do We Really Need Deep Learning Models for Time Series Forecasting?"** [arXiv:2101.02118](https://arxiv.org/abs/2101.02118) — Pencere tabanlı çok-çıktılı regresyon olarak kurulmuş bir GBRT, elektrik dahil dokuz veri kümesinde NeurIPS/KDD/ICML/ICLR'dan sekiz SOTA derin modelin **tamamını geçti**. Kazanç tamamen girdi/çıktı özellik mühendisliğinden, öğreniciden değil.

Derin öğrenme denenecekse en uygun aday **TiDE** (Das, A. ve ark., 2023, TMLR, [arXiv:2304.08424](https://arxiv.org/pdf/2304.08424)) — MLP encoder-decoder, Transformer'lardan 5-10x hızlı, **bilinen gelecek kovaryatlarını** (takvim, hava) almak için tasarlanmış. Alternatifler: TSMixer (Ekambaram ve ark., KDD 2023, [DOI](https://doi.org/10.1145/3580305.3599533)), PatchTST (Nie ve ark., ICLR 2023).

### Foundation modeller: kapsam dışı

| Kaynak | Bulgu |
| --- | --- |
| [arXiv:2410.09487](https://arxiv.org/pdf/2410.09487) | TSFM'ler sıfırdan eğitilmiş Transformer'larla yalnızca *karşılaştırılabilir* |
| BuildSys 2024, [DOI](https://doi.org/10.1145/3671127.3698708) | 1.900+ binada Chronos-46M NRMSE 13,56 vs **LightGBM 12,96** — LightGBM kazandı |
| [arXiv:2412.12834](https://arxiv.org/html/2412.12834) | "current TSFMs cannot incorporate external conditions, which is their primary limitation" |
| [arXiv:2605.24381](https://arxiv.org/html/2605.24381v1) | Enerji: **XGBoost MASE 0,573** vs TimesFM 1,046, Chronos 1,102 |
| [arXiv:2602.05390](https://arxiv.org/pdf/2602.05390) | Düz LSTM+RevIN sıklıkla tüm FM'leri geçiyor; FM'ler yalnızca değişken iklimlerde yardımcı |

Bizim problemimiz hava ve takvim güdümlü, 122 adım ufuklu, hedef geri beslemesi yok — FM'lerin en zayıf olduğu rejim. **Beklenen etki negatif. Atlanıyor.**

## 3. Trafo / fider seviyesi tahmin

**"Optimization of Electric Transformer Operation Through Load Estimation Based on the K-Means Algorithm"** (Universidad de Zaragoza, [tam metin](https://zaguan.unizar.es/record/162912/files/texto_completo.pdf)) — bizim cold-start tarifimizin neredeyse aynısı:

1. Sayaçlı trafoların günlük yük eğrileri üzerinde **DTW ile K-Means** → temsili tüketim desenleri.
2. **LightGBM sınıflandırıcı** her sayaçsız trafoyu yalnızca **müşteri nitelikleriyle** (bağlı kullanıcı tipi ve sayısı) bir kümeye atıyor.

**16.864 trafoluk** gerçek bir sistemde doğrulanmış: bireysel trafo seviyesinde %95,6, fider seviyesinde %95,3, sistem genelinde %98,1 doğruluk. Statik metadata'nın tek başına makul bir yük şekli atamaya yeterli olduğunun yayınlanmış kanıtı.

Tamamlayıcı: **"Load Profile Assignment for Planning and Operation Support in Distribution Networks Under Partial Smart Meter Penetration," *Processes* 14(10), 1505 (2026)** ([MDPI](https://www.mdpi.com/2227-9717/14/10/1505)) — günlük eğrilerde k-means + **aylık tüketime** göre gruplama, ardından günlük küme dizisi üzerinde **Markov zinciri** ile statik değil dinamik profil seçimi.

Hibrit mimari referansı: *Frontiers in Energy Research* (2026), [DOI](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2026.1715418/full) — TFT + LightGBM artık düzeltmesi, tek başına TFT'yi %11,7 RMSE geçiyor. Bizde artık-düzeltme deseni olarak ilginç, ana mimari olarak değil.

## 4. Cold-start

**(a) En doğrudan: küme bilgili lag yeniden kurulumu.** *"Contextual Variables, Customer Segmentation and Cold-Start Conditions in Household Load Forecasting," **Algorithms 2026, 19, 114**,* [DOI](https://doi.org/10.3390/a19020114). Müşterileri ikili bağlamsal özniteliklerle kümeleyip yeni müşteriyi **en yakın merkeze** atıyor, sonra lag özelliklerini **küme ortalamalarından yeniden kuruyor**.

Aynı çalışma dürüst negatif kontrolü de veriyor: kümeler atanamıyorsa lag özellikleri yeniden kurulamaz. **Bizim için çevirisi: cold-start yolunun tüm geçerliliği `guc` + `lokasyon`'un kümeleri tanımlamaya yetecek kadar bilgilendirici olmasına bağlı.** Bu yüzden önce tavan ölçümü.

**(b) Meta-öğrenme hedef geçmişi gerektirmez, transfer öğrenme gerektirir.** Wu, D. ve ark. (2021), *IEEE Access* 9, [DOI](https://doi.org/10.1109/access.2021.3053317): *"transfer learning requires the customer's past for individualization. Meta learning, however, does not require the past data of the target customer."* Sıfır geçmişli 2.024 trafo için transfer öğrenme **yapısal olarak kullanılamaz**.

**(c) Few-shot / MAML.** Sarmas ve ark., *Energies* 18(3), 742 (2025), [MDPI](https://www.mdpi.com/1996-1073/18/3/742) — MAML, transfer öğrenmeyi ve göreve özel ML'i %12,5 geçiyor.

**(d) Açık/ticari metadata'dan profil atama.** Vercamer, D. ve ark. (2016), *IEEE Trans. Power Systems* 31(5), 3693–3701, [DOI](http://doi.org/10.1109/TPWRS.2015.2493083): AMI okuması olmayan yeni müşteriler kartografik veriyle sınıflandırılıyor ve *"Using external data alone, the model was still able to adequately place customers into their relevant load profile."* Coğrafyaya yaslanmanın yayınlanmış gerekçesi.

**(e) Endüstriyel emsal.** AWS Amazon Forecast cold-start yeniden tasarımı ([blog](https://aws.amazon.com/blogs/machine-learning/generate-cold-start-forecasts-for-products-with-no-historical-data-using-amazon-forecast-now-up-to-45-more-accurate/)) tek bir değişiklikten **%45'e varan iyileşme** raporluyor: her şeyi havuzlamak yerine *"identify explicit products... that have the most similar characteristics to the cold start products [and] focus on this subset."* Yani **metadata üzerinde kNN, global ortalamayı geçiyor.**

**(f) Yeniden kümelemeden kNN atama.** Lin, Z. ve ark., *IEEE Trans. Smart Grid* (2026), [DOI](https://doi.org/10.1109/tsg.2026.3651940). Kod: `github.com/U-T-G/AGSC`.

## 5. Hiyerarşik uzlaştırma (MinT) — ana yöntem değil

**Wickramasuriya, S.L., Athanasopoulos, G., Hyndman, R.J. (2019).** *JASA* 114(526), 804–819. [PDF](https://robjhyndman.com/papers/MinT.pdf)

Metriğimiz yalnızca **alt seviyede** satır bazlı RMSLE. Tutarlılık (coherence) şartı yok ve MinT **ham ölçekte** kareli hatayı optimize ediyor, log ölçekte değil — yani yanlış kaybı optimize ediyor. Ayrıca $W_h$ seriler gözlemlerden fazla olduğunda tahmin edilemez (5.344 seri, 455 gün).

Yararlı rolü: ilçe × güç bandı toplamını (çok daha düzgün seri) tahmin edip her trafonun tarihsel payıyla **yukarıdan aşağıya dağıtmak**. Cold trafolar için de geçerli tahmin üretir. Orta etki, düşük risk, yalnızca ensemble üyesi.

## 6. Hava modellemesi: Ege yazı

Türkiye'ye özel çapa: Ember (2024), [**"Solar surge: Meeting two-thirds of the rise in Türkiye's peak electricity demand in 2024"**](https://ember-energy.org/latest-insights/solar-surge-meeting-two-thirds-of-the-rise-in-turkiyes-peak-electricity-demand-in-2024/). Metodolojisi doğrudan yeniden kullanılabilir:

- CDD, MGM'nin resmi ısıtma/soğutma derece gün sayfalarından; **saatlik il değerleri Open-Meteo Historical Weather API'den** — bizim seçtiğimiz kaynağı bağımsız olarak doğruluyor.
- Bölgesel toplama **nüfus ağırlıklı** il ortalamalarıyla (TÜİK ADNKS).
- **Haziran 2024 CDD'si** önceki beş yıl Haziran ortalamasının **üç katı**; **Temmuz 2024 1,5 katı**; CDD 81 ilin 74'ünde arttı ve **en büyük artış Ege bölgesinde** — yani tam olarak İzmir/Manisa.
- Yaz tepe talebi **saat 14:00'te** — doluluk değil sıcaklık güdümlü soğutma mekanizması.

Panel-ekonometrik destek: *Journal of Energy Systems*, [DOI](https://doi.org/10.30521/jes.1969238) — 81 il, 2015–2023, iki yönlü sabit etkiler + Random Forest; **sanayinin GSYH payı ve CDD pozitif anlamlı**; kentleşme, yenilenebilir kapasite ve gaz tüketimi değil.

**Sıralı öneriler:**

1. **Parçalı/spline sıcaklık tepkisi, doğrusal değil.** `CDD = max(0, T_ort - 22)` (Ember'in Türkiye bazı) **ve** `HDD = max(0, 18 - T_ort)`. Nisan'da İzmir'de hâlâ ısıtma yükü var, Temmuz saf soğutma; tek doğrusal terim V şeklini temsil edemez.
2. **Hareketli/gecikmeli CDD:** 3/7/14 günlük iz süren CDD toplamları termal ataleti ve klima açma davranışını yakalar. İnsanlar sıcak bir *günde* değil sıcak bir *dönemden sonra* klimayı açıyor.
3. **Savitzky-Golay yumuşatılmış** günlük ortalama sıcaklık (ASHRAE 1.'sinin özellik setinde).
4. `apparent_temperature_mean` ve `relative_humidity_2m_mean` — nem/hissedilen sıcaklık kanalı.
5. `shortwave_radiation_sum` — burada **negatif** yönde önemli: sayaç arkası güneş PV'si yüksek ışınım günlerinde ölçülen `tuketim`'i bastırıyor ve Türkiye'nin güneş filosu 2024'te tepe talep artışının üçte ikisini karşılayacak kadar büyüdü. Net tüketim ölçen bir dağıtım trafosunda bu, Nisan–Temmuz 2026 için gerçek ve büyüyen bir aşağı yönlü bias.

## 7. Türkiye tatilleri — kesin tarihler

Diyanet İşleri Başkanlığı dini günler takvimi ve bağımsız kaynaklarla çapraz kontrol edildi ([Habertürk 2026](https://www.haberturk.com/2026-resmi-tatil-takvimi-belli-oldu-bu-yil-kac-gun-tatil-var-resmi-tatiller-hangi-gunlere-denk-geliyor-3850434), [CottGroup](https://www.cottgroup.com/tr/blog/calisma-hayati/item/resmi-tatil-gunleri-ve-mesai-hesaplama), [Diyanet 2025](https://vakithesaplama.diyanet.gov.tr/dinigunler.php?yil=2025)).

### Test ufku, 1 Nisan – 31 Temmuz 2026

| Tarih | Gün | Tatil |
| --- | --- | --- |
| 23 Nis 2026 | Per | Ulusal Egemenlik ve Çocuk Bayramı |
| 1 May 2026 | Cum | Emek ve Dayanışma Günü |
| 19 May 2026 | Sal | Atatürk'ü Anma, Gençlik ve Spor Bayramı |
| **26 May 2026** | Çar öncesi Sal | **Kurban Bayramı arifesi (yarım gün, 13:00'ten)** |
| **27–30 May 2026** | Çar–Cmt | **Kurban Bayramı 1.–4. gün** |
| 15 Tem 2026 | Çar | Demokrasi ve Millî Birlik Günü |

### Train dönemi, 1 Ocak 2025 – 31 Mart 2026

| Tarih | Tatil |
| --- | --- |
| 1 Oca 2025 | Yılbaşı |
| **1–29 Mar 2025** | Ramazan ayı |
| 29 Mar 2025 | Ramazan Bayramı arifesi |
| **30 Mar – 1 Nis 2025** | Ramazan Bayramı 1.–3. gün |
| 23 Nis 2025 | Çocuk Bayramı |
| 1 May 2025 | Emek ve Dayanışma |
| 19 May 2025 | Gençlik ve Spor |
| 5 Haz 2025 | Kurban Bayramı arifesi |
| **6–9 Haz 2025** | Kurban Bayramı 1.–4. gün |
| 15 Tem 2025 | Demokrasi ve Millî Birlik |
| 30 Ağu 2025 | Zafer Bayramı |
| 28 Eki (yarım) / 29 Eki 2025 | Cumhuriyet Bayramı arifesi / Cumhuriyet Bayramı |
| 1 Oca 2026 | Yılbaşı |
| **19 Şub – 19 Mar 2026** | Ramazan ayı |
| 19 Mar 2026 | Ramazan Bayramı arifesi (yarım gün) |
| **20–22 Mar 2026** | Ramazan Bayramı 1.–3. gün |

### İki sonuç, ele alınmazsa sessizce zarar verir

1. **Kurban Bayramı yıl başına ~10 gün erkene kayıyor: 6–9 Haz 2025 → 27–30 May 2026.** Herhangi bir `lag_365` veya "geçen yıl aynı gün" özelliği 2026 bayramını sıradan Haziran 2025 iş günleriyle hizalar ve tersi. **Çözüm:** işaretli `bayrama_kalan_gun` ve/veya hicri hizalı lag (2025 referans penceresini gözlenen -10 günlük ofsetle kaydır), Gregoryen lag'in yanında.
2. **Ramazan 2026 test ufkunda hiç yok** (Şub–Mar 2026), ama *Nisan–Temmuz 2025* mevsimsel naif referans penceresi **1 Nisan 2025 = Ramazan Bayramı 3. günü** ile başlıyor. Geçen-yıl özellikleri kurulurken 1 Nisan 2025 maskelenmeli veya özel işaretlenmeli.

**Doğrulanamadı:** 2026 MEB yaz tatili başlangıç tarihi. Turizm bölgesinde konut yükü açısından önemli; MEB çalışma takvimi doğrudan kontrol edilmeli.

Mekanizma kanıtı: *Boğaziçi Journal*, [DOI](https://doi.org/10.21773/boun.37.2.3) — Ramazan ayı, tatil ve hafta sonu göstergeleri saatlik tüketimin **düzenliliğini anlamlı ölçüde azaltıyor** (12s/24s/168s bantlarında wavelet gücü). Pratik okuma: tatiller yükün yalnızca *seviyesini* değil *şeklini ve varyansını* değiştiriyor, dolayısıyla tek bir toplamsal tatil kukla değişkeni yetersiz. Etkileşim kullan: `tatil × ilçe`, `tatil × guc bandı`, `bayram_gun_indeksi`.

## 8. RMSLE'ye özel rehberlik

### 8.1 Temel kimlik ve bias düzeltmesi tuzağı

$$\mathrm{RMSLE}(p,a) = \sqrt{\tfrac{1}{n}\sum(\log(1{+}p_i)-\log(1{+}a_i))^2} = \mathrm{RMSE}\big(\log1p(p), \log1p(a)\big)$$

RMSLE, log1p uzayında RMSE'nin **kendisidir**. Dolayısıyla:

- **`z = log1p(tuketim)` üzerinde düz L2/RMSE ile eğit.** Doğrudan yarışma metriğini minimize ediyorsun. Özel amaç fonksiyonu gerekmiyor. ASHRAE 1.'sinin yaptığı tam olarak bu: *"The meter values to be predicted were transformed by adding 1 and taking the natural logarithm."*
- **`expm1(ẑ)` ile tahmin et ve Duan smearing veya `exp(σ²/2)` lognormal düzeltmesini UYGULAMA.**

İkinci nokta yaygın genel tavsiyeyle çelişiyor, gerekçesi açıkça belirtilmeli. Duan smearing tahmincisi — $\hat{Y}_j = \exp(\widehat{\ln Y}_j)\cdot\frac{1}{N}\sum_i \exp(e_i)$ (Duan, N., 1983, *JASA* 78(383), 605–610) — medyan yerine $E[Y]$'yi kurtarmak için var, çünkü Jensen eşitsizliği $E[e^\epsilon] > 1$ veriyor. Bu düzeltme **kaybın ham ölçekte olduğu durumda doğru, log ölçekte olduğu durumda yanlış.** RMSLE'nin Bayes-optimal tahmini tam olarak $\hat{z} = E[\log1p(y)\mid x]$ olduğu için, $>1$ olan her smearing çarpanı seni optimumdan **uzaklaştırır**. Smearing çarpanı daima $\geq 1$ olduğundan RMSLE'yi daima kötüleştirir.

Aynısı `exp(pred + σ²/2)` için de geçerli. Bu araçları yalnızca iş raporu için ham ölçekli kWh tahmini gerekirse kullan, leaderboard için asla.

### 8.2 Çarpımsal faktörün işe yaradığı durum

Ham ölçekte çarpan $c$, log uzayında **toplamsal $\log c$ kaydırması**. Yani *sabit log-uzayı bias'ını* düzeltmek için doğru araç, başka hiçbir şey için değil.

**ASHRAE 2.'si (cHa0s) post-processing'i "Very critical" olarak etiketledi.** [Write-up](https://www.kaggle.com/competitions/ashrae-energy-prediction/writeups/cha0s-2nd-place-solution): *"Since we remove a lot of low value observations from training data, it artificially increases the mean of the target variable and hence the model's raw predictions on test data also has an inflated mean… reducing the mean of predictions of test data by a reducing factor helps bring it down to its true mean."* Çoğu model için **0,8–0,85** — yani **aşağı** yönlü çarpan, çünkü agresif aykırı değer filtreleme onları yukarı biaslamıştı. Sık tekrarlanan "×1,05" versiyonu **yanlış**.

**M5 1.'si (YeonJun In) bilinçli olarak hiç kullanmadı:** *"Without post-processing. (e.g.) magic multiplier."* ([write-up](https://www.kaggle.com/competitions/m5-forecasting-accuracy/writeups/yeonjun-in-stu-1st-place-solution))

**Karar:** doğrulama fold'unda segment başına (`{cold, warm} × il` veya `× guc bandı`) tek bir skaler $\delta$ (log-uzayı kesişim kaydırması) tahmin et. Yalnızca (i) işareti fold'lar arasında kararlıysa ve (ii) büyüklüğü fold'lar arası standart sapmaya göre büyükse uygula. Aykırı değer temizliği yaptıysan $\delta < 0$ beklenir. Orta-yüksek etki, **yüksek aşırı uyum riski.**

### 8.3 Diğer RMSLE sonuçları

- **Tahminleri $\geq 0$'a kırp**, sonra küçük bir tabana. `log1p` -1'in altında tanımsız ve RMSLE sıfıra yakın tahminlerde patlar.
- **Küçük trafolar metriğe hâkim.** RMSLE ölçek-bağımsız: 200 kVA'da 50 kWh/gün ve 1600 kVA'da 5.000 kWh/gün satır başına eşit katkı yapar. Ham ölçekli RMSE sezgisi (veya ham ölçekli modelin özellik önemi) yol gösterici olmasın. **`log1p(tuketim) - log(guc)`** modellemeyi düşün — metriğin kendi ölçeğinde tam bir yeniden parametrizasyon, `guc` aralığında varyansı homojenleştirir. ASHRAE 1.'si aynısını taban alanıyla yaptı: *"Models were also trained on meter values per unit floor area."*
- **Sıfırlar ve sıfıra yakınlar en yüksek kaldıraçlı satırlar.** Devre dışı bir trafo 0 okurken 500 kWh tahmin etmek $(\log 501)^2 \approx 38{,}6$ kare-log hata, tipik satırın ~40 katı. İki aşamalı model düşün: `tuketim ≈ 0` sınıflandırıcısı regresörü kapılar. ASHRAE ilk 5'in hepsi agresif sıfır/sabit-seri işlemesi yaptı: 3. sıra *"eliminated 0s in the same period in the same site"*; 5. sıra *"dropped long streaks of constant values and zero target values."*

## 9. Kazanan Kaggle çözümleri

### 9.1 ASHRAE Great Energy Predictor III — en yakın analog

**Genel bakış makalesi:** Miller, C. ve ark. (2020), *Science and Technology for the Built Environment*, [DOI](https://doi.org/10.1080/23744731.2020.1795514) · [arXiv:2007.06933](https://arxiv.org/abs/2007.06933). Tüm ilk-5'in kodu ve yeniden üretimi: [buds-lab/ashrae-great-energy-predictor-3-solution-analysis](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis).

Ölçek: 20M eğitim noktası, 2.380 sayaç, 1.448 bina, 16 site, 41M test noktası, 3.614 takım. İlk 5'in private skorları **1,231 / 1,232 / 1,234 / 1,235 / 1,237** — beş sıra arasında %0,5 aralık. Yarışmanın kendi sonucu: büyük ağırlıkla gradient boosting ensemble'ları kazandı ve **"preprocessing of the data sets emerged as a key differentiator."**

**1. — Matthew Motoki & Isamu Yamashita.** Private 1,231, public 0,938 (**public sıra 14** — lokal CV'ye güvenmenin çarpıcı kanıtı).

- *Ön işleme:* elle, sayaç başına anomali temizliği; eksik hava sıcaklığında doğrusal interpolasyon; **hava dosyası ile sayaç dosyası arasında saat dilimi hizalama**.
- *Hedef:* `log1p`, ayrıca birim taban alanı başına tüketim için paralel hedef.
- *Özellikler (28 adet):* ham sayaç/metadata/hava; bina metadata'sı ile sayaç arasında **kategorik etkileşimler**; zamansal (tatiller, gündüz/gece); **sıcaklık lag özellikleri**; **Savitzky-Golay yumuşatılmış sıcaklık**; **sin/cos döngüsel kodlama**; **frekans sayımları**; birkaç **hedef kodlama** varyantı.
- *Modeller:* eğitim alt kümesi başına ayrı modeller — sayaç tipi (4), site_id (16), building_id × meter (2.380) — CatBoost, LightGBM ve MLP ile.
- *Validasyon:* **ardışık onbir aylık bloklarla 12-fold CV.**
- *Ensemble:* optimize edilmiş ağırlıklarla **genelleştirilmiş ağırlıklı ortalama**.
- Toplam hesap: g4dn.4xlarge'da 8.320 dakika — ilk 5'in en pahalısı (5. sıra 1,237 için 61 dakika).

**2. — cHa0s.** Private 1,232. 1.449 binanın tamamında elle görsel aykırı değer temizliği (*"we just decided to spend few minutes on every building"*); **kasıtlı olarak basit hava istatistikleri, gelişmiş lag özelliği yok**; XGBoost + LightGBM + CatBoost + FFNN. Prophet denenip bırakıldı. Validasyon: **ay üzerinde 4-fold ve 5-fold CV**. Ensemble ağırlıkları site başına, sayaç tipi başına farklı. Post-processing ×0,8–0,85.

**3. — Xavier Capdepon.** Private 1,234. Aynı sitede aynı dönemdeki sıfırları eledi; 21 özellik; Keras CNN + LightGBM + CatBoost.

**4. — Jun Yang.** **Hava lag özellikleri ve hedef kodlama** dahil 23 özellik, özellikler **alt eğitim kümeleriyle seçildi**; XGBoost (2-fold, 5-fold) + LightGBM (3-fold).

**Sızıntı tuzağı.** Public test setinin bir kısmı internetten kazınabiliyordu. Yarışmacılar bulup kodu açıkça paylaştı; düzenleyicilerin post-mortem'i: *"All publicly-available data sets discovered by the contestants were shared openly and no contestant had an unfair advantage… these data were not used to determine the winner."* Düzenleyicilerin önerisi: **"Future machine learning competitions should avoid using publicly available data in even the public leaderboard component."**

> **Bizim kontrolümüz (2026-08-21):** EPİAŞ Şeffaflık Platformu tüketim verisini yalnızca **il bazında aylık** ve dağıtım şirketi seviyesinde yayınlıyor; ilçe veya trafo kırılımı yok ([veri listesi](https://www.epias.com.tr/wp-content/uploads/2023/05/seffaflik-platformunda-yayimlanacak-veri-listesi.pdf), [teknik doküman](https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html)). İlçe seviyesinde yayınlanan tek şey serbest tüketici **sayaç adedi**, tüketim miktarı değil. Dolayısıyla ASHRAE'deki gibi kazınabilir bir gerçek-değer kaynağı **yok**. Bu açık uç kapandı.

**Sıralı aktarılabilir çıkarımlar:**

1. Veri temizliği > özellik mühendisliği > model seçimi. Düzenleyicilerin ve beş kazananın hepsinin açık uzlaşması.
2. `log1p` hedef + ağaç ensemble'ı.
3. Aynı verinin **farklı bölümlemeleri üzerinde çok sayıda model** eğitip harmanla (il, güç bandı, ilçe × güç bandı) — Montero-Manso karmaşıklık ayarının pratiği.
4. Zaman bloklu CV (ay veya çok-aylık bloklar), asla rastgele.
5. CV'ye public leaderboard'dan çok güven (1. sıra public'te 14.).
6. Sıcaklık *lag*'leri ve *yumuşatma*, yalnızca aynı gün sıcaklığı değil.

### 9.2 M5 Forecasting Accuracy — validasyon tasarımı dersi

**Kazanan: YeonJun In (YJ_STU).** [Write-up](https://www.kaggle.com/competitions/m5-forecasting-accuracy/writeups/yeonjun-in-stu-1st-place-solution). Resmi post-mortem: Makridakis, S. ve ark. (2022), *IJF*, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0169207021001874).

- **Validasyon şeması (kopyalanacak kısım):** *"Time based split: mimic train/test split"* — cv1 d1830–1857, cv2 d1858–1885, cv3 d1886–1913, public d1914–1941, private d1942–1969; **her pencere tam olarak yarışma ufku uzunluğunda** ve **early stopping yok**.
- **Model seçim kriteri (neredeyse hiç kimsenin kopyalamadığı kısım):** *"select final model using mean(cvs, public score) and std(cvs, public score) (especially, focusing on std)."* En iyi ortalamayı değil **fold'lar arası sağlamlığı** seçti. Write-up'ın en değerli fikri.
- **Mimari:** 220 LightGBM — magaza (10), magaza×kategori (30), magaza×departman (70) bazında havuzlanmış, her biri **rekursif ve rekursif olmayan (dogrudan)** varyantlarda; her seri **altı modelin eşit ağırlıklı ortalaması**. Amaç fonksiyonu: **Tweedie** (`tweedie_variance_power ≈ 1,1`), negatif olmayan, sağa çarpık, sıfırda kütlesi olan hedef için.
- **Ensemble gerekçesi:** rekursif genel olarak ve public'te daha iyi, rekursif olmayan cv3'te en iyi; ikisi de tek başına yüksek fold-arası varyanslı; harmanlama varyansı düşürdü.
- **Magic multiplier yok.**

**Bizim için uyarlama.** Ufkumuzda **hiç hedef geri beslemesi yok**, dolayısıyla rekursif seçenek kapalı ve rekursif/doğrudan çeşitlilik ekseni yok. Yerine: (a) ufuk-bloğu başına model (test ayı başına bir model) vs tek ufuk-agnostik model; (b) farklı bölümlemeler (il, güç bandı, ilçe); (c) farklı öğreniciler; (d) `log1p(tuketim)` vs `log1p(tuketim) - log(guc)` hedefleri. Sonra onun ortalama-ve-**std** seçim kuralını uygula.

**Tweedie üzerine:** denenmeye değer, ama ham ölçekte deviance optimize ettiği, RMSLE'yi değil, unutulmamalı. `log1p`+L2 formülasyonunun yerine değil, RMSLE'de değerlendirilen bir ensemble üyesi olarak test edilmeli.

### 9.3 Corporación Favorita — doğrudan çok-ufuklu mimari

Metrik NWRMSLE, 16 günlük ufuk, 125M satır — bizim *doğrudan çok-ufuklu* problemimize yapısal olarak en yakını.

**1. (public 0,504 / private 0,509)** — [detaylı özet](https://www.zhihu.com/en/article/33671208):

- `model_1` (0,506/0,511): **16 LightGBM, her gelecek gün için bir tane.**
- `model_2` (0,507/0,513): 16 sinir ağı, her gün için bir tane.
- `model_3` (0,512/0,515): **16 günün tamamı için 1 LightGBM**, neredeyse aynı özelliklerle.
- `model_4` (0,517/0,519): 1 sinir ağı.
- Final: `0,42·m1 + 0,28·m2 + 0,18·m3 + 0,12·m4`.
- Özellikler: aynı lag/rolling bataryası **üç granülaritede** — ürün×mağaza, ürün, mağaza×sınıf — artı yalnızca LightGBM için promosyon/promosyon-dışı günlere ayrılmış ortalamalar ve üstel ağırlıklı azalan toplamlar.

**5. (Lenz Du)** — [write-up](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47556), [kod](https://github.com/LenzDu/Kaggle-Competition-Favorita). Bizim için çok önemli iki detay:

- **Çok-origin eğitim seti inşası:** *"I split the raw time sequences into time sliding windows of different lengths, and extract features from these sliding windows to predict the upcoming 16 values… I use multiple time stamps to construct my dataset. Validation period is 2017.7.26–2017.8.10, and training periods are collected randomly during 2017.1.1–2017.7.5. However, the time sequences used for constructing features are extended to much earlier."* Tek bir forecast origin'i entity başına yüzlerce eğitim satırına çevirmenin yolu.
- **Cold-start filtreleme:** *"only store-item combinations that are appeared in the test data and have at least one record during 2017 in the training data are included."* Sıfır geçmişli entity'leri düşürdüler — bizim sahip **olmadığımız** bir lüks, çünkü skorlanan satırlarımızın %22,2'si tam olarak o entity'ler.

**Top-%3 referans uygulaması** açık validasyon notuyla: [btrotta/kaggle-favorita](https://github.com/btrotta/kaggle-favorita/) — 1–15 Ağu 2017'de doğruladı ama **modelleri yalnızca son 10 günde seçti**, çünkü public leaderboard ilk 5 günü kapsıyordu; ve aynı modeli **üç farklı eğitim dönemi uzunluğunda** eğitti, *"the longest training period gave the best validation score, but the others contributed to the ensemble."*

Modern yeniden uygulama ([taramelli13/favorita-forecasting](https://github.com/taramelli13/favorita-forecasting)) anti-sızıntı kuralını somutlaştırıyor: `lags: [16,17,...,365]` ile **tüm lag'ler ≥ 16 = ufuk uzunluğu**, ve her rolling penceresinde `rolling_shift: 16`. **Bizim karşılığı: her lag ve rolling penceresi ≥ 122 gün kaydırılmalı** veya daha iyisi ve daha basiti, kesin olarak 2026-03-31 cutoff'ına göre hesaplanmalı.

### 9.4 Rossmann Store Sales — validasyon felsefesi

**1. Gert Jacobusse.** [Kazanan röportajı](https://medium.com/kaggle-blog/rossmann-store-sales-winners-interview-1st-place-gert-jacobusse-a14b271659b). 20+ XGBoost ensemble'ı, çoğu tek başına ilk-3'e girecek nitelikte. Zaman bütçesi: **"50% on feature engineering, 40% on feature selection plus model ensembling, and less than 10% on model selection and tuning."**

En önemli içgörüsü: *"I could reliably predict performance improvements based on a hold out set within the trainset. Because of this insight, I did not overfit the public test set, so my model worked very well on the public test set as well as the unseen private test set that was four weeks further ahead."* Ve neden bu yapıyı tercih ettiği: *"The advantage of a holdout set is that I can use the public test set as a real test set, not a set that gives me feedback to improve my model."*

İmza özellikleri: **çoklu ufukta hareketli pencere ortalamaları, hafta günü ve promosyon gibi faktörlere ayrılmış** — bizim hareketli ortalamalarımızı hafta günü ve tatil/tatil-dışı olarak ayırmanın doğrudan analogu.

### 9.5 Güncel (2024–2026) enerji yarışmaları

**2025 Electricity Consumption Forecasting AI Competition (DACON × Kore Enerji Ajansı), 934 takım içinde 2. — Team SKKU brAIn.** [Repo](https://github.com/Bellissimo-AI/2025_Electricity_Consumption_Forecasting_AI_Competition). Bina seviyesi saatlik kWh. Pipeline: **yaz/yaz-dışı mevsimsel ayrım** → **kümeleme tabanlı ayrıştırma** → üç granülaritede **grup bazlı XGBoost** (bina *tipi*, bina *numarası*, *global*) → elle ayarlanmış ağırlıklı ensemble (yazın `tip·0,2 + no·0,3 + global·0,5`; diğer zamanlarda `0,25/0,25/0,5`) → **belirli bina/zaman tahminlerini tarihsel değerlerin kırpılmış ortalamalarıyla değiştiren tatil post-processing'i** → negatifleri 0'a kırpma. Ayrıca **Wind Chill Temperature** türev özelliği.

Açık yaz/yaz-dışı model ayrımı bizim için özellikle ilgili: ufkumuz bir geçiş ayından (Nisan) tepe soğutmaya (Temmuz) uzanıyor.

**Önceki halka açık GDZ Elektrik Datathon'ları** — aynı dağıtım şirketi, aynı İzmir/Manisa coğrafyası, dolayısıyla alan bilgisi transferi gerçek:

- **GDZ Datathon 2023** (şebeke merkezi bazında saatlik enerji dağıtımı, MAPE): [arukemre/Energy-consumption-based-on-hourly-data](https://github.com/arukemre/Energy-consumption-based-on-hourly-data). `TimeSeriesSplit(n_splits=3, test_size=2200)`, sonra basit son-3-ay holdout'a geçmiş (*"Çok fazla fark yaratmadı"*). Dış veri: **İzmir için Meteostat hava verisi**, Türkiye toplam saatlik üretim ve tüketim, WorldWeatherOnline — **hepsi kullanımdan önce kaydırılmış**.
- **GDZ Datathon 2024** (ilçe bazında plansız kesinti, MAE). Team Pikachow'un [sunum destesi](http://www.anilozturk.net/wp-content/uploads/2024/05/GDZ24-Datathon-Sunum-Pikachow.pdf) metodolojik olarak en yararlı belge: CatBoost / LightGBM / XGBoost / lokal modeller (ARIMA, Croston, EMA) / global NN (RNN, TFT, TimesNet) taraması, ardından **SHAP tabanlı özyinelemeli özellik seçimiyle 7 fold × 15 adım × 25 özellik ile 490 özelliği 97'ye indirme**, Optuna/TPESampler ile tuning. Kendi retrospektif tavsiyeleri: özellik üretim hızı için pandas → Polars. Diğer repolar: [alperengulunay (10.)](https://github.com/alperengulunay/2024-GDZ-Datathon-10-th-Place-Submission) (47 ilçenin her birini bağımsız modelliyor; sıfır-kesinti günlerini filtrelemek için RF sınıflandırıcı + AutoGluon TimeSeries regresyonu — bizim sıfır tüketim satırlarımız için doğrudan yeniden kullanılabilir **iki aşamalı sıfır kapısı**).

## 10. Validasyon tasarımı

Yönetici referanslar: Bergmeir, C., Hyndman, R.J. & Koo, B. (2018) AR modelleri için CV'nin ne zaman geçerli olduğu üzerine; **Hewamalage, H., Ackermann, K., Bergmeir, C. (2023). "Forecast evaluation for data scientists: common pitfalls and best practices," *DMKD* 37, 788–832, [DOI](https://doi.org/10.1007/s10618-022-00894-5)** — **sabit-origin** ile **kayan-origin** (tsCV / prequential) değerlendirmeyi ayırıyor. Ayrıca Cerqueira, V., Torgo, L., Mozetič, I. (2020), *Machine Learning* 109, [DOI](https://dl.acm.org/doi/10.1007/s10994-020-05910-7): **durağan olmayan** seriler için *"the most accurate estimates are produced by out-of-sample methods, particularly the holdout approach repeated in multiple testing periods."*

Bizim panel güçlü şekilde durağan değil (entity sayısı ikiye katlanıyor), dolayısıyla: **her biri 122 gün olan tekrarlı sabit-origin holdout'lar, artı yapay cold-start maskelemesi.**

### 10.1 Fold tasarımı

| Fold | Train ≤ | Doğrulama | Amaç |
| --- | --- | --- | --- |
| **A — mevsim hizalı** | 2025-03-31 | **2025-04-01 → 2025-07-31** | *Asıl* fold. Aynı takvim konumu, aynı Nis–Tem soğutma rampası, aynı 23 Nis / 1 May / 19 May / 15 Tem tatilleri, Kurban içeride. Yalnızca 90 günlük önceki geçmiş — kısa geçmişli entity'ler için iyi bir stres testi. |
| **B — en güncel** | 2025-11-30 | 2025-12-01 → 2026-03-31 | En güncel entity popülasyonu ve büyüme rejimi; Ramazan + Ramazan Bayramı 2026 içeride. Kış, dolayısıyla hava tepki katsayıları transfer edilemez. |
| **C — ara** | 2025-09-30 | 2025-10-01 → 2026-01-31 | Ortalama/std sağlamlık kriteri için üçüncü origin. |

Fold A, yaz sıcaklık tepkisini ve Kurban Bayramı işlemesini test eden tek fold. En ağır ağırlıklandır ama asla tek başına seçim yapma — o 122 günlük pencereye aşırı uymanın yolu bu. **A/B/C arası ortalama ve standart sapmayı** raporla ve düşük std'li konfigürasyonları tercih et.

### 10.2 Cold-start maskeleme tarifi

Her fold'da, cutoff seçildikten sonra:

1. Doğrulama penceresinde aktif tüm entity'leri belirle.
2. **Gerçek cold-start popülasyonuna tabakalı eşleştir.** Gerçekten cold olan 2.024 test trafosu üzerinde (ilçe, `guc` bandı, test-satır-sayısı bandı) ampirik ortak dağılımını kur. Doğrulama entity'lerini o ortak dağılıma uyacak şekilde örnekle — düzgün rastgele %28,8 **değil**. Bu önemli, çünkü cold entity'ler coğrafi veya kapasite olarak neredeyse kesinlikle düzgün dağılmamış (yeni bağlantılar gelişen ilçelerde ve belirli `guc` sınıflarında kümeleniyor).
3. **Cutoff öncesi geçmişlerinin tamamını** eğitim çerçevesinden sil — yalnızca lag özelliklerini değil. Sadece lag'leri null yaparsan grup hedef kodlamaları geçmişi sessizce emer ve cold-start performansını olduğundan iyi tahmin edersin.
4. **Ufuk ortası girişi de taklit et:** maskelenen entity'lerin bir alt kümesi için doğrulama penceresinin erken satırlarını düşür, böylece ilk kez ufkun 2. veya 3. ayında görünsünler (gözlenen ~78'e karşı ~113 satır asimetrisi).
5. **Üç sayıyı ayrı skorla:** `RMSLE_warm`, `RMSLE_cold` ve gerçek test oranlarıyla (%77,8 / %22,2) satır ağırlıklı harman. Harmanı optimize et ama iki bileşeni daima izle — warm'ı 0,01 iyileştirip cold'u 0,05 kötüleştiren bir değişiklik toplamda kayıptır ve toplu sayıda gözden kaçar.

Dürüst %77,8/%22,2 harmanını hesaplayabildiğin için cold-start'ın özel bir model hak edip etmediğine rasyonel karar verebilirsin. `RMSLE_cold` ≈ 2× `RMSLE_warm` ise cold satırlar kare-hata toplamına satır paylarının ~4 katı katkı yapar ve cold-start çalışması en yüksek getirili faaliyet olur.

### 10.3 Sızıntı denetim listesi

- [ ] **Hiçbir özellik fold cutoff'ından sonraki veriyi kullanmıyor.** Grup ortalamaları, hedef kodlamalar, `guc` istatistikleri, küme merkezleri, smearing/çarpan faktörleri ve tüm çerçeveye fit edilmiş herhangi bir ölçekleyici dahil.
- [ ] **Her lag/rolling özelliği ya as-of-cutoff ya da ≥ ufuk mesafesi kadar kaydırılmış.** Favorita kazananları tam olarak ufuk uzunluğu kadar kaydırdı (16/16 gün); bizimki 122.
- [ ] **Her `shift`/`rolling` öncesi `groupby('tanim')`.** Sıralanmamış veya gruplanmamış çerçevede `shift` bir trafonun geçmişini sessizce diğerine sızdırır.
- [ ] **Hedef kodlamaları out-of-fold**, yumuşatmalı. 5.344 seviyeli `tanim`'ı fold içinde hedef kodlamak ders kitabı aşırı uyumu; CatBoost'un sıralı hedef istatistiklerini tercih et veya cold-start modelinden `tanim`'ı tamamen çıkar.
- [ ] **`guc`'ün `tanim` başına sabit olduğunu kontrol et.** (Kontrol edildi: sabit. Grup başına tekil değer sayısı 1.)
- [ ] **Hava: gerçekleşen mi tahmin mi sorusunu karara bağla ve belgele.**
- [ ] **Test dosyası metadata'sı yasal ama işaretlenmeli.** Test CSV'si hangi `tanim`'ın hangi `tarih`te göründüğünü söylüyor, dolayısıyla ilk görülme tarihi, son görülme tarihi ve trafo başına toplam satır sayısı meşru olarak türetilebilir ve muhtemelen bilgilendirici (Haziran 2026'da ilk kez görünen bir trafo, rampa profili olan yeni bir devreye alma). Bu sızıntı değil — verilen veri — ama gerçek bir operasyonel tahminde mevcut *olmaz*.
- [ ] **Yayınlanmış gerçek değerler için kontrol.** (Kontrol edildi: EPİAŞ yalnızca il bazında aylık yayınlıyor, ilçe/trafo kırılımı yok.)

### 10.4 Model kodu yazmadan önce yapılacak 30 dakikalık deney

*Algorithms 2026, 19, 114* makalesinin negatif kontrolü şunu ima ediyor: tüm cold-start stratejisi tek bir ampirik soruya bağlı — **`log1p(tuketim)` varyansının ne kadarı yalnızca `guc` ve `lokasyon` ile açıklanıyor?**

Yalnızca eğitim verisi üzerinde `log1p(tuketim) - log(guc)` değişkenini `ilçe × ay × hafta günü` üzerinde modelle ve artık standart sapmasını ölç. Trafo kimliği eklendiğindeki artık std ile karşılaştır. Aradaki fark, metadata'dan elde edilebilecek cold-start doğruluğunun sert tavanı.

Bu ölçüm, ayrıntılı metadata-kNN retrieval'a yatırım yapmayı mı yoksa cold satırların indirgenemez gürültülü olduğunu kabul edip bunun yerine **sistematik olarak yansız olmaya** odaklanmayı mı gerektiğini hemen söyler. RMSLE için yansızlık, `log1p`'in **medyanını** doğru tutmak demektir — sağlam bir grup medyanı, ortalama değil.

## 11. Bu yarışma için en önemli 10 eylem

Birim efor başına beklenen RMSLE etkisine göre sıralı.

1. **`log1p(tuketim)` üzerinde düz L2 ile eğit, `expm1` ile tahmin et, Duan smearing veya `exp(σ²/2)` UYGULAMA.** RMSLE log uzayında RMSE'nin *kendisi*, dolayısıyla `log1p`'in koşullu ortalaması zaten Bayes-optimal; her smearing çarpanı ≥ 1 ve RMSLE'yi kesinlikle artırır. Sonra ≥ 0'a kırp. İkinci hedef olarak `log1p(tuketim) - log(guc)` (kVA başına log yoğunluk) test et.
2. **Mevsim hizalı validasyon fold'unu kur: train ≤ 2025-03-31, doğrulama 2025-04-01 → 2025-07-31.** Nis–Tem soğutma rampasını, 23 Nis / 1 May / 19 May / 15 Tem tatillerini ve bir Kurban Bayramı'nı çalıştıran tek fold. İki tane daha 122 günlük sabit-origin fold ekle ve fold'lar arası **ortalama *ve* standart sapmaya** göre seç.
3. **Her fold'da tabakalı cold-start maskelemesi uygula — lag'leri değil geçmişi sil.** Gerçek 2.024 cold trafonun (ilçe, `guc` bandı, satır sayısı bandı) ortak dağılımına eşleştir. Bazı entity'lerin erken doğrulama satırlarını da maskele. `RMSLE_warm`, `RMSLE_cold` ve gerçek satır-ağırlıklı harmanı ayrı skorla. Bu olmadan skorunun %22,2'si üzerinde hiçbir sinyalin yok.
4. **Naif tabanları önce kur, özellikle geçen yıl mevsimsel naifi — meşru ve muhtemelen güçlü.** Nis–Tem **2025** gerçekleşmeleri eğitim verisinin içinde, dolayısıyla "aynı trafo, yılın aynı günü, bir yıl geri" yalnızca cutoff öncesi bilgi kullanıyor. Kur: (a) train'in son 28/91 gününün trafo bazlı ortalaması; (b) Nis–Tem 2025'ten mevsimsel naif; (c) mevsimsel naif × güncel seviye oranı; (d) cold entity'ler için `ilçe × guc bandı × ay × hafta günü` bazında `log1p(tuketim/guc)` **medyanı**, `guc` ile yeniden ölçeklenmiş. Ortalama değil medyan.
5. **Bayram hizasızlığını, her yıl-üzeri özelliği sessizce bozmadan önce düzelt.** Kurban **6–9 Haz 2025 → 27–30 May 2026** (~10 gün erken); Ramazan Bayramı **30 Mar–1 Nis 2025 → 20–22 Mar 2026**; ve mevsimsel naif referans penceresinin ilk günü olan 1 Nis 2025 Ramazan Bayramı'nın 3. günü. İşaretli `bayrama_kalan_gun`, `bayram_gun_indeksi`, arife yarım gün bayrağı ve Gregoryen lag'in yanında hicri hizalı lag ekle.
6. **Hava: Open-Meteo'yu bir kez çek (2026-07-31'e kadar çalıştığı doğrulandı) ve sıcaklığı doğru değil V şekli olarak modelle.** ~50 tekilleştirilmiş ERA5-Land ızgara noktası × 577 gün ≈ 2.050 ağırlıklı çağrı, 10.000/gün ücretsiz kotanın çok içinde; CC-BY 4.0 atıf zorunlu. `CDD = max(0, T-22)` **ve** `HDD = max(0, 18-T)`; termal atalet için 3/7/14 günlük iz süren CDD; Savitzky-Golay yumuşatılmış sıcaklık; `apparent_temperature_mean` + `relative_humidity_2m_mean`; `shortwave_radiation_sum` — son maddenin sayaç arkası PV üzerinden **negatif** çalışması muhtemel. **Sonra iklim normalleri ablasyonunu çalıştır.**
7. **Cold-start'ı ortalama alma değil metadata-retrieval problemi yap — ama önce tavanı ölç.** Amazon Forecast yalnızca "en benzer ürünleri belirle ve o alt kümeye odaklan"a geçişten **%45'e varan iyileşme** raporluyor. Uygula: warm trafoların normalize günlük profilleri üzerinde (DTW) k-means → cold trafoları (`guc`, ilçe, coğrafya) uzayında en yakın merkeze ata → geçmişten türeyen özelliklerini **küme medyanlarından** yeniden kur. Kurmadan önce §10.4'teki varyans ayrıştırma kontrolünü çalıştır.
8. **Tek global gradient-boosted model, bölümlemeler üzerinde ensemble'lanmış — bu ölçekte ampirik olarak baskın mimari.** Montero-Manso & Hyndman global modellerin sabit karmaşıklık taşıdığını kanıtlıyor. İlgili her kazanan tam olarak bunu yaptı. Bizim bölümlemelerimiz: il, `guc` bandı, `ilçe × guc bandı`, artı ayrı cold ve warm modelleri. Yüksek kardinaliteli `tanim`/ilçe için CatBoost. **Foundation modelleri atla.**
9. **Zamanını veri temizliğine ve iki aşamalı sıfır modeline harca, bu sırayla.** ASHRAE düzenleyicilerinin 3.614 takımdan çıkardığı sonuç: *"Pre-processing of the training data, including outlier removal, was the key differentiator in the top winners and was usually not an automated process."* RMSLE için özellikle sıfırlar en yüksek kaldıraçlı satırlar.
10. **Segment başına log-uzayı kesişim kaydırması fit et — ama yalnızca sağlamlık testini geçerse.** ASHRAE 2.'si bunu "çok kritik" dedi ve **0,8–0,85** *aşağı* yönlü kullandı (sık yanlış aktarılan 1,05 değil). M5 kazananı hiç kullanmadı. Yani: doğrulamada `{cold, warm} × il` başına δ tahmin et ve yalnızca işaret üç fold'da kararlıysa ve |δ| fold'lar arası std'ye göre büyükse uygula. Leaderboard'a aşırı uymanın en kolay yeri.

## 12. Doğrulanamayan noktalar

- **2026 MEB yaz tatili kesin başlangıç tarihi** (İzmir/Manisa). Turizm bölgesinde konut yükü için ilgili; MEB çalışma takvimi doğrudan kontrol edilmeli.
- **47 ilçe koordinat listesinin tamamı.** Kullanılabilir topluluk listeleri 2013-14 büyükşehir reorganizasyonundan önceye ait, ilçeler eksik (Yunusemre, Şehzadeler, Menderes) ve en az iki koordinat il merkezinden kopyalanmış. Bakımlı bir kaynağa karşı yeniden geokodlanmalı ve il merkeziyle aynı koordinata düşen her ilçe elle doğrulanmalı.

## 13. Dış veri kaynakları: doğrulanmış detaylar

### Open-Meteo Historical Weather API

Canlı test edilmiş istek:

```
https://archive-api.open-meteo.com/v1/archive
  ?latitude=38.4189&longitude=27.1287
  &start_date=2026-07-25&end_date=2026-07-31
  &daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,
         shortwave_radiation_sum,relative_humidity_2m_mean,
         precipitation_sum,sunshine_duration,apparent_temperature_mean
  &timezone=Europe%2FIstanbul
```

Gerçek yanıt: İzmir 25–31 Tem 2026 `temperature_2m_mean` = [25,3 / 25,5 / 28,5 / 31,0 / 30,1 / 28,6 / 28,4] °C, `temperature_2m_max` 37,6 °C'ye kadar, nem %33–45, `shortwave_radiation_sum` 25,2–28,8 MJ/m².

**Doğrulanan olgular:**

- **2026-07-31'e kadar İzmir kapsaması canlı ve tam.** Eski OpenAPI enum'unda olmayan `temperature_2m_mean`, `relative_humidity_2m_mean` ve `apparent_temperature_mean` de çalışıyor.
- **Izgara yuvarlaması:** (38,4189, 27,1287) isteği `latitude: 38.488575, longitude: 27.109905` döndürdü. ERA5-Land'in ~0,1° / ~11 km ızgarası ([open-meteo/open-data](https://github.com/open-meteo/open-data/blob/main/README.md): `copernicus_era5` 0,25°/~25 km, `copernicus_era5_land` 0,1°/~11 km, 1940/1950→bugün, 5 gün gecikme). **Pratik sonuç: birçok İzmir metropol ilçesi aynı ızgara hücresine düşüyor.** Çekmeden önce koordinat listesini tekilleştir.
- **Kota** ([şartlar](https://open-meteo.com/en/terms)): ücretsiz katman **günde 10.000 çağrı, saatte 5.000, dakikada 600, ayda 300.000**, IP bazlı, **yalnızca ticari olmayan**, veri **CC-BY 4.0** (atıf zorunlu). Arşiv ağırlık formülü kabaca `max(1, degisken/10) × max(1, gun/14) × konum`.
- **Bizim bütçe:** ~50 tekilleştirilmiş ızgara noktası × 577 gün (2025-01-01 → 2026-07-31) × ~10 günlük değişken ≈ konum başına `1 × 41 × 1` ≈ **~2.050 toplam** — bir günlük ücretsiz kotanın rahatça içinde. Bir kez çek, Parquet'e cache'le, bir daha çekme.
- **Kendi kendine barındırma çıkışı:** `docker pull ghcr.io/open-meteo/open-meteo`, `docker run open-meteo sync copernicus_era5_land temperature_2m --past-days 730` (~8 GB), sonra `/v1/archive`'ı lokal sun. Sunucu kodu AGPLv3.

**Karara bağlanan nokta:** `archive-api` **ERA5 reanalizi — retrospektif gerçekleşmeler** döndürüyor. Ufuk (Nis–Tem 2026) geçmişte olduğu için erişilebilir, ama bu *mükemmel hava öngörüsü* demek ve gerçek bir 4 ay ileri operasyonel tahminde bulunmaz. **Seçim: ERA5 gerçekleşmelerini kullan (yarışma dış veriyi açıkça teşvik ediyor) VE iklim normalleri ablasyonunu çalıştır**, böylece skorun ne kadarının hava öngörüsünden geldiği bilinir. Bu sayı jüri sunumu için gerekli.

Dürüst operasyonel alternatif (referans olarak): Open-Meteo'nun **Historical Forecast API** ve **Previous Runs API** (sabit 1–7 günlük öncü sürede sürekli seri, Oca 2024'ten itibaren) — gerçek bir tahmin pipeline'ının doğru analogu, ama hâlâ 4 aylık öncü süre değil.

### Alternatifler

| Kaynak | Erişim | Notlar |
| --- | --- | --- |
| **NASA POWER** | `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN,RH2M,PRECTOTCORR&community=re&latitude=38.42&longitude=27.13&start=20250101&end=20260731&format=JSON` | Kimlik doğrulama yok. Meteoroloji 1981→, ışınım 1984→, 300+ parametre, 2–7 gün gecikme, istek başına **maks 20 parametre**; 0,5°×0,5° ızgara. [Docs](https://power.larc.nasa.gov/docs/services/api/temporal/daily/). Python: [`pynasapower`](https://github.com/alekfal/pynasapower). ERA5 için iyi bağımsız çapraz kontrol, ışınımda en güçlü. |
| **Meteostat** | `pip install meteostat`; `Daily(Point(38.42, 27.13, 25), start, end).fetch()` | İstasyon tabanlı (NOAA, DWD, ulusal servisler). [Docs](https://dev.meteostat.net/python/). **GDZ Datathon 2023 çözümünde bu coğrafyada kullanılmış.** Uyarı: reanalizin aksine gerçek istasyon boşlukları var. |
| **MGM** | [mgm.gov.tr](https://mgm.gov.tr/site/bilgi-edinme.aspx?r=j) | Otoriter ulusal kaynak ve Ember'in kullandığı resmi HDD/CDD tablolarının yayıncısı. **Ama açık programatik API yok; toplu veri ücretli tarife ve resmi talep süreci arkasında.** Datathon için tekrarüretilebilir değil. Yayınlanmış aylık HDD/CDD tablolarını Open-Meteo türevi CDD'nin *doğrulama referansı* olarak kullan. |

### Türkiye tatil kütüphanesi

**`holidays` (vacanza/python-holidays) — önerilen.** [Kaynak](https://github.com/vacanza/holidays/blob/e0cc9dab/holidays/countries/turkey.py) ve [docs](https://holidays.readthedocs.io/en/main/auto_gen_docs/turkey/) üzerinden doğrulandı:

- `country = "TR"`, `start_year = 1936`, `default_language = "tr"`.
- **`supported_categories = (HALF_DAY, PUBLIC)`** — arife günleri ayrı **`HALF_DAY`** kategorisinde, etiket `"%s (saat 13.00'ten)"`. Özellikle `_add_eid_al_fitr_eve(...)`, `_add_arafah_day(...)` ve 28 Ekim.
- Dini tatiller: Ramazan Bayramı (`eid_al_fitr` 1.–3. gün) ve Kurban Bayramı (`eid_al_adha` 1.–4. gün) `TurkeyIslamicHolidays` üzerinden, **`EID_AL_FITR_DATES_CONFIRMED_YEARS = (1936, 2032)`** ve **`EID_AL_ADHA_DATES_CONFIRMED_YEARS = (1936, 2032)`**. Bizim 2025 ve 2026 tarihleri *doğrulanmış* aralığın içinde — tahmini değil — ve Diyanet'in hicri çeviricisinden geliyor.
- `TurkeyIslamicHolidays(calendar_delta_days=...)` tüm İslami tarihleri kaydırmayı sağlıyor, hicri-ofset lag özellikleri için kullanışlı.

```python
import holidays
from holidays.constants import PUBLIC, HALF_DAY
tr = holidays.country_holidays("TR", years=[2025, 2026],
                               categories=(PUBLIC, HALF_DAY),
                               language="tr")
```

**`workalendar`** alternatif; avantajları `is_working_day()` ve `add_working_days()`. Türk yarım günlerini **modellemiyor**. Okul tatillerini kavram olarak destekliyor ama Türkiye için değil. `holidays` birincil, `workalendar` yalnızca iş günü aritmetiği için.

### İlçe koordinatları

Topluluk kaynağı: [gist.github.com/altinsoft](https://gist.github.com/altinsoft/f81b59a40447c35a63de729927c32794). İl merkezleri: **İzmir 38,41885 / 27,12872; Manisa 38,619099 / 27,428921.**

**Üç açık uyarı.** (1) Gist **2013–2014 büyükşehir reorganizasyonundan önceye ait**: Manisa'nın merkez ilçesi **Yunusemre** ve **Şehzadeler** olarak bölündü, İzmir'in artık 30 ilçesi var. Eksikler: Menderes, Yunusemre, Şehzadeler ve birkaçı daha. GDZ'nin kendi materyalleri iki ilde **47 ilçe** diyor, dolayısıyla boşluk beklenmeli. (2) Birkaç kayıt il merkezinin kopyası (Buca ve Karşıyaka ikisi de 38,41885/27,12872 gösteriyor) — açıkça yanlış, düzeltilmeli. (3) Gist topluluk derlemesi, belirtilmiş lisansı yok.

**Öneri:** bu listeyi hızlı başlangıç olarak kullan, ama tekrarüretilebilirlik için `lokasyon` ilçe adlarını bakımlı bir kaynağa karşı geokodla — [dilan01hall/turkiye-il-ilce-koordinatlari](https://huggingface.co/datasets/dilan01hall/turkiye-il-ilce-koordinatlari) veya [adilmustafayilmaz/turkiye-il-ilce-mahalle-verileri](https://github.com/adilmustafayilmaz/turkiye-il-ilce-mahalle-verileri) — ve kopyalanmış koordinata düşen her ilçeyi elle doğrula. Sonra hava çekmeden önce ERA5-Land ~11 km ızgarasına tekilleştir.

**Bu projedeki uygulama (2026-08-21):** 47 lokasyonun tamamı `src/external/coords.py` içinde, reorganizasyon sonrası ilçeler dahil, Buca/Karşıyaka il-merkezi kopyaları düzeltilmiş halde. Ayrıntı: `docs/external-data.md`.

## 12. EPİAŞ sızıntı taraması (2026-08-21)

ASHRAE GEPIII'de public test setinin bir kısmı internetten kazınabiliyordu. Analog kontrol: EPİAŞ Şeffaflık Platformu Nisan–Temmuz 2026 için **trafo-gün** gerçekleşen tüketim yayımlamıyor. Yayımlanan tüketim **il × ay** (`/v1/consumption/data/consumption-quantity`) ve ulusal saatlik sistem yükü. Skorlanan birimle eşleşmediği için hedef sızıntısı yok. İl-aylık seri bu sürümde kullanılmadı.

