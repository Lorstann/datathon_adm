# Dorduncu tur: istatistiksel analiz, TabArena, cold lokasyon ofseti

Uretim tarihi: 2026-08-23

LB: **1.06713** (degismedi; gunun uc hakki dun gece kullanilmisti).

## 1. Varyans ayristirmasi — uc gun gec kalinmis analiz

Ayrinti: `reports/43_variance.md`. Ozet, `log1p` sifir olmayan satirlar,
toplam varyans 2.6787:

| bilesen | pay |
| --- | --- |
| trafolar arasi | **91.9%** |
| ortak zaman (mevsim + hava + takvim) | **1.4%** |
| trafo x gun | 7.1% |

Bu tablo, onceki uc turun deneme gecmisini aciklıyor. Hava ve takvim
muhendisligi %1.4'luk bir kutunun icinde; trafo-bazli hava egimleri bile
(ORNEKLEM-ICI, yani sisirilmis ust sinir) toplamin %3.75'ine kadar cikiyor.
`reports/07`'nin "hava notr" bulgusu, `reports/33`'un carpan reddi ve
`reports/39`'un mevsim-agirligi reddi ayni yapisal gercegin uc yuzu.

Cold tavani iki bagimsiz yoldan ayni yere cikti:
- varyans: guc+lokasyon trafo seviyesinin %57.7'sini acikliyor, kalan %42.3
  cold'da tanim geregi erisilemez -> RMSLE ~1.11
- `reports/02`: kahin sifir kapisi ile cold RMSLE 1.09-1.14

## 2. TabArena / RealMLP — denendi, reddedildi

Onceki turlarda topluluk denemeleri yalnizca CatBoost ve XGBoost'laydi; ikisi
de agac, yani LightGBM ile yuksek korelasyonlu. Mimari olarak farkli bir model
bu varsayimi ilk kez test etti.

Hata korelasyonu **0.86-0.93** — agac-agac korelasyonunun belirgin altinda,
yani cesitlilik gercek. Ama model yeterince iyi degil:

| yapilandirma | realmlp tek | blend etkisi |
| --- | --- | --- |
| 400k satir, 48 epoch | 1.2150 | w=0.1: B -0.0004, C -0.0041 |
| 1M satir, 64 epoch | 1.3160 | w=0.15: A +0.0046, B +0.0054, C -0.0055 |

Daha fazla veri ve epoch modeli KOTULESTIRDI ve blend'i pozitife cevirdi.
Yalnizca fold C'de tutarli kazanc var. Karar: **RET**.

Not: CPU-only kurulum (Python 3.13 icin CUDA wheel yok). GPU'lu ve ayarlanmis
bir RealMLP daha iyi olabilir; su anki kanit ret yonunde.

## 3. Cold seviyesine lokasyon ofseti — KABUL adayi

Varyans ayristirmasinin dogrudan isaret ettigi bosluktu: cold seviyemiz
`global_z + log_guc`, yani yalnizca guc (R2 0.490). Lokasyonun ekledigi 0.087
bostaydi. `reports/02` lokasyonu reddetmisti ama o olcum kirli `is_cold`
tanimiyla ve duzenlilestirilmemis grup ortalamasiyla yapilmisti.

Ampirik Bayes ofseti, prior taramasi (`reports/42_cold_lokasyon.md`):

| prior | A | B | C | ortalama |
| --- | --- | --- | --- | --- |
| 0 (mevcut) | — | — | — | 1.1948 |
| 20 | +0.0027 | -0.0017 | -0.0041 | 1.1938 |
| **50** | +0.0020 | **-0.0023** | **-0.0061** | **1.1927** |
| 150 | +0.0027 | -0.0029 | -0.0025 | 1.1939 |

cold RMSLE'de de ayni yon (C: -0.0066). Yalnizca fold A kaybettiriyor; fold
A'nin egitim cercevesi kis-agirlikli ve 45 gunluk ufuklu, yani gonderim
yapilandirmasini olcmuyor.

Boru hattina yerlestirildi (`level_shape.lok_offset_table`, `LOK_OFFSET_PRIOR
= 50`), egitim ve tahmin taraflarinda ayni yoldan uygulanıyor. **Yarinki ilk
gonderim adayi.**

## 4. Iki hatali iddia, ikisi de geri alindi

**"RealMLP projede ilk kez kazandiran topluluk."** Zayif koşunun fold B
kazanci -0.0004'tu, yani sifira yakin. Guclu koşu tersini gosterdi. Fazla
erken ve fazla olumlu sunuldu.

**"LightGBM'de ~0.002 belirlenimsizlik var."** Test edildi: LightGBM ayni
tohumla bit-birebir ayni sonucu veriyor; tum boru hatti da oyle (fold B iki
kez 1.020709). Iddia yanlisti.

Gercek sebep daha onemli: fold B ayni yapilandirmada farkli koşularda
1.0207 / 1.0193 / 1.0171 verdi cunku **koşular arasinda kod durumu degisti**.
Sonuc: **farkli zamanlarda calisan script'lerin sayilari kiyaslanamaz.**
Deney tasarimi dogru (her script kendi icinde varyant karsilastiriyor) ama
birkac kez script'ler arasi kiyas yapildi (or. `reports/31` v2_base 1.1946 ile
`reports/32` lgbm_m7 1.1968). O kiyaslar gecersiz; her raporun kendi taban
satiri kullanilmali.

## 5. Teslim edilenler

- `reports/43_variance.md` — varyans ayristirmasi, tavan hesaplari
- `notebooks/01_veri_analizi.ipynb` — 16 hucre, her biri bir karara bagli
- `src/models/neural_prep.py` — sinir agi on islemesi (NaN'i bilgi olarak korur)
- `src/models/level_shape.lok_offset_table` — cold lokasyon ofseti
- `scripts/40..42` — RealMLP, cold lokasyon deneyleri
- 30 test geciyor

## 6. Yarin

1. Cold lokasyon ofseti (prior=50) — LB hakemligi.
2. Sonuca gore: kapi gucu k'yi lokasyon ofsetiyle birlikte yeniden ayarlamak
   (ikisi de cold seviyesine dokunuyor, etkilesebilirler).
3. RealMLP'yi GPU'lu ortamda yeniden denemek (su an CPU-only).
