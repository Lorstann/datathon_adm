# Ucuncu tur: alti deneme, alti ret

Uretim tarihi: 2026-08-23

LB degismedi: **1.06899** (`submission_multiorigin.csv`, v2 + kapi k=0.75).
Bu turda gonderim yapilmadi, cunku uc fold'un tamaminda kazandiran hicbir
degisiklik cikmadi.

## Denenenler

| deneme | fold A | fold B | fold C | karar |
| --- | --- | --- | --- | --- |
| Ozellik paketi v3 (tumu) | -0.0059 | +0.0026 | +0.0098 | RET |
| v3 yalniz hava carpanlari | -0.0068 | +0.0058 | +0.0046 | RET |
| `cdd_x_slope` tek basina | 0.0000 | -0.0009 | +0.0012 | RET (etkisiz) |
| `hdd_x_slope` tek basina | -0.0068 | +0.0035 | +0.0064 | RET |
| Seviye cesitliligi (m7+m28 ortalamasi) | -0.0046 | -0.0034 | +0.0054 | RET |
| XGBoost (kategorikler onarildi) | +0.3167 | +0.4968 | +0.1760 | RET |
| Kapi kalibrasyon gridi (24 nokta) | — | — | — | RET (3/3 iyilestiren yok) |
| Dis veri (TÜİK + EPİAŞ), temiz metrik | -0.0010 | +0.0042 | -0.0030 | RET |

## Ogrenilen: doz-tepki tuzagi

v3'un hava carpanlari fold A'da kazandirip B/C'de kaybettirdi. Fold A'nin ufku
Nis-Tem, yani gonderim ufkuyla ayni takvim penceresi, dolayisiyla "yalniz
mevsimsel olarak benzer fold'a bakalim" demek cazipti. Doz-tepki taramasi da
destekler gorundu: fold A'da kazanc ayla (Nis +1003, May -1524, Haz -2595,
Tem -3253) ve CDD dortte birleriyle (-866 / -2375 / -3128) MONOTON artiyordu.

Carpanlari ayirinca tablo tersine dondu:

| fold A varyanti | blend |
| --- | --- |
| v2_base | 1.3586 |
| `cdd_only` | 1.3586 (bit-birebir ayni) |
| `hdd_only` | 1.3518 |
| `cdd_hdd` | 1.3518 |

`cdd_only` ile taban **birebir ayni** cikti: fold A'nin egitim cercevesi
15 Sub - 31 Mar arasini kapsiyor, orada CDD ~ 0, yani `cdd_x_slope` egitimde
sabit-sifir bir kolon ve agac hic dallanmiyor. Kazancin tamami
`hdd_x_slope`'tan geliyordu. CDD ile monoton gorunen desen nedensel degil
takvimsel bir es-degisimdi: Temmuz'a dogru CDD de ay indeksi de artiyor, HDD
azaliyor.

Sonuc: mekanizma anlatisi doz-tepki egrisiyle desteklenmis gibi gorunse bile,
bileseni izole etmeden karar verilmemeli. Izolasyon olmasa yanlis gerekceyle
dogru olmayan bir ozellik gonderilecekti.

## Fold A neden bu turda tekrar tekrar yaniltti

Fold A origin'i 2025-03-31, veri 2025-01-01'de basliyor. Cok-origin
cercevesinde 45 gun arayla tek bir kullanilabilir origin kaliyor ve o
origin'in ufku da 45 gunle kirpiliyor. Yani fold A:

- egitim satirlari **yalnizca 1-45 gun ufuklu**, dogrulama 1-122 gun,
- egitim penceresi **yalnizca kis** (Sub-Mar), dogrulama Nis-Tem.

Dolayisiyla fold A, gonderimde kurdugumuz yapilandirmayi hic olcmuyor. Daha
sik origin (14 gun arayla 4 origin) satir sayisini artirir ama ufuk kirpmasini
COZMEZ: 122 gunluk ufuklu bir egitim ornegi bu veriyle fold A'da uretilemez.
Yapisal kisit.

Buna ragmen fold A karar kuralindan CIKARILMADI. Tur ortasinda, kaybeden
denemeleri kazanana cevirecek sekilde fold atmak, hayatta kalan fold'lara
asiri uyumun tam tanimi.

## Kod durumu

v3 ozellikleri koddan kaldirildi. Onemliydi: olculmus olarak daha kotu
olduklari halde kod tabaninda duruyorlardi, yani `22_submit_multiorigin.py`
o haliyle calistirilsa gonderilmis modelden (1.1946) daha kotu bir model
(1.1968) uretirdi.

Kalanlar (v2, gonderimdeki hali): `hist_mean_14/21`, `hist_mom_*`,
`hist_yoy_gap`, `hist_q25/q75/iqr_91`, `hist_dow_rel_*`, `hist_cdd_slope`,
`hist_hdd_slope`, `hist_rel_peer`.

`fit_xgboost` onarimi KORUNDU (kategorikleri artik dusurmuyor,
`enable_categorical=True`). Model yine de cok kotu (1.5266) ama artik
karsilastirma adil; eski surumde XGBoost `lokasyon_cat`'i hic gormuyordu.

29 test geciyor.

## Sonraki tur icin durus

Alti bagimsiz denemenin altisi da reddedildi. Bu, model yerel bir optimumda
demek. Kalan bilinen tavan (`reports/24_zero_shrink.md`) cold trafolarin olu
olup olmadigini bilmekte ve o bilgi veride yok (siniflandirici AUC 0.584 <
lokasyonun 0.616).

Yeni gonderim, yeni bir bilgi kaynagi ya da yeni bir yapisal fikir
gerektiriyor; parametre ve ozellik taramasi bu noktada getiri uretmiyor.
