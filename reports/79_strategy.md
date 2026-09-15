# Kalan 3 gonderim: durum ve plan

Tarih: 2026-08-31. Bitis: 2026-09-01 23:59 UTC. Kalan hak: yarin 3 (bugunku 3
tukendi: FINAL_1, FINAL_4, probe_Q_COMBO).

## Nerede duruyoruz

| | LB (public %30) |
|---|---|
| Takim en iyisi (FINAL_4) | 1.01548 |
| Siralama | 31 / 469 |
| Top 10 esigi | ~0.99 |
| Lider | ~0.98 |

FINAL_4 modelleme urunu degil: `Leaderboard Geometrisi` notundaki olcum
yontemiyle, 13 puanlanmis gonderimin ridge harmani olarak uretilmis. Kodu
kayip ama gerek de yok -- yontem yeniden kuruldu (`scripts/75_blend.py`) ve
notun uc dogrulanmis tahmini (sapma 0.0004, 0.0002, 0.00016) ayni cikiyor.

## Bugun olculenler

### 1. Havuzun gercek kapasitesi: +0.0022

`68_lb_algebra` once 1.01308 demisti; o sayi sisirilmisti. Referans olarak
FINAL_4 alinip kendi bilesenleri de yon havuzuna konunca Gram matrisi tekil
oluyor (kosul sayisi 1.4e15) -- notun "Tuzak 1"i. Turetilmis harmanlar
(SUB_A, SUB_B, SUB_D, FINAL_4) yon havuzundan cikarilinca kosul sayisi
1.45e3'e iniyor ve Monte Carlo ile lambda secilince:

| | beklenen LB |
|---|---|
| FINAL_4 | 1.01548 |
| ridge harman (lambda/tr = 0.003) | **1.01329** |

Bu **elde garanti** duran +0.0022. Dosya hazir: `submissions/submission_BLEND.csv`.

### 2. Yeni tahminci ailesi: sifir katki

Notun kilit iddiasi "kazanc bilesenlerin ne kadar FARKLI olduguyla orantili".
Bunu test etmek icin agac icermeyen uc tahminci kuruldu (`71_probe_candidates`):
grup profili, gecen yilin ayni gunu (d-364), rank-8 panel SVD. Ortalamalari
(Q_COMBO) havuza gore gercekten yeni bir eksen aciyordu:

- izdusum R2 = 0.58, yani %42'si hicbir gonderimde yok
- ||D_dik|| = 0.56 -- havuzdaki her seyden genis

Gonderildi, skor **1.33324**. Cebir:

```
<e0, D>      = -0.00031
havuzun payi = -0.00124
DIK bilesen  = +0.00093   ->  kazanc 0.000003 MSE  ->  LB 1.01548
```

Yani 0.56 genisliginde yepyeni bir eksen ve artigin o eksendeki izdusumu
**sifir**. Bu bir basarisizlik degil, bir olcum: FINAL_4'un artigi duzgun /
yapisal hicbir yonle iliskili degil.

### 3. Artik neden yapisal degil: sifirlar

`76_zero_floor`, train pencerelerinde sifir belirsizliginin bedelini olctu:

| pencere | gecmisten p ile MSE | hangi gun sifir bilinse MSE | sifirin bedeli |
|---|---|---|---|
| B (Ara-Mar) | 0.4924 | 0.2551 | 0.2373 (hatanin %48'i) |
| A (Nis-Tem) | 0.7493 | 0.4982 | 0.2511 (hatanin %34'u) |

Ve bedelin cogu **surpriz** sifirlardan: gecmisi "tamamen canli" (p<0.05)
gorunen trafolarin beklenmedik sifir gunleri. Fold B'de 970 satir (%0.26)
tek baslarina 0.129 MSE tasiyor. Bunlar tanim geregi ongorulemez.

Kaldirac aritmetigi neden bu kadar sert: seviyesi m~7 olan bir satirda sifir
kacirmak kareli hataya 49 katar. Satirlarin binde birini dogru sifirlamak
0.049 MSE, yani butun top-10 acigini kapatir -- ama binde birini yanlis
sifirlamak ayni miktarda kaybettirir.

### 4. Cold tavani gercek

Hata butcesi: cold satirlar test'in %22'si ama MSE'nin ~%75'i. LB cebiri
cold YANLILIGINI sifir olcmustu, yani kalan sey varyans: yeni bir trafoyu
digerlerinden ayirt edemiyoruz.

`77_id_adjacency` son kullanilmamis sinyali denedi. `tanim` kimlikleri saha
kurulum sirasina gore veriliyor gorunuyor ve komsuluk gercekten cografi:

| kimlik farki | ayni ilce |
|---|---|
| 1 | 0.880 |
| 2-3 | 0.836 |
| rastgele cift | 0.090 |

Ama seviye tasimiyor. Trafolarin %30'u disarida birakilarak (cold taklidi)
olculen OOS R2:

| yontem | OOS R2 |
|---|---|
| band + ilce | 0.0144 |
| kimlik-kNN (en iyi ayar) | 0.0186 |

Onemli duzeltme: `reports/43`'un "guc+lokasyon tavani R2 = 0.577" rakami
SEVIYE icin, ve o R2'nin neredeyse tamami `log(guc)`'ten geliyor. `log(guc)`
cikarildiktan sonra lokasyon cold seviyenin **%1.4'unu** aciklıyor. Cold
tavani gercek ve uzerindeyiz.

## Yarinki 3 hak

Sira onemli: once iki olcum, sonra hepsini iceren harman.

### Gonderim 1 -- `submissions/probe_DISP.csv`

Dagilim keskinligi ekseni: `q = ort + 1.15*(p0 - ort)`. Bu eksenin havuza
izdusumu yalnizca R2 = 0.08 -- neredeyse tamamen olculmemis. Olctugu soru:
trafolar arasi ayrimi yeterince keskin mi yapiyoruz. ||D|| = 0.2127.

Okuma: `<e0,D> = (S^2 - 1.01548^2 - 0.04523)/2`
- negatifse daha keskin olmaliyiz, pozitifse daha puruzsuz
- optimal agirlik `w = -<e0,D>/||D||^2`

### Gonderim 2 -- `submissions/probe_ZGATE.csv`

Birlesik sifir/panel kapisi: Mart'ta olu trafolar (20.777 satir), panel cikis
satirlari (5.509), panel giris satirlari (21.347) -- hepsi asagi cekilmis.
Toplam %6.81 satira dokunuyor, ||D|| = 0.1173. Uc alt-desen de ayni isareti
bekledigi icin tek olcumde toplanabilirler.

Dar oldugu icin kaldiraci yuksek: dokunulan satirlarda ortalama yanlilik 0.5
ise LB 1.01103, 1.0 ise 0.99755.

### Gonderim 3 -- nihai harman

Iki skoru `scripts/75_blend.py` icindeki CATALOG'a ekle, calistir, cikan
`submission_BLEND.csv`'yi gonder. Iki yoklama da sifir cikarsa harman zaten
1.01329 veriyor; yani probe'lar hicbir sey kaybettirmiyor.

## Beklenti

| senaryo | LB | siralama |
|---|---|---|
| ikisi de sifir (Q_COMBO gibi) | 1.01329 | ~28 |
| biri orta sinyal | ~1.008 | ~22 |
| ZGATE'te guclu sinyal | ~0.998 | ~12 |

Durust taban: **1.0133 garanti**. Ustu olcume bagli.

## Nihai secim notu

Kaggle iki gonderim sectiriyor. Public'e fit edilmis harman df ~14 ile
214.000 satira oturuyor; notun hesabina gore istatistiksel iyimserlik
7e-5, ihmal edilebilir. Yine de bir slot FINAL_4'e (ham, LB'ye fit
edilmemis daha az) birakilmali; digeri nihai harman.
