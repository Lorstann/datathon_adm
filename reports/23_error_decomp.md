# Multiorigin C hata kirilimi
Uretim: 2026-08-22 10:46:42.783473

Butun paylar **kare hata** (log1p uzayi) uzerinden. RMSLE'lerin kendisi toplanamaz; harcanan hatayi ancak kare uzayda bolusturebiliriz.

## Fold B_guncel  (egitim satiri 1,200,000, dogrulama 439,925)

RMSLE all = 0.9319

### cold/warm x sifir/sifir-degil

| dilim | satir | satir payi | kare hata payi | RMSLE |
| --- | --- | --- | --- | --- |
| warm / sifir degil | 354,686 | 80.6% | **24.0%** | 0.5086 |
| warm / sifir | 16,124 | 3.7% | **18.2%** | 2.0754 |
| cold / sifir degil | 65,720 | 14.9% | **15.9%** | 0.9610 |
| cold / sifir | 3,395 | 0.8% | **41.9%** | 6.8681 |

### Varlik yogunlasmasi

| en kotu N varlik | kare hata payi |
| --- | --- |
| 10 / 4,637 | 19.7% |
| 20 / 4,637 | 34.4% |
| 50 / 4,637 | 60.2% |
| 100 / 4,637 | 72.9% |
| 500 / 4,637 | 88.6% |

### Ufka gore

| ufuk | RMSLE | kare hata payi |
| --- | --- | --- |
| 1-30 | 0.7236 | 13.6% |
| 31-60 | 0.8933 | 22.5% |
| 61-90 | 0.9817 | 28.3% |
| 91-130 | 1.0616 | 35.6% |

### Yanlilik (ortalama log1p artik = gercek - tahmin)

| dilim | n | ortalama artik |
| --- | --- | --- |
| warm, sifir degil | 354,686 | +0.0346 |
| cold, sifir degil | 65,720 | +0.1612 |
| ay 1 (sifir degil) | 106,646 | +0.1033 |
| ay 2 (sifir degil) | 100,036 | +0.0600 |
| ay 3 (sifir degil) | 115,234 | +0.0267 |
| ay 12 (sifir degil) | 98,490 | +0.0281 |

## Fold C_ara  (egitim satiri 1,167,286, dogrulama 384,988)

RMSLE all = 1.2367

### cold/warm x sifir/sifir-degil

| dilim | satir | satir payi | kare hata payi | RMSLE |
| --- | --- | --- | --- | --- |
| warm / sifir degil | 289,000 | 75.1% | **31.2%** | 0.7970 |
| warm / sifir | 8,036 | 2.1% | **4.7%** | 1.8604 |
| cold / sifir degil | 81,222 | 21.1% | **13.4%** | 0.9853 |
| cold / sifir | 6,730 | 1.7% | **50.7%** | 6.6607 |

### Varlik yogunlasmasi

| en kotu N varlik | kare hata payi |
| --- | --- |
| 10 / 4,030 | 11.5% |
| 20 / 4,030 | 19.5% |
| 50 / 4,030 | 38.7% |
| 100 / 4,030 | 62.1% |
| 500 / 4,030 | 87.2% |

### Ufka gore

| ufuk | RMSLE | kare hata payi |
| --- | --- | --- |
| 1-30 | 0.6383 | 5.3% |
| 31-60 | 1.0682 | 16.8% |
| 61-90 | 1.4004 | 34.0% |
| 91-130 | 1.4703 | 43.9% |

### Yanlilik (ortalama log1p artik = gercek - tahmin)

| dilim | n | ortalama artik |
| --- | --- | --- |
| warm, sifir degil | 289,000 | +0.0294 |
| cold, sifir degil | 81,222 | +0.0998 |
| ay 1 (sifir degil) | 107,649 | +0.1427 |
| ay 10 (sifir degil) | 77,131 | -0.1169 |
| ay 11 (sifir degil) | 84,094 | -0.0414 |
| ay 12 (sifir degil) | 101,348 | +0.1356 |
