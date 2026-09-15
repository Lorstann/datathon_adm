# Cold gelistirme denemeleri — sonuc

Uretim ozeti (havayla CV, ayni 3 fold).

| Deneme | blend | cold | vs C |
| --- | --- | --- | --- |
| level_shape_C (baseline) | **1.336** | **1.875** | — |
| Router C + B (z, hist dusur) | 1.362 | 1.955 | cold kotu |
| Router C + cold-shape | 1.343 | 1.896 | cold kotu |
| C mask50 / mask70 / augment | ≥1.336 | ≥1.874 | esik altinda |

**Bulgu:** Mevcut C’nin cold yolu (`global_z + log_guc` + hist-maskeli sekil) denenen tum cold-ozel modellerden iyi. Ayrı B/cold-shape cold’u bozuyor; mask/augment warm’i biraz bozup cold’u iyilestirmiyor.

**Gonderim:** Kapı tutulmadi — yeni submission yok. LB’deki C (1.14884) duruyor.

**Sonraki gercekci adaylar (henuz denenmedi):**
1. Cold seviye icin `guc_band × month` medyan (peer degil, ay kosullu)
2. Nested: fold-ici cold residual → kucuk dogrusal ay/tatil duzeltmesi (oracle ay bias tavanı Fold A’da ~0.03; B/C’de zarar)
3. Warm’i oldugu gibi birakip cold’ta **quantile / pinball** veya sifir-agirlikli kayip
4. Dis veri: yalniz cold’a yaz turizm ilce bayragi (onceki planda dusuk oncelik)
