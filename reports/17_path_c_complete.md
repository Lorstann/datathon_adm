# Yol C tamamlandi — LB gonderim yok

## Tavan (Adim 1)
- Rapor: `reports/15_hierarchy_ceiling.md`
- **NO-GO**: hicbir naif/hiyerarsi tavani C'yi blend-0.05 veya cold-0.08 ile gecemedi.
- En yakin: `profile_kmeans` blend 1.372 (+0.036 vs C). Cold hala ~1.88.

## Dar C (Adim 3)
- Rapor: `reports/16_narrow_c.md`
- YoY cold seviye, sequential cold bias, soft-zero: hepsi baseline'dan kotu veya notr.
- **submit_ok=False** → `submission_narrow_c_v3` uretilmedi, Kaggle'a gonderilmedi.

## Dis veri P0
- `data/external/epias/README.md` — kayitli API / UI CSV talimati
- `data/external/tuik/README.md` — ADNKS ilce nufus talimati
- Bu ortamda otomatik indirme basarisiz (SSL/403). Dosya gelince ozellik eklenir.

## Framing
- `docs/science-superpowers/questions/2026-08-21-hierarchy-ceiling.md` — H0 kabul (tavan esigi tutmadi).

## LB durumu
- Mevcut en iyi public: **level_shape_C ~1.14884**
- Lokal kapı yeni aday uretmedi; gunluk hak saklandi.
