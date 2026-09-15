# Grid Up Datathon — trafo bazlı günlük tüketim tahmini

GDZ/ADM Grid Up: 2026-04-01 – 2026-07-31 arası trafo-gün aktif tüketim (kWh), metrik RMSLE.

## Hızlı başlangıç

```powershell
pip install -r requirements.txt
python -m pip install -e .
python scripts/01_data_quality.py
python scripts/02_cold_start_ceiling.py
python scripts/03_eda.py
python scripts/04_fetch_weather.py
python scripts/05_baselines.py
python scripts/06_train.py
python scripts/20_multiorigin_cv.py    # duzeltilmis metrik + cok-origin CV
python scripts/22_submit_multiorigin.py   # LB 1.06899
```

Kaggle gönderimi `submissions/` altına yazılır; yükleme sozlu onay + Kaggle MCP ile.

## Kurallar

- `data/raw/` asla değiştirilmez.
- Validasyon rastgele split değil; 3 sabit-origin fold + tabakalı cold-start maskelemesi.
- `is_cold` = origin'de gecmisi olmayan satir; maskelenmis olmak yetmez.
- Egitim satirlari test satirlariyla ayni sekilde uretilir: donmus gecmis + ufuk 1..122.
- `log1p` hedef, L2, `expm1` geri dönüş. Smearing yok.
- Her deney `experiments.csv` satırı.

## Dokümanlar

- `docs/science-superpowers/questions/2026-08-21-trafo-tuketim-tahmini.md` — çerçeve
- `docs/superpowers/specs/2026-08-21-grid-up-datathon-design.md` — tasarım
- `docs/prior-work.md` — literatür
- `docs/external-data.md` — dış veri atıfları
- `reports/30_round2_summary.md` — ikinci tur: hata kirilimi, sifir kapisi, ozellik paketi
- `reports/22_pipeline_fix.md` — metrik ve egitim cercevesi duzeltmesi (once bunu oku)
- `reports/` — üretilmiş analiz çıktıları
