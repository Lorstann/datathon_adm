# Feature family ablation (level_shape_C, with weather baseline)
Uretim: 2026-08-21 16:51:03.926832

## Fold A_mevsim
| Family | blend | warm | cold | delta_vs_full |
| --- | --- | --- | --- | --- |
| full | 1.4165 | 1.0040 | 2.3480 | +0.0000 |
| no_weather | 1.4169 | 1.0056 | 2.3467 | +0.0004 |
| no_calendar | 1.4163 | 1.0036 | 2.3481 | -0.0002 |
| no_history | 1.4214 | 1.0115 | 2.3502 | +0.0050 |

## Fold B_guncel
| Family | blend | warm | cold | delta_vs_full |
| --- | --- | --- | --- | --- |
| full | 1.2019 | 1.0175 | 1.6976 | +0.0000 |
| no_weather | 1.2051 | 1.0222 | 1.6980 | +0.0032 |
| no_calendar | 1.2012 | 1.0158 | 1.6989 | -0.0007 |
| no_history | 1.2563 | 1.0994 | 1.6959 | +0.0544 |

Pozitif delta = aileyi cikarmak skoru kotulestirir (aile faydali). Negatif delta = aile zararli veya gurultu.

## Yorum

- **Fold A (mevsim / hava ekstrapolasyonu):** hava neredeyse notr (+0.0004). Beklenen "hava > 0" etkisi bu fold'da cok zayif.
- **Fold B (guncel):** gecmis baskin (+0.054); hava kucuk ama pozitif (+0.003).
- **Takvim (tatil/hafta sonu):** her iki fold'da da cikarmak hafifce iyilestiriyor (−0.0002 / −0.0007) — gurultu veya asiri uyum; aileyi tutuyoruz ama genisletmeye oncelik yok.
