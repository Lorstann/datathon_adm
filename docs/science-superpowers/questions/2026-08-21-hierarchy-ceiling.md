# Hiyerarşik top-down tavanı C'yi geçer mi?

**Research question:** Origin-öncesi bilinen veriyle kurulan hiyerarşik top-down (grup gün toplamı × varlık payı) veya guc-bandı profil kümesi naif tahmincileri, aynı 3-fold cold-maskeli validasyonda `level_shape_C` havayla CV skorunu blend'te en az 0.05 veya cold'ta en az 0.08 geçebilir mi?

**Background / motivation:** Public LB öncelikli. Mevcut C warm'ta güçlü, cold'ta zayıf (~%22 test satırı). Ayrı cold kolları C'yi geçemedi. Kahin seviye ~0.5 vs metadata ~2.0; seviye tavanı şüpheli. Mimariye geçmeden önce naif tavan ölçülür (yol C).

**Hypotheses:**
- H0: Hiçbir Adım-1 naif/hiyerarşi tavani, 3-fold ortalamada `rmsle_blend ≤ C_blend − 0.05` veya (`rmsle_cold ≤ C_cold − 0.08` ve blend kötüleşmez) koşulunu sağlamaz.
- H1: En az bir tavan bu eşiği geçer → hiyerarşi mimarisine yatırım haklıdır.

**Population & unit of analysis:** GDZ/ADM trafo-gün paneli; birim = (`tanim`, `tarih`). Fold'lar `src/config.FOLDS`; cold maskesi gerçek test cold profiline tabakalı.

**Key variables:**
- Outcome: `tuketim` → RMSLE / `rmsle_blend` / `rmsle_cold`
- Exposure: tahminci ailesi (top-down pay, YoY, guc-kume profili, guc_band×month z)
- Referans: `level_shape_C` havayla CV (blend ≈ 1.336, cold ≈ 1.875)

**What counts as an answer:**
- **Go (H1):** eşik tutar → Adım 2 hiyerarşi modeli.
- **No-go (H0):** eşik tutmaz → Adım 3 dar C (YoY cold seviye / bias); mimari yatırımı yok.

**Scope & exclusions:** Foundation modeller, ağır ensemble, lokasyon-kNN, MinT ana yöntem, agresif soft-zero seviye çarpanı dışarıda. Trafo-gün gerçekleşen hedef aranmaz (sızıntı).

**Open questions for prior-work survey:** Zaten `docs/prior-work.md` (global vs local, cold metadata, hiyerarşi). Yeni tarama gerekmez; tavan ölçümü karar verir.
