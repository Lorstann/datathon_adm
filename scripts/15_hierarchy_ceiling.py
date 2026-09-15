"""Hiyerarsik top-down ve cold profil tavan olcumu (naif, vektorize).

Kullanim: python scripts/15_hierarchy_ceiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src import config as C
from src.data.load import build_panel, entity_static, load_test, load_train, split_location
from src.tracking import log_experiment
from src.validation.folds import build_cold_profile, guc_band
from src.validation.metrics import segment_report, summarize_folds
from src.validation.split import make_fold

C_BLEND_REF = 1.3359
C_COLD_REF = 1.8746


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "il" not in out.columns:
        out = out.join(split_location(out[C.LOCATION]))
    out["guc_band"] = guc_band(out[C.POWER]).astype("string")
    out["dow"] = out[C.DATE].dt.dayofweek.astype("int8")
    out["month"] = out[C.DATE].dt.month.astype("int8")
    out["doy"] = out[C.DATE].dt.dayofyear.astype("int16")
    out["year"] = out[C.DATE].dt.year.astype("int16")
    out["ilce_key"] = out["ilce"].fillna(out["bolge"]).fillna(out["il"]).astype("string")
    return out


def topdown_predict(
    hist: pd.DataFrame,
    valid: pd.DataFrame,
    is_cold: np.ndarray,
    *,
    group_cols: list[str],
    yoy: bool = False,
) -> tuple[np.ndarray, float]:
    h, v = hist, valid
    day_grp = h.groupby([C.DATE, *group_cols], observed=True)[C.TARGET].sum().reset_index()
    day_grp["dow"] = pd.to_datetime(day_grp[C.DATE]).dt.dayofweek.astype("int8")
    day_grp["doy"] = pd.to_datetime(day_grp[C.DATE]).dt.dayofyear.astype("int16")
    day_grp["year"] = pd.to_datetime(day_grp[C.DATE]).dt.year.astype("int16")

    key_dow = [*group_cols, "dow"]
    grp_dow = day_grp.groupby(key_dow, observed=True)[C.TARGET].median().rename("grp_day")

    # warm share: entity / group total consumption
    esum = h.groupby([C.ENTITY, *group_cols], observed=True)[C.TARGET].sum().rename("esum")
    gsum = h.groupby(group_cols, observed=True)[C.TARGET].sum().rename("gsum")
    es = esum.reset_index().merge(gsum.reset_index(), on=group_cols, how="left")
    es["share_warm"] = np.where(es["gsum"] > 0, es["esum"] / es["gsum"], np.nan)
    share_warm = es.set_index(C.ENTITY)["share_warm"]

    # cold share by guc within group (entities seen in hist or valid)
    meta = pd.concat(
        [h[[C.ENTITY, C.POWER, *group_cols]], v[[C.ENTITY, C.POWER, *group_cols]]],
        ignore_index=True,
    ).drop_duplicates(C.ENTITY)
    guc_sum = meta.groupby(group_cols, observed=True)[C.POWER].sum().rename("guc_sum")
    meta = meta.merge(guc_sum.reset_index(), on=group_cols, how="left")
    meta["share_guc"] = np.where(meta["guc_sum"] > 0, meta[C.POWER] / meta["guc_sum"], np.nan)
    share_guc = meta.set_index(C.ENTITY)["share_guc"]

    n_ent = h.groupby(group_cols, observed=True)[C.ENTITY].nunique().rename("n_ent")
    meta = meta.merge(n_ent.reset_index(), on=group_cols, how="left")
    meta["share_eq"] = np.where(meta["n_ent"] > 0, 1.0 / meta["n_ent"], np.nan)
    share_eq = meta.set_index(C.ENTITY)["share_eq"]

    out = v[[C.ENTITY, *group_cols, "dow", "doy", "month"]].copy()
    out = out.merge(grp_dow.reset_index(), on=[*group_cols, "dow"], how="left")
    hit = float(out["grp_day"].notna().mean())
    dow_fb = day_grp.groupby("dow")[C.TARGET].median()
    out["grp_day"] = out["grp_day"].fillna(out["dow"].map(dow_fb)).fillna(0.0)

    if yoy:
        day_grp["month"] = pd.to_datetime(day_grp[C.DATE]).dt.month.astype("int8")
        wide = (
            day_grp.groupby([*group_cols, "doy", "year"], observed=True)[C.TARGET]
            .mean()
            .unstack("year")
        )
        years = sorted(wide.columns.tolist())
        if len(years) >= 2:
            y_ref, y_prev = years[-1], years[-2]
            ratio = (
                (wide[y_ref] / wide[y_prev].replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .rename("yoy_scale")
                .reset_index()
            )
            mwide = (
                day_grp.groupby([*group_cols, "month", "year"], observed=True)[C.TARGET]
                .mean()
                .unstack("year")
            )
            mratio = (
                (mwide[y_ref] / mwide[y_prev].replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .rename("yoy_m")
                .reset_index()
            )
            out = out.merge(ratio, on=[*group_cols, "doy"], how="left")
            out = out.merge(mratio, on=[*group_cols, "month"], how="left")
            scale = out["yoy_scale"].fillna(out["yoy_m"]).fillna(1.0)
            out["grp_day"] = out["grp_day"] * scale

    share = np.where(
        is_cold,
        share_guc.reindex(out[C.ENTITY]).to_numpy(),
        share_warm.reindex(out[C.ENTITY]).to_numpy(),
    )
    share = np.where(np.isfinite(share), share, share_eq.reindex(out[C.ENTITY]).to_numpy())
    share = np.where(np.isfinite(share), share, 0.0)
    pred = np.clip(out["grp_day"].to_numpy(dtype="float64") * share, 0.0, None)
    return pred, hit


def guc_month_z_predict(hist: pd.DataFrame, valid: pd.DataFrame, *, yoy_fallback: bool) -> tuple[np.ndarray, float]:
    h = hist.copy()
    h["z"] = np.log1p(h[C.TARGET].clip(lower=0)) - np.log(h[C.POWER].clip(lower=1).astype("float64"))
    tab = h.groupby(["guc_band", "month"], observed=True)["z"].median().rename("z")
    tab_ym = h.groupby(["guc_band", "month", "year"], observed=True)["z"].median().rename("z_ym")

    out = valid[["guc_band", "month", "year", C.POWER]].copy()
    out = out.merge(tab.reset_index(), on=["guc_band", "month"], how="left")
    if yoy_fallback:
        # fill from any year same month in hist (prefer recent)
        fill = tab_ym.reset_index().sort_values("year").drop_duplicates(["guc_band", "month"], keep="last")
        out = out.merge(fill[["guc_band", "month", "z_ym"]], on=["guc_band", "month"], how="left")
        out["z"] = out["z"].fillna(out["z_ym"])
    hit = float(out["z"].notna().mean())
    out["z"] = out["z"].fillna(h["z"].median())
    pred = np.clip(np.expm1(out["z"] + np.log(out[C.POWER].clip(lower=1).astype("float64"))), 0.0, None)
    return pred.to_numpy(), hit


def profile_cluster_predict(hist: pd.DataFrame, valid: pd.DataFrame, is_cold: np.ndarray, seed: int = C.SEED) -> np.ndarray:
    h = hist.copy()
    h["log1p"] = np.log1p(h[C.TARGET].clip(lower=0))
    piv = h.pivot_table(index=C.ENTITY, columns="dow", values="log1p", aggfunc="mean")
    piv = piv.reindex(columns=range(7))
    piv = piv.apply(lambda r: r.fillna(r.mean()), axis=1).dropna(how="all")
    X = piv.to_numpy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    Xn = X / norms
    k = int(min(8, max(2, len(piv) // 30)))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Xn)
    ent_cl = pd.Series(labels, index=piv.index, name="cluster")

    hh = h.merge(ent_cl, left_on=C.ENTITY, right_index=True, how="inner")
    cl_prof = hh.groupby(["cluster", "dow"], observed=True)["log1p"].mean()
    cl_mean = hh.groupby("cluster")["log1p"].mean()
    level_warm = h.groupby(C.ENTITY)["log1p"].mean()
    global_z = float((h["log1p"] - np.log(h[C.POWER].clip(lower=1))).median())

    ent_band = h.groupby(C.ENTITY)["guc_band"].first()
    band_cl = (
        pd.DataFrame({"guc_band": ent_band, "cluster": ent_cl})
        .dropna()
        .groupby("guc_band")["cluster"]
        .agg(lambda s: int(s.mode().iloc[0]))
    )

    pred = np.empty(len(valid), dtype="float64")
    for i, (ent, dow, guc, gb) in enumerate(
        zip(valid[C.ENTITY], valid["dow"], valid[C.POWER], valid["guc_band"], strict=True)
    ):
        log_guc = np.log(max(float(guc), 1.0))
        if (not is_cold[i]) and ent in level_warm.index and ent in ent_cl.index:
            cl = int(ent_cl[ent])
            prof = cl_prof.get((cl, int(dow)), level_warm[ent])
            shape = float(prof) - float(cl_mean.get(cl, level_warm[ent]))
            pred[i] = max(0.0, float(np.expm1(float(level_warm[ent]) + shape)))
        else:
            cl = int(band_cl.get(gb, 0))
            prof = cl_prof.get((cl, int(dow)), global_z + log_guc)
            shape = float(prof) - float(cl_mean.get(cl, global_z + log_guc))
            pred[i] = max(0.0, float(np.expm1(global_z + log_guc + shape)))
    return pred


def main() -> None:
    C.ensure_dirs()
    raw_train = load_train()
    test = load_test()
    static = entity_static(build_panel(raw_train, test))
    profile = build_cold_profile(static, set(raw_train[C.ENTITY].unique()))

    rows = []
    lines = [
        "# Hierarchy / profile ceiling\n",
        f"Uretim: {pd.Timestamp.now()}\n",
        f"Referans C: blend={C_BLEND_REF:.4f}, cold={C_COLD_REF:.4f}\n",
        "Go: blend <= C-0.05 VEYA (cold <= C-0.08 ve blend kotulesmesin).\n",
    ]
    methods = [
        "td_guc_dow",
        "td_ilce_guc_dow",
        "td_guc_dow_yoy",
        "td_ilce_guc_dow_yoy",
        "guc_month_z",
        "guc_month_z_yoy",
        "profile_kmeans",
    ]

    for fold in C.FOLDS:
        print(f"=== {fold.name} ===")
        split = make_fold(raw_train, fold, profile, seed=C.SEED)
        hist = _prep(split.train)
        valid = _prep(split.valid)
        cold = np.asarray(split.is_cold, dtype=bool)

        preds: dict[str, np.ndarray] = {}
        covs: dict[str, float] = {}

        for name, gcols, yoy in (
            ("td_guc_dow", ["guc_band"], False),
            ("td_ilce_guc_dow", ["ilce_key", "guc_band"], False),
            ("td_guc_dow_yoy", ["guc_band"], True),
            ("td_ilce_guc_dow_yoy", ["ilce_key", "guc_band"], True),
        ):
            p, cov = topdown_predict(hist, valid, cold, group_cols=gcols, yoy=yoy)
            preds[name], covs[name] = p, cov

        p, cov = guc_month_z_predict(hist, valid, yoy_fallback=False)
        preds["guc_month_z"], covs["guc_month_z"] = p, cov
        p, cov = guc_month_z_predict(hist, valid, yoy_fallback=True)
        preds["guc_month_z_yoy"], covs["guc_month_z_yoy"] = p, cov

        preds["profile_kmeans"] = profile_cluster_predict(hist, valid, cold)
        covs["profile_kmeans"] = 1.0

        lines.append(f"\n## Fold {fold.name}\n")
        lines.append("| method | warm | cold | blend | coverage |\n| --- | --- | --- | --- | --- |\n")
        for m in methods:
            metrics = segment_report(valid[C.TARGET], preds[m], split.segment, is_cold=split.is_cold)
            rows.append({"model": m, "fold": fold.name, **metrics, "coverage": covs[m]})
            log_experiment(
                exp_id=f"ceil_{m}",
                model=m,
                target="log1p",
                feature_set="hierarchy_ceiling",
                fold=fold.name,
                metrics=metrics,
                notes=f"cov={covs[m]:.3f}",
            )
            lines.append(
                f"| {m} | {metrics.get('rmsle_warm', float('nan')):.4f} | "
                f"{metrics.get('rmsle_cold', float('nan')):.4f} | "
                f"{metrics.get('rmsle_blend', float('nan')):.4f} | {covs[m]:.3f} |\n"
            )
            print(f"  {m}: blend={metrics['rmsle_blend']:.4f} cold={metrics['rmsle_cold']:.4f}")

    df = pd.DataFrame(rows)
    lines.append(
        "\n## Ozet\n| method | blend | cold | vs C blend | vs C cold |\n| --- | --- | --- | --- | --- |\n"
    )
    summary = {}
    best_go = None
    for m, g in df.groupby("model"):
        b = summarize_folds(g, "rmsle_blend")["mean"]
        c = summarize_folds(g, "rmsle_cold")["mean"]
        summary[m] = {"blend": float(b), "cold": float(c)}
        lines.append(
            f"| {m} | {b:.4f} | {c:.4f} | {b - C_BLEND_REF:+.4f} | {c - C_COLD_REF:+.4f} |\n"
        )
        go = (b <= C_BLEND_REF - 0.05) or (
            c <= C_COLD_REF - 0.08 and b <= C_BLEND_REF + 0.005
        )
        if go and (best_go is None or b < summary[best_go]["blend"]):
            best_go = m

    decision = {
        "c_blend_ref": C_BLEND_REF,
        "c_cold_ref": C_COLD_REF,
        "summary": summary,
        "go": best_go is not None,
        "best_method": best_go,
        "path": "hierarchy" if best_go is not None else "narrow_C",
    }
    lines.append(
        f"\n**Go/No-go: {'GO — ' + str(best_go) if best_go else 'NO-GO'}** → `{decision['path']}`\n"
    )
    (C.REPORTS_DIR / "15_hierarchy_ceiling.md").write_text("".join(lines), encoding="utf-8")
    (C.REPORTS_DIR / "15_hierarchy_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("DECISION", decision)


if __name__ == "__main__":
    main()
