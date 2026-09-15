"""Fold bolumu: egitim / dogrulama cerceveleri, maskeleme, sizinti denetimi."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config as C
from src.audit.leakage import run_all_checks
from src.validation.folds import (
    ColdProfile,
    apply_entry_delay,
    assign_entry_offsets,
    cold_start_mask,
    history_length,
    history_segment,
    mask_history,
)


@dataclass
class FoldSplit:
    fold: C.Fold
    train: pd.DataFrame
    valid: pd.DataFrame
    masked_entities: pd.Index
    masked_offsets: pd.Series  # maskelenen trafonun ufka giris gecikmesi (gun)
    history_days: pd.Series  # valid satirlari ile hizali, maskeleme SONRASI
    is_cold: pd.Series
    segment: pd.Series
    checks: dict[str, str]


def make_fold(
    observed: pd.DataFrame,
    fold: C.Fold,
    profile: ColdProfile,
    *,
    seed: int = C.SEED,
    apply_mask: bool = True,
) -> FoldSplit:
    """Tek fold icin sizintisiz egitim/dogrulama cerceveleri.

    `observed` hedefi bilinen ham train satirlari (maskesiz).
    """
    valid = observed.loc[
        (observed[C.DATE] >= fold.valid_start) & (observed[C.DATE] <= fold.valid_end)
    ].copy()
    pre = observed.loc[observed[C.DATE] <= fold.origin].copy()

    hist_before = history_length(pre, fold.origin)
    cand_idx = valid[C.ENTITY].unique()
    cand_idx = [e for e in cand_idx if hist_before.get(e, 0) > 0]
    cand = (
        valid.loc[valid[C.ENTITY].isin(cand_idx)]
        .groupby(C.ENTITY, sort=False)
        .agg(
            lokasyon=(C.LOCATION, "first"),
            guc=(C.POWER, "first"),
            n_valid_days=(C.DATE, "nunique"),
        )
    )

    # Dogrulama penceresinde origin sonrasi ilk kez gorunen trafolar zaten
    # gecmissiz, yani DOGAL cold. Maskeleme kotasi bunlarin ustune degil,
    # yerine gelir: aksi halde fold'un cold orani hedefi asiyordu (olculdu:
    # C_ara fold'unda satirlarin %39,8'i gecmissizken yalnizca %15,1'i cold
    # sayiliyordu).
    n_valid_entities = valid[C.ENTITY].nunique()
    n_natural_cold = n_valid_entities - len(cand)
    n_target = int(round(n_valid_entities * profile.entity_rate)) - n_natural_cold

    if apply_mask and len(cand) and n_target > 0:
        masked = cold_start_mask(cand, profile, seed=seed, n_target=n_target)
        offsets = assign_entry_offsets(masked, profile, seed=seed)
    else:
        masked = pd.Index([], name=C.ENTITY)
        offsets = pd.Series(dtype="int64")

    train = mask_history(pre, masked, fold.origin)
    valid = apply_entry_delay(valid, offsets, fold.valid_start)
    valid = valid.sort_values([C.ENTITY, C.DATE], ignore_index=True)
    train = train.sort_values([C.ENTITY, C.DATE], ignore_index=True)

    hist_after = history_length(train, fold.origin)
    history_days = valid[C.ENTITY].map(hist_after).fillna(0).astype("int32")
    # Cold = origin'de gecmisi olmayan satir. Maskelenmis olmak yeterli degil:
    # dogal yeni girisler de gercek test'teki cold trafolarla ayni durumda ve
    # onlari warm saymak `rmsle_warm`'i cold satirlarla kirletir.
    is_cold = history_days.eq(0)
    segment = history_segment(history_days)

    checks = run_all_checks(
        observed=train, origin=fold.origin, masked_entities=masked
    )

    return FoldSplit(
        fold=fold,
        train=train,
        valid=valid,
        masked_entities=masked,
        masked_offsets=offsets,
        history_days=history_days,
        is_cold=is_cold.reset_index(drop=True),
        segment=segment.reset_index(drop=True),
        checks=checks,
    )
