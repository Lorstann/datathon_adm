"""Ham veri yukleme, sema dogrulama ve panel insasi.

data/raw/ asla degistirilmez (workspace kurali 15). Bu modul yalnizca okur.
"""

from __future__ import annotations

import pandas as pd

from src import config as C

_TRAIN_SCHEMA = {C.ENTITY, C.POWER, C.DATE, C.TARGET, C.LOCATION}
_TEST_SCHEMA = {"id", C.ENTITY, C.POWER, C.DATE, C.LOCATION}

# `tanim` sayisal gorunse de bir kimlik dizesi: train'de 9, test'te 12 tekil
# alfanumerik deger var ("202917T", "Iskele DM", "M-3115" gibi). Sayiya
# zorlamak bu trafolari dusurur.
_DTYPES = {
    C.ENTITY: "string",
    C.POWER: "int32",
    C.LOCATION: "string",
}


def _validate(df: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{name}: eksik kolon(lar) {sorted(missing)}")


def load_train() -> pd.DataFrame:
    df = pd.read_csv(
        C.TRAIN_CSV,
        dtype={**_DTYPES, C.TARGET: "float64"},
        parse_dates=[C.DATE],
        encoding="utf-8",
    )
    _validate(df, _TRAIN_SCHEMA, "train.csv")
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(
        C.TEST_CSV,
        dtype={**_DTYPES, "id": "string"},
        parse_dates=[C.DATE],
        encoding="utf-8",
    )
    _validate(df, _TEST_SCHEMA, "test.csv")
    return df


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(C.SAMPLE_SUBMISSION_CSV, dtype={"id": "string"})


def split_location(s: pd.Series) -> pd.DataFrame:
    """`lokasyon` hiyerarsisini il / bolge / ilce olarak ayirir.

    Format IL>BOLGE>ILCE (Izmir) veya IL>BOLGE (Manisa). Jenerik "GEDIZ EDAS"
    gibi tek parcali degerlerde il disindaki seviyeler NA kalir. Eksik seviye
    NA birakilir, ust seviyeyle doldurulmaz: doldurmak "Manisa'nin ilcesi
    Manisa" gibi yanlis bir kategori uretir ve agac modelinde gercek bir ilce
    ile karisir.
    """
    parts = s.str.split(">", expand=True)
    while parts.shape[1] < 3:
        parts[parts.shape[1]] = pd.NA
    out = pd.DataFrame(
        {
            "il": parts[0].astype("string").str.strip(),
            "bolge": parts[1].astype("string").str.strip(),
            "ilce": parts[2].astype("string").str.strip(),
        },
        index=s.index,
    )
    out["lokasyon_derinlik"] = s.str.count(">").astype("int8") + 1
    return out


def build_panel(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Train ve test'i tek panele birlestirir.

    `is_test` bayragi satirin kaynagini isaretler. `tuketim` test satirlarinda
    NaN. Lokasyon hiyerarsisi ayristirilir.

    Panelin birlesik olmasi transduktif ozellikler (trafonun panelde ilk
    gorulme tarihi, devreye alinma yasi) icin gerekli. Bu ozellikler
    `test.csv` icinde verildigi icin mesru, ama operasyonel bir tahminde
    mevcut olmayacagi docs/prior-work.md 10.3'te not edildi.
    """
    tr = train.copy()
    tr["id"] = pd.NA
    tr["id"] = tr["id"].astype("string")
    tr["is_test"] = False

    te = test.copy()
    te[C.TARGET] = pd.NA
    te[C.TARGET] = pd.to_numeric(te[C.TARGET])
    te["is_test"] = True

    cols = ["id", C.ENTITY, C.POWER, C.DATE, C.TARGET, C.LOCATION, "is_test"]
    panel = pd.concat([tr[cols], te[cols]], ignore_index=True)
    panel = panel.join(split_location(panel[C.LOCATION]))
    panel = panel.sort_values([C.ENTITY, C.DATE], ignore_index=True)
    return panel


def entity_static(panel: pd.DataFrame) -> pd.DataFrame:
    """Trafo basina sabit nitelikler.

    `guc` ve `lokasyon`'un trafo basina sabit oldugu dogrulandi, dolayisiyla
    ilk gozlem temsili. Panel uyeligi tarihleri de burada toplanir.
    """
    g = panel.groupby(C.ENTITY, sort=False)
    out = g.agg(
        guc=(C.POWER, "first"),
        lokasyon=(C.LOCATION, "first"),
        il=("il", "first"),
        bolge=("bolge", "first"),
        ilce=("ilce", "first"),
        lokasyon_derinlik=("lokasyon_derinlik", "first"),
        panel_ilk_tarih=(C.DATE, "min"),
        panel_son_tarih=(C.DATE, "max"),
        panel_gun_sayisi=(C.DATE, "nunique"),
    )
    test_rows = panel.loc[panel["is_test"]].groupby(C.ENTITY, sort=False)[C.DATE]
    out["test_gun_sayisi"] = test_rows.nunique().reindex(out.index).fillna(0).astype("int32")
    out["test_ilk_tarih"] = test_rows.min().reindex(out.index)
    return out
