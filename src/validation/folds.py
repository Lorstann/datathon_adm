"""Fold uretimi, gecmis uzunlugu olcumu ve tabakali cold-start maskelemesi.

Test setinde trafolarin %28,8'i (satirlarin %22,2'si) hic gecmisi olmayan
trafolardan geliyor. Hicbir dogal fold bu orani tasimiyor (dogal fold'larda
%7,5-13,9), dolayisiyla maskeleme zorunlu.

Maskeleme uniform rastgele DEGIL: gercek cold trafolarin (ilce, guc bandi,
test satir sayisi bandi) ortak dagilimina tabakali eslestirilir. Yeni
baglantilar gelisen ilcelerde ve belirli guc siniflarinda kumelendigi icin
uniform ornekleme cold segmenti yanlis temsil eder.

Kritik: maskelenen trafonun origin oncesi gecmisi egitim cercevesinden
TAMAMEN silinir. Yalnizca lag ozelliklerini NaN yapmak yetmez; grup hedef
kodlamalari gecmisi sessizce emer ve cold-start performansini oldugundan iyi
gosterir (docs/prior-work.md 10.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config as C

# Sinirlar standart trafo anma guclerinin arasina konuldu, boylece nominal
# degerler (160, 250, 400, 630, 1000, 1250, 1600 kVA) bir bandin icine tam
# oturur. Son band (>=2600 kVA) dagitim trafosu degil, 10-36 MVA'lik dagitim
# merkezi sinifi: 38 varlik, davranisi diger bandlardan yapisal olarak farkli.
GUC_BAND_EDGES = (0, 100, 160, 250, 400, 630, 1000, 1600, 2600, np.inf)
GUC_BAND_LABELS = (
    "<100",
    "100-160",
    "160-250",
    "250-400",
    "400-630",
    "630-1000",
    "1000-1600",
    "1600-2600",
    "2600+",
)

ROW_BAND_EDGES = (0, 30, 60, 90, 110, np.inf)
ROW_BAND_LABELS = ("<30", "30-60", "60-90", "90-110", "110+")


def guc_band(guc: pd.Series) -> pd.Series:
    return pd.cut(
        guc.astype("float64"),
        bins=list(GUC_BAND_EDGES),
        labels=list(GUC_BAND_LABELS),
        right=False,
    )


def row_band(n_rows: pd.Series) -> pd.Series:
    return pd.cut(
        n_rows.astype("float64"),
        bins=list(ROW_BAND_EDGES),
        labels=list(ROW_BAND_LABELS),
        right=False,
    )


def history_length(observed: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """Her trafo icin origin'e kadar (dahil) gozlenmis gun sayisi.

    `observed` yalnizca hedefi bilinen satirlari icermelidir. Sonuc, origin'de
    duran bir tahmincinin o trafo hakkinda sahip oldugu gozlem sayisidir.
    """
    hist = observed.loc[observed[C.DATE] <= origin]
    return hist.groupby(C.ENTITY, sort=False)[C.DATE].nunique()


def history_segment(days: pd.Series) -> pd.Series:
    """Gecmis uzunlugunu rapor segmentlerine cevirir."""
    return pd.cut(
        days.astype("float64"),
        bins=list(C.HISTORY_SEGMENT_EDGES) + [np.inf],
        labels=list(C.HISTORY_SEGMENT_LABELS),
        right=False,
    )


@dataclass(frozen=True)
class ColdProfile:
    """Gercek test cold-start popülasyonunun ampirik profili.

    joint: (ilce, guc_bandi, satir_bandi) uzerinde olasilik dagilimi.
    entry_offsets: cold trafolarin ufkun kacinci gununde ilk kez gorundugu.
    entity_rate: cold trafolarin tum test trafolarina orani.
    """

    joint: pd.Series
    entry_offsets: np.ndarray
    entity_rate: float


def build_cold_profile(static: pd.DataFrame, train_entities: set) -> ColdProfile:
    """Test setindeki gercek cold trafolardan hedef profili cikarir.

    `static` entity_static() cikisi, `train_entities` train'de gorulen trafolar.

    Cografi anahtar `ilce` degil `lokasyon`: Manisa kayitlari IL>BOLGE
    formatinda, yani ilce seviyesi hic yok ve 1.788 Manisa trafosunun tamami
    tek bir NA tabakasinda cokuyor. `lokasyon` (47 tekil deger) her iki ilde de
    mevcut olan en ince cozunurluk.
    """
    test_ents = static.loc[static["test_gun_sayisi"] > 0].copy()
    test_ents["is_cold"] = ~test_ents.index.isin(train_entities)
    cold = test_ents.loc[test_ents["is_cold"]]

    key = pd.MultiIndex.from_arrays(
        [
            cold["lokasyon"].astype(str),
            guc_band(cold["guc"]).astype(str),
            row_band(cold["test_gun_sayisi"]).astype(str),
        ],
        names=["lokasyon", "guc_band", "row_band"],
    )
    joint = pd.Series(1.0, index=key).groupby(level=[0, 1, 2]).sum()
    joint = joint / joint.sum()

    offsets = (cold["test_ilk_tarih"] - C.TEST_START).dt.days.to_numpy(dtype="int64")

    return ColdProfile(
        joint=joint,
        entry_offsets=offsets,
        entity_rate=float(test_ents["is_cold"].mean()),
    )


def cold_start_mask(
    candidates: pd.DataFrame,
    profile: ColdProfile,
    *,
    seed: int = C.SEED,
    n_target: int | None = None,
) -> pd.Index:
    """Maskelenecek trafolari tabakali ornekle secer.

    `candidates` dogrulama penceresinde aktif ve origin'e kadar gecmisi OLAN
    trafolar; index `tanim`, kolonlar `lokasyon`, `guc`, `n_valid_days`.

    Strateji: gercek cold profilinin her tabakasi icin hedef sayiyi hesapla,
    o tabakadaki adaylardan ornekle. Tabaka bos veya yetersizse eksik kalan
    kota, kalan tabakalar arasinda gercek profil agirliklariyla yeniden
    dagitilir; boylece toplam maskeleme orani korunur.
    """
    rng = np.random.default_rng(seed)
    if n_target is None:
        n_target = int(round(len(candidates) * profile.entity_rate))
    n_target = int(max(0, min(n_target, len(candidates))))
    if n_target == 0 or len(candidates) == 0:
        return pd.Index([], name=C.ENTITY)

    cand = candidates.copy()
    cand["_lok"] = cand["lokasyon"].astype(str)
    cand["_guc"] = guc_band(cand["guc"]).astype(str)
    cand["_row"] = row_band(cand["n_valid_days"]).astype(str)
    groups = cand.groupby(["_lok", "_guc", "_row"], observed=True).groups

    chosen: list = []
    used: set = set()

    def take(pool, k: int) -> None:
        avail = [e for e in pool if e not in used]
        if not avail or k <= 0:
            return
        k = min(k, len(avail))
        picked = rng.choice(len(avail), size=k, replace=False)
        for i in picked:
            e = avail[int(i)]
            used.add(e)
            chosen.append(e)

    # 1) Tam ucluye (ilce, guc bandi, satir bandi) gore kota.
    quota = (profile.joint * n_target).to_dict()
    for key, target in sorted(quota.items(), key=lambda kv: -kv[1]):
        k = int(round(target))
        if k <= 0:
            continue
        take(list(groups.get(key, [])), k)

    # 2) Eksik kalani lokasyon x guc marjinaline gore dagit. Marjinal
    #    olasiliklar kalan kotayi bolusur, boylece profilin sekli korunur.
    if len(chosen) < n_target:
        marginal = profile.joint.groupby(level=[0, 1]).sum().sort_values(ascending=False)
        by_pair = cand.groupby(["_lok", "_guc"], observed=True).groups
        remaining = n_target - len(chosen)
        for pair, w in marginal.items():
            if len(chosen) >= n_target:
                break
            take(list(by_pair.get(pair, [])), int(np.ceil(remaining * w)))

    # 3) Hala eksikse global havuzdan tamamla. Toplam maskeleme orani profil
    #    sadakatinden onceliklidir, cunku raporlanan harman satir oranlarina
    #    dayaniyor ve oran sapmasi harmani dogrudan bozar.
    if len(chosen) < n_target:
        take(list(cand.index), n_target - len(chosen))

    return pd.Index(chosen[:n_target], name=C.ENTITY)


def assign_entry_offsets(
    masked: pd.Index, profile: ColdProfile, *, seed: int = C.SEED
) -> pd.Series:
    """Maskelenen trafolara ufuk-ici giris gecikmesi atar.

    Gercek cold trafolar ufkun basinda degil ortasinda devreye giriyor
    (ortalama 78 satir, warm olanlar 113). Ampirik ofset dagilimindan ornekle.
    """
    rng = np.random.default_rng(seed + 1)
    if len(profile.entry_offsets) == 0:
        return pd.Series(0, index=masked, dtype="int64")
    draws = rng.choice(profile.entry_offsets, size=len(masked), replace=True)
    return pd.Series(draws, index=masked, dtype="int64")


def mask_history(
    observed: pd.DataFrame,
    masked_entities: pd.Index,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """Maskelenen trafolarin origin oncesi TUM satirlarini duser.

    Lag'leri NaN yapmak yerine satirlari silmek zorunlu: aksi halde grup
    hedef kodlamalari ve akran istatistikleri o trafonun gecmisini emer.
    """
    if len(masked_entities) == 0:
        return observed
    drop = observed[C.ENTITY].isin(set(masked_entities)) & (observed[C.DATE] <= origin)
    return observed.loc[~drop]


def apply_entry_delay(
    valid: pd.DataFrame,
    offsets: pd.Series,
    valid_start: pd.Timestamp,
) -> pd.DataFrame:
    """Maskelenen trafolarin dogrulama penceresi basindaki satirlarini duser."""
    if offsets.empty:
        return valid
    delay = valid[C.ENTITY].map(offsets)
    first_allowed = valid_start + pd.to_timedelta(delay.fillna(0), unit="D")
    return valid.loc[valid[C.DATE] >= first_allowed]
