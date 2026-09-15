"""Bugunun LB-cebiri deneylerini `experiments.csv`'ye isler (workspace kurali 14).

Bu deneylerin CV metrikleri yok -- olcum dogrudan public leaderboard'dan
okunuyor. `rmsle_all` sutununa gonderilen dosyanin gercek LB skoru,
`rmsle_blend` sutununa o olcumun harmana kattigi beklenen skor yaziliyor.
"""

from __future__ import annotations

import pandas as pd

from src import config as C

ROWS = [
    dict(exp_id="lb_pool_rank", model="lb_algebra_ridge", feature_set="pool16",
         fold="lb_public", rmsle_all=1.01548, rmsle_blend=1.01329,
         lb_public=None,
         notes="turetilmis harmanlar cikarildi; kosul 1.4e15->1.45e3; "
               "lambda/tr=0.003 Monte Carlo; garanti kazanc +0.00219"),
    dict(exp_id="probe_Q_COMBO", model="grup_profil+d364+SVD8",
         feature_set="agacsiz", fold="lb_public", rmsle_all=1.33324,
         rmsle_blend=1.01329, lb_public=1.33324,
         notes="izdusum R2=0.58 ||D_dik||=0.56 ama <e0,D_dik>=+0.00093; "
               "dik bilesenin kazanci 0.000003 MSE = SIFIR"),
    dict(exp_id="zero_floor_B", model="tanisal", feature_set="gecmis_p",
         fold="B_guncel", rmsle_all=0.7017, rmsle_blend=0.5051,
         notes="sifir belirsizliginin bedeli 0.2373 MSE = hatanin %48'i; "
               "surpriz sifir 970 satir tek basina 0.129 MSE"),
    dict(exp_id="zero_floor_A", model="tanisal", feature_set="gecmis_p",
         fold="A_mevsim", rmsle_all=0.8656, rmsle_blend=0.7058,
         notes="sifir belirsizliginin bedeli 0.2511 MSE = hatanin %34'u"),
    dict(exp_id="id_adjacency", model="kimlik_kNN", feature_set="tanim_sayisal",
         fold="entity_holdout_30", rmsle_all=None, rmsle_blend=None,
         notes="komsu kimlik %88 ayni ilce (rastgele %9) ama seviye tasimiyor; "
               "OOS R2 0.0186 vs band+ilce 0.0144; log(guc) sonrasi lokasyon "
               "cold seviyenin %1.4'unu acikliyor -> cold tavani gercek"),
]


def main() -> None:
    df = pd.read_csv(C.EXPERIMENTS_CSV)
    ts = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
    new = pd.DataFrame([{**{c: None for c in df.columns},
                         **r, "timestamp": ts, "target": "log1p", "seed": C.SEED}
                        for r in ROWS])
    out = pd.concat([df, new[df.columns]], ignore_index=True)
    out.to_csv(C.EXPERIMENTS_CSV, index=False)
    print(f"{len(ROWS)} satir eklendi -> {C.EXPERIMENTS_CSV} "
          f"(toplam {len(out)})")


if __name__ == "__main__":
    main()
