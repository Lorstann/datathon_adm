"""47 lokasyonun ilce merkez koordinatlari.

Kaynak: OpenStreetMap / Wikipedia ilce merkezleri, 2013-14 buyuksehir
reorganizasyonu sonrasi Yunusemre, Sehzadeler, Menderes, Karaburun dahil.
Buca ve Karsiyaka icin il merkezi kopyasi KULLANILMADI (topluluk gist'indeki
bilinen hata). Hava cekiminde ERA5-Land ~11 km izgarasina snap edilecek.
"""

from __future__ import annotations

# (latitude, longitude) WGS84. Anahtar: train/test'teki tam `lokasyon` dizesi.
LOCATION_COORDS: dict[str, tuple[float, float]] = {
    "MANİSA>AHMETLİ": (38.5197, 27.9386),
    "MANİSA>AKHİSAR": (38.9186, 27.8378),
    "MANİSA>ALAŞEHİR": (38.3508, 28.5171),
    "MANİSA>DEMİRCİ": (39.0461, 28.6589),
    "MANİSA>GÖLMARMARA": (38.7139, 27.9147),
    "MANİSA>GÖRDES": (38.9328, 28.2897),
    "MANİSA>KIRKAĞAÇ": (39.1064, 27.6692),
    "MANİSA>KULA": (38.5472, 28.6497),
    "MANİSA>KÖPRÜBAŞI": (38.7497, 28.4047),
    "MANİSA>SALİHLİ": (38.4826, 28.1393),
    "MANİSA>SARIGÖL": (38.2394, 28.6967),
    "MANİSA>SARUHANLI": (38.7346, 27.5681),
    "MANİSA>SELENDİ": (38.7444, 28.8678),
    "MANİSA>SOMA": (39.1855, 27.6094),
    "MANİSA>TURGUTLU": (38.4953, 27.6997),
    "MANİSA>YUNUSEMRE": (38.6191, 27.4100),
    "MANİSA>ŞEHZADELER": (38.6136, 27.4550),
    "İZMİR>GÜNEY BÖLGE>BAYINDIR": (38.2174, 27.6481),
    "İZMİR>GÜNEY BÖLGE>BEYDAĞ": (38.0869, 28.2088),
    "İZMİR>GÜNEY BÖLGE>KARABURUN": (38.6375, 26.5111),
    "İZMİR>GÜNEY BÖLGE>KEMALPAŞA": (38.4263, 27.4173),
    "İZMİR>GÜNEY BÖLGE>KİRAZ": (38.2306, 28.2044),
    "İZMİR>GÜNEY BÖLGE>MENDERES": (38.2497, 27.1342),
    "İZMİR>GÜNEY BÖLGE>SEFERİHİSAR": (38.1975, 26.8386),
    "İZMİR>GÜNEY BÖLGE>SELÇUK": (37.9500, 27.3681),
    "İZMİR>GÜNEY BÖLGE>TORBALI": (38.1618, 27.3584),
    "İZMİR>GÜNEY BÖLGE>TİRE": (38.0888, 27.7353),
    "İZMİR>GÜNEY BÖLGE>URLA": (38.3222, 26.7647),
    "İZMİR>GÜNEY BÖLGE>ÇEŞME": (38.3228, 26.3064),
    "İZMİR>GÜNEY BÖLGE>ÖDEMİŞ": (38.2278, 27.9719),
    "İZMİR>KUZEY BÖLGE>ALİAĞA": (38.7998, 26.9720),
    "İZMİR>KUZEY BÖLGE>BERGAMA": (39.1207, 27.1805),
    "İZMİR>KUZEY BÖLGE>DİKİLİ": (39.0710, 26.8906),
    "İZMİR>KUZEY BÖLGE>FOÇA": (38.6703, 26.7566),
    "İZMİR>KUZEY BÖLGE>KINIK": (39.0872, 27.3833),
    "İZMİR>KUZEY BÖLGE>MENEMEN": (38.6145, 27.0694),
    "İZMİR>METROPOL>BALÇOVA": (38.3948, 27.0594),
    "İZMİR>METROPOL>BAYRAKLI": (38.4622, 27.1667),
    "İZMİR>METROPOL>BORNOVA": (38.4698, 27.2211),
    "İZMİR>METROPOL>BUCA": (38.3865, 27.1784),
    "İZMİR>METROPOL>GAZİEMİR": (38.3215, 27.1277),
    "İZMİR>METROPOL>GÜZELBAHÇE": (38.3744, 26.8881),
    "İZMİR>METROPOL>KARABAĞLAR": (38.3733, 27.1350),
    "İZMİR>METROPOL>KARŞIYAKA": (38.4595, 27.1118),
    "İZMİR>METROPOL>KONAK": (38.4192, 27.1287),
    "İZMİR>METROPOL>NARLIDERE": (38.3910, 27.0029),
    "İZMİR>METROPOL>ÇİĞLİ": (38.4978, 27.0698),
}


def location_coordinates() -> dict[str, tuple[float, float]]:
    return dict(LOCATION_COORDS)
