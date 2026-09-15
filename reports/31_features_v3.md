# Ozellik paketi v3
Uretim: 2026-08-23 05:14:57.511730

Hepsine cold olu-trafo kapisi k=0.75. v2_base = gonderimdeki yapilandirma.

## Fold A_mevsim

| variant | ozellik | warm | cold | blend |
| --- | --- | --- | --- | --- |
| v2_base | 122 | 0.8771 | 2.3721 | 1.3586 |
| v3_weather_only | 124 | 0.8711 | 2.3622 | 1.3518 |
| v3_all | 132 | 0.8738 | 2.3611 | 1.3526 |

## Fold B_guncel

| variant | ozellik | warm | cold | blend |
| --- | --- | --- | --- | --- |
| v2_base | 122 | 0.6515 | 1.7855 | 1.0183 |
| v3_weather_only | 124 | 0.6626 | 1.7862 | 1.0241 |
| v3_all | 132 | 0.6556 | 1.7869 | 1.0209 |

## Fold C_ara

| variant | ozellik | warm | cold | blend |
| --- | --- | --- | --- | --- |
| v2_base | 122 | 0.8316 | 2.0361 | 1.2070 |
| v3_weather_only | 124 | 0.8387 | 2.0383 | 1.2117 |
| v3_all | 132 | 0.8485 | 2.0377 | 1.2168 |

## Ozet (rmsle_blend)

| variant | mean | std | warm | cold |
| --- | --- | --- | --- | --- |
| v2_base | 1.1946 | 0.1392 | 0.7867 | 2.0646 |
| v3_all | 1.1968 | 0.1362 | 0.7927 | 2.0619 |
| v3_weather_only | 1.1958 | 0.1342 | 0.7908 | 2.0622 |

### Fold bazinda v2_base farki (negatif = iyi)

- v2_base: {'A_mevsim': 0.0, 'B_guncel': 0.0, 'C_ara': 0.0}
- v3_all: {'A_mevsim': -0.0059, 'B_guncel': 0.0026, 'C_ara': 0.0098}
- v3_weather_only: {'A_mevsim': -0.0068, 'B_guncel': 0.0058, 'C_ara': 0.0046}

En iyi: **v2_base**
