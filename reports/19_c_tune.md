# C tune (seed / trees / hyper)
Uretim: 2026-08-21 19:13:34.318344
Ref: blend=1.3359 cold=1.8746
use_external=False (LB C iskeleti).

## Fold A_mevsim
| variant | warm | cold | blend |
| --- | --- | --- | --- |
| baseline_400 | 1.0052 | 2.3479 | 1.4171 |
| trees_700 | 1.0037 | 2.3491 | 1.4167 |
| deep_leaves | 1.0052 | 2.3483 | 1.4172 |
| seed_avg3_500 | 1.0042 | 2.3486 | 1.4168 |
| seed_avg3_deep | 1.0049 | 2.3489 | 1.4173 |

## Fold B_guncel
| variant | warm | cold | blend |
| --- | --- | --- | --- |
| baseline_400 | 1.0156 | 1.6973 | 1.2005 |
| trees_700 | 1.0150 | 1.6973 | 1.2001 |
| deep_leaves | 1.0175 | 1.6967 | 1.2016 |
| seed_avg3_500 | 1.0145 | 1.6989 | 1.2003 |
| seed_avg3_deep | 1.0169 | 1.6976 | 1.2015 |

## Fold C_ara
| variant | warm | cold | blend |
| --- | --- | --- | --- |
| baseline_400 | 1.3335 | 1.5787 | 1.3916 |
| trees_700 | 1.3336 | 1.5804 | 1.3921 |
| deep_leaves | 1.3354 | 1.5775 | 1.3927 |
| seed_avg3_500 | 1.3329 | 1.5788 | 1.3911 |
| seed_avg3_deep | 1.3320 | 1.5781 | 1.3903 |

## Ozet
| variant | blend | cold |
| --- | --- | --- |
| baseline_400 | 1.3364 | 1.8746 |
| deep_leaves | 1.3372 | 1.8741 |
| seed_avg3_500 | 1.3361 | 1.8754 |
| seed_avg3_deep | 1.3363 | 1.8749 |
| trees_700 | 1.3363 | 1.8756 |

Best: **seed_avg3_deep**; submit_ok=True
