# Bandung negative densify levers

Record of levers **banned from subsequent climb densify** (clear negative ROI or cross-city prior).

| key | decision | reason |
| --- | --- | --- |
| `min_history_length` | ban | Clear negative ROI on Bandung f63_w1 epoch-test Acc: hmin runs are bottom-3 (0.165/0.156/0.132) vs wave median ≈0.179; raising min_history above 1 hurts. |
| `seed` | ban | Cross-city prior (MEL): seed sweeps rarely lift Acc; waste densify budget. |
| `transition_2hop_bias` | ban | Cross-city prior (MEL): hop bias rarely helps. |
| `pop_score_bias` | ban | Cross-city prior (MEL): pop score bias rarely helps. |
| `hour_pop_score_bias` | ban | Cross-city prior (MEL): hour-pop bias rarely helps. |
| `gauss_geo_bias` | ban | Optional early loser; keep geo_score_bias instead. |

## Bandung f63_w1 evidence (`min_history_length`)

Wave median epoch-test Acc@1 ≈ **0.1789**. All `hmin` runs were bottom:

| tag | Acc@1 |
| --- | ---: |
| `..._hmin2_bpr29` | 0.1647 |
| `..._hmin2_ep30_pat5` | 0.1557 |
| `..._hmin3_ep50_pat8` | 0.1317 |

Keep fixed `min_history_length: 1` in base; do not densify this knob.

