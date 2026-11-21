# Design notes — why each Acc climb lever

Intent only. Flags → [`model-settings.md`](model-settings.md). Numbers → [`experiments.md`](experiments.md).

**Base** remains the Basic_Updated model design. Climbs ask: *what inductive bias, on top of SURD MoE, still moves known_only Acc@1?*

---

## Keep stack (what it was meant to test)

| Layer | Intent |
| --- | --- |
| dim256 / heads8 | Capacity over Basic 128/4 for Tokyo Acc hillclimb |
| λ_bpr 0.29 | Push hard-neg margin so Acc@1 (not only NDCG) is optimized |
| tsb 2.0 + unique_p residual ssb | Causal last→cand transitions, gated by Unique-P SURD mass so prior does not dominate weak candidates |
| cand-context last_slot | Fuse cand category + Δgeo (+ last time slot) into the query without re-prepare |
| transition_attn `gate` | Let transition counts modulate P/C attention, not only add a score bias |
| geo 0.15 | Mild proximity prior (needs context on); stronger λ collapsed Acc |
| revisit 0.10 | Prefer POIs the user already visits; 0.20 over-hubbed Acc |
| soft β0.10 prior=revisit | Mix one-hot CE with a revisit soft target so gradients match the locked revisit bias |

---

## Leftover / discarded levers (intent → outcome summary)

| Lever | Design question | Empirically |
| --- | --- | --- |
| Soft prior ≠ revisit | Does tsb/EMA soft label help more than revisit soft? | No — only revisit@0.10 Keep |
| hour-pop (#8) | Hour-conditioned popularity as score prior | Acc↓ on soft10r |
| 2-hop A² (#4) | Multi-hop transition mass beyond last→cand | Acc↓ on soft10r (Discard) |
| geo-power (#6) | Sharper distance kernel than log1p | Acc↓ |
| cat MTL (#45) | Dense next-category aux without score prior | Acc↓ |
| ep>1 (#47) | Longer Acc1-selected training | ep1 still best |
| radiation / jump / return-gate | Mobility models beyond flat geo | Acc≤ or − vs peak |
| spatial candidate filter (#10) | Shrink pool by gravity/coverage | Inflates Acc only when coverage unfair |
| T / Platt / TTA / MC-dropout (#5/#34/#36/#46) | Free fusion on frozen logits | Flat or − Acc; plain `--score-temperature` ÷T **cannot** change Acc@1 (order-preserving) |
| seed soup (#42) | Variance reduction without new bias | Only seed42 held peak |
| KD / laplace (#41/#40) | Distill peak / geo-smooth embeddings | No Acc lift |
| hist-hit / recency / session-gap | Finer revisit / tsb timing | Redundant with locked stack |

---

## How to propose the next lever

1. Name the **mechanism** (score bias, loss, representation, eval fusion).  
2. State what it should capture that soft10r does **not**.  
3. Prefer one flag on soft10r; Keep only if Acc@1 > 0.267650.  
4. Record Added knobs in [`model-settings.md`](model-settings.md) and the run row in [`experiments.md`](experiments.md).
