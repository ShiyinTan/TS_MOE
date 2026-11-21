# Acc@1 climb — deltas vs Basic_Updated

**Authoritative Base:** [`TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md`](../TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md) (repo root).  
Not WIN alone. Every climb knob below is **Added** on top of that doc (or an explicit change from its defaults).

Mirrored on the GPU box as `results/tokyo/climb_delta_from_basic.md`.

---

## Base (from Basic_Updated §1 / §19 + code that implements it)

| Item | Basic setting |
| --- | --- |
| Sources | P/C/T/L aligned sequences; **no user_id as model input** |
| Experts | 15 U/S MoE (4 Unique + 11 Synergy); Top-K=**4**; T=1.0 start |
| SURD | 26-D offline prior; train-all / fold tables; path A joint then marginalize |
| Offline | `hash_dim=64`, `n_clusters=4`, `max_length`/`N_max=50`, `tau_pos=10`, seed=0 |
| T/L features | STHGCN-aligned 48-D time / 18-D location |
| Hidden | dim **128**, heads **4**, Expert FFN 128→256→128, GELU, dropout 0.1 |
| Loss | `L_rec + λ_SURD L_SURD + λ_balance L_balance` |
| λ starts | `lambda_prior=0.1`, `lambda_SURD=0.05`, `lambda_balance=0.001` |
| Optimizer | AdamW (lr / weight decay set in train protocol) |
| Router (code default implementing Basic) | SURD-guided; codebase default `hard_prior` = π from candidate U/S SURD mass |

Anything below that is **not** in Basic (or changes a Basic default) is **Added**.

---

## Added layers that produced Keeps

### A. LOCKED WIN `hc_sg_Atime_bpr029_ssb026` — Acc@1 **0.260274**

**Added on Base:**

| flag / knob | value | note |
| --- | --- | --- |
| `--dim` / `--heads` | **256** / **8** | vs Basic 128 / 4 |
| `--lambda-bpr` | **0.29** | Basic has no BPR aux |
| `--transition-score-bias` | **2.0** | Basic has no tsb |
| `--surd-bias-gate` | `unique_p` | |
| `--tsb-gate-mode` | `residual` | |
| `--surd-score-bias` | **0.26** | |
| `--candidate-context` | `on` | Basic has no cand-context fuse |
| `--candidate-context-time` | `last_slot` | |
| `--epochs` / `--checkpoint-metric` | `1` / `acc1` | protocol |
| cache | `runs/tokyo/cache_abl_tgt_last` | Tokyo prepare |

### B. Peak `on_rev_soft10r` — Acc@1 **0.267650** · **current locked peak**

**Added on Base** = WIN Added (A) **plus**:

| flag | value |
| --- | --- |
| `--transition-attn` | `gate` |
| `--geo-score-bias` | `0.15` |
| `--revisit-score-bias` | `0.10` |
| `--soft-target-beta` | `0.10` |
| `--soft-target-prior` | `revisit` |

### Intermediate Keeps (superseded)

| run | Acc@1 | Added beyond WIN (A) |
| --- | ---: | --- |
| `hc_win_trgate_gsb015` | 0.263435 | `transition_attn gate` + `geo 0.15` |
| `on_rev010` | 0.265543 | above + `revisit 0.10` |

---

## All other Added settings tried (Discard unless noted)

Stack for leftovers = soft10r (= Base + A + B). Extra = only the lever under test.

| idea / run | Extra vs soft10r (or vs Base→WIN path) | Acc@1 | 判定 |
| --- | --- | ---: | --- |
| soft β≠0.10 / prior≠revisit | soft05, soft10 tsb+revisit, soft10t | ≤peak | Discard |
| revisit λ=0.20 | `--revisit-score-bias 0.20` | 0.2476 | Discard |
| #32 hist-hit / rev-recency | history_hit / revisit_recency | 0.2624 | Discard |
| #33 tsb session-gap | `--tsb-session-gap` | =peak | Discard |
| #38/#34/#36 | locked-ε / gamma / TTA | ≤peak | Discard |
| #10 spatial filter | candidate spatial prior | cov≪0.98 | Discard |
| #37 adaptive neg | `--neg-sample adaptive_mix` | ≤0.258 | Discard |
| #39 EMA soft | soft prior includes `ema` | 0.2561 | Discard |
| #43 ssb/tsb micro | ssb/tsb retune | 0.2624 | Discard |
| #41 KD / #40 laplace | `--lambda-kd` / `--poi-geo-laplace` | ≤0.259 | Discard |
| #42 seed soup | multi-seed avg | no lift | Discard |
| #44 fam blend | `--fam-area-blend on` | 0.2561 | Discard |
| #46 MC-dropout | eval `--mc-dropout 10/20` | 0.2655/0.2634 | Discard |
| #45 cat MTL | `--lambda-cat-aux 0.1/0.3` | 0.257/0.256 | Discard |
| #47 ep2–3 | `--epochs 2/3` | best=peak; later↓ | Discard |
| #6 geo power | `--geo-score-power 1.5/2.0` | 0.2634/0.256 | Discard |
| #8 hour-pop | `--hour-pop-score-bias 0.10/0.15` | 0.2518 / 0.2624 | Discard |
| earlier: radiation/jump/return-gate/Δt-bins/cat-stage/focal/BPR-max/calib/tsb-decay | various | ≪ | Discard |

### Queued Added (implement / run)

| # | Extra flags | status |
| --- | --- | --- |
| #4 | `--transition-2hop-bias 0.3/0.5` | relaunched after A2-after-split fix (`on_s10r_2hop03/05` b) |
| #9 | `--cat-stage-bias` (mild) | queued |
| #11 | learned bilinear transition (TBD) | queued |
| #5 | eval `--score-temperature` / Platt on frozen peak | queued |

---

## How to read a run

1. Start from **Base** (this file §Base / Basic_Updated).  
2. Apply **WIN Added (A)** if the run is on the Tokyo Acc climb stack.  
3. Apply **peak Added (B)** if on soft10r.  
4. Apply the run’s **Extra** row only.
