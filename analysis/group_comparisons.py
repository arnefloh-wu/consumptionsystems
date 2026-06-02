"""
Focal group comparisons (spill-over contrasts)
Study: Value_StudentPool_Nov2026

Comparison 1  NEGATIVE spill-over:  A_Spos  vs  C_Spos-Hneg
              (does adding a BAD phone to a GOOD service drag evaluations down?)
Comparison 2  POSITIVE spill-over:  B_Sneg  vs  D_Sneg-Hpos
              (does adding a GOOD phone to a BAD service lift evaluations up?)

Aggregated scales compared (1..7, higher = more favourable):
    satisfaction   mean of 5 items   (alpha = .97)
    value          mean of 2 items   *** COMPROMISED: items exported as endpoints
                                          only -> effectively 3 values {1,4,7},
                                          ~50% missing. Nonparametric test + caveat.
    loyalty_wom    mean of 7 content-aligned items (alpha = .95; duplicate removed)

Tests per scale:
    Primary    Welch's t-test (unequal variance/n) + Cohen's d + 95% CI of difference
    Robustness Mann-Whitney U + rank-biserial r   (primary for the binary value scale)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.width", 130)

CSV = "Value_StudentPool_Nov2026_27.Februar2026_10.29.csv"
FIG = "analysis/figures"

# ----------------------------------------------------------------- data prep
df = pd.read_csv(CSV, dtype=str, keep_default_na=False).iloc[2:].reset_index(drop=True)

def to_num(s):
    m = {"very low":1,"sehr niedrig":1,"very high":7,"sehr hoch":7,
         "lehne völlig ab":1,"stimme völlig zu":7}
    s = (s or "").strip().lower()
    if s in m: return float(m[s])
    if s == "": return np.nan
    try: return float(s)
    except: return np.nan

def coalesce(cols):
    return df[cols].apply(lambda c: c.map(to_num)).bfill(axis=1).iloc[:, 0]

CONDS = ["A_Spos","B_Sneg","C_Spos-Hneg","D_Sneg-Hpos"]
SIT = {"A_Spos":"Q25_1","B_Sneg":"Q33_1","C_Spos-Hneg":"Q41_1","D_Sneg-Hpos":"Q49_1"}
SAT = {"A_Spos":[f"Q29_{i}" for i in range(1,6)],"B_Sneg":[f"Q37_{i}" for i in range(1,6)],
       "C_Spos-Hneg":[f"Q45_{i}" for i in range(1,6)],"D_Sneg-Hpos":[f"Q53_{i}" for i in range(1,6)]}

def assign(r):
    seen = [c for c in CONDS if any(r[col].strip() for col in SAT[c]) or r[SIT[c]].strip()]
    return seen[0] if len(seen)==1 else ("none" if not seen else "multiple")
df["Condition"] = df.apply(assign, axis=1)

for i in range(1,6):
    df[f"sat_{i}"] = coalesce([f"Q29_{i}",f"Q37_{i}",f"Q45_{i}",f"Q53_{i}"])
LOY_MAP = {  # 7-item content-aligned loyalty (block-A duplicate removed, online item dropped)
    "loy_1":["Q30_1","Q38_1","Q46_1","Q54_1"], "loy_2":["Q30_2","Q38_2","Q46_2","Q54_2"],
    "loy_3":["Q30_3","Q38_3","Q46_3","Q54_3"], "loy_4":["Q30_4","Q38_4","Q46_4","Q54_4"],
    "loy_5":["Q30_5","Q38_5","Q46_5","Q54_5"], "loy_6":["Q30_6","Q38_6","Q46_6","Q54_6"],
    "loy_7":["Q30_7","Q38_8","Q46_8","Q54_8"]}
for name, cols in LOY_MAP.items():
    df[name] = coalesce(cols)
df["pval_1"] = coalesce(["Q27_1","Q35_1","Q43_1","Q51_1"])
df["pval_2"] = coalesce(["Q27_2","Q35_2","Q43_2","Q51_2"])

df["satisfaction"] = df[[f"sat_{i}" for i in range(1,6)]].mean(axis=1)
df["loyalty"]      = df[list(LOY_MAP)].mean(axis=1)
df["value"]        = df[["pval_1","pval_2"]].mean(axis=1)

ana = df[df.Condition.isin(CONDS)].copy()

SCALES = ["satisfaction", "value", "loyalty"]
SCALE_LABEL = {"satisfaction":"Satisfaction", "value":"Value", "loyalty":"Loyalty/WOM"}
COMPARISONS = [
    ("Comparison 1 — NEGATIVE spill-over", "A_Spos", "C_Spos-Hneg"),
    ("Comparison 2 — POSITIVE spill-over", "B_Sneg", "D_Sneg-Hpos"),
]

# ----------------------------------------------------------------- statistics
def compare(g1, g2):
    """Full two-group comparison; returns a dict of stats."""
    g1 = g1.dropna(); g2 = g2.dropna()
    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    diff = m1 - m2
    # Welch t + CI
    se = np.sqrt(s1**2/n1 + s2**2/n2)
    dfw = se**4 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    t = diff / se
    p_t = 2 * stats.t.sf(abs(t), dfw)
    tcrit = stats.t.ppf(0.975, dfw)
    ci = (diff - tcrit*se, diff + tcrit*se)
    # Cohen's d (pooled, n-weighted)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    d = diff / sp
    # Mann-Whitney U + rank-biserial
    U, p_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    rrb = 2*U/(n1*n2) - 1
    return dict(n1=n1, n2=n2, m1=m1, m2=m2, s1=s1, s2=s2, diff=diff,
                se=se, df=dfw, t=t, p_t=p_t, ci=ci, d=d, U=U, p_u=p_u, rrb=rrb)

def stars(p):
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

def h(t): print("\n" + "="*86 + f"\n{t}\n" + "="*86)

results = {}
for title, c1, c2 in COMPARISONS:
    h(f"{title}:  {c1}  vs  {c2}")
    print(f"{'Scale':<14}{'M1 (SD)':>14}{'M2 (SD)':>14}{'n1/n2':>9}"
          f"{'diff [95% CI]':>22}{'Welch t (df)':>16}{'p':>9}{'d':>7}{'MWU p':>9}")
    print("-"*114)
    results[title] = {}
    for sc in SCALES:
        r = compare(ana.loc[ana.Condition==c1, sc], ana.loc[ana.Condition==c2, sc])
        results[title][sc] = r
        c_m1   = f"{r['m1']:.2f} ({r['s1']:.2f})"
        c_m2   = f"{r['m2']:.2f} ({r['s2']:.2f})"
        c_n    = f"{r['n1']}/{r['n2']}"
        c_diff = f"{r['diff']:+.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]"
        c_t    = f"{r['t']:.2f} ({r['df']:.0f})"
        c_pt   = f"{r['p_t']:.3g} {stars(r['p_t'])}"
        c_pu   = f"{r['p_u']:.3g} {stars(r['p_u'])}"
        flag   = "  <- value: caution" if sc == "value" else ""
        print(f"{SCALE_LABEL[sc]:<14}{c_m1:>14}{c_m2:>14}{c_n:>9}{c_diff:>22}"
              f"{c_t:>16}{c_pt:>9}{r['d']:>+7.2f}{c_pu:>9}{flag}")
    print("\n  (value: items exported as endpoints only -> 3 distinct values {1,4,7}, "
          "~50% missing.\n   Mann-Whitney U is the appropriate test; interpret with caution.)")

# ----------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9.5, "figure.dpi": 120})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)

GREY_PURE = "#cfcfcf"   # baseline (pure service) condition
GREY_MIX  = "#5a5a5a"   # incongruent (+phone) condition

def ci_halfwidth(g):
    g = g.dropna();
    return stats.t.ppf(0.975, len(g)-1) * g.std(ddof=1)/np.sqrt(len(g)) if len(g) > 1 else 0

for ax, (title, c1, c2) in zip(axes, COMPARISONS):
    x = np.arange(len(SCALES)); w = 0.36
    m1 = [results[title][sc]["m1"] for sc in SCALES]
    m2 = [results[title][sc]["m2"] for sc in SCALES]
    e1 = [ci_halfwidth(ana.loc[ana.Condition==c1, sc]) for sc in SCALES]
    e2 = [ci_halfwidth(ana.loc[ana.Condition==c2, sc]) for sc in SCALES]

    b1 = ax.bar(x - w/2, m1, w, yerr=e1, capsize=4, color=GREY_PURE,
                edgecolor="black", label=c1, error_kw=dict(lw=1.2))
    b2 = ax.bar(x + w/2, m2, w, yerr=e2, capsize=4, color=GREY_MIX,
                edgecolor="black", label=c2, error_kw=dict(lw=1.2))

    # hatch the value bars to flag compromised data
    vi = SCALES.index("value")
    b1[vi].set_hatch("///"); b2[vi].set_hatch("///")

    # value labels on bars
    for xi, m in zip(x - w/2, m1):
        ax.text(xi, 0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, m in zip(x + w/2, m2):
        ax.text(xi, 0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color="white")

    # significance brackets (Welch p for sat/loy, Mann-Whitney for value)
    for i, sc in enumerate(SCALES):
        r = results[title][sc]
        p_use = r["p_u"] if sc == "value" else r["p_t"]
        top = max(m1[i]+e1[i], m2[i]+e2[i])
        y = top + 0.35
        ax.plot([i-w/2, i-w/2, i+w/2, i+w/2], [y, y+0.1, y+0.1, y], lw=1.1, color="black")
        mark = stars(p_use) + ("†" if sc == "value" else "")
        ax.text(i, y+0.12, mark, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels([SCALE_LABEL[s] for s in SCALES])
    ax.set_ylim(0, 7.6)
    ax.set_title(title.split(" — ")[1].title() + f"\n{c1}  vs  {c2}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.axhline(4, ls=":", color="grey", lw=0.7)  # scale midpoint

axes[0].set_ylabel("Aggregated scale score (1–7, higher = more favourable)")
fig.suptitle("Spill-over comparisons across satisfaction, value, and loyalty  "
             "(bars = mean ±95% CI;  *** p<.001 ** p<.01 * p<.05;  † value = binary/partial data)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{FIG}/07_spillover_comparisons.png"
fig.savefig(out, dpi=130); plt.close(fig)
print(f"\nFigure saved: {out}")
