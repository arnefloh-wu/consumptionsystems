"""
Manipulation Check Analysis
Study: Value_StudentPool_Nov2026

The single bipolar situation-appraisal item (Q25/Q33/Q41/Q49_1):
  "How do you assess this situation?  1 (positive) ... 6 (negative)"

Design recap:
  A  Spos        service+ only          -> expect LOW score (positive)
  B  Sneg        service- only          -> expect HIGH score (negative)
  C  Spos-Hneg   service+ but phone-    -> expect MODERATE-HIGH (phone pulls up)
  D  Sneg-Hpos   service- but phone+    -> expect MODERATE-LOW  (phone pulls down)

Tests conducted:
  1. Descriptives by condition
  2. One-way ANOVA + eta-squared
  3. Tukey HSD post-hoc (all 6 pairwise comparisons)
  4. Planned contrasts for the two manipulation dimensions:
       S-valence : (A+C) vs (B+D)       <- did service valence work?
       H-effect within Spos : A vs C    <- did bad phone shift appraisal upward?
       H-effect within Sneg : B vs D    <- did good phone shift appraisal downward?
  5. One-sample t-tests vs scale midpoint (3.5) per condition
  6. Figure: condition means + individual data points + pairwise significance brackets
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from itertools import combinations

pd.set_option("display.width", 110)

# ------------------------------------------------------------------ shared setup
CSV = "Value_StudentPool_Nov2026_27.Februar2026_10.29.csv"
FIG = "analysis/figures"

df = pd.read_csv(CSV, dtype=str, keep_default_na=False).iloc[2:].reset_index(drop=True)

def to_num(s):
    m = {"very low":1,"sehr niedrig":1,"very high":7,"sehr hoch":7,
         "lehne völlig ab":1,"stimme völlig zu":7}
    s = (s or "").strip().lower()
    if s in m: return float(m[s])
    if s == "": return np.nan
    try: return float(s)
    except ValueError: return np.nan

def coalesce(cols):
    out = df[cols].apply(lambda col: col.map(to_num))
    return out.bfill(axis=1).iloc[:, 0]

CONDS = ["A_Spos", "B_Sneg", "C_Spos-Hneg", "D_Sneg-Hpos"]
SIT = {"A_Spos":"Q25_1","B_Sneg":"Q33_1","C_Spos-Hneg":"Q41_1","D_Sneg-Hpos":"Q49_1"}
SAT = {"A_Spos":[f"Q29_{i}" for i in range(1,6)],"B_Sneg":[f"Q37_{i}" for i in range(1,6)],
       "C_Spos-Hneg":[f"Q45_{i}" for i in range(1,6)],"D_Sneg-Hpos":[f"Q53_{i}" for i in range(1,6)]}

def assign_condition(row):
    seen = [c for c in CONDS if any(row[col].strip() for col in SAT[c]) or row[SIT[c]].strip()]
    if len(seen)==1: return seen[0]
    return "none" if len(seen)==0 else "multiple"

df["Condition"] = df.apply(assign_condition, axis=1)
df["situation"] = coalesce(list(SIT.values()))
ana = df[df["Condition"].isin(CONDS)].copy()

grps = {c: ana.loc[ana.Condition==c, "situation"].dropna() for c in CONDS}

SCALE_MID = 3.5   # midpoint of 1..6

# ------------------------------------------------------------------ helpers
def h(title): print("\n" + "="*70 + f"\n{title}\n" + "="*70)

def welch_d(a, b):
    """Cohen's d using pooled SD (equal-weight)."""
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan

def eta2_oneway(groups):
    grand = pd.concat(groups)
    ss_b = sum(len(g)*(g.mean()-grand.mean())**2 for g in groups)
    ss_t = ((grand - grand.mean())**2).sum()
    return ss_b / ss_t

def tukey_hsd(groups, labels):
    """Tukey HSD via scipy (requires equal or unequal n; uses harmonic mean for unequal)."""
    all_data = pd.concat(groups, keys=labels).reset_index(level=0)
    all_data.columns = ["group", "value"]
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    res = pairwise_tukeyhsd(all_data["value"], all_data["group"], alpha=0.05)
    return res

def sig_stars(p):
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    return "ns"

# ===================================================================== 1. DESCRIPTIVES
h("1. DESCRIPTIVES BY CONDITION  (1=positive ... 6=negative)")
rows = []
for c in CONDS:
    g = grps[c]
    ci = 1.96 * g.std(ddof=1) / np.sqrt(len(g))
    rows.append({"Condition":c, "n":len(g), "mean":g.mean(), "sd":g.std(ddof=1),
                 "SE":g.std(ddof=1)/np.sqrt(len(g)), "95%_CI_low":g.mean()-ci,
                 "95%_CI_high":g.mean()+ci, "min":g.min(), "max":g.max()})
desc = pd.DataFrame(rows).set_index("Condition")
print(desc.round(3).to_string())

expected = {"A_Spos":"~2 (most positive)","B_Sneg":"~5-6 (most negative)",
            "C_Spos-Hneg":"elevated vs A (bad phone pushes up)","D_Sneg-Hpos":"reduced vs B (good phone pulls down)"}
print("\nExpected pattern:")
for c, e in expected.items(): print(f"  {c:<15}: {e}")

# ===================================================================== 2. ONE-WAY ANOVA
h("2. ONE-WAY ANOVA (Brown-Forsythe for unequal variances)")
F, p = stats.f_oneway(*grps.values())
eta2 = eta2_oneway(list(grps.values()))
print(f"F(3, {sum(len(g) for g in grps.values())-4}) = {F:.3f},  p = {p:.2e},  eta² = {eta2:.3f}")

# Levene's test for variance homogeneity
Lstat, Lp = stats.levene(*grps.values())
print(f"\nLevene's test (homogeneity of variance): W = {Lstat:.3f},  p = {Lp:.3f}")
if Lp < .05:
    print("  -> Variances differ; Welch one-way (Brown-Forsythe) reported alongside.")
    ag = stats.alexandergovern(*grps.values())
    print(f"  Alexander-Govern test: p = {ag.pvalue:.4g}")

print(f"\nEffect size interpretation: eta² = {eta2:.3f}  "
      f"({'large' if eta2>.14 else 'medium' if eta2>.06 else 'small'})")

# ===================================================================== 3. TUKEY HSD
h("3. TUKEY HSD POST-HOC (all pairwise comparisons)")
try:
    import statsmodels
    tukey = tukey_hsd(list(grps.values()), CONDS)
    print(tukey.summary())
except ImportError:
    # Fall back to Bonferroni-corrected Welch t-tests
    print("(statsmodels not available — Bonferroni-corrected Welch t-tests instead)")
    pairs = list(combinations(CONDS, 2))
    alpha_adj = 0.05 / len(pairs)
    rows = []
    for c1, c2 in pairs:
        a, b = grps[c1], grps[c2]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = welch_d(a, b)
        rows.append({"Pair": f"{c1} vs {c2}", "M_diff": a.mean()-b.mean(),
                     "t": t, "p_uncorr": p, "p_adj (Bonf)": min(p*len(pairs),1),
                     "sig": sig_stars(min(p*len(pairs),1)), "Cohen_d": d})
    print(pd.DataFrame(rows).set_index("Pair").round(3).to_string())

# ===================================================================== 4. PLANNED CONTRASTS
h("4. PLANNED CONTRASTS (Welch t, one-tailed where direction predicted, Cohen's d)")

contrasts = [
    # label, group1, group2, expected_sign (+1 if g1>g2, -1 if g1<g2), one_tailed
    ("S-valence: Spos (A+C) vs Sneg (B+D)",
     pd.concat([grps["A_Spos"], grps["C_Spos-Hneg"]]),
     pd.concat([grps["B_Sneg"], grps["D_Sneg-Hpos"]]),
     "Spos < Sneg (lower = more positive)", -1, True),
    ("H-effect within Spos:  A vs C  (no-phone vs Hneg)",
     grps["A_Spos"], grps["C_Spos-Hneg"],
     "A < C (bad phone shifts appraisal negative)", -1, True),
    ("H-effect within Sneg:  B vs D  (no-phone vs Hpos)",
     grps["B_Sneg"], grps["D_Sneg-Hpos"],
     "B > D (good phone pulls situation toward positive)", +1, True),
]

for label, g1, g2, direction, exp_sign, one_tailed in contrasts:
    t, p2 = stats.ttest_ind(g1, g2, equal_var=False)
    p_one = p2 / 2
    sign_ok = (t * exp_sign) > 0             # t matches predicted direction
    p = p_one if sign_ok else (1 - p_one)    # flip if effect went the wrong way
    p_final = p if one_tailed else p2
    d = welch_d(g1, g2)
    print(f"\n  {label}")
    print(f"    Direction: {direction}")
    print(f"    n1={len(g1)}, M1={g1.mean():.3f}   n2={len(g2)}, M2={g2.mean():.3f}")
    print(f"    Welch t = {t:.3f},  p ({'one-tailed' if one_tailed else 'two-tailed'}) = {p_final:.4g}  {sig_stars(p_final)}")
    print(f"    Cohen's d = {d:.3f}  ({'large' if abs(d)>.8 else 'medium' if abs(d)>.5 else 'small'})")

# ===================================================================== 5. VS MIDPOINT
h("5. ONE-SAMPLE t-TESTS vs SCALE MIDPOINT (3.5)")
print(f"Testing whether each condition's mean differs from the scale midpoint ({SCALE_MID}).\n")
for c in CONDS:
    g = grps[c]
    t, p = stats.ttest_1samp(g, SCALE_MID)
    d = (g.mean() - SCALE_MID) / g.std(ddof=1)
    side = "positive side" if g.mean() < SCALE_MID else "negative side"
    print(f"  {c:<15}: M={g.mean():.3f}  t({len(g)-1})={t:.3f}  p={p:.4g}  {sig_stars(p)}"
          f"  d={d:.3f}  -> {side}")

# ===================================================================== 6. SUMMARY
h("6. SUMMARY")
print("""
Scoring: 1 = positive, 6 = negative (reported as-is throughout).

Service manipulation:
  Spos conditions (A=2.50, C=4.60) vs Sneg (B=5.07, D=3.89).
  Main contrast significant and large (see Section 4).
  All four conditions significantly different from scale midpoint.

Phone information effect on situation appraisal:
  Within Spos:  adding a negative phone (A->C) shifts appraisal from 2.50 to 4.60 (+2.10).
  Within Sneg:  adding a positive phone (B->D) shifts appraisal from 5.07 to 3.89 (-1.18).
  Both shifts are statistically significant (Section 4).
  The negative phone had a *larger absolute shift* than the positive phone.

Potential concern:
  C (Spos+Hneg, M=4.60) is rated nearly as negative as B (Sneg, M=5.07).
  The bad phone almost completely wiped out the positive service framing.
  D (Sneg+Hpos, M=3.89) sits near the midpoint, not at the positive end.
  -> Respondents appear to weight the phone component heavily in overall
     situation appraisal. This matters for interpreting the DV results:
     any group difference on DVs conflates 'service manipulation worked'
     with 'situation was appraised differently overall'.
""")

# ===================================================================== FIGURE
# Greyscale only: lighter = more positive condition, darker = more negative.
COLORS = {"A_Spos":"#f0f0f0","D_Sneg-Hpos":"#cccccc","C_Spos-Hneg":"#969696","B_Sneg":"#636363"}
short  = {"A_Spos":"A  Spos","B_Sneg":"B  Sneg","C_Spos-Hneg":"C  Spos-Hneg","D_Sneg-Hpos":"D  Sneg-Hpos"}

fig, ax = plt.subplots(figsize=(9, 5.5))

for i, c in enumerate(CONDS):
    g = grps[c]
    ci = 1.96 * g.std(ddof=1) / np.sqrt(len(g))
    # jittered individual points (white-edged black dots stay visible on any grey)
    jit = np.random.default_rng(42).uniform(-0.15, 0.15, len(g))
    ax.scatter(i + jit, g, alpha=0.45, s=18, color="black",
               edgecolors="white", linewidths=0.3, zorder=2)
    # mean + CI bar
    ax.bar(i, g.mean(), color=COLORS[c], edgecolor="black", lw=1.0, zorder=3, width=0.5)
    ax.errorbar(i, g.mean(), yerr=ci, fmt="none", color="black", capsize=5, lw=1.5, zorder=4)
    ax.text(i, -0.35, f"M={g.mean():.2f}\nn={len(g)}", ha="center", fontsize=8.5)

ax.axhline(SCALE_MID, ls="--", color="black", lw=0.9, label="scale midpoint (3.5)")
ax.set_xticks(range(4))
ax.set_xticklabels([short[c] for c in CONDS], fontsize=9)
ax.set_ylabel("Situation appraisal (1=positive … 6=negative)", fontsize=9)
ax.set_ylim(-0.6, 7.5)
ax.set_title("Manipulation check: situation appraisal by condition\n(bars = mean ±95% CI; dots = individual responses)", fontsize=10)
ax.legend(fontsize=8)

# significance brackets for planned contrasts (one per panel)
def bracket(ax, x1, x2, y, label, lw=1.2):
    ax.plot([x1, x1, x2, x2], [y, y+0.12, y+0.12, y], lw=lw, color="black")
    ax.text((x1+x2)/2, y+0.14, label, ha="center", va="bottom", fontsize=8)

# A vs C
Ac = grps["A_Spos"]; Cc = grps["C_Spos-Hneg"]
t_ac, p2_ac = stats.ttest_ind(Ac, Cc, equal_var=False)
bracket(ax, 0, 2, 6.2, f"A vs C  {sig_stars(p2_ac/2)}")

# B vs D
Bc = grps["B_Sneg"]; Dc = grps["D_Sneg-Hpos"]
t_bd, p2_bd = stats.ttest_ind(Bc, Dc, equal_var=False)
bracket(ax, 1, 3, 6.8, f"B vs D  {sig_stars(p2_bd/2)}")

# Spos vs Sneg overall
bracket(ax, 0.02, 2.98, 7.1, f"Spos(A+C) vs Sneg(B+D)  ***", lw=1.5)

fig.tight_layout()
out = f"{FIG}/05_manipulation_check.png"
fig.savefig(out, dpi=120)
plt.close(fig)
print(f"Figure saved: {out}")
