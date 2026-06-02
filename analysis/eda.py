"""
Exploratory Data Analysis
Study: Value_StudentPool_Nov2026  (WU Vienna - consumption systems / spill-over study)

Design (from .qsf survey flow + .docx):
  Between-subjects, BlockRandomizer SubSet=1 -> each respondent sees exactly ONE of 4
  scenario conditions describing a *bundled* product: a mobile SERVICE (S) and the
  phone hardware (H = "Handy"), which cannot be used without each other.

    A  Spos        service described positively              (single-component baseline)
    B  Sneg        service described negatively              (single-component baseline)
    C  Spos-Hneg   service positive  BUT phone negative      (incongruent -> neg. spill-over?)
    D  Sneg-Hpos   service negative  BUT phone positive      (incongruent -> pos. spill-over?)

  All dependent variables are measured about the SERVICE/provider. Research aim
  (debrief Q58): test spill-over effects between phone and service perceptions.

Constructs (same battery repeats inside every condition block, by slot position):
    situation       1 bipolar item   1..6   (1 = positive ... 6 = negative)
    perc_value      2 Likert items   1..7
    overall_value   1 MC item        1..7   (text endpoints: very low / very high)
    satisfaction    5 Likert items   1..7
    loyalty_wom     7 Likert items   1..7   (content-aligned; block-A duplicate removed,
                                             online-WOM item dropped - see note in code)
    repurchase      1 MC item        1..7   (text endpoints: very low / very high)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.simplefilter("ignore", category=FutureWarning)
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 60)

CSV = "Value_StudentPool_Nov2026_27.Februar2026_10.29.csv"
FIG = "analysis/figures"

# ----------------------------------------------------------------------------- load
# Row 0 = Q-codes (header), row 1 = question labels, row 2 = ImportId JSON. Data row 3+.
df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
df = df.iloc[2:].reset_index(drop=True)          # drop the 2 descriptive header rows

n_raw = len(df)

# ------------------------------------------------------------------- helper recodes
def to_num(s):
    """Recode a survey cell to numeric. Labelled endpoints -> 1/7, blanks -> NaN."""
    m = {
        "very low": 1, "sehr niedrig": 1, "very high": 7, "sehr hoch": 7,
        "lehne völlig ab": 1, "stimme völlig zu": 7,           # German strongly dis/agree
        "strongly disagree": 1, "strongly agree": 7,
    }
    s = (s or "").strip().lower()
    if s in m:
        return float(m[s])
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def coalesce(cols):
    """Combine the parallel per-condition columns (only one is filled per respondent)."""
    out = df[cols].apply(lambda col: col.map(to_num))
    return out.bfill(axis=1).iloc[:, 0]

# ----------------------------------------------------------- condition assignment
# Block -> its satisfaction items (most reliably populated marker of "saw this block")
SAT = {
    "A_Spos":      [f"Q29_{i}" for i in range(1, 6)],
    "B_Sneg":      [f"Q37_{i}" for i in range(1, 6)],
    "C_Spos-Hneg": [f"Q45_{i}" for i in range(1, 6)],
    "D_Sneg-Hpos": [f"Q53_{i}" for i in range(1, 6)],
}
SIT = {"A_Spos": "Q25_1", "B_Sneg": "Q33_1", "C_Spos-Hneg": "Q41_1", "D_Sneg-Hpos": "Q49_1"}

def assign_condition(row):
    seen = []
    for cond, cols in SAT.items():
        if any(row[c].strip() for c in cols) or row[SIT[cond]].strip():
            seen.append(cond)
    if len(seen) == 1:
        return seen[0]
    if len(seen) == 0:
        return "none"
    return "multiple"  # should not happen in a between-subjects design

df["Condition"] = df.apply(assign_condition, axis=1)

# ------------------------------------------------------------- coalesce constructs
df["situation"]     = coalesce(list(SIT.values()))
df["pval_1"]        = coalesce(["Q27_1", "Q35_1", "Q43_1", "Q51_1"])
df["pval_2"]        = coalesce(["Q27_2", "Q35_2", "Q43_2", "Q51_2"])
df["overall_value"] = coalesce(["Q28", "Q36", "Q44", "Q52"])
for i in range(1, 6):
    df[f"sat_{i}"] = coalesce([f"Q29_{i}", f"Q37_{i}", f"Q45_{i}", f"Q53_{i}"])

# --- Loyalty/WOM: content-aligned 7-item scale (see note below) -----------------
# The raw 8 slots are NOT comparable across blocks:
#   * Block A (Q30) repeats "encourage friends/relatives to purchase" in slots 6 AND 8
#     (a duplicate; the two ratings correlate r=.86) and OMITS the online-WOM item.
#   * Blocks B/C/D (Q38/Q46/Q54) use slot 7 = "express positive opinions online" and
#     slot 8 = "speak positively in personal conversations".
# Coalescing by raw slot therefore mixed different items in slots 7 & 8. We rebuild
# the scale by CONTENT: drop the block-A duplicate (Q30_8) and the online-WOM item
# (present only in B/C/D, so unusable in a cross-condition scale), and map the
# "personal conversations" item from its true per-block slot.
LOY_MAP = {
    "loy_1": ["Q30_1", "Q38_1", "Q46_1", "Q54_1"],  # continue using for years
    "loy_2": ["Q30_2", "Q38_2", "Q46_2", "Q54_2"],  # first choice
    "loy_3": ["Q30_3", "Q38_3", "Q46_3", "Q54_3"],  # would choose if free
    "loy_4": ["Q30_4", "Q38_4", "Q46_4", "Q54_4"],  # recommend to advice-seekers
    "loy_5": ["Q30_5", "Q38_5", "Q46_5", "Q54_5"],  # recommend to friends/relatives
    "loy_6": ["Q30_6", "Q38_6", "Q46_6", "Q54_6"],  # encourage to purchase
    "loy_7": ["Q30_7", "Q38_8", "Q46_8", "Q54_8"],  # speak positively in conversations
}
for name, cols in LOY_MAP.items():
    df[name] = coalesce(cols)
# Online-WOM item kept separately (B/C/D only; NaN for condition A) for optional use.
df["loy_online_BCD"] = coalesce(["Q38_7", "Q46_7", "Q54_7"])

df["repurchase"]    = coalesce(["Q31", "Q39", "Q47", "Q55"])

SAT_ITEMS = [f"sat_{i}" for i in range(1, 6)]
LOY_ITEMS = list(LOY_MAP.keys())             # 7 content-aligned items (duplicate removed)
PVAL_ITEMS = ["pval_1", "pval_2"]

df["perc_value"]   = df[PVAL_ITEMS].mean(axis=1)
df["satisfaction"] = df[SAT_ITEMS].mean(axis=1)
df["loyalty_wom"]  = df[LOY_ITEMS].mean(axis=1)

# Robust vs compromised measures (see section 5b): perceived value & overall value
# exported only their two scale ENDPOINTS (no mid-scale 2..6) and are sparsely
# populated -> excluded from inferential analysis.
DV_ROBUST = ["satisfaction", "loyalty_wom", "repurchase"]
COMPROMISED = ["perc_value", "overall_value"]

# --------------------------------------------------------------------- metadata
df["Duration"] = pd.to_numeric(df["Duration (in seconds)"], errors="coerce")
df["Progress"] = pd.to_numeric(df["Progress"], errors="coerce")
df["Finished_b"] = df["Finished"].str.strip().eq("True")
df["StartDate_dt"] = pd.to_datetime(df["StartDate"], errors="coerce")

# age: strip whitespace/newlines, coerce; flag implausible for a student pool
age = pd.to_numeric(df["Q5"].str.strip(), errors="coerce")
df["age"] = age.where((age >= 16) & (age <= 80))     # 'tes' -> NaN, keep numeric only
df["gender"] = df["Q4"].str.strip().replace("", np.nan)

CONDS = ["A_Spos", "B_Sneg", "C_Spos-Hneg", "D_Sneg-Hpos"]

# ============================================================= reporting helpers
def h(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)

def cronbach_alpha(frame):
    d = frame.dropna()
    k = d.shape[1]
    if k < 2 or len(d) < 3:
        return np.nan, len(d)
    item_var = d.var(axis=0, ddof=1).sum()
    total_var = d.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_var / total_var), len(d)

# =================================================================== 1. OVERVIEW
h("1. DATASET OVERVIEW")
print(f"File              : {CSV}")
print(f"Response records  : {n_raw}")
print(f"Variables (raw)   : {df.shape[1] - 0} columns in file")
print(f"Collection window : {df['StartDate_dt'].min():%Y-%m-%d}  ->  {df['StartDate_dt'].max():%Y-%m-%d}")
print(f"Finished = True   : {df['Finished_b'].sum()}  ({df['Finished_b'].mean()*100:.1f}%)")
print(f"Distribution chan.: {df['DistributionChannel'].value_counts().to_dict()}")
print(f"User language     : {df['UserLanguage'].value_counts().to_dict()}")

# ============================================================= 2. DATA QUALITY
h("2. DATA QUALITY & COMPLETION")
print("Progress (% of survey completed) summary:")
print(df["Progress"].describe()[["count", "mean", "min", "25%", "50%", "75%", "max"]].round(1).to_string())
print(f"\nProgress == 100 : {(df['Progress'] == 100).sum()}")
print(f"Progress <  100 : {(df['Progress'] < 100).sum()}  (partial / drop-out)")

print("\nDuration (seconds) for finished respondents:")
fd = df.loc[df["Finished_b"], "Duration"]
print(fd.describe()[["count", "mean", "min", "25%", "50%", "75%", "max"]].round(0).to_string())
print(f"Speeders < 60s  (finished): {(fd < 60).sum()}")
print(f"Very long > 1h  (finished): {(fd > 3600).sum()}  (Qualtrics keeps tab open -> inflated)")
print(f"Median minutes  (finished): {fd.median()/60:.1f} min")

# straightlining on the loyalty battery (zero variance across items)
loy = df[LOY_ITEMS]
n_loy_items = len(LOY_ITEMS)
straight = (loy.notna().sum(axis=1) >= n_loy_items) & (loy.std(axis=1, ddof=0) == 0)
print(f"\nStraight-lining on {n_loy_items}-item loyalty battery (all identical): {int(straight.sum())} respondents")

# ===================================================== 3. CONDITION ASSIGNMENT
h("3. EXPERIMENTAL CONDITION ASSIGNMENT (between-subjects)")
vc = df["Condition"].value_counts()
print(vc.to_string())
ana = df[df["Condition"].isin(CONDS)].copy()
print(f"\nAnalytic sample (assigned to a condition): n = {len(ana)}")
expected = len(ana) / 4
chi = stats.chisquare([vc.get(c, 0) for c in CONDS])
print(f"Equal-cell chi-square test: chi2={chi.statistic:.2f}, p={chi.pvalue:.3f}  "
      f"(EvenPresentation=False in qsf -> unequal cells expected)")

# completeness of each DV within the analytic sample
h("3b. ITEM / SCALE COMPLETENESS within analytic sample")
for name in ["situation", "perc_value", "overall_value", "satisfaction", "loyalty_wom", "repurchase"]:
    n_ok = ana[name].notna().sum()
    print(f"  {name:<14}: {n_ok:3d}/{len(ana)}  non-missing  ({n_ok/len(ana)*100:4.0f}%)")

# ============================================================= 4. DEMOGRAPHICS
h("4. SAMPLE DESCRIPTION (analytic sample)")
print("Gender:")
print(ana["gender"].value_counts(dropna=False).to_string())
print("\nAge:")
print(ana["age"].describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(1).to_string())
print(f"Has mobile phone (Q2): {ana['Q2'].str.strip().value_counts().to_dict()}")

# =========================================================== 5. SCALE QUALITY
h("5. SCALE RELIABILITY (Cronbach's alpha, analytic sample)")
a_sat, n_sat = cronbach_alpha(ana[SAT_ITEMS])
a_loy, n_loy = cronbach_alpha(ana[LOY_ITEMS])
print(f"  Satisfaction    (5 items) : alpha = {a_sat:.3f}   (n={n_sat})")
print(f"  Loyalty / WOM   (7 items) : alpha = {a_loy:.3f}   (n={n_loy})")
print("  NOTE: loyalty rebuilt as 7 content-aligned items: block-A duplicate")
print("        (Q30_8 = Q30_6) removed; online-WOM item dropped (absent in block A).")

print("\nScale score descriptives (1..7; situation is 1=positive..6=negative):")
desc = ana[["situation"] + DV_ROBUST].describe().T
print(desc[["count", "mean", "std", "min", "50%", "max"]].round(2).to_string())

# ----------------------------------------------------- 5b. COMPROMISED MEASURES
h("5b. COMPROMISED MEASURES - perceived value & overall value (EXCLUDED)")
print("These two questions exported ONLY their scale endpoints, with no mid-scale (2..6)")
print("values, and are sparsely populated. The same endpoint is coded inconsistently")
print("(e.g. both 'stimme voellig zu' and '7'), the signature of a choice-set edit")
print("mid-fielding (the two CSV exports already differed by the Q56 block).\n")
for v, items in [("perc_value", PVAL_ITEMS), ("overall_value", ["overall_value"])]:
    vals = sorted(ana[v].dropna().unique().tolist())
    print(f"  {v:<14}: n={ana[v].notna().sum():3d}/{len(ana)}   distinct values present = {vals}")
print("\n  -> Excluded from all inferential analysis below.")

# ===================================================== 6. MANIPULATION CHECK
h("6. MANIPULATION CHECK - situation appraisal by condition")
print("Lower = more positive (1) ... higher = more negative (6).")
print("Expect S-positive (A, C) LOW; S-negative (B, D) HIGH.\n")
sit = ana.groupby("Condition")["situation"].agg(["count", "mean", "std"]).reindex(CONDS).round(2)
print(sit.to_string())
spos = ana.loc[ana["Condition"].isin(["A_Spos", "C_Spos-Hneg"]), "situation"].dropna()
sneg = ana.loc[ana["Condition"].isin(["B_Sneg", "D_Sneg-Hpos"]), "situation"].dropna()
t = stats.ttest_ind(spos, sneg, equal_var=False)
print(f"\nS-positive (A,C) M={spos.mean():.2f}  vs  S-negative (B,D) M={sneg.mean():.2f}")
print(f"Welch t={t.statistic:.2f}, p={t.pvalue:.4g}  -> manipulation {'OK' if t.pvalue<.05 else 'WEAK'}")

# ===================================================== 7. DV BY CONDITION + ANOVA
h("7. DEPENDENT VARIABLES BY CONDITION")
DVs = DV_ROBUST
means = ana.groupby("Condition")[DVs].mean().reindex(CONDS).round(2)
ns = ana.groupby("Condition")[DVs].count().reindex(CONDS)
print("Means:")
print(means.to_string())
print("\nn per cell:")
print(ns.to_string())

print("\nOne-way ANOVA across the 4 conditions:")
for dv in DVs:
    groups = [ana.loc[ana["Condition"] == c, dv].dropna() for c in CONDS]
    F, p = stats.f_oneway(*groups)
    # eta squared
    grand = pd.concat(groups)
    ss_b = sum(len(g) * (g.mean() - grand.mean())**2 for g in groups)
    ss_t = ((grand - grand.mean())**2).sum()
    eta2 = ss_b / ss_t
    print(f"  {dv:<13}: F={F:6.2f}  p={p:.3g}  eta^2={eta2:.3f}")

# ===================================================== 8. SPILL-OVER CONTRASTS
h("8. FOCAL SPILL-OVER CONTRASTS (the research question)")
def contrast(label, c1, c2, dv):
    a = ana.loc[ana["Condition"] == c1, dv].dropna()
    b = ana.loc[ana["Condition"] == c2, dv].dropna()
    if len(a) < 2 or len(b) < 2:
        print(f"  {label:<40} {dv:<13}: insufficient n")
        return
    t = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
    print(f"  {label:<40} {dv:<13}: "
          f"{a.mean():.2f} vs {b.mean():.2f}  Welch t={t.statistic:5.2f} p={t.pvalue:.3g} d={d:+.2f}")

print("Negative spill-over  (does a BAD phone drag down a GOOD service?)  A_Spos vs C_Spos-Hneg:")
for dv in DVs:
    contrast("  Spos  ->  Spos-Hneg", "A_Spos", "C_Spos-Hneg", dv)
print("\nPositive spill-over  (does a GOOD phone lift a BAD service?)      B_Sneg vs D_Sneg-Hpos:")
for dv in DVs:
    contrast("  Sneg  ->  Sneg-Hpos", "B_Sneg", "D_Sneg-Hpos", dv)

# ===================================================== 9. CORRELATIONS
h("9. CONSTRUCT CORRELATIONS (pooled analytic sample, Pearson)")
corr = ana[["situation"] + DV_ROBUST].corr().round(2)
print(corr.to_string())
print("(situation is reverse-keyed: 1=positive..6=negative, hence negative correlations)")

# =================================================================== FIGURES
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
COLORS = {"A_Spos": "#2c7fb8", "B_Sneg": "#d95f0e",
          "C_Spos-Hneg": "#7fcdbb", "D_Sneg-Hpos": "#fec44f"}
short = {"A_Spos": "A\nSpos", "B_Sneg": "B\nSneg",
         "C_Spos-Hneg": "C\nSpos-Hneg", "D_Sneg-Hpos": "D\nSneg-Hpos"}

# Fig 1: sample / quality dashboard
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
vc.reindex(CONDS).plot.bar(ax=ax[0, 0], color=[COLORS[c] for c in CONDS])
ax[0, 0].set_title("Condition cell sizes (analytic sample)")
ax[0, 0].set_xticklabels([short[c] for c in CONDS], rotation=0)
ax[0, 0].set_ylabel("respondents")

ax[0, 1].hist(df["Progress"].dropna(), bins=20, color="#888", edgecolor="white")
ax[0, 1].set_title("Survey progress (all records)")
ax[0, 1].set_xlabel("% completed"); ax[0, 1].set_ylabel("count")

fdmin = (fd / 60).clip(upper=40)
ax[1, 0].hist(fdmin, bins=30, color="#3182bd", edgecolor="white")
ax[1, 0].set_title("Completion time (finished, capped 40 min)")
ax[1, 0].set_xlabel("minutes"); ax[1, 0].set_ylabel("count")

ana["age"].dropna().plot.hist(ax=ax[1, 1], bins=range(17, 32), color="#31a354", edgecolor="white")
ax[1, 1].set_title("Age distribution (analytic sample)")
ax[1, 1].set_xlabel("age")
fig.tight_layout(); fig.savefig(f"{FIG}/01_sample_quality.png"); plt.close(fig)

# Fig 2: manipulation check + DV means with 95% CI
def ci95(s):
    s = s.dropna()
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
m = sit["mean"]; e = [ci95(ana.loc[ana.Condition == c, "situation"]) for c in CONDS]
ax[0].bar(range(4), m, yerr=e, color=[COLORS[c] for c in CONDS], capsize=4)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([short[c] for c in CONDS])
ax[0].set_title("Manipulation check: situation appraisal\n(1=positive ... 6=negative)")
ax[0].axhline(3.5, ls="--", c="grey", lw=.8); ax[0].set_ylabel("mean")

x = np.arange(4); w = 0.25
for j, dv in enumerate(DVs):
    mv = [ana.loc[ana.Condition == c, dv].mean() for c in CONDS]
    ev = [ci95(ana.loc[ana.Condition == c, dv]) for c in CONDS]
    ax[1].bar(x + (j-1)*w, mv, w, yerr=ev, capsize=2, label=dv)
ax[1].set_xticks(x); ax[1].set_xticklabels([short[c] for c in CONDS])
ax[1].set_title("Service evaluations by condition (1..7, mean ±95% CI)")
ax[1].set_ylabel("mean"); ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(f"{FIG}/02_manipulation_and_DVs.png"); plt.close(fig)

# Fig 3: correlation heatmap
fig, ax = plt.subplots(figsize=(6.5, 5.4))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right"); ax.set_yticklabels(corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.values[i, j]) > .5 else "black", fontsize=8)
ax.set_title("Construct correlations (pooled)")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(f"{FIG}/03_correlations.png"); plt.close(fig)

# Fig 4: distribution of key DVs by condition (boxplots)
fig, ax = plt.subplots(1, len(DVs), figsize=(11, 4))
for k, dv in enumerate(DVs):
    data = [ana.loc[ana.Condition == c, dv].dropna() for c in CONDS]
    bp = ax[k].boxplot(data, patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], CONDS):
        patch.set_facecolor(COLORS[c])
    ax[k].set_xticklabels(["A", "B", "C", "D"])
    ax[k].set_title(dv); ax[k].set_ylim(0.5, 7.5)
fig.suptitle("Distribution of service evaluations by condition (A=Spos B=Sneg C=Spos-Hneg D=Sneg-Hpos)")
fig.tight_layout(); fig.savefig(f"{FIG}/04_DV_distributions.png"); plt.close(fig)

h("FIGURES WRITTEN")
for f in ["01_sample_quality", "02_manipulation_and_DVs", "03_correlations", "04_DV_distributions"]:
    print(f"  {FIG}/{f}.png")
print("\nDone.")
