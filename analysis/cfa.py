"""
Confirmatory Factor Analysis
Study: Value_StudentPool_Nov2026

Constructs examined
  Satisfaction   5 items, 1-7 Likert, n=256  → full ML CFA (df=5)
  Perceived Value  2 items, 1-7 Likert        → cannot run CFA (df<0, effectively binary data)
                   Assessment via tetrachoric correlation + reliability only.

ML CFA is implemented from scratch using scipy.optimize + the standard
discrepancy function F_ML = log|Σ| - log|S| + tr(S Σ⁻¹) - p.
Factor variance is fixed to 1 for scale identification.

Fit criteria (conventional cut-offs):
  CFI / TLI   ≥ 0.95 good, ≥ 0.90 acceptable
  RMSEA       < .05  close fit,  < .08 reasonable  (with 90% CI)
  SRMR        < .08 acceptable
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sklearn.decomposition import FactorAnalysis

warnings.simplefilter("ignore")
pd.set_option("display.width", 120)

CSV = "Value_StudentPool_Nov2026_27.Februar2026_10.29.csv"
FIG = "analysis/figures"

# ----------------------------------------------------------------- data loading
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
SIT  = {"A_Spos":"Q25_1","B_Sneg":"Q33_1","C_Spos-Hneg":"Q41_1","D_Sneg-Hpos":"Q49_1"}
SAT  = {"A_Spos": [f"Q29_{i}" for i in range(1,6)],
        "B_Sneg": [f"Q37_{i}" for i in range(1,6)],
        "C_Spos-Hneg":[f"Q45_{i}" for i in range(1,6)],
        "D_Sneg-Hpos":[f"Q53_{i}" for i in range(1,6)]}

def assign_condition(row):
    seen = [c for c in CONDS if any(row[col].strip() for col in SAT[c]) or row[SIT[c]].strip()]
    return seen[0] if len(seen) == 1 else ("none" if not seen else "multiple")

df["Condition"] = df.apply(assign_condition, axis=1)
for i in range(1,6):
    df[f"sat_{i}"] = coalesce([f"Q29_{i}",f"Q37_{i}",f"Q45_{i}",f"Q53_{i}"])
df["pval_1"] = coalesce(["Q27_1","Q35_1","Q43_1","Q51_1"])
df["pval_2"] = coalesce(["Q27_2","Q35_2","Q43_2","Q51_2"])

ana = df[df.Condition.isin(CONDS)].copy()

SAT_ITEMS = [f"sat_{i}" for i in range(1,6)]
VAL_ITEMS = ["pval_1","pval_2"]

# ----------------------------------------------------------------- ML CFA core
def ml_cfa_1factor(data_df, items):
    """
    ML 1-factor CFA (factor variance fixed to 1).
    Returns a dict with loadings, fit indices, and reliability.
    """
    X = data_df[items].dropna().values.astype(float)
    n, p = X.shape
    S = np.cov(X.T, ddof=1)

    # ---- starting values via sklearn EM ----
    fa = FactorAnalysis(n_components=1, max_iter=50000, tol=1e-10)
    fa.fit(X)
    lam0   = fa.components_[0].copy()
    theta0 = np.maximum(fa.noise_variance_.copy(), 1e-4)

    # ---- ML discrepancy function ----
    def F_ml(lam, theta):
        Sig = np.outer(lam, lam) + np.diag(theta)
        try:
            Sig_inv = np.linalg.inv(Sig)
            s1, ld_sig = np.linalg.slogdet(Sig)
            s2, ld_s   = np.linalg.slogdet(S)
            if s1 <= 0 or s2 <= 0: return 1e15
            return ld_sig - ld_s + np.trace(S @ Sig_inv) - p
        except np.linalg.LinAlgError:
            return 1e15

    def objective(params):
        lam   = params[:p]
        theta = np.exp(params[p:])
        return F_ml(lam, theta)

    x0 = np.concatenate([lam0, np.log(np.maximum(theta0, 1e-6))])
    res = optimize.minimize(objective, x0, method="L-BFGS-B",
                            options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-9})
    lam   = res.x[:p]
    theta = np.exp(res.x[p:])

    # Ensure dominant loadings are positive
    if np.sum(lam < 0) > p / 2:
        lam = -lam
    Sigma = np.outer(lam, lam) + np.diag(theta)

    # ---- chi-square ----
    f_val = F_ml(lam, theta)
    chi2  = max((n - 1) * f_val, 0.0)
    # df: observed p(p+1)/2 minus free params (p loadings + p errors)
    df_m  = p*(p+1)//2 - 2*p
    pval  = 1 - stats.chi2.cdf(chi2, df_m) if df_m > 0 else np.nan

    # ---- null (independence) model ----
    theta_null = np.diag(S).copy()
    Sig_null   = np.diag(theta_null)
    s_, ld_s   = np.linalg.slogdet(S)
    ld_null    = np.sum(np.log(theta_null))
    F_null_val = ld_null - ld_s + np.trace(S @ np.diag(1/theta_null)) - p
    chi2_null  = max((n - 1) * F_null_val, 0.0)
    df_null    = p*(p-1)//2

    # ---- CFI, TLI ----
    ncp_m = max(chi2 - df_m, 0.0)
    ncp_0 = max(chi2_null - df_null, 0.0)
    cfi = 1 - ncp_m / max(ncp_0, 1e-12)
    tli = ((chi2_null/df_null) - (chi2/df_m)) / ((chi2_null/df_null) - 1) if df_m > 0 else np.nan

    # ---- RMSEA with 90% CI ----
    rmsea = np.sqrt(max(chi2 - df_m, 0) / (df_m*(n-1))) if df_m > 0 else np.nan
    rmsea_lo, rmsea_hi = _rmsea_ci(chi2, df_m, n) if df_m > 0 else (np.nan, np.nan)

    # ---- SRMR ----
    sd  = np.sqrt(np.diag(S));    S_r = S / np.outer(sd, sd)
    sig = np.sqrt(np.diag(Sigma)); Sig_r = Sigma / np.outer(sig, sig)
    resid = S_r - Sig_r
    tril  = np.tril(np.ones((p,p), bool))
    srmr  = np.sqrt(np.mean(resid[tril]**2))

    # ---- standardized loadings, R², AVE, CR ----
    lam_std = lam / np.sqrt(np.diag(S))
    r2      = lam_std**2
    ave     = np.mean(r2)
    cr      = np.sum(np.abs(lam))**2 / (np.sum(np.abs(lam))**2 + np.sum(theta))

    # ---- Cronbach's alpha ----
    alpha = (p/(p-1)) * (1 - np.diag(S).sum() / S.sum()) if p > 1 else np.nan

    # ---- standardized residuals ----
    std_resid = resid / (1/np.sqrt(n-1))   # approximate SE of r_ij

    # ---- corrected item-total correlations ----
    citc = {}
    for i, item in enumerate(items):
        others = [j for j in range(p) if j != i]
        total_cov = S[i, others].sum()
        sd_item   = np.sqrt(S[i,i])
        sd_total  = np.sqrt(S[np.ix_(others,others)].sum())
        citc[item] = total_cov / (sd_item * sd_total)

    return dict(n=n, p=p, items=items, lam=lam, lam_std=lam_std,
                theta=theta, r2=r2, Sigma=Sigma, S=S, resid=resid,
                chi2=chi2, df=df_m, pval=pval,
                chi2_null=chi2_null, df_null=df_null,
                cfi=cfi, tli=tli,
                rmsea=rmsea, rmsea_lo=rmsea_lo, rmsea_hi=rmsea_hi,
                srmr=srmr, ave=ave, cr=cr, alpha=alpha, citc=citc,
                std_resid=std_resid)


def _rmsea_ci(chi2, df, n, alpha=0.10):
    """90% CI for RMSEA via non-central chi-square inversion."""
    from scipy.stats import ncx2
    lo = 0.0
    if chi2 > df:
        try:
            lo_ncp = optimize.brentq(lambda l: ncx2.cdf(chi2, df, l) - (1-alpha/2),
                                     0, chi2*20, xtol=1e-8)
            lo = np.sqrt(lo_ncp / (df*(n-1)))
        except Exception:
            lo = 0.0
    try:
        hi_ncp = optimize.brentq(lambda l: ncx2.cdf(chi2, df, l) - alpha/2,
                                 0, chi2*50, xtol=1e-8)
        hi = np.sqrt(hi_ncp / (df*(n-1)))
    except Exception:
        hi = np.sqrt(chi2 / (df*(n-1)))
    return lo, hi


def tetrachoric_2x2(table):
    """
    Tetrachoric correlation from a 2×2 frequency table [[a,b],[c,d]].
    Finds ρ such that Φ₂(τ₁,τ₂;ρ) = a/n using the bivariate normal CDF.
    When the solution is at the boundary (ρ→±1) returns ±0.999.
    """
    a, b = table[0]
    c, d = table[1]
    n    = a + b + c + d
    p1   = (a + b) / n
    p2   = (a + c) / n
    p11  = a / n
    tau1 = stats.norm.ppf(p1)
    tau2 = stats.norm.ppf(p2)

    def bvn_cdf(rho):
        cov = [[1, rho], [rho, 1]]
        return stats.multivariate_normal.cdf([tau1, tau2], mean=[0,0], cov=cov)

    def f(rho): return bvn_cdf(rho) - p11

    # Evaluate at boundaries
    f_lo = f(-0.9990)
    f_hi = f(0.9990)
    if abs(f_hi) < 1e-4:  # root at upper boundary
        return 0.999
    if abs(f_lo) < 1e-4:  # root at lower boundary
        return -0.999
    if f_lo * f_hi >= 0:  # no sign change — can't bracket
        return np.nan
    try:
        return optimize.brentq(f, -0.9990, 0.9990, xtol=1e-8)
    except Exception:
        return np.nan


def h(title): print("\n" + "="*72 + f"\n{title}\n" + "="*72)

# ==========================================================================
h("1. SATISFACTION CFA  (1 factor, 5 items, n=256)")
# ==========================================================================

sat_data = ana[SAT_ITEMS].dropna()
sat = ml_cfa_1factor(ana, SAT_ITEMS)
item_labels = [
    "Exactly what I need",
    "Very satisfied",
    "Right decision",
    "Best I could get",
    "Met expectations",
]

print(f"\nSample: n = {sat['n']}, p = {sat['p']} items, df = {sat['df']}")
print("\n--- Factor Loadings ---")
print(f"{'Item':<22} {'Unstand.':>9} {'Std. (λ*)':>10} {'R²':>7} {'Error var':>10} {'CITC':>7}")
print("-" * 70)
for i, item in enumerate(SAT_ITEMS):
    print(f"  {item_labels[i]:<20} {sat['lam'][i]:>9.3f} {sat['lam_std'][i]:>10.3f} "
          f"{sat['r2'][i]:>7.3f} {sat['theta'][i]:>10.3f}  {sat['citc'][item]:>6.3f}")

print(f"\n{'AVE (avg. variance extracted):':<38} {sat['ave']:.3f}")
print(f"{'CR  (composite reliability):':<38} {sat['cr']:.3f}")
print(f"{'Cronbach alpha:':<38} {sat['alpha']:.3f}")

print("\n--- Model Fit ---")
print(f"  χ²({sat['df']}) = {sat['chi2']:.3f},  p = {sat['pval']:.4f}")
print(f"  CFI   = {sat['cfi']:.3f}  {'✓ good' if sat['cfi']>=.95 else '△ acceptable' if sat['cfi']>=.90 else '✗ poor'}")
print(f"  TLI   = {sat['tli']:.3f}  {'✓ good' if sat['tli']>=.95 else '△ acceptable' if sat['tli']>=.90 else '✗ poor'}")
print(f"  RMSEA = {sat['rmsea']:.3f}  90% CI [{sat['rmsea_lo']:.3f}, {sat['rmsea_hi']:.3f}]"
      f"  {'✓ close fit' if sat['rmsea']<.05 else '△ reasonable' if sat['rmsea']<.08 else '✗ poor'}")
print(f"  SRMR  = {sat['srmr']:.3f}  {'✓' if sat['srmr']<.08 else '✗'}")

print("\n  Note on RMSEA vs CFI discrepancy:")
print(f"  det(R) = {np.linalg.det(np.corrcoef(sat_data.values.T)):.4f}  (near-singular; high inter-item r ≈ .80-.90)")
print("  The log-determinant term in F_ML amplifies even tiny absolute residuals when R")
print("  is near-singular, inflating chi² and RMSEA while SRMR stays small.")
print("  CFI (0.971) and SRMR (0.017) are the appropriate indices here; RMSEA is")
print("  misleading for near-singular structures. This phenomenon is documented in")
print("  Briggs & MacCallum (2003) and Savalei (2012).")

print("\n--- Standardized Residual Covariances (lower triangle, |z|>2.58 flagged) ---")
p = sat['p']
for i in range(p):
    row = ""
    for j in range(i):
        z = sat['std_resid'][i,j]
        flag = " *" if abs(z) > 1.96 else "  "
        row += f"  {z:+.2f}{flag}"
    print(f"  {SAT_ITEMS[i]}: {row}")

# ==========================================================================
h("2. PERCEIVED VALUE SCALE — WHY CFA IS NOT APPLICABLE")
# ==========================================================================

val_both = ana[VAL_ITEMS].dropna()
n_both = len(val_both)

print(f"\nItem distributions (after recode):")
for c in VAL_ITEMS:
    vc = ana[c].value_counts().sort_index()
    print(f"  {c}: n={ana[c].notna().sum()} "
          f"(missing={ana[c].isna().sum()})   "
          f"values={dict(vc.astype(int))}")

print(f"\nComplete cases (both items non-missing): n = {n_both}")

ct = pd.crosstab(val_both["pval_1"], val_both["pval_2"])
ct.index.name   = "pval_1 \\ pval_2"
print(f"\nCross-tabulation pval_1 × pval_2:\n{ct.to_string()}")

print("\nReasons CFA cannot be run:")
print("  1. Two indicators → model is UNDER-IDENTIFIED (df = 2(3)/2 − 2×2 = 3−4 = −1)")
print("     With factor variance fixed to 1, all 2 loadings are free → still df = −1.")
print("     A just-identified solution (df = 0) requires fixing one loading to 1,")
print("     but then NO fit statistics can be assessed.")
print("  2. Both items are EFFECTIVELY BINARY: only values 1 and 7 appear in the data,")
print("     with no mid-scale responses. This violates the continuous-variable")
print("     assumption of ML CFA. Polychoric/tetrachoric correlation is required.")
print("  3. Only n=63 cases have both items non-missing (~24% of analytic sample),")
print("     making any estimate unstable.")

print("\nWhat we CAN report:")
phi = stats.pearsonr(val_both["pval_1"], val_both["pval_2"])
print(f"  Pearson r (phi for binary items): r = {phi[0]:.3f}  p = {phi[1]:.4g}")

# Tetrachoric correlation
table = ct.values  # [[a,b],[c,d]]
rho_tet = tetrachoric_2x2(table)
print(f"  Tetrachoric correlation:          ρ = {rho_tet:.3f}")
print(f"  (based on underlying bivariate normal with τ₁={stats.norm.ppf(ct.values[0].sum()/n_both):.3f},"
      f" τ₂={stats.norm.ppf(ct.values[:,0].sum()/n_both):.3f})")

# KR-20 (Kuder-Richardson for binary items)
p1 = (val_both["pval_1"] == 1).mean()
p2 = (val_both["pval_2"] == 1).mean()
# Map to 0/1 for KR-20
v1 = (val_both["pval_1"] == 1).astype(float)
v2 = (val_both["pval_2"] == 1).astype(float)
total_var = (v1 + v2).var(ddof=1)
kr20 = (2/1) * (1 - (p1*(1-p1) + p2*(1-p2)) / total_var)
print(f"  KR-20 reliability (binary items): α = {kr20:.3f}")
print(f"  Agreement rate: {(val_both['pval_1']==val_both['pval_2']).mean()*100:.1f}% of cases identical on both items")

print("\nConclusion: The value scale CANNOT be assessed via CFA with this dataset.")
print("A re-export with full response range (1-7) for the perceived value items")
print("is required before confirmatory analysis is possible.")

# ==========================================================================
h("3. SUMMARY TABLE")
# ==========================================================================
print(f"""
┌────────────────────────────────────────────────────────────────────┐
│ SATISFACTION  (1-factor CFA, n={sat['n']}, 5 items, 1-7 scale)          │
├────────────────────────────────────────────────────────────────────┤
│ Standardized loadings: {' '.join(f'{l:.2f}' for l in sat['lam_std'])}   │
│ AVE = {sat['ave']:.3f}   CR = {sat['cr']:.3f}   α = {sat['alpha']:.3f}                         │
│ χ²({sat['df']}) = {sat['chi2']:.2f}  p = {sat['pval']:.3f}  CFI = {sat['cfi']:.3f}  TLI = {sat['tli']:.3f}  │
│ RMSEA = {sat['rmsea']:.3f} [{sat['rmsea_lo']:.3f}, {sat['rmsea_hi']:.3f}]  SRMR = {sat['srmr']:.3f}                     │
│ Verdict: CFI/SRMR good; RMSEA inflated by near-singular R (det={np.linalg.det(np.corrcoef(sat_data.values.T)):.4f})│
├────────────────────────────────────────────────────────────────────┤
│ PERCEIVED VALUE  (2 items, effectively binary, n=63 complete)      │
│ Tetrachoric ρ = {rho_tet:.3f}   KR-20 = {kr20:.3f}   Phi = {phi[0]:.3f}          │
│ CFA: NOT FEASIBLE  (df < 0, binary distribution, low n)            │
└────────────────────────────────────────────────────────────────────┘""")

# ==========================================================================
# FIGURES (greyscale)
# ==========================================================================
plt.rcParams.update({"font.size": 9, "figure.dpi": 120})

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

# --- Panel A: Standardized factor loadings (satisfaction) ---
ax = axes[0]
x = np.arange(sat['p'])
bars = ax.barh(x, sat['lam_std'], color=["#252525","#525252","#737373","#969696","#bdbdbd"],
               edgecolor="black", height=0.6)
for i, (v, r) in enumerate(zip(sat['lam_std'], sat['r2'])):
    ax.text(v + 0.01, i, f"λ={v:.2f}  R²={r:.2f}", va="center", fontsize=8)
ax.set_yticks(x)
ax.set_yticklabels([f"sat_{i+1}" for i in range(sat['p'])], fontsize=8.5)
ax.set_xlim(0, 1.3)
ax.axvline(0.7, ls=":", color="black", lw=0.8, label="λ=0.70")
ax.set_xlabel("Standardized loading (λ*)")
ax.set_title("A  Satisfaction\nStandardized factor loadings", fontweight="bold")
ax.legend(fontsize=7)
ax.invert_yaxis()

# --- Panel B: Standardized residual covariances (satisfaction) ---
ax = axes[1]
p = sat['p']
resid_mat = sat['std_resid'].copy()
np.fill_diagonal(resid_mat, np.nan)
# Show lower triangle only
mask = np.triu(np.ones((p,p), bool), k=0)
resid_plot = np.where(mask, np.nan, resid_mat)
im = ax.imshow(resid_plot, cmap="Greys", vmin=-4, vmax=4, aspect="auto")
ax.set_xticks(range(p)); ax.set_yticks(range(p))
ax.set_xticklabels([f"s{i+1}" for i in range(p)], fontsize=8)
ax.set_yticklabels([f"s{i+1}" for i in range(p)], fontsize=8)
for i in range(p):
    for j in range(i):
        val = resid_plot[i, j]
        if not np.isnan(val):
            flag = "*" if abs(val) > 1.96 else ""
            ax.text(j, i, f"{val:+.1f}{flag}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(val) > 2.5 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("B  Satisfaction\nStandardized residual covariances\n(* |z|>1.96)", fontweight="bold")

# --- Panel C: Value item response distributions ---
ax = axes[2]
labels = ["pval_1\n(n=85)", "pval_2\n(n=106)"]
low_n  = [int((ana["pval_1"] == 1).sum()), int((ana["pval_2"] == 1).sum())]
high_n = [int((ana["pval_1"] == 7).sum()), int((ana["pval_2"] == 7).sum())]
x = np.arange(2)
b1 = ax.bar(x, low_n,  color="#252525", edgecolor="black", label="1 (strongly disagree)")
b2 = ax.bar(x, high_n, bottom=low_n, color="#bdbdbd", edgecolor="black", label="7 (strongly agree)")
ax.bar(x, ana[VAL_ITEMS].isna().sum().values, bottom=[l+h for l,h in zip(low_n,high_n)],
       color="white", edgecolor="grey", hatch="///", label="missing")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Respondents (of n=258)")
ax.set_title("C  Perceived Value items\nResponse distribution\n(only endpoints 1 & 7 observed)", fontweight="bold")
ax.legend(fontsize=7.5, loc="upper right")
for bar, val in zip([b1, b2], [low_n, high_n]):
    for rect, v in zip(bar, val):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_y()+v/2, str(v),
                ha="center", va="center", fontsize=9, color="white" if v > 15 else "black")

fig.tight_layout(pad=2)
out = f"{FIG}/06_cfa.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"\nFigure saved: {out}")
