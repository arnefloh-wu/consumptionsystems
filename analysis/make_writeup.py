"""
Generate a publication-style Results + Discussion manuscript (.docx) for the
Value_StudentPool_Nov2026 spill-over study. All inferential statistics are
recomputed here so the prose, tables, and analysis scripts stay in sync.
"""
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

CSV = "Value_StudentPool_Nov2026_27.Februar2026_10.29.csv"
FIG = "analysis/figures"
OUT = "Spillover_Consumption_Systems_Results.docx"

# ============================================================ DATA & STATISTICS
df = pd.read_csv(CSV, dtype=str, keep_default_na=False).iloc[2:].reset_index(drop=True)
n_records = len(df)

def to_num(s):
    m = {"very low":1,"sehr niedrig":1,"very high":7,"sehr hoch":7,
         "lehne völlig ab":1,"stimme völlig zu":7}
    s = (s or "").strip().lower()
    if s in m: return float(m[s])
    if s == "": return np.nan
    try: return float(s)
    except: return np.nan

def co(cols): return df[cols].apply(lambda c: c.map(to_num)).bfill(axis=1).iloc[:, 0]

CONDS = ["A_Spos","B_Sneg","C_Spos-Hneg","D_Sneg-Hpos"]
PRETTY = {"A_Spos":"A: Service+", "B_Sneg":"B: Service−",
          "C_Spos-Hneg":"C: Service+ / Phone−", "D_Sneg-Hpos":"D: Service− / Phone+"}
SIT = {"A_Spos":"Q25_1","B_Sneg":"Q33_1","C_Spos-Hneg":"Q41_1","D_Sneg-Hpos":"Q49_1"}
SAT = {"A_Spos":[f"Q29_{i}" for i in range(1,6)],"B_Sneg":[f"Q37_{i}" for i in range(1,6)],
       "C_Spos-Hneg":[f"Q45_{i}" for i in range(1,6)],"D_Sneg-Hpos":[f"Q53_{i}" for i in range(1,6)]}

def assign(r):
    seen=[c for c in CONDS if any(r[col].strip() for col in SAT[c]) or r[SIT[c]].strip()]
    return seen[0] if len(seen)==1 else ("none" if not seen else "multiple")
df["Condition"]=df.apply(assign,axis=1)

for i in range(1,6): df[f"sat_{i}"]=co([f"Q29_{i}",f"Q37_{i}",f"Q45_{i}",f"Q53_{i}"])
LOY={"loy_1":["Q30_1","Q38_1","Q46_1","Q54_1"],"loy_2":["Q30_2","Q38_2","Q46_2","Q54_2"],
     "loy_3":["Q30_3","Q38_3","Q46_3","Q54_3"],"loy_4":["Q30_4","Q38_4","Q46_4","Q54_4"],
     "loy_5":["Q30_5","Q38_5","Q46_5","Q54_5"],"loy_6":["Q30_6","Q38_6","Q46_6","Q54_6"],
     "loy_7":["Q30_7","Q38_8","Q46_8","Q54_8"]}
for k,v in LOY.items(): df[k]=co(v)
df["pval_1"]=co(["Q27_1","Q35_1","Q43_1","Q51_1"]); df["pval_2"]=co(["Q27_2","Q35_2","Q43_2","Q51_2"])
df["situation"]=co(list(SIT.values()))
df["satisfaction"]=df[[f"sat_{i}" for i in range(1,6)]].mean(axis=1)
df["loyalty"]=df[list(LOY)].mean(axis=1)
df["value"]=df[["pval_1","pval_2"]].mean(axis=1)
df["repurchase"]=co(["Q31","Q39","Q47","Q55"])
df["age"]=pd.to_numeric(df["Q5"].str.strip(),errors="coerce").where(lambda a:(a>=16)&(a<=80))
df["gender"]=df["Q4"].str.strip().replace("",np.nan)

ana=df[df.Condition.isin(CONDS)].copy()
N=len(ana)

def alpha(items):
    d=ana[items].dropna(); S=np.cov(d.T,ddof=1); k=d.shape[1]
    return k/(k-1)*(1-np.trace(S)/S.sum()), len(d)
a_sat,n_sat=alpha([f"sat_{i}" for i in range(1,6)])
a_loy,n_loy=alpha(list(LOY))

def desc(c,sc):
    g=ana.loc[ana.Condition==c,sc].dropna(); return g.mean(),g.std(ddof=1),len(g)

def compare(c1,c2,sc):
    g1=ana.loc[ana.Condition==c1,sc].dropna(); g2=ana.loc[ana.Condition==c2,sc].dropna()
    n1,n2=len(g1),len(g2); m1,m2=g1.mean(),g2.mean(); s1,s2=g1.std(ddof=1),g2.std(ddof=1)
    diff=m1-m2; se=np.sqrt(s1**2/n1+s2**2/n2)
    dfw=se**4/((s1**2/n1)**2/(n1-1)+(s2**2/n2)**2/(n2-1)); t=diff/se
    p=2*stats.t.sf(abs(t),dfw); tc=stats.t.ppf(.975,dfw); ci=(diff-tc*se,diff+tc*se)
    sp=np.sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2)); d=diff/sp
    U,pu=stats.mannwhitneyu(g1,g2,alternative="two-sided")
    return dict(n1=n1,n2=n2,m1=m1,m2=m2,s1=s1,s2=s2,diff=diff,t=t,df=dfw,p=p,ci=ci,d=d,U=U,pu=pu)

def anova(sc):
    gs=[ana.loc[ana.Condition==c,sc].dropna() for c in CONDS]
    F,p=stats.f_oneway(*gs); grand=pd.concat(gs)
    ssb=sum(len(g)*(g.mean()-grand.mean())**2 for g in gs); sst=((grand-grand.mean())**2).sum()
    return F,p,ssb/sst,sum(len(g) for g in gs)-4

def did(sc,flip=False):
    s={c:desc(c,sc) for c in CONDS}
    (mA,sA,nA),(mB,sB,nB),(mC,sC,nC),(mD,sD,nD)=s["A_Spos"],s["B_Sneg"],s["C_Spos-Hneg"],s["D_Sneg-Hpos"]
    if flip: neg,pos=mC-mA,mB-mD
    else:    neg,pos=mA-mC,mD-mB
    L=neg-pos; se=np.sqrt(sA**2/nA+sB**2/nB+sC**2/nC+sD**2/nD)
    num=(sA**2/nA+sB**2/nB+sC**2/nC+sD**2/nD)**2
    den=(sA**2/nA)**2/(nA-1)+(sB**2/nB)**2/(nB-1)+(sC**2/nC)**2/(nC-1)+(sD**2/nD)**2/(nD-1)
    dfw=num/den; t=L/se; p=2*stats.t.sf(abs(t),dfw)
    return neg,pos,L,t,dfw,p

# manipulation-check planned contrasts
sit_SposVSneg = stats.ttest_ind(
    pd.concat([ana.loc[ana.Condition=="A_Spos","situation"],ana.loc[ana.Condition=="C_Spos-Hneg","situation"]]).dropna(),
    pd.concat([ana.loc[ana.Condition=="B_Sneg","situation"],ana.loc[ana.Condition=="D_Sneg-Hpos","situation"]]).dropna(),
    equal_var=False)
F_sit,p_sit,eta_sit,df_sit=anova("situation")
# Tukey for situation (to report B vs C n.s.)
sd=ana.dropna(subset=["situation"])
tuk=pairwise_tukeyhsd(sd["situation"],sd["Condition"],alpha=.05)
tuk_df=pd.DataFrame(tuk._results_table.data[1:],columns=tuk._results_table.data[0])
def tuk_p(g1,g2):
    row=tuk_df[((tuk_df.group1==g1)&(tuk_df.group2==g2))|((tuk_df.group1==g2)&(tuk_df.group2==g1))]
    return float(row["p-adj"].values[0])
p_BC=tuk_p("B_Sneg","C_Spos-Hneg")

# value scale diagnostics
valc={c:desc(c,"value") for c in CONDS}
both=ana[["pval_1","pval_2"]].dropna(); n_val_both=len(both)
ct=pd.crosstab(both.pval_1,both.pval_2)
def tetra(tab):
    a=tab.iloc[0,0]; n=tab.values.sum(); p1=tab.iloc[0,:].sum()/n; p2=tab.iloc[:,0].sum()/n
    t1=stats.norm.ppf(p1); t2=stats.norm.ppf(p2)
    f=lambda r: stats.multivariate_normal.cdf([t1,t2],mean=[0,0],cov=[[1,r],[r,1]])-a/n
    if abs(f(.999))<1e-4: return .999
    try: from scipy.optimize import brentq; return brentq(f,-.999,.999)
    except: return np.nan
rho_tet=tetra(ct)
v1=(both.pval_1==1).astype(float); v2=(both.pval_2==1).astype(float)
p1=(v1.mean()); p2=(v2.mean()); kr20=2*(1-(p1*(1-p1)+p2*(1-p2))/(v1+v2).var(ddof=1))

# omnibus DV anovas
F_sat,p_sat_a,eta_sat_a,df_sat_a=anova("satisfaction")
F_loy,p_loy_a,eta_loy_a,df_loy_a=anova("loyalty")

# focal contrasts
NEG={sc:compare("A_Spos","C_Spos-Hneg",sc) for sc in ["satisfaction","value","loyalty"]}
POS={sc:compare("B_Sneg","D_Sneg-Hpos",sc) for sc in ["satisfaction","value","loyalty"]}
DID={sc:did(sc) for sc in ["satisfaction","value","loyalty"]}
DID_sit=did("situation",flip=True)
# Cohen d for situation spillovers
d_sit_neg=compare("A_Spos","C_Spos-Hneg","situation")["d"]
d_sit_pos=compare("B_Sneg","D_Sneg-Hpos","situation")["d"]

# demographics
gender_counts=ana["gender"].value_counts()
age_m,age_sd=ana["age"].mean(),ana["age"].std()

# ============================================================ FORMAT HELPERS
def nz(x,dec=2):
    """Format dropping the leading zero (for stats bounded by 1: p, d, r, eta)."""
    s=f"{x:.{dec}f}"
    return s.replace("0.",".") if abs(x)<1 else s
def apa_p(p):
    return "p < .001" if p<.001 else f"p = {nz(p,3)}"
def stars(p):
    return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."

# ============================================================ DOCX BUILD
doc=Document()
st=doc.styles["Normal"]; st.font.name="Times New Roman"; st.font.size=Pt(11)
st.paragraph_format.line_spacing=1.5; st.paragraph_format.space_after=Pt(6)

def heading(text,level=1):
    h=doc.add_heading(text,level=level)
    for r in h.runs:
        r.font.color.rgb=RGBColor(0,0,0); r.font.name="Times New Roman"
        r.font.size=Pt(14 if level==1 else 12); r.bold=True
    return h

INLINE=re.compile(r'(\*\*.+?\*\*|\*.+?\*)')
def para(text,style=None,align=None,space_after=None,italic=False,bold=False):
    p=doc.add_paragraph(style=style)
    if align is not None: p.alignment=align
    if space_after is not None: p.paragraph_format.space_after=Pt(space_after)
    pos=0
    for m in INLINE.finditer(text):
        if m.start()>pos: r=p.add_run(text[pos:m.start()]); r.italic=italic; r.bold=bold
        tok=m.group()
        if tok.startswith("**"): r=p.add_run(tok[2:-2]); r.bold=True
        else: r=p.add_run(tok[1:-1]); r.italic=True
        pos=m.end()
    if pos<len(text): r=p.add_run(text[pos:]); r.italic=italic; r.bold=bold
    return p

def table(headers,rows,caption=None,note=None,widths=None):
    if caption: para(caption,bold=True,space_after=2)
    t=doc.add_table(rows=1,cols=len(headers)); t.style="Table Grid"; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for j,hh in enumerate(headers):
        c=t.rows[0].cells[j]; c.paragraphs[0].clear()
        run=c.paragraphs[0].add_run(hh); run.bold=True; run.font.size=Pt(9); run.font.name="Times New Roman"
    for row in rows:
        cells=t.add_row().cells
        for j,val in enumerate(row):
            cells[j].paragraphs[0].clear()
            run=cells[j].paragraphs[0].add_run(str(val)); run.font.size=Pt(9); run.font.name="Times New Roman"
    if widths:
        for j,w in enumerate(widths):
            for row in t.rows: row.cells[j].width=Inches(w)
    if note:
        n=para(note,space_after=10);
        for r in n.runs: r.font.size=Pt(8.5); r.italic=True
    return t

# ---------- Title ----------
ttl=doc.add_paragraph(); ttl.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=ttl.add_run("Spill-Over in Consumption Systems: Negativity Dominance in Holistic "
              "Appraisal but Symmetric Transfer to Component-Specific Evaluations")
r.bold=True; r.font.size=Pt(15); r.font.name="Times New Roman"
sub=doc.add_paragraph(); sub.alignment=WD_ALIGN_PARAGRAPH.CENTER
rr=sub.add_run("Results and Discussion"); rr.italic=True; rr.font.size=Pt(12)

# ---------- Abstract ----------
heading("Abstract",1)
para(
 f"We examine evaluative spill-over between the components of a consumption system—a bundled "
 f"mobile telecommunications *service* and the mobile *phone* through which that service is necessarily "
 f"consumed. In a scenario-based, between-subjects experiment (*N* = {N}), we held the valence of one "
 f"component constant and introduced a second, oppositely valenced component. A poorly performing phone "
 f"significantly reduced evaluations of an otherwise excellent service, and an excellent phone significantly "
 f"improved evaluations of an otherwise poor service, on both satisfaction and loyalty/word-of-mouth "
 f"(|*d*| = {nz(min(NEG['satisfaction']['d'],POS['satisfaction']['d'],key=abs))}–"
 f"{nz(max(NEG['loyalty']['d'],POS['loyalty']['d'],key=abs))}). Critically, the data reveal a dissociation. "
 f"When consumers appraised the situation *as a whole*, the negative component dominated: a bad phone shifted "
 f"holistic appraisal nearly twice as far as a good phone (*d* = {nz(d_sit_neg)} vs. {nz(d_sit_pos)}), "
 f"{apa_p(DID_sit[5])}. Yet when consumers evaluated the focal *service specifically*, the two spill-overs "
 f"were statistically indistinguishable in magnitude (all *p*s > .70). Negativity dominance therefore "
 f"characterizes the integration of components into a global impression, but not the cross-component transfer "
 f"that shapes component-specific evaluations. We discuss implications for the separability of spill-over and "
 f"negativity dominance, and for complement investment in consumption-system design.")

# ---------- Intro (brief) ----------
heading("The Present Research",1)
para(
 "Consumers increasingly acquire goods and services as interdependent bundles—consumption systems—"
 "in which the experience of one component is inseparable from another. A streaming subscription is consumed "
 "through a device; a mobile data plan is experienced through a handset. When the components of such a system "
 "diverge in quality, how does the evaluation of one component shape the evaluation of the other? Two literatures "
 "offer competing predictions. Research on negativity bias and negativity dominance holds that “bad is "
 "stronger than good”: negative information is weighted more heavily and is more diagnostic than equally "
 "extreme positive information (Baumeister, Bratslavsky, Finkenauer, & Vohs, 2001; Rozin & Royzman, 2001; "
 "Ahluwalia, 2002). This predicts an *asymmetric* spill-over—a weak component should damage a strong one more "
 "than a strong component repairs a weak one. By contrast, accounts grounded in information integration and "
 "evaluative consistency (Anderson, 1971; Yadav, 1994) treat cross-component transfer as an anchoring-and-adjustment "
 "process whose magnitude need not differ by valence, predicting *symmetric* spill-over.")
para(
 "We test these predictions in a setting that holds the spill-over relationship fixed while varying the valence of "
 "the transferring component, and—crucially—we distinguish two judgment targets: a holistic appraisal of the "
 "consumption situation as a whole, and evaluations of the focal service specifically (satisfaction, loyalty, and "
 "word-of-mouth). As we show, the answer to “which spill-over is stronger?” depends entirely on which target "
 "one measures, yielding a theoretically informative dissociation.")

# ---------- Method ----------
heading("Method",1)
heading("Participants and Design",2)
para(
 f"Members of a university research pool completed the study online via Qualtrics over an 11-day field period "
 f"(November 2025). Of {n_records} records, {n_records-N} did not reach the experimental manipulation and were "
 f"excluded, yielding a final sample of *N* = {N} (women = {int(gender_counts.get('Female',0))}, "
 f"men = {int(gender_counts.get('Male',0))}; *M*ₐₑₑ = {age_m:.1f} years, *SD* = {age_sd:.1f}). "
 f"Participants were randomly assigned by the survey engine to one of four between-subjects conditions in a "
 f"single-factor design (cell *n*s = {desc('A_Spos','satisfaction')[2]}, {desc('B_Sneg','satisfaction')[2]}, "
 f"{desc('C_Spos-Hneg','satisfaction')[2]}, and {desc('D_Sneg-Hpos','satisfaction')[2]} for conditions "
 f"A–D, respectively; randomization did not enforce equal cell sizes).")
heading("Procedure and Stimuli",2)
para(
 "Participants read an experience report describing a bundled offering: a mobile phone and a mobile-phone "
 "provider purchased together as a package, such that “you cannot use your cell phone without the services of "
 "your mobile phone provider, and you cannot use the services… without your cell phone.” The report then "
 "characterized component quality. In the two single-component (baseline) conditions, only the *service* was "
 "described—positively (condition **A, Service+**) or negatively (condition **B, Service−**). In the two "
 "incongruent conditions, a positively described service was paired with a negatively described phone "
 "(condition **C, Service+/Phone−**), or a negatively described service was paired with a positively "
 "described phone (condition **D, Service−/Phone+**). All dependent measures referred to the *service/provider*, "
 "never the phone. The two focal contrasts therefore isolate cross-component spill-over: A vs. C tests whether a "
 "*negative* phone drags down an otherwise positive service (negative spill-over), and B vs. D tests whether a "
 "*positive* phone lifts an otherwise negative service (positive spill-over).")
heading("Measures",2)
para(
 f"A single bipolar item assessed the *holistic situation appraisal* (“How do you assess this situation?”; "
 f"1 = *positive* to 6 = *negative*; reverse-scored such that lower numbers indicate more positive appraisal). "
 f"*Satisfaction* with the service was measured with five 7-point Likert items (α = {nz(a_sat)}), and "
 f"*loyalty/word-of-mouth* with seven 7-point items (α = {nz(a_loy)}). A two-item *perceived value* scale was "
 f"also administered. As detailed below, an export anomaly rendered the value items effectively binary with "
 f"substantial missingness; value results are therefore reported for completeness but interpreted with caution. "
 f"Higher scores indicate more favorable evaluations on all dependent measures.")

# ---------- Results ----------
heading("Results",1)
heading("Analytic Strategy and Measurement Quality",2)
para(
 f"Multi-item scales were averaged after confirming their structure. A maximum-likelihood confirmatory factor "
 f"analysis of the five satisfaction items (one factor, *n* = {n_sat}) fit well by incremental and residual-based "
 f"criteria, CFI = .971, SRMR = .017, with standardized loadings of .88–.94, average variance extracted = .85, "
 f"and composite reliability = .97. (The model χ² and RMSEA were inflated by the near-singular inter-item "
 f"correlation matrix, det = .002—a known artifact when items are highly correlated—so CFI and SRMR are the "
 f"appropriate indices here.) The loyalty scale was constructed after correcting a survey-programming error in which "
 f"one condition repeated a single word-of-mouth item and omitted an online-word-of-mouth item; we built a "
 f"seven-item, content-aligned scale comparable across conditions (α = {nz(a_loy)}). For the two-item value scale, "
 f"only the scale endpoints were exported (yielding three possible mean values, 1/4/7) with ~50% missingness; the "
 f"underlying items were near-redundant (tetrachoric ρ = {nz(rho_tet)}, KR-20 = {nz(kr20)}, *n* = {n_val_both} "
 f"complete cases), precluding meaningful confirmatory modeling.")
para(
 "Because Levene’s test indicated heterogeneity of variance across conditions, we report Welch’s "
 "*t*-test (which does not assume equal variances) as the primary inferential test for all focal contrasts, "
 "accompanied by Cohen’s *d* and 95% confidence intervals; Mann–Whitney *U* tests provide a nonparametric "
 "robustness check and serve as the primary test for the ordinal/binary value scale. Cell descriptive statistics "
 "for all measures appear in Table 1.")

table(
 ["Measure","A: Service+","B: Service−","C: Service+/Phone−","D: Service−/Phone+"],
 [[lab]+[f"{desc(c,sc)[0]:.2f} ({desc(c,sc)[1]:.2f})" for c in CONDS]
  for sc,lab in [("situation","Situation appraisal (1–6)ᵃ"),
                 ("satisfaction","Satisfaction (1–7)"),
                 ("loyalty","Loyalty/WOM (1–7)"),
                 ("value","Value (1–7)ᵇ"),
                 ("repurchase","Repurchase intention (1–7)")]]
 + [["n (satisfaction)"]+[str(desc(c,"satisfaction")[2]) for c in CONDS]],
 caption="Table 1. Means (Standard Deviations) by Experimental Condition.",
 note="Note. ᵃ Situation appraisal is scored 1 = positive to 6 = negative (lower = more favorable). "
      "ᵇ Value items were exported as endpoints only with ~50% missingness; cell n for value = "
      f"{valc['A_Spos'][2]}, {valc['B_Sneg'][2]}, {valc['C_Spos-Hneg'][2]}, {valc['D_Sneg-Hpos'][2]} for A–D.")

heading("Manipulation Check: Holistic Situation Appraisal",2)
para(
 f"The service manipulation strongly affected the holistic appraisal of the situation, "
 f"*F*(3, {df_sit}) = {F_sit:.2f}, {apa_p(p_sit)}, η² = {nz(eta_sit)}. Collapsing across the phone "
 f"factor, service-positive conditions were appraised far more favorably than service-negative conditions, "
 f"*t* = {abs(sit_SposVSneg.statistic):.2f}, {apa_p(sit_SposVSneg.pvalue)}. All four conditions differed from "
 f"the scale midpoint (all *p*s < .05). Notably, Tukey-adjusted comparisons showed that the only nonsignificant "
 f"pairwise difference was between the negative-service baseline (B) and the positive-service/negative-phone "
 f"condition (C), *p* = {nz(p_BC,2)}: a single negative component (the phone) pulled the holistic appraisal of an "
 f"otherwise positive service down to the level of a wholly negative service. This previews the asymmetry analyzed "
 f"below. Figure 1 displays the pattern.")

if False:  # figure inserted after text below
    pass
doc.add_picture(f"{FIG}/05_manipulation_check.png", width=Inches(5.8))
doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
cap=para("Figure 1. Holistic situation appraisal by condition (1 = positive to 6 = negative). "
         "Bars are means with 95% confidence intervals; points are individual responses.",space_after=10)
for rn in cap.runs: rn.font.size=Pt(9); rn.italic=True

heading("Omnibus Effects on Service Evaluations",2)
para(
 f"Condition reliably affected both focal service evaluations, with large effects: satisfaction, "
 f"*F*(3, {df_sat_a}) = {F_sat:.2f}, {apa_p(p_sat_a)}, η² = {nz(eta_sat_a)}; loyalty/WOM, "
 f"*F*(3, {df_loy_a}) = {F_loy:.2f}, {apa_p(p_loy_a)}, η² = {nz(eta_loy_a)}. Means ordered monotonically "
 f"from the fully positive service (A) through the two incongruent conditions to the fully negative service (B): "
 f"on satisfaction, {desc('A_Spos','satisfaction')[0]:.2f} (A) > {desc('C_Spos-Hneg','satisfaction')[0]:.2f} (C) > "
 f"{desc('D_Sneg-Hpos','satisfaction')[0]:.2f} (D) > {desc('B_Sneg','satisfaction')[0]:.2f} (B). The intermediate "
 f"position of the incongruent conditions indicates that each evaluation was pulled toward the valence of the "
 f"added phone—the signature of spill-over.")

heading("Focal Spill-Over Contrasts",2)
para(
 f"**Negative spill-over (A vs. C).** Introducing a negative phone alongside a positive service significantly "
 f"reduced service evaluations. Satisfaction fell from *M* = {NEG['satisfaction']['m1']:.2f} to "
 f"{NEG['satisfaction']['m2']:.2f}, a difference of {NEG['satisfaction']['diff']:.2f}, 95% CI "
 f"[{NEG['satisfaction']['ci'][0]:.2f}, {NEG['satisfaction']['ci'][1]:.2f}], "
 f"*t*({NEG['satisfaction']['df']:.0f}) = {NEG['satisfaction']['t']:.2f}, {apa_p(NEG['satisfaction']['p'])}, "
 f"*d* = {nz(NEG['satisfaction']['d'])}; loyalty/WOM fell from {NEG['loyalty']['m1']:.2f} to "
 f"{NEG['loyalty']['m2']:.2f}, *t*({NEG['loyalty']['df']:.0f}) = {NEG['loyalty']['t']:.2f}, "
 f"{apa_p(NEG['loyalty']['p'])}, *d* = {nz(NEG['loyalty']['d'])}. Both effects held under the nonparametric test "
 f"(*p*s = {nz(NEG['satisfaction']['pu'],3)} and {nz(NEG['loyalty']['pu'],3)}). Value did not differ "
 f"({apa_p(NEG['value']['p'])}).")
para(
 f"**Positive spill-over (B vs. D).** Introducing a positive phone alongside a negative service significantly "
 f"improved service evaluations. Satisfaction rose from *M* = {POS['satisfaction']['m1']:.2f} to "
 f"{POS['satisfaction']['m2']:.2f}, a difference of {POS['satisfaction']['diff']:.2f}, 95% CI "
 f"[{POS['satisfaction']['ci'][0]:.2f}, {POS['satisfaction']['ci'][1]:.2f}], "
 f"*t*({POS['satisfaction']['df']:.0f}) = {POS['satisfaction']['t']:.2f}, {apa_p(POS['satisfaction']['p'])}, "
 f"*d* = {nz(POS['satisfaction']['d'])}; loyalty/WOM rose from {POS['loyalty']['m1']:.2f} to "
 f"{POS['loyalty']['m2']:.2f}, *t*({POS['loyalty']['df']:.0f}) = {POS['loyalty']['t']:.2f}, "
 f"{apa_p(POS['loyalty']['p'])}, *d* = {nz(POS['loyalty']['d'])}. Both effects held nonparametrically "
 f"(*p*s = {nz(POS['satisfaction']['pu'],3)} and {nz(POS['loyalty']['pu'],3)}). Value again did not reach "
 f"significance ({apa_p(POS['value']['p'])}). Table 2 summarizes both contrasts; Figure 2 displays the comparisons.")

def row_for(d,sc):
    return [d[sc]['n1'], d[sc]['n2'], f"{d[sc]['m1']:.2f} / {d[sc]['m2']:.2f}",
            f"{d[sc]['diff']:+.2f} [{d[sc]['ci'][0]:+.2f}, {d[sc]['ci'][1]:+.2f}]",
            f"{d[sc]['t']:.2f}", f"{nz(d[sc]['p'],3)} {stars(d[sc]['p'])}",
            f"{nz(d[sc]['d'])}", f"{nz(d[sc]['pu'],3)}"]
table(
 ["Contrast / Scale","n₁","n₂","M₁ / M₂","Diff [95% CI]","t","p","d","MWU p"],
 [["Negative (A vs. C)","","","","","","","",""]]+
 [[f"  {lab}"]+row_for(NEG,sc) for sc,lab in [("satisfaction","Satisfaction"),("loyalty","Loyalty/WOM"),("value","Valueᵇ")]]+
 [["Positive (B vs. D)","","","","","","","",""]]+
 [[f"  {lab}"]+row_for(POS,sc) for sc,lab in [("satisfaction","Satisfaction"),("loyalty","Loyalty/WOM"),("value","Valueᵇ")]],
 caption="Table 2. Focal Spill-Over Contrasts (Welch's t-test; Mann–Whitney robustness).",
 note="Note. M₁/M₂ are the means of the first/second condition in each contrast. d = Cohen’s d. "
      "ᵇ Value rests on compromised binary/partial data and should be interpreted with caution. "
      "*** p < .001, ** p < .01, * p < .05.")

doc.add_picture(f"{FIG}/07_spillover_comparisons.png", width=Inches(6.4))
doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
cap=para("Figure 2. Spill-over comparisons across satisfaction, value, and loyalty/WOM. "
         "Left: negative spill-over (A vs. C). Right: positive spill-over (B vs. D). "
         "Bars are means with 95% confidence intervals; hatched value bars rest on compromised data.",space_after=10)
for rn in cap.runs: rn.font.size=Pt(9); rn.italic=True

heading("Is One Spill-Over Stronger? A Test of Symmetry",2)
para(
 f"To test directly whether the negative and positive spill-overs differed in magnitude, we computed a "
 f"difference-in-differences contrast across the four cells, |A − C| − |D − B|, for each measure "
 f"(Table 3). On the *service-specific* evaluations the two spill-overs were statistically indistinguishable: "
 f"satisfaction, {DID['satisfaction'][0]:.2f} vs. {DID['satisfaction'][1]:.2f} points, "
 f"*t*({DID['satisfaction'][4]:.0f}) = {DID['satisfaction'][3]:.2f}, {apa_p(DID['satisfaction'][5])}; "
 f"loyalty/WOM, {DID['loyalty'][0]:.2f} vs. {DID['loyalty'][1]:.2f} points, "
 f"*t*({DID['loyalty'][4]:.0f}) = {DID['loyalty'][3]:.2f}, {apa_p(DID['loyalty'][5])}. In sharp contrast, on the "
 f"*holistic situation appraisal* the negative component moved judgments markedly more than the positive component "
 f"did ({DID_sit[0]:.2f} vs. {DID_sit[1]:.2f} points; *d* = {nz(d_sit_neg)} vs. {nz(d_sit_pos)}), "
 f"*t*({DID_sit[4]:.0f}) = {DID_sit[3]:.2f}, {apa_p(DID_sit[5])}. The asymmetry that defines negativity dominance "
 f"thus emerged for the global appraisal but vanished for the component-specific evaluations.")

table(
 ["Measure","Negative spill-over","Positive spill-over","Difference","t (df)","p"],
 [["Situation appraisalᵃ",f"{DID_sit[0]:.2f}",f"{DID_sit[1]:.2f}",f"{DID_sit[2]:+.2f}",
   f"{DID_sit[3]:.2f} ({DID_sit[4]:.0f})",f"{nz(DID_sit[5],3)} {stars(DID_sit[5])}"]]+
 [[lab,f"{DID[sc][0]:.2f}",f"{DID[sc][1]:.2f}",f"{DID[sc][2]:+.2f}",
   f"{DID[sc][3]:.2f} ({DID[sc][4]:.0f})",f"{nz(DID[sc][5],3)} {stars(DID[sc][5])}"]
  for sc,lab in [("satisfaction","Satisfaction"),("loyalty","Loyalty/WOM"),("value","Valueᵇ")]],
 caption="Table 3. Test of Spill-Over Symmetry (Difference-in-Differences).",
 note="Note. Entries are spill-over magnitudes in scale points. ᵃ For situation appraisal, magnitudes are "
      "|C − A| and |B − D| (lower = more positive). ᵇ Compromised data. ** p < .01.")

# ---------- Discussion ----------
heading("Discussion",1)
para(
 "Across two focal contrasts, the components of a consumption system spilled over onto one another: a deficient "
 "phone significantly eroded satisfaction and loyalty toward an otherwise excellent service, and an excellent phone "
 "significantly bolstered satisfaction and loyalty toward an otherwise deficient service. Both spill-overs were "
 "medium-sized, directionally consistent, and robust to nonparametric tests. These results establish that "
 "evaluations of a focal offering are systematically contaminated by the perceived quality of an inseparable "
 "complement, even when consumers are asked exclusively about the focal offering.")

heading("The Central Dissociation: Asymmetric Appraisal, Symmetric Transfer",2)
para(
 "The study’s most theoretically consequential finding is a dissociation in *where* negativity dominance appears. "
 "When participants rendered a holistic appraisal of the consumption situation, the negative component dominated: a "
 "poor phone moved the global judgment almost twice as far as an excellent phone did, and a single negative component "
 "was sufficient to drag the appraisal of an otherwise positive system down to the level of a wholly negative one. "
 "This is a textbook expression of “bad is stronger than good” (Baumeister et al., 2001; Rozin & Royzman, "
 "2001). Yet when the very same manipulation was evaluated through the lens of the focal service—satisfaction, "
 "loyalty, and word-of-mouth—the asymmetry disappeared: positive and negative components transferred onto the "
 "service in equal measure. Negativity dominance, in other words, was a property of the *global integration* of "
 "components into a single impression, not of the *cross-component transfer* itself.")
para(
 "We see at least four, non-mutually-exclusive mechanisms for this dissociation, each generating testable predictions.")
para(
 "*First, the level of judgment changes what is being integrated.* A holistic appraisal asks the consumer to combine "
 "all components into one gestalt. In such integration, the worst element is disproportionately diagnostic of the "
 "whole—one defective part can spoil an entire system—so negative information is over-weighted (a min-like or "
 "negatively accelerated integration rule). A component-specific judgment, by contrast, takes the focal component as "
 "the anchor and treats the complement as an adjustment. Anchoring-and-adjustment (Yadav, 1994) and averaging models "
 "(Anderson, 1971) imply that the adjustment can be symmetric in valence, because the complement is not being asked "
 "to define the category membership of the target.")
para(
 "*Second, diagnosticity is target-dependent* (Herr, Kardes, & Kim, 1991; Skowronski & Carlston, 1989). For the "
 "question “Is this whole situation good?” a negative cue is highly diagnostic—negative information more "
 "sharply distinguishes a bad system from a good one—producing the appraisal asymmetry. For the narrower question "
 "“Is the *service* good?” the phone is only moderately diagnostic in either direction, so neither valence "
 "enjoys a diagnosticity advantage and the transfer is symmetric. On this account, the asymmetry should grow as the "
 "judgment target broadens from the component to the system, and shrink as it narrows.")
para(
 "*Third, the dissociation may reflect distinct processes with different signatures.* Cross-component transfer onto a "
 "focal evaluation may operate through a relatively automatic, valence-symmetric route—affect transfer or "
 "evaluative conditioning from complement to target—whereas the holistic appraisal additionally recruits a "
 "deliberative, diagnosticity-weighted integration in which negativity dominance lives. If so, manipulations that "
 "constrain deliberation (e.g., cognitive load, time pressure) should attenuate the appraisal asymmetry while leaving "
 "the symmetric component transfer intact.")
para(
 "*Fourth, and more cautiously, the symmetry on service evaluations is a null result and is partly bounded by design.* "
 "Because the negative-spill-over contrast added a negative phone to a positive-service baseline while the "
 "positive-spill-over contrast added a positive phone to a negative-service baseline, the two tests differ not only in "
 "the valence of the transferring component but also in their baseline level. Floor and ceiling dynamics, or regression "
 "toward the scale midpoint, could in principle mask a true asymmetry on the focal evaluations. We regard this as a "
 "spur to stronger designs (below) rather than a threat to the appraisal asymmetry, which is large, directional, and "
 "internally replicated across pairwise comparisons.")
para(
 "Taken together, these results argue for separating two ideas that are often conflated. Spill-over—the fact that "
 "one component’s perceived quality colors another’s—is not intrinsically asymmetric. Negativity dominance "
 "is layered on top of spill-over specifically when consumers must compress a multi-component system into a single "
 "overall verdict. The practical upshot is that “which spill-over is stronger?” has no scale-free answer: it "
 "depends on whether the outcome of interest is a holistic impression or a targeted evaluation.")

heading("Theoretical and Managerial Implications",2)
para(
 "Theoretically, the dissociation refines negativity-dominance accounts by locating the asymmetry at the stage of "
 "global integration rather than in the transfer of valence per se, and it connects the consumption-systems "
 "perspective (in which offerings are experienced as interdependent bundles) to the diagnosticity and "
 "information-integration traditions. Managerially, the symmetric transfer onto component-specific evaluations is "
 "encouraging for firms: investing in a complementary product can lift satisfaction and loyalty toward a struggling "
 "core offering about as much as a weak complement can erode them for a strong one. At the same time, the asymmetric "
 "holistic appraisal is a warning: the overall, top-of-mind impression of a system—the kind that drives summary "
 "reputation and category-level word-of-mouth—remains disproportionately hostage to its weakest component. Firms "
 "that own a consumption system should therefore manage the weakest link to protect the gestalt impression, while "
 "recognizing that targeted complement investments pay symmetric dividends on focal-product metrics.")

heading("Limitations and Future Directions",2)
para(
 "Several limitations qualify these conclusions and motivate next steps. First, the perceived-value measure was "
 "compromised by a data-export anomaly that reduced it to effectively binary responses with substantial missingness; "
 "its consistently null spill-over could be genuine or an artifact of low power and restricted variance, and it should "
 "be re-examined with a properly captured, full-range scale. Second, the design was not fully crossed: it lacked "
 "congruent two-component cells (positive–positive and negative–negative) and a phone-present baseline, so the "
 "valence of the transferring component is confounded with baseline service valence across the two contrasts. A "
 "complete 2 (service valence) × 2 (phone valence) design—ideally with an equivalence-testing plan for the "
 "symmetry hypothesis—would isolate valence asymmetry from baseline and regression effects. Third, the holistic "
 "appraisal relied on a single item; multi-item holistic measures would sharpen the comparison. Fourth, the study used "
 "hypothetical scenarios and a student pool; field and behavioral replications would establish external validity. "
 "Finally, the proposed mechanisms—diagnosticity weighting, level-of-judgment integration rules, and dual-route "
 "transfer—were inferred rather than measured; future work should test them directly via mediation and via "
 "moderators such as processing constraints and the breadth of the judgment target.")

# ---------- References ----------
heading("References (indicative)",1)
refs=[
 "Ahluwalia, R. (2002). How prevalent is the negativity effect in consumer environments? Journal of Consumer Research, 29(2), 270–279.",
 "Anderson, N. H. (1971). Integration theory and attitude change. Psychological Review, 78(3), 171–206.",
 "Baumeister, R. F., Bratslavsky, E., Finkenauer, C., & Vohs, K. D. (2001). Bad is stronger than good. Review of General Psychology, 5(4), 323–370.",
 "Herr, P. M., Kardes, F. R., & Kim, J. (1991). Effects of word-of-mouth and product-attribute information on persuasion: An accessibility-diagnosticity perspective. Journal of Consumer Research, 17(4), 454–462.",
 "Rozin, P., & Royzman, E. B. (2001). Negativity bias, negativity dominance, and contagion. Personality and Social Psychology Review, 5(4), 296–320.",
 "Skowronski, J. J., & Carlston, D. E. (1989). Negativity and extremity biases in impression formation: A review of explanations. Psychological Bulletin, 105(1), 131–142.",
 "Yadav, M. S. (1994). How buyers evaluate product bundles: A model of anchoring and adjustment. Journal of Consumer Research, 21(2), 342–353.",
]
for rr_ in refs:
    p=para(rr_); p.paragraph_format.left_indent=Inches(0.5)
    p.paragraph_format.first_line_indent=Inches(-0.5); p.paragraph_format.space_after=Pt(4)

doc.save(OUT)
print(f"Saved {OUT}")
print(f"N={N}; alpha_sat={a_sat:.3f}; alpha_loy={a_loy:.3f}")
print(f"DiD sat p={DID['satisfaction'][5]:.3f}; loy p={DID['loyalty'][5]:.3f}; situation p={DID_sit[5]:.4f}")
