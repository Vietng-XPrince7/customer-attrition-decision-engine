"""
Phase 2 — Financial EDA & Pain-Point Discovery
================================================
Run: python scripts/phase2_eda.py
Outputs:
    reports/figures/01_pareto_revenue_at_risk.png
    reports/figures/02_revenue_compression_dynamics.png
    reports/figures/03_tipping_points_eqpdays_overage.png
    reports/figures/04_correlation_heatmap_top_drivers.png
    + Full Summary Report printed to stdout
"""

import warnings, sys, os
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "matplotlib")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# -- Style ----------------------------------------------------------------------
PALETTE = {"churn": "#E53935", "stay": "#1565C0", "vip": "#F57F17", "mass": "#546E7A"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F9FAFB",
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
})
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300
EPS = 1e-5

# -- Load data -----------------------------------------------------------------
print("=" * 70)
print("  PHASE 2 — FINANCIAL EDA & PAIN-POINT DISCOVERY")
print("=" * 70)
df = pd.read_parquet(ROOT / "data/processed/merged_clean.parquet")
print(f"\n[DATA] Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

# Impute rev columns for analysis only (not for modelling)
df["rev_Mean_f"]  = df["rev_Mean"].fillna(df["rev_Mean"].median())
df["avg3rev_f"]   = df["avg3rev"].fillna(df["avg3rev"].median())
df["avg6rev_f"]   = df["avg6rev"].fillna(df["avg6rev"].median())
df["avgrev_f"]    = df["avgrev"].fillna(df["avgrev"].median())
df["change_rev_f"]= df["change_rev"].fillna(0.0)
df["avg3mou_f"]   = df["avg3mou"].fillna(df["avg3mou"].median())
df["avg6mou_f"]   = df["avg6mou"].fillna(df["avg6mou"].median())
df["eqpdays_f"]   = df["eqpdays"].fillna(df["eqpdays"].median())
df["ovrrev_f"]    = df["ovrrev_Mean"].fillna(0.0)
df["totmrc_f"]    = df["totmrc_Mean"].fillna(df["totmrc_Mean"].median())

# ===============================================================================
# CHART 1 — Pareto: Revenue at Risk (VIP segment)
# ===============================================================================
print("\n" + "-" * 70)
print("  TASK 1 — PARETO: REVENUE AT RISK IN VIP SEGMENT")
print("-" * 70)

# Sort by rev_Mean desc -> cumulative share
df_sorted = df.sort_values("rev_Mean_f", ascending=False).copy()
df_sorted["cum_rev_pct"] = df_sorted["rev_Mean_f"].cumsum() / df_sorted["rev_Mean_f"].sum() * 100
df_sorted["cum_cust_pct"] = (np.arange(1, len(df_sorted)+1)) / len(df_sorted) * 100

# VIP: top 20% by rev_Mean
vip_cutoff = df["rev_Mean_f"].quantile(0.80)
df["vip_flag"] = (df["rev_Mean_f"] >= vip_cutoff).astype(int)

vip   = df[df["vip_flag"] == 1]
mass  = df[df["vip_flag"] == 0]
total_rev = df["rev_Mean_f"].sum()

vip_rev_share = vip["rev_Mean_f"].sum() / total_rev * 100
vip_churn_rate = vip["churn"].mean() * 100
mass_churn_rate = mass["churn"].mean() * 100

# Revenue at risk (monthly): VIP churners' rev_Mean
vip_churners = vip[vip["churn"] == 1]
rev_at_risk_monthly = vip_churners["rev_Mean_f"].sum()
rev_at_risk_annual  = rev_at_risk_monthly * 12

# Quartile analysis
df["rev_quartile"] = pd.qcut(df["rev_Mean_f"], 4, labels=["Q1\n(Bottom)", "Q2", "Q3", "Q4\n(Top)"])
q_stats = df.groupby("rev_quartile", observed=True).agg(
    n=("churn", "count"),
    churn_rate=("churn", "mean"),
    total_rev=("rev_Mean_f", "sum"),
    mean_rev=("rev_Mean_f", "mean"),
).assign(rev_share=lambda x: x["total_rev"] / total_rev * 100)

print(f"  VIP Threshold (80th pct of rev_Mean): ${vip_cutoff:.2f}/month")
print(f"  VIP Count: {len(vip):,} ({len(vip)/len(df)*100:.1f}% of customers)")
print(f"  VIP Revenue Share: {vip_rev_share:.1f}% of total monthly revenue")
print(f"  VIP Churn Rate: {vip_churn_rate:.1f}% vs. Mass: {mass_churn_rate:.1f}%")
print(f"  VIP Churners: {len(vip_churners):,} customers")
print(f"  Monthly Revenue at Risk (VIP churners): ${rev_at_risk_monthly/1e6:.3f}M")
print(f"  Annual Revenue at Risk (VIP churners):  ${rev_at_risk_annual/1e6:.3f}M")
print("\n  Quartile Revenue Distribution:")
print(q_stats[["n","mean_rev","rev_share","churn_rate"]].round(3).to_string())

# Plot
fig = plt.figure(figsize=(16, 7))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# Ax1: Lorenz / Pareto curve
ax1 = fig.add_subplot(gs[0])
ax1.plot(df_sorted["cum_cust_pct"], df_sorted["cum_rev_pct"],
         color="#1565C0", linewidth=2.5, label="Revenue Concentration Curve")
ax1.plot([0,100],[0,100], "k--", linewidth=1, alpha=0.5, label="Perfect Equality")
ax1.axvline(20, color=PALETTE["vip"], linestyle=":", linewidth=2, label="Top 20% Customers")
ax1.axhline(vip_rev_share, color=PALETTE["vip"], linestyle=":", linewidth=2)
ax1.annotate(f"Top 20% -> {vip_rev_share:.0f}%\nof Revenue",
             xy=(20, vip_rev_share), xytext=(30, vip_rev_share-18),
             fontsize=9, color=PALETTE["vip"], fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=PALETTE["vip"], lw=1.5))
ax1.fill_between(df_sorted["cum_cust_pct"], df_sorted["cum_rev_pct"],
                 df_sorted["cum_cust_pct"], alpha=0.08, color="#1565C0")
ax1.set_xlabel("Cumulative % of Customers (sorted by Revenue)")
ax1.set_ylabel("Cumulative % of Total Monthly Revenue")
ax1.set_title("Pareto Revenue Concentration\n(Lorenz Curve)")
ax1.legend(fontsize=8)

# Ax2: Revenue share & Churn rate by Quartile
ax2 = fig.add_subplot(gs[1])
x = np.arange(4)
bars = ax2.bar(x, q_stats["rev_share"], color=["#90CAF9","#42A5F5","#1E88E5","#0D47A1"],
               edgecolor="white", width=0.55, label="Revenue Share (%)")
ax2b = ax2.twinx()
ax2b.plot(x, q_stats["churn_rate"]*100, "o-", color=PALETTE["churn"],
          linewidth=2.5, markersize=8, label="Churn Rate (%)")
for bar, val in zip(bars, q_stats["rev_share"]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{val:.1f}%",
             ha="center", va="bottom", fontsize=8, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(q_stats.index.tolist())
ax2.set_ylabel("Revenue Share (%)"); ax2b.set_ylabel("Churn Rate (%)", color=PALETTE["churn"])
ax2b.tick_params(axis="y", labelcolor=PALETTE["churn"])
ax2.set_title("Revenue Share & Churn Rate\nby Revenue Quartile")
ax2.set_facecolor("#F9FAFB")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper left")

# Ax3: Revenue at Risk decomposition
ax3 = fig.add_subplot(gs[2])
labels_bar = ["VIP\nStayers", "VIP\nChurners\n(Rev@Risk)", "Mass\nChurners"]
rev_vals = [
    vip[vip["churn"]==0]["rev_Mean_f"].sum()/1e6,
    rev_at_risk_monthly/1e6,
    mass[mass["churn"]==1]["rev_Mean_f"].sum()/1e6,
]
colors_bar = [PALETTE["stay"], PALETTE["churn"], "#EF9A9A"]
bh = ax3.bar(labels_bar, rev_vals, color=colors_bar, edgecolor="white", width=0.55)
for bar, val in zip(bh, rev_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"${val:.2f}M",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax3.set_ylabel("Monthly Revenue (USD Millions)")
ax3.set_title(f"Monthly Revenue at Risk\nVIP Churners = ${rev_at_risk_monthly/1e6:.2f}M\n(Annual: ${rev_at_risk_annual/1e6:.2f}M)")
ax3.set_facecolor("#F9FAFB")

fig.suptitle("PAIN POINT #1 — Revenue Concentration & VIP Revenue at Risk",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "01_pareto_revenue_at_risk.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"\n  [SAVED] 01_pareto_revenue_at_risk.png")

# ===============================================================================
# CHART 2 — Silent Attrition / Revenue Compression Dynamics
# ===============================================================================
print("\n" + "-" * 70)
print("  TASK 2 — REVENUE COMPRESSION DYNAMICS (SILENT ATTRITION)")
print("-" * 70)

# Derived signals
df["rev_decay_3v6"]  = df["avg3rev_f"] / (df["avg6rev_f"] + EPS)
df["mou_decay_3v6"]  = df["avg3mou_f"] / (df["avg6mou_f"] + EPS)
df["rev_decay_life"] = df["avg3rev_f"] / (df["avgrev_f"]  + EPS)

# Compression thresholds
compress_threshold = 0.85  # avg3rev < 85% of avg6rev
vip_compress = (
    (df["vip_flag"] == 1)
    & (df["churn"] == 1)
    & (df["rev_decay_3v6"] < compress_threshold)
)
n_vip_compress = vip_compress.sum()
pct_vip_compress = n_vip_compress / len(vip_churners) * 100

neg_change_vip = (df["vip_flag"]==1) & (df["churn"]==1) & (df["change_rev_f"] < -10)
pct_neg_change = neg_change_vip.sum() / len(vip_churners) * 100

print(f"  Revenue Decay Ratio (avg3rev/avg6rev) by Churn:")
for label, grp in df.groupby("churn"):
    name = "Churn" if label == 1 else "Stay"
    vals = grp["rev_decay_3v6"]
    print(f"    {name}: mean={vals.mean():.4f}  median={vals.median():.4f}  std={vals.std():.4f}")

print(f"\n  MOU Decay Ratio (avg3mou/avg6mou) by Churn:")
for label, grp in df.groupby("churn"):
    name = "Churn" if label == 1 else "Stay"
    vals = grp["mou_decay_3v6"]
    print(f"    {name}: mean={vals.mean():.4f}  median={vals.median():.4f}")

print(f"\n  change_rev by Churn:")
for label, grp in df.groupby("churn"):
    name = "Churn" if label == 1 else "Stay"
    vals = grp["change_rev_f"]
    print(f"    {name}: mean={vals.mean():.3f}  median={vals.median():.3f}")

print(f"\n  VIP Churners with rev_decay_3v6 < {compress_threshold}: {n_vip_compress:,} ({pct_vip_compress:.1f}%)")
print(f"  VIP Churners with change_rev < -10: {neg_change_vip.sum():,} ({pct_neg_change:.1f}%)")

# T-test on rev_decay_3v6
t_stat, p_val = stats.ttest_ind(
    df[df["churn"]==1]["rev_decay_3v6"].dropna(),
    df[df["churn"]==0]["rev_decay_3v6"].dropna()
)
print(f"\n  T-test rev_decay_3v6 (Churn vs Stay): t={t_stat:.4f}, p={p_val:.2e}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 2a — Distribution of change_rev
ax = axes[0,0]
for label, color, name in [(0,PALETTE["stay"],"Stayers"),(1,PALETTE["churn"],"Churners")]:
    vals = df[df["churn"]==label]["change_rev_f"].clip(-200, 200)
    ax.hist(vals, bins=80, alpha=0.55, color=color, label=name, density=True)
ax.axvline(0, color="black", lw=1.5, linestyle="--", alpha=0.7)
ax.set_xlim(-150, 100)
ax.set_xlabel("Monthly Revenue Change ($)"); ax.set_ylabel("Density")
ax.set_title("Distribution of Revenue Change\n(Churners vs Stayers)")
ax.legend()
ax.annotate(f"Churners median: ${df[df['churn']==1]['change_rev_f'].median():.1f}",
            xy=(df[df['churn']==1]['change_rev_f'].median(), 0.005),
            xytext=(-100, 0.012), fontsize=8, color=PALETTE["churn"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["churn"]))

# 2b — Box plot rev_decay_3v6 by Churn & VIP
ax = axes[0,1]
df["churn_vip"] = df["churn"].map({0:"Stay",1:"Churn"}) + " / " + df["vip_flag"].map({0:"Mass",1:"VIP"})
order = ["Stay / Mass","Stay / VIP","Churn / Mass","Churn / VIP"]
palette_map = {"Stay / Mass":"#90CAF9","Stay / VIP":PALETTE["stay"],
               "Churn / Mass":"#EF9A9A","Churn / VIP":PALETTE["churn"]}
data_box = [df[df["churn_vip"]==g]["rev_decay_3v6"].clip(0,2).dropna() for g in order]
bp = ax.boxplot(data_box, tick_labels=order, patch_artist=True, widths=0.55,
                medianprops=dict(color="gold", linewidth=2.5))
for patch, key in zip(bp["boxes"], order):
    patch.set_facecolor(palette_map[key])
ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1.2, label="No decay (ratio=1)")
ax.axhline(compress_threshold, color=PALETTE["vip"], linestyle=":", linewidth=1.5,
           label=f"Compression threshold ({compress_threshold})")
ax.set_ylabel("Revenue Decay Ratio (avg3rev/avg6rev)")
ax.set_title("Revenue Decay by Churn & Value Segment\n(Ratio < 0.85 = Silent Attrition)")
ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=15)

# 2c — rev_decay_3v6 vs change_rev scatter (VIP churners)
ax = axes[1,0]
vip_ch = df[(df["vip_flag"]==1) & (df["churn"]==1)].copy()
sc = ax.scatter(vip_ch["rev_decay_3v6"].clip(0,2),
                vip_ch["change_rev_f"].clip(-200, 100),
                c=vip_ch["rev_Mean_f"], cmap="RdYlGn_r",
                alpha=0.3, s=8, rasterized=True)
ax.axvline(compress_threshold, color=PALETTE["churn"], linestyle="--", lw=1.5,
           label=f"Decay < {compress_threshold}")
ax.axhline(-10, color=PALETTE["vip"], linestyle=":", lw=1.5, label="change_rev < -$10")
ax.set_xlabel("Revenue Decay Ratio (avg3rev/avg6rev)")
ax.set_ylabel("Monthly Revenue Change ($)")
ax.set_title(f"VIP Churners: Decay vs. Revenue Change\n"
             f"({pct_vip_compress:.0f}% show Silent Attrition pattern)")
plt.colorbar(sc, ax=ax, label="rev_Mean ($)")
ax.legend(fontsize=8)

# 2d — Monotonic trend: avg6 -> avg3 -> change_rev by churn cohort
ax = axes[1,1]
grp_means = df.groupby("churn")[["avg6rev_f","avg3rev_f","change_rev_f","rev_Mean_f"]].mean()
x_pts = [0, 1, 2]
labels_x = ["Avg Rev\n6-Month", "Avg Rev\n3-Month", "Avg Rev\nMost Recent"]
norm_factor = grp_means.loc[0, "avg6rev_f"]  # normalise to stayer avg6

for label, color, name in [(0,PALETTE["stay"],"Stayers"),(1,PALETTE["churn"],"Churners")]:
    row = grp_means.loc[label]
    vals = [row["avg6rev_f"]/norm_factor, row["avg3rev_f"]/norm_factor, row["rev_Mean_f"]/norm_factor]
    ax.plot(x_pts, vals, "o-", color=color, linewidth=2.5, markersize=10, label=name)
    for xi, val in zip(x_pts, vals):
        ax.annotate(f"{val:.3f}", xy=(xi, val), xytext=(xi+0.05, val+0.01), fontsize=8, color=color)

ax.set_xticks(x_pts); ax.set_xticklabels(labels_x)
ax.set_ylabel("Normalised Revenue (vs. Stayer Avg6)")
ax.set_title("Revenue Trajectory Over Time\n(Normalised to Stayer Avg6 Baseline)")
ax.legend()

fig.suptitle("PAIN POINT #2 — Revenue Compression Dynamics & Silent Attrition Detection",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "02_revenue_compression_dynamics.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"\n  [SAVED] 02_revenue_compression_dynamics.png")

# ===============================================================================
# CHART 3 — Behavioral & Physical Tipping Points
# ===============================================================================
print("\n" + "-" * 70)
print("  TASK 3 — BEHAVIORAL & PHYSICAL TIPPING POINTS")
print("-" * 70)

# eqpdays analysis
df["eqp_bin"] = pd.cut(df["eqpdays_f"].clip(0, 1000),
                        bins=[0,180,365,548,730,1000],
                        labels=["<6m","6-12m","12-18m","18-24m",">24m"])
eqp_churn = df.groupby("eqp_bin", observed=True)["churn"].agg(["mean","count"]).reset_index()
eqp_churn.columns = ["eqp_bin","churn_rate","count"]

# Overage analysis
df["overage_bucket"] = pd.cut(df["ovrrev_f"],
                               bins=[-0.01, 0, 5, 20, 50, 5000],
                               labels=["None","$0-5","$5-20","$20-50",">$50"])
ovr_churn = df.groupby("overage_bucket", observed=True)["churn"].agg(["mean","count"]).reset_index()
ovr_churn.columns = ["overage","churn_rate","count"]

# Network quality
df["net_quality"] = df["comp_vce_Mean"] / (df["attempt_Mean"] + 1)
df["drop_rate"]   = df["drop_blk_Mean"] / (df["attempt_Mean"] + 1)

print("  eqpdays bin vs Churn Rate:")
print(eqp_churn.to_string(index=False))
print("\n  Overage bucket vs Churn Rate:")
print(ovr_churn.to_string(index=False))

for col in ["drop_blk_Mean","comp_vce_Mean","custcare_Mean","ovrrev_Mean"]:
    r, p = stats.spearmanr(df[col].fillna(0), df["churn"])
    print(f"  Spearman {col} vs churn: r={r:.4f}, p={p:.2e}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 3a — Churn rate vs eqpdays bin
ax = axes[0,0]
colors_eqp = ["#66BB6A","#FFCA28","#FFA726","#EF5350","#B71C1C"]
bars = ax.bar(eqp_churn["eqp_bin"], eqp_churn["churn_rate"]*100,
              color=colors_eqp, edgecolor="white")
ax2b = ax.twinx()
ax2b.plot(range(len(eqp_churn)), eqp_churn["count"]/1000, "D--",
          color="#546E7A", linewidth=1.5, markersize=7, label="Count (k)")
ax2b.set_ylabel("Customer Count (thousands)", color="#546E7A")
for bar, rate in zip(bars, eqp_churn["churn_rate"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{rate*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("Device Age (eqpdays)"); ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Device Age\n(Equipment Obsolescence Tipping Point)")
ax.set_facecolor("#F9FAFB")
# Find max churn
max_idx = eqp_churn["churn_rate"].idxmax()
ax.bar(eqp_churn.loc[max_idx,"eqp_bin"], eqp_churn.loc[max_idx,"churn_rate"]*100,
       color="#B71C1C", edgecolor="gold", linewidth=2)

# 3b — Churn rate vs overage bucket
ax = axes[0,1]
colors_ovr = ["#81C784","#FFD54F","#FFB300","#F4511E","#B71C1C"]
bars2 = ax.bar(ovr_churn["overage"], ovr_churn["churn_rate"]*100,
               color=colors_ovr[:len(ovr_churn)], edgecolor="white")
for bar, rate in zip(bars2, ovr_churn["churn_rate"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{rate*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("Monthly Overage Revenue (Bill Shock)"); ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Bill Shock Severity\n(Overage Revenue Bucket)")
ax.set_facecolor("#F9FAFB")

# 3c — Drop/block rate vs churn
ax = axes[1,0]
df["drop_bin"] = pd.cut(df["drop_rate"].clip(0, 0.5),
                         bins=[0, 0.05, 0.10, 0.20, 0.30, 0.51],
                         labels=["<5%","5-10%","10-20%","20-30%",">30%"])
net_churn = df.groupby("drop_bin", observed=True)["churn"].mean() * 100
bars3 = ax.bar(net_churn.index, net_churn.values,
               color=["#66BB6A","#FFCA28","#FF7043","#EF5350","#B71C1C"], edgecolor="white")
for bar, val in zip(bars3, net_churn.values):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.3, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("Call Failure Rate (drop+block / attempt)"); ax.set_ylabel("Churn Rate (%)")
ax.set_title("Network Quality Impact on Churn\n(Call Failure Rate Tipping Point)")
ax.set_facecolor("#F9FAFB")

# 3d — custcare_Mean vs churn rate binned
ax = axes[1,1]
df["cc_bin"] = pd.cut(df["custcare_Mean"].clip(0, 20),
                       bins=[0, 0.01, 1, 3, 7, 20.01],
                       labels=["0 calls","0-1","1-3","3-7",">7"])
cc_churn = df.groupby("cc_bin", observed=True)["churn"].mean() * 100
bars4 = ax.bar(cc_churn.index, cc_churn.values,
               color=["#81C784","#FFF176","#FFB300","#F4511E","#B71C1C"], edgecolor="white")
for bar, val in zip(bars4, cc_churn.values):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.3, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("Avg Monthly Customer Care Calls"); ax.set_ylabel("Churn Rate (%)")
ax.set_title("Customer Care Intensity Impact on Churn\n(Service Friction Tipping Point)")
ax.set_facecolor("#F9FAFB")

fig.suptitle("PAIN POINT #3 — Behavioral & Physical Tipping Points",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "03_tipping_points_eqpdays_overage.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"\n  [SAVED] 03_tipping_points_eqpdays_overage.png")

# ===============================================================================
# CHART 4 — Correlation Heatmap: Top Drivers of Churn & Revenue Change
# ===============================================================================
print("\n" + "-" * 70)
print("  TASK 4 — TOP FEATURE CORRELATIONS WITH CHURN & CHANGE_REV")
print("-" * 70)

numeric_df = df.select_dtypes("number").drop(columns=[
    "Customer_ID","vip_flag","churn_vip","rev_Mean_f","avg3rev_f","avg6rev_f",
    "avgrev_f","change_rev_f","ovrrev_f","totmrc_f","avg3mou_f","avg6mou_f",
    "eqpdays_f","rev_decay_3v6","mou_decay_3v6","rev_decay_life","net_quality","drop_rate"
], errors="ignore")

# Spearman with churn
spear_churn = {}
for col in numeric_df.columns:
    try:
        r, p = stats.spearmanr(numeric_df[col].fillna(0), df["churn"])
        spear_churn[col] = r
    except Exception:
        pass
spear_churn_s = pd.Series(spear_churn).abs().sort_values(ascending=False)
top20_churn = spear_churn_s.head(20).index.tolist()

# Spearman with change_rev
spear_rev = {}
for col in numeric_df.columns:
    try:
        r, p = stats.spearmanr(numeric_df[col].fillna(0), df["change_rev_f"])
        spear_rev[col] = r
    except Exception:
        pass
spear_rev_s = pd.Series(spear_rev).abs().sort_values(ascending=False)

print("\n  Top 15 Spearman |r| with CHURN:")
for i, (col, val) in enumerate(spear_churn_s.head(15).items(), 1):
    direction = "+" if spear_churn.get(col, 0) > 0 else "-"
    print(f"    {i:2d}. {col:<35} |r|={val:.4f}  (dir={direction})")

print("\n  Top 15 Spearman |r| with CHANGE_REV:")
for i, (col, val) in enumerate(spear_rev_s.head(15).items(), 1):
    direction = "+" if spear_rev.get(col, 0) > 0 else "-"
    print(f"    {i:2d}. {col:<35} |r|={val:.4f}  (dir={direction})")

# Correlation heatmap on top features
union_top = list(dict.fromkeys(top20_churn + spear_rev_s.head(10).index.tolist()))[:22]
corr_cols = union_top + ["churn"]
corr_df = df[corr_cols].fillna(0).corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Heatmap
mask = np.zeros_like(corr_df, dtype=bool)
sns.heatmap(corr_df, mask=mask, ax=axes[0], cmap="RdBu_r", center=0,
            annot=True, fmt=".2f", annot_kws={"size":7},
            linewidths=0.4, cbar_kws={"shrink":0.8},
            xticklabels=True, yticklabels=True)
axes[0].set_title("Spearman Correlation Matrix\n(Top Feature Drivers + Churn)",
                  fontsize=12, fontweight="bold")
axes[0].tick_params(axis="x", rotation=45, labelsize=8)
axes[0].tick_params(axis="y", rotation=0, labelsize=8)

# Bar chart: top drivers
ax2 = axes[1]
top_n = 20
top_churn_signed = pd.Series({k: spear_churn[k] for k in spear_churn_s.head(top_n).index})
top_churn_signed = top_churn_signed.sort_values()
colors_bar2 = [PALETTE["churn"] if v > 0 else PALETTE["stay"] for v in top_churn_signed]
bars_h = ax2.barh(range(top_n), top_churn_signed.values, color=colors_bar2,
                  edgecolor="white", height=0.65)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels(top_churn_signed.index, fontsize=8)
ax2.axvline(0, color="black", linewidth=1)
ax2.set_xlabel("Spearman r (signed)")
ax2.set_title(f"Top {top_n} Feature Correlations\nwith Churn (Signed Spearman r)")
ax2.set_facecolor("#F9FAFB")
for bar, val in zip(bars_h, top_churn_signed.values):
    ax2.text(val + (0.003 if val >= 0 else -0.003),
             bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=7.5,
             ha="left" if val >= 0 else "right")

fig.suptitle("PAIN POINT #4 — Top Feature Correlations with Churn & Revenue Change",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "04_correlation_heatmap_top_drivers.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"\n  [SAVED] 04_correlation_heatmap_top_drivers.png")

# ===============================================================================
# FINAL SUMMARY REPORT
# ===============================================================================
print("\n" + "=" * 70)
print("  EXECUTIVE SUMMARY — FINANCIAL EDA FINDINGS")
print("=" * 70)
print(f"\n  DATASET: {df.shape[0]:,} customers × {df.shape[1]} features")
print(f"  Churn Rate: {df['churn'].mean()*100:.2f}% ({df['churn'].sum():,} churners)")
print(f"\n  -- PARETO (PAIN POINT #1) -------------------------------------------")
print(f"  VIP Threshold (rev_Mean 80th pct): ${vip_cutoff:.2f}/month")
print(f"  Top 20% customers -> {vip_rev_share:.1f}% of monthly revenue  [Pareto Rule]")
print(f"  VIP Churn Rate:  {vip_churn_rate:.1f}%  |  Mass Churn Rate: {mass_churn_rate:.1f}%")
print(f"  VIP Churners:    {len(vip_churners):,} customers")
print(f"  Monthly Revenue at Risk:  ${rev_at_risk_monthly/1e6:.3f}M USD")
print(f"  Annual  Revenue at Risk:  ${rev_at_risk_annual/1e6:.3f}M USD")
print(f"\n  -- SILENT ATTRITION (PAIN POINT #2) ---------------------------------")
print(f"  VIP Churners with rev_decay_3v6 < {compress_threshold}: {pct_vip_compress:.1f}% -> Silent Attrition")
print(f"  VIP Churners with change_rev < -$10/mo:               {pct_neg_change:.1f}%")
print(f"  T-test rev_decay_3v6 (Churn vs Stay): t={t_stat:.2f}, p={p_val:.1e}")
print(f"\n  -- TIPPING POINTS (PAIN POINT #3) -----------------------------------")
print(f"  eqpdays peak churn: {eqp_churn.loc[eqp_churn['churn_rate'].idxmax(),'eqp_bin']} bin")
print(f"  Overage peak churn: {ovr_churn.loc[ovr_churn['churn_rate'].idxmax(),'overage']} bucket")
print(f"  Top churn driver:  {spear_churn_s.index[0]}  (|r|={spear_churn_s.iloc[0]:.4f})")
print(f"  2nd churn driver:  {spear_churn_s.index[1]}  (|r|={spear_churn_s.iloc[1]:.4f})")
print(f"  3rd churn driver:  {spear_churn_s.index[2]}  (|r|={spear_churn_s.iloc[2]:.4f})")
print(f"\n  Figures saved to: {FIG_DIR}")
print("=" * 70)

