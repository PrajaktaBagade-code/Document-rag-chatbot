"""
STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
Produces 12 publication-quality charts saved as PNGs:
  01 - Overall churn rate KPI card
  02 - Churn by contract type
  03 - Churn by tenure cohort (cohort analysis)
  04 - Churn by internet service
  05 - Churn by payment method
  06 - Monthly charges distribution: churned vs retained
  07 - Churn by number of services
  08 - Churn by senior citizen status
  09 - Churn funnel (customer lifecycle stages)
  10 - Revenue at risk by segment
  11 - Churn by charge tier
  12 - Correlation heatmap of churn drivers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

CLEAN = "cleaned_telco_churn.csv"
DIR   = "outputs/charts"
os.makedirs(DIR, exist_ok=True)

df = pd.read_csv(CLEAN)
df["TenureBucket"] = pd.Categorical(
    df["TenureBucket"],
    categories=["0–6 mo","7–12 mo","13–24 mo","25–48 mo","49–72 mo"],
    ordered=True
)

# ── Design system ───────────────────────────────────────────
CHURN_C   = "#E24B4A"    # red  — churned
RETAIN_C  = "#1A56A0"    # blue — retained
ACCENT    = "#1D9E75"    # green — highlights
AMBER     = "#EF9F27"
BG        = "#F8FAFC"
GRID      = "#E2E8F0"
TEXT      = "#0F172A"
SUB       = "#475569"
PALETTE   = [RETAIN_C, CHURN_C, ACCENT, AMBER, "#7F77DD", "#D85A30"]

def ax_style(ax, title, xl="", yl="", grid_axis="y"):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax.set_xlabel(xl, fontsize=9, color=SUB)
    ax.set_ylabel(yl, fontsize=9, color=SUB)
    ax.tick_params(colors=SUB, labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(GRID)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

def save(name):
    plt.tight_layout()
    plt.savefig(f"{DIR}/{name}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ {name}")

pct = mtick.PercentFormatter(1.0)

print("\nGenerating EDA charts...")

# ─────────────────────────────────────────────────────────────
# CHART 01 — KPI Overview Card
# ─────────────────────────────────────────────────────────────
total        = len(df)
churned      = df["ChurnFlag"].sum()
retained     = total - churned
churn_rate   = churned / total
rev_at_risk  = df["RevenueAtRisk"].sum()
avg_tenure_c = df[df["ChurnFlag"]==1]["tenure"].mean()
avg_tenure_r = df[df["ChurnFlag"]==0]["tenure"].mean()

fig, axes = plt.subplots(1, 4, figsize=(14, 3), facecolor="white")
kpis = [
    ("Total Customers",    f"{total:,}",          "#EFF6FF", RETAIN_C),
    ("Churned Customers",  f"{churned:,} ({churn_rate*100:.1f}%)", "#FCEBEB", CHURN_C),
    ("Monthly Rev at Risk",f"${rev_at_risk:,.0f}", "#FAEEDA", AMBER),
    ("Avg Tenure (Churned)",f"{avg_tenure_c:.1f} mo vs {avg_tenure_r:.1f} mo retained", "#EAF3DE", ACCENT),
]
for ax, (label, val, bg, col) in zip(axes, kpis):
    ax.set_facecolor(bg)
    ax.text(0.5, 0.62, val,   ha="center", va="center", fontsize=16, fontweight="bold",
            color=col, transform=ax.transAxes)
    ax.text(0.5, 0.28, label, ha="center", va="center", fontsize=9,  color=SUB,
            transform=ax.transAxes)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(1.2); spine.set_visible(True)
plt.suptitle("Churn Analysis — KPI Dashboard", fontsize=13, fontweight="bold",
             color=TEXT, y=1.04)
save("01_kpi_overview.png")

# ─────────────────────────────────────────────────────────────
# CHART 02 — Churn by Contract Type
# ─────────────────────────────────────────────────────────────
ct = df.groupby("Contract")["ChurnFlag"].agg(["mean","sum","count"]).reset_index()
ct.columns = ["Contract","ChurnRate","Churned","Total"]
ct = ct.sort_values("ChurnRate", ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
bars = ax1.barh(ct["Contract"], ct["ChurnRate"],
                color=[CHURN_C, AMBER, RETAIN_C], alpha=0.88)
ax1.xaxis.set_major_formatter(pct)
for bar, rate in zip(bars, ct["ChurnRate"]):
    ax1.text(rate + 0.005, bar.get_y()+bar.get_height()/2,
             f"{rate*100:.1f}%", va="center", fontsize=9, fontweight="bold", color=TEXT)
ax_style(ax1, "Churn Rate by Contract Type", xl="Churn Rate", grid_axis="x")

x = np.arange(len(ct))
w = 0.35
ax2.bar(x-w/2, ct["Total"]-ct["Churned"], w, label="Retained", color=RETAIN_C, alpha=0.85)
ax2.bar(x+w/2, ct["Churned"],             w, label="Churned",  color=CHURN_C,  alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(ct["Contract"])
ax2.legend(fontsize=8)
ax_style(ax2, "Volume by Contract Type", yl="Customers")
save("02_churn_by_contract.png")

# ─────────────────────────────────────────────────────────────
# CHART 03 — Cohort Analysis: Churn by Tenure Bucket
# ─────────────────────────────────────────────────────────────
cohort = (df.groupby("TenureBucket", observed=True)["ChurnFlag"]
            .agg(["mean","sum","count"]).reset_index())
cohort.columns = ["TenureBucket","ChurnRate","Churned","Total"]

fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
bar_colors = [CHURN_C if r > 0.25 else AMBER if r > 0.15 else RETAIN_C
              for r in cohort["ChurnRate"]]
bars = ax.bar(cohort["TenureBucket"].astype(str), cohort["ChurnRate"],
              color=bar_colors, alpha=0.88, zorder=3)
ax.yaxis.set_major_formatter(pct)
for bar, row in zip(bars, cohort.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{row.ChurnRate*100:.1f}%\n({row.Churned} customers)",
            ha="center", fontsize=8, color=TEXT, fontweight="bold")
ax.axhline(df["ChurnFlag"].mean(), color=CHURN_C, linestyle="--",
           linewidth=1.2, label=f"Overall avg ({df['ChurnFlag'].mean()*100:.1f}%)")
ax.legend(fontsize=9)
ax_style(ax, "Cohort Analysis: Churn Rate by Customer Tenure",
         xl="Tenure Cohort", yl="Churn Rate")
save("03_cohort_churn_by_tenure.png")

# ─────────────────────────────────────────────────────────────
# CHART 04 — Churn by Internet Service
# ─────────────────────────────────────────────────────────────
inet = df.groupby("InternetService")["ChurnFlag"].agg(["mean","count"]).reset_index()
inet.columns = ["InternetService","ChurnRate","Count"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
colors_inet = [CHURN_C, AMBER, RETAIN_C]
wedges, texts, autos = ax1.pie(
    inet["Count"], labels=inet["InternetService"],
    autopct="%1.1f%%", colors=colors_inet, startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=1.5), pctdistance=0.78
)
for auto in autos: auto.set_fontsize(9)
ax1.set_title("Internet Service Mix", fontsize=11, fontweight="bold", color=TEXT)

bars = ax2.bar(inet["InternetService"], inet["ChurnRate"],
               color=colors_inet, alpha=0.88)
ax2.yaxis.set_major_formatter(pct)
for bar, rate in zip(bars, inet["ChurnRate"]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{rate*100:.1f}%", ha="center", fontsize=10, fontweight="bold", color=TEXT)
ax_style(ax2, "Churn Rate by Internet Service", yl="Churn Rate")
plt.suptitle("Internet Service: Churn Analysis", fontsize=13,
             fontweight="bold", color=TEXT, y=1.02)
save("04_churn_by_internet.png")

# ─────────────────────────────────────────────────────────────
# CHART 05 — Churn by Payment Method
# ─────────────────────────────────────────────────────────────
pay = df.groupby("PaymentMethod")["ChurnFlag"].agg(["mean","count"]).reset_index()
pay.columns = ["PaymentMethod","ChurnRate","Count"]
pay = pay.sort_values("ChurnRate", ascending=True)
short_labels = [p.replace(" (automatic)","*").replace("Electronic check","E-check")
                .replace("Mailed check","Mail check")
                .replace("Bank transfer","Bank transfer")
                .replace("Credit card","Credit card")
                for p in pay["PaymentMethod"]]

fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
bar_colors = [CHURN_C if r > 0.35 else AMBER if r > 0.20 else RETAIN_C
              for r in pay["ChurnRate"]]
bars = ax.barh(short_labels, pay["ChurnRate"], color=bar_colors, alpha=0.88)
ax.xaxis.set_major_formatter(pct)
for bar, rate, cnt in zip(bars, pay["ChurnRate"], pay["Count"]):
    ax.text(rate+0.004, bar.get_y()+bar.get_height()/2,
            f"{rate*100:.1f}%  (n={cnt:,})", va="center", fontsize=9, color=TEXT)
ax.set_xlim(0, 0.60)
ax_style(ax, "Churn Rate by Payment Method\n(* = automatic payment)",
         xl="Churn Rate", grid_axis="x")
save("05_churn_by_payment.png")

# ─────────────────────────────────────────────────────────────
# CHART 06 — Monthly Charges Distribution: Churned vs Retained
# ─────────────────────────────────────────────────────────────
churned_df  = df[df["ChurnFlag"]==1]["MonthlyCharges"]
retained_df = df[df["ChurnFlag"]==0]["MonthlyCharges"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
ax1.hist(retained_df, bins=35, color=RETAIN_C, alpha=0.7, label="Retained", density=True)
ax1.hist(churned_df,  bins=35, color=CHURN_C,  alpha=0.7, label="Churned",  density=True)
ax1.axvline(churned_df.mean(),  color=CHURN_C,  linestyle="--", linewidth=1.5,
            label=f"Churned avg: ${churned_df.mean():.0f}")
ax1.axvline(retained_df.mean(), color=RETAIN_C, linestyle="--", linewidth=1.5,
            label=f"Retained avg: ${retained_df.mean():.0f}")
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:.0f}"))
ax_style(ax1, "Monthly Charge Distribution", xl="Monthly Charges ($)", yl="Density")

bp = ax2.boxplot([retained_df, churned_df], tick_labels=["Retained","Churned"],
                 patch_artist=True,
                 medianprops=dict(color=AMBER, linewidth=2))
bp["boxes"][0].set_facecolor(RETAIN_C + "44")
bp["boxes"][1].set_facecolor(CHURN_C  + "44")
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:.0f}"))
ax_style(ax2, "Monthly Charges Boxplot", yl="Monthly Charges ($)", grid_axis="y")
plt.suptitle("Monthly Charges: Churned vs Retained Customers",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
save("06_monthly_charges_dist.png")

# ─────────────────────────────────────────────────────────────
# CHART 07 — Churn by Number of Services
# ─────────────────────────────────────────────────────────────
svc = df.groupby("NumServices")["ChurnFlag"].agg(["mean","count"]).reset_index()
svc.columns = ["NumServices","ChurnRate","Count"]

fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
bar_c = [CHURN_C if r > 0.30 else AMBER if r > 0.20 else RETAIN_C
         for r in svc["ChurnRate"]]
bars = ax.bar(svc["NumServices"], svc["ChurnRate"], color=bar_c, alpha=0.88, zorder=3)
ax2b = ax.twinx()
ax2b.plot(svc["NumServices"], svc["Count"], color=AMBER,
          marker="o", linewidth=2, markersize=6, label="Customer count")
ax2b.set_ylabel("Number of Customers", fontsize=9, color=AMBER)
ax2b.tick_params(colors=AMBER, labelsize=8)
ax.yaxis.set_major_formatter(pct)
for bar, row in zip(bars, svc.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{row.ChurnRate*100:.1f}%", ha="center", fontsize=8,
            fontweight="bold", color=TEXT)
ax.set_xlabel("Number of Add-on Services Subscribed", fontsize=9, color=SUB)
ax_style(ax, "Churn Rate vs Number of Subscribed Services",
         xl="Services Subscribed", yl="Churn Rate")
save("07_churn_by_services.png")

# ─────────────────────────────────────────────────────────────
# CHART 08 — Churn by Demographics
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor="white")
demo_pairs = [
    ("SeniorCitizen", {0:"Non-Senior", 1:"Senior"}),
    ("Partner",       None),
    ("Dependents",    None),
]
for ax, (col, mapping) in zip(axes, demo_pairs):
    grp = df.groupby(col)["ChurnFlag"].mean().reset_index()
    grp.columns = [col,"ChurnRate"]
    if mapping:
        grp[col] = grp[col].map(mapping)
    colors_d = [CHURN_C if r > 0.25 else RETAIN_C for r in grp["ChurnRate"]]
    bars = ax.bar(grp[col].astype(str), grp["ChurnRate"], color=colors_d, alpha=0.88)
    ax.yaxis.set_major_formatter(pct)
    for bar, rate in zip(bars, grp["ChurnRate"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{rate*100:.1f}%", ha="center", fontsize=10,
                fontweight="bold", color=TEXT)
    ax_style(ax, f"Churn by {col}", yl="Churn Rate")
plt.suptitle("Demographic Churn Breakdown",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
save("08_churn_by_demographics.png")

# ─────────────────────────────────────────────────────────────
# CHART 09 — Customer Funnel (Lifecycle Stages)
# ─────────────────────────────────────────────────────────────
stages = ["All Customers","Active Subscribers","Monthly Contract","Fiber + Monthly","Churned"]
counts = [
    len(df),
    len(df[df["InternetService"] != "No"]),
    len(df[(df["InternetService"] != "No") & (df["Contract"]=="Month-to-month")]),
    len(df[(df["InternetService"]=="Fiber optic") & (df["Contract"]=="Month-to-month")]),
    df["ChurnFlag"].sum()
]

fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
bar_heights = [1.0]*len(stages)
bar_widths  = [c/counts[0] for c in counts]
colors_f    = [RETAIN_C, "#378ADD", AMBER, "#D85A30", CHURN_C]
y_positions = np.arange(len(stages))[::-1]

for i, (stage, count, width, col) in enumerate(
        zip(stages, counts, bar_widths, colors_f)):
    y = y_positions[i]
    ax.barh(y, width, left=(1-width)/2, height=0.55,
            color=col, alpha=0.85, zorder=3)
    ax.text(0.5, y, f"{stage}\n{count:,} ({count/counts[0]*100:.1f}%)",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="white", zorder=4)

ax.set_xlim(0, 1); ax.axis("off")
ax.set_title("Customer Churn Funnel — Who Churns and Where?",
             fontsize=12, fontweight="bold", color=TEXT, pad=12)
save("09_churn_funnel.png")

# ─────────────────────────────────────────────────────────────
# CHART 10 — Revenue at Risk by Segment
# ─────────────────────────────────────────────────────────────
risk = (df[df["ChurnFlag"]==1]
        .groupby("Contract")["MonthlyCharges"]
        .agg(["sum","count","mean"])
        .reset_index())
risk.columns = ["Contract","TotalRisk","Customers","AvgCharge"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
bars = ax1.bar(risk["Contract"], risk["TotalRisk"],
               color=[CHURN_C, AMBER, RETAIN_C], alpha=0.88)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
for bar, row in zip(bars, risk.itertuples()):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
             f"${row.TotalRisk/1000:.1f}K\n({row.Customers} cust.)",
             ha="center", fontsize=8, color=TEXT)
ax_style(ax1, "Monthly Revenue at Risk by Contract", yl="Monthly Revenue ($)")

bars2 = ax2.bar(risk["Contract"], risk["AvgCharge"],
                color=[CHURN_C, AMBER, RETAIN_C], alpha=0.88)
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:.0f}"))
for bar, row in zip(bars2, risk.itertuples()):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"${row.AvgCharge:.0f}", ha="center", fontsize=9,
             fontweight="bold", color=TEXT)
ax_style(ax2, "Avg Monthly Charge of Churned Customers", yl="Avg Charge ($)")
plt.suptitle("Revenue at Risk — Churn Impact by Contract",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
save("10_revenue_at_risk.png")

# ─────────────────────────────────────────────────────────────
# CHART 11 — Churn by Charge Tier
# ─────────────────────────────────────────────────────────────
tier = (df.groupby("ChargeTier", observed=True)["ChurnFlag"]
          .agg(["mean","count"]).reset_index())
tier.columns = ["ChargeTier","ChurnRate","Count"]

fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
colors_t = [RETAIN_C, AMBER, "#D85A30", CHURN_C]
bars = ax.bar(tier["ChargeTier"].astype(str), tier["ChurnRate"],
              color=colors_t, alpha=0.88, zorder=3)
ax.yaxis.set_major_formatter(pct)
for bar, row in zip(bars, tier.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
            f"{row.ChurnRate*100:.1f}%\n(n={row.Count:,})",
            ha="center", fontsize=8.5, color=TEXT, fontweight="bold")
ax_style(ax, "Churn Rate by Monthly Charge Tier",
         xl="Charge Tier", yl="Churn Rate")
save("11_churn_by_charge_tier.png")

# ─────────────────────────────────────────────────────────────
# CHART 12 — Correlation Heatmap of Churn Drivers
# ─────────────────────────────────────────────────────────────
encode_map = {
    "Contract":        {"Month-to-month":2,"One year":1,"Two year":0},
    "InternetService": {"Fiber optic":2,"DSL":1,"No":0},
    "PaymentMethod":   {"Electronic check":3,"Mailed check":2,
                        "Bank transfer (automatic)":1,"Credit card (automatic)":0},
}
df_corr = df[["tenure","MonthlyCharges","NumServices","SeniorCitizen",
              "AutoPay","Contract","InternetService","PaymentMethod","ChurnFlag"]].copy()
for col, mapping in encode_map.items():
    df_corr[col] = df_corr[col].map(mapping)

rename = {
    "tenure":"Tenure", "MonthlyCharges":"Monthly Charges",
    "NumServices":"Num Services", "SeniorCitizen":"Senior Citizen",
    "AutoPay":"Auto Pay", "Contract":"Contract Type",
    "InternetService":"Internet Service", "PaymentMethod":"Payment Method",
    "ChurnFlag":"CHURN"
}
df_corr.rename(columns=rename, inplace=True)
corr = df_corr.corr()

fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink":0.8}, ax=ax,
            annot_kws={"size":8})
ax.set_title("Correlation Matrix — Churn Drivers",
             fontsize=12, fontweight="bold", color=TEXT, pad=12)
ax.tick_params(colors=SUB, labelsize=8)
save("12_correlation_heatmap.png")

print(f"\n  ✓ All 12 charts saved to  {DIR}/")
