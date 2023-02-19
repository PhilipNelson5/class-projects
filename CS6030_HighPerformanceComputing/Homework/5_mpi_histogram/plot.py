#!/usr/bin/env python3
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

##### DATA #####

df = pd.read_csv("./timing.csv")
df = df.groupby(["rank_count", "data_count"]).mean().reset_index()

def speedup(row, col):
    return df[df["data_count"] == row["data_count"]][col].iloc[0] / row[col]

def efficiency(row, col):
    return df[df["data_count"] == row["data_count"]][col].iloc[0] / row["rank_count"] / row[col]

for col in ["app", "histogram", "total"]:
    df[f"{col}_speedup"] = df.apply(lambda row: speedup(row, f"{col}_time"), axis=1)
    df[f"{col}_efficiency"] = df.apply(lambda row: efficiency(row, f"{col}_time"), axis=1)

print(df)

##### PLOT #####

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,10))

##### Time #####

col = 0

ax = axes[0][col]
ax.set_title("50e6 elements")
ax.set_ylabel("time (ms)")
ax.set_xlabel("number of processes")
sns.lineplot(data=df[df["data_count"]==50000000], x="rank_count", y="histogram_time", label="histogram", ax=ax)
sns.lineplot(data=df[df["data_count"]==50000000], x="rank_count", y="app_time", label="app", ax=ax)
sns.lineplot(data=df[df["data_count"]==50000000], x="rank_count", y="total_time", label="total", ax=ax)

ax = axes[1][col]
ax.set_title("100e6 elements")
ax.set_ylabel("time (ms)")
ax.set_xlabel("number of processes")
sns.lineplot(data=df[df["data_count"]==100000000], x="rank_count", y="histogram_time", label="histogram", ax=ax)
sns.lineplot(data=df[df["data_count"]==100000000], x="rank_count", y="app_time", label="app", ax=ax)
sns.lineplot(data=df[df["data_count"]==100000000], x="rank_count", y="total_time", label="total", ax=ax)

ax = axes[2][col]
ax.set_title("200e6 elements")
ax.set_ylabel("time (ms)")
ax.set_xlabel("number of processes")
sns.lineplot(data=df[df["data_count"]==200000000], x="rank_count", y="histogram_time", label="histogram", ax=ax)
sns.lineplot(data=df[df["data_count"]==200000000], x="rank_count", y="app_time", label="app", ax=ax)
sns.lineplot(data=df[df["data_count"]==200000000], x="rank_count", y="total_time", label="total", ax=ax)

##### Speedup #####

col = 1
max_speedup = max([df[f"{col}_speedup"].max() for col in ["app", "histogram", "total"]])
min_speedup = min([df[f"{col}_speedup"].min() for col in ["app", "histogram", "total"]])

ax = axes[0][col]
ax.set_ylim([min_speedup*.9, max_speedup*1.05])
ax.set_title("50e6 elements")
ax.set_ylabel("speedup")
ax.set_xlabel("number of processes")
sns.lineplot(data=df[df["data_count"]==50000000], x="rank_count", y="histogram_speedup", label="histogram", ax=ax)
sns.lineplot(data=df[df["data_count"]==50000000], x="rank_count", y="app_speedup", label="app", ax=ax)
sns.lineplot(data=df[df["data_count"]==50000000], x="rank_count", y="total_speedup", label="total", ax=ax)

ax = axes[1][col]
ax.set_ylim([min_speedup*.9, max_speedup*1.05])
ax.set_title("100e6 elements")
ax.set_ylabel("speedup")
ax.set_xlabel("number of processes")
sns.lineplot(data=df[df["data_count"]==100000000], x="rank_count", y="histogram_speedup", label="histogram", ax=ax)
sns.lineplot(data=df[df["data_count"]==100000000], x="rank_count", y="app_speedup", label="app", ax=ax)
sns.lineplot(data=df[df["data_count"]==100000000], x="rank_count", y="total_speedup", label="total", ax=ax)

ax = axes[2][col]
ax.set_ylim([min_speedup*.9, max_speedup*1.05])
ax.set_title("200e6 elements")
ax.set_ylabel("speedup")
ax.set_xlabel("number of processes")
sns.lineplot(data=df[df["data_count"]==200000000], x="rank_count", y="histogram_speedup", label="histogram", ax=ax)
sns.lineplot(data=df[df["data_count"]==200000000], x="rank_count", y="app_speedup", label="app", ax=ax)
sns.lineplot(data=df[df["data_count"]==200000000], x="rank_count", y="total_speedup", label="total", ax=ax)

##### Efficiency #####

col = 2
max_efficiency = max([df[f"{col}_efficiency"].max() for col in ["app", "histogram", "total"]])
min_efficiency = min([df[f"{col}_efficiency"].min() for col in ["app", "histogram", "total"]])

ax = axes[0][col]
# ax.set_ylim([min_speedup*.9, max_speedup*1.05])
ax.set_title("50e6 elements")
ax.set_ylabel("efficiency")
ax.set_xlabel("number of processes")
sns.lineplot(data=df[df["data_count"]==50000000], x="rank_count", y="histogram_efficiency", label="histogram", ax=ax)
sns.lineplot(data=df[df["data_count"]==50000000], x="rank_count", y="app_efficiency", label="app", ax=ax)
sns.lineplot(data=df[df["data_count"]==50000000], x="rank_count", y="total_efficiency", label="total", ax=ax)

ax = axes[1][col]
# ax.set_ylim([min_speedup*.9, max_speedup*1.05])
ax.set_title("100e6 elements")
ax.set_ylabel("efficiency")
ax.set_xlabel("number of processes")
sns.lineplot(data=df[df["data_count"]==100000000], x="rank_count", y="histogram_efficiency", label="histogram", ax=ax)
sns.lineplot(data=df[df["data_count"]==100000000], x="rank_count", y="app_efficiency", label="app", ax=ax)
sns.lineplot(data=df[df["data_count"]==100000000], x="rank_count", y="total_efficiency", label="total", ax=ax)

ax = axes[2][col]
# ax.set_ylim([min_speedup*.9, max_speedup*1.05])
ax.set_title("200e6 elements")
ax.set_ylabel("efficiency")
ax.set_xlabel("number of processes")
sns.lineplot(data=df[df["data_count"]==200000000], x="rank_count", y="histogram_efficiency", label="histogram", ax=ax)
sns.lineplot(data=df[df["data_count"]==200000000], x="rank_count", y="app_efficiency", label="app", ax=ax)
sns.lineplot(data=df[df["data_count"]==200000000], x="rank_count", y="total_efficiency", label="total", ax=ax)

##### Save #####

fig.align_ylabels(axes)
fig.tight_layout()
plt.savefig("plot.pdf")
