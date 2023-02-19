#!/usr/bin/env python3
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

##### DATA #####
file="mpi"

df = pd.read_csv(f"./data/timing_{file}.csv")
df = df.groupby(["threads", "curves"]).mean().reset_index()

def speedup(row, col):
    return df[(df["curves"] == row["curves"])][col].iloc[0] / row[col]

def efficiency(row, col):
    return df[df["curves"] == row["curves"]][col].iloc[0] / row["threads"] / row[col]

for col in ["program", "curve"]:
    df[f"{col}_speedup"] = df.apply(lambda row: speedup(row, f"{col}_time"), axis=1)
    df[f"{col}_efficiency"] = df.apply(lambda row: efficiency(row, f"{col}_time"), axis=1)

print(df)

##### PLOT #####

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

##### Time #####
row = 0; col = 0
ax = axes[row][col]
ax.set_title("Total Program Time")
ax.set_ylabel("Time (ms)")
ax.set_xlabel("Number of Processes")
for x in np.flip(df['curves'].unique()):
    sns.lineplot(data=df[df["curves"]==x], x="threads", y="program_time", label=f"{x}", ax=ax)

row = 0; col = 1
ax = axes[row][col]
ax.set_title("Streamline Calculation Time")
ax.set_ylabel("Time (ms)")
ax.set_xlabel("Number of Processes")
for x in np.flip(df['curves'].unique()):
    sns.lineplot(data=df[df["curves"]==x], x="threads", y="curve_time", label=f"{x}", ax=ax)

##### Speedup #####
row = 1; col = 0
ax = axes[row][col]
ax.set_title("Total Program Speedup")
ax.set_ylabel("Speedup")
ax.set_xlabel("Number of Processes")
for x in np.flip(df['curves'].unique()):
    sns.lineplot(data=df[df["curves"]==x], x="threads", y="program_speedup", label=f"{x}", ax=ax)

row = 1; col = 1
ax = axes[row][col]
ax.set_title("Streamline Calculation Speedup")
ax.set_ylabel("Speedup")
ax.set_xlabel("Number of Processes")
for x in np.flip(df['curves'].unique()):
    sns.lineplot(data=df[df["curves"]==x], x="threads", y="curve_speedup", label=f"{x}", ax=ax)

##### Efficiency #####
row = 2; col = 0
ax = axes[row][col]
ax.set_title("Total Program Efficiency")
ax.set_ylabel("Efficiency")
ax.set_xlabel("Number of Processes")
for x in np.flip(df['curves'].unique()):
    sns.lineplot(data=df[df["curves"]==x], x="threads", y="program_efficiency", label=f"{x}", ax=ax)

row = 2; col = 1
ax = axes[row][col]
ax.set_title("Streamline Calculation Efficiency")
ax.set_ylabel("Efficiency")
ax.set_xlabel("Number of Processes")
for x in np.flip(df['curves'].unique()):
    sns.lineplot(data=df[df["curves"]==x], x="threads", y="curve_efficiency", label=f"{x}", ax=ax)

##### Save #####

fig.align_ylabels(axes)
fig.tight_layout()
plt.savefig(f"plot_{file}.pdf")
