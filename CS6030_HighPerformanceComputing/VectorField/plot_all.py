#!/usr/bin/env python3
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

##### DATA #####
def read(file):
    df = pd.read_csv(f"./data/timing_{file}.csv")
    df = df.groupby(["threads", "curves"]).mean().reset_index()

    def speedup(row, col):
        return df[(df["curves"] == row["curves"])][col].iloc[0] / row[col]

    def efficiency(row, col):
        return df[df["curves"] == row["curves"]][col].iloc[0] / row["threads"] / row[col]

    for col in ["program", "curve"]:
        df[f"{col}_speedup"] = df.apply(lambda row: speedup(row, f"{col}_time"), axis=1)
        df[f"{col}_efficiency"] = df.apply(lambda row: efficiency(row, f"{col}_time"), axis=1)
    
    return df

df_c = read('cuda')
df_t = read('thread')
df_m = read('mpi')

sns.lineplot(data=df_t[df_t.curves == 1024], x='threads', y='curve_time', label='threads')
sns.lineplot(data=df_m[df_m.curves == 1024], x='threads', y='curve_time', label='mpi')
sns.lineplot(data=df_c[(df_c.curves == 1024)&(df_c.threads < 100)], x='threads', y='curve_time', label='cuda')

# plt.title('Comparison of methods')
plt.xlabel('Threads / Processes / Processes per Block')
plt.ylabel('Time (ms)')

plt.tight_layout()
plt.savefig(f"plot_all.pdf")