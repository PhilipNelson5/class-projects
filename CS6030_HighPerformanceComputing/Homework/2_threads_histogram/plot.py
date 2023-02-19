#!/usr/bin/env python3
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

timings = pd.read_csv("./timings.csv", skipinitialspace=True)
timings["speedup"] = timings["time_ms"].iloc[0] / timings["time_ms"]
timings["efficiency"] = timings["time_ms"].iloc[0] / timings["threads"] / timings["time_ms"]
print(timings)

fig, axes = plt.subplots(3, figsize=(6,10))
# fig.suptitle("Timing results")

sns.lineplot(data=timings, x="threads", y="time_ms", ax=axes[0])
# plt.title("Execution Time")
# plt.xlabel("thread count")
axes[0].set_ylabel("time (ms)")
# plt.savefig("./images/time.png")

# plt.figure()
sns.lineplot(data=timings, x="threads", y="speedup", ax=axes[1])
# plt.title("Speedup")
# plt.xlabel("thread count")
# plt.savefig("./images/speedup.png")

# plt.figure()
sns.lineplot(data=timings, x="threads", y="efficiency", ax=axes[2])
# plt.title("Efficiency")
# plt.xlabel("thread count")
# plt.savefig("./images/efficiency.png")

fig.align_ylabels(axes)
fig.tight_layout()
plt.savefig("./images/plots.png")
