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

# timings_a = pd.read_csv("./timings_outer_only_array.csv", skipinitialspace=True)
# timings_b = pd.read_csv("./timings_move_array.csv", skipinitialspace=True)
# timings_a["speedup"] = timings_a["time_ms"].iloc[0] / timings_a["time_ms"]
# timings_a["efficiency"] = timings_a["time_ms"].iloc[0] / timings_a["threads"] / timings_a["time_ms"]
# timings_b["speedup"] = timings_b["time_ms"].iloc[0] / timings_b["time_ms"]
# timings_b["efficiency"] = timings_b["time_ms"].iloc[0] / timings_b["threads"] / timings_b["time_ms"]

# fig, axes = plt.subplots(3, figsize=(6,10))
# # fig.suptitle("Timing results")

# sns.lineplot(data=timings_a, x="threads", y="time_ms", ax=axes[0], label="outer only")
# sns.lineplot(data=timings_b, x="threads", y="time_ms", ax=axes[0], label="outer + memcpy")
# # plt.title("Execution Time")
# # plt.xlabel("thread count")
# axes[0].set_ylabel("time (ms)")
# # plt.savefig("./images/time.png")

# # plt.figure()
# sns.lineplot(data=timings_a, x="threads", y="speedup", ax=axes[1], label="outer only")
# sns.lineplot(data=timings_b, x="threads", y="speedup", ax=axes[1], label="outer + memcpy")
# # plt.title("Speedup")
# # plt.xlabel("thread count")
# # plt.savefig("./images/speedup.png")

# # plt.figure()
# sns.lineplot(data=timings_a, x="threads", y="efficiency", ax=axes[2], label="outer only")
# sns.lineplot(data=timings_b, x="threads", y="efficiency", ax=axes[2], label="outer + memcpy")
# # plt.title("Efficiency")
# # plt.xlabel("thread count")
# # plt.savefig("./images/efficiency.png")

# fig.align_ylabels(axes)
# fig.tight_layout()
# plt.savefig("./images/plots.png")