#!/usr/bin/env python3
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

timings = pd.read_csv("./data/timings_data_count.csv", skipinitialspace=True)
print(timings)

sns.lineplot(data=timings, x="data_count", y="time_ms")
plt.xlabel("data count")
plt.ylabel("time (ms)")
plt.tight_layout()
plt.savefig("./images/data_count.png")

# timings_iterator = pd.read_csv("./data/timings_data_count_iterator.csv", skipinitialspace=True)
# timings_array = pd.read_csv("./data/timings_data_count_array.csv", skipinitialspace=True)

# plt.figure()
# sns.lineplot(data=timings_iterator, x="data_count", y="time_ms", label="iterator")
# sns.lineplot(data=timings_array, x="data_count", y="time_ms", label="array")
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
# plt.xlabel("data count")
# plt.ylabel("time (ms)")
# plt.tight_layout()
# plt.savefig("./images/data_count.png")