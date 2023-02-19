#!/usr/bin/env python3
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

timings = pd.read_csv("./data/notch.txt", delim_whitespace=True)

sns.lineplot(data=timings[timings.kernel == "Copy"], x="dim", y="ebw", label="Copy")
sns.lineplot(data=timings[timings.kernel == "Tiled_Transpose"], x="dim", y="ebw", label="Tiled Transpose")
sns.lineplot(data=timings[timings.kernel == "Naive_Transpose"], x="dim", y="ebw", label="Naive Transpose")
plt.legend()
plt.xlabel(R'Tile size $2^n x 2^n$')
plt.ylabel('Effective bandwidth GB/s')
# plt.title('Effective Bandwidth Comparison on NVIDIA Tesla K80')
plt.tight_layout()
plt.savefig('./plot.png')