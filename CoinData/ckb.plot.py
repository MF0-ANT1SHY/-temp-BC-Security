"""
Usage: python ckb.plot.py
Required files: ckb_blocks_analysis.csv
Description: This script reads a CSV file containing timestamp gaps and miner addresses for CKB blocks and plots the time gap distribution for the top 20 miners.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('ckb_blocks_analysis.csv')

df['time_gap'] = (df['time_gap'] / 1000).round()


total_blocks = len(df)


top_20_miners = df['miner_address'].value_counts().head(20)

# 计算top 20矿工的总出块数和占比
top_20_total_blocks = top_20_miners.sum()
total_ratio = (top_20_total_blocks / total_blocks) * 100


fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.ravel()  


for idx, (address, count) in enumerate(top_20_miners.items()):

    miner_data = df[df['miner_address'] == address]['time_gap']
    

    miner_ratio = (count / total_blocks) * 100
    

    axes[idx].hist(miner_data, bins=50, range=(0, 50), edgecolor='black')
    axes[idx].set_title(f'Miner: {address[:10]}...\nBlocks: {count} ({miner_ratio:.2f}%)', 
                       fontsize=8)
    axes[idx].tick_params(axis='both', which='major', labelsize=6)
    axes[idx].set_xlim(0, 50)

plt.tight_layout()

plt.suptitle(f'Time Gap Distribution for Top 20 Miners\nTotal Blocks: {total_blocks:,}, '
             f'Top 20 Miners Total: {top_20_total_blocks:,} ({total_ratio:.2f}%)', 
             y=1.02, fontsize=16)

plt.savefig('miner_time_gap_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nTop 20 miners statistics:")
print("Total blocks:", total_blocks)
print("Top 20 miners total blocks:", top_20_total_blocks)
print("Top 20 miners ratio: {:.2f}%".format(total_ratio))
print("\nIndividual miners statistics:")
for address, count in top_20_miners.items():
    ratio = (count / total_blocks) * 100
    print(f"{address}: {count} blocks ({ratio:.2f}%)")