import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

top_value = 20
os.makedirs('./plot/zcash', exist_ok=True) # Create directory if it doesn't exist
df = pd.read_csv('zcash_mining_data.csv') # Read the CSV file
global_min = -100 # Calculate global min and max for time gaps (excluding extreme outliers)
global_max = df['time_gap'].quantile(0.99)  # Using 99th percentile to exclude extreme outliers

# df = df[df['height'] <= 653600] # before
df = df[df['height'] >= 653600] # after

# Calculate miner statistics 
miner_stats = df['miner_address'].value_counts()
total_blocks = len(df)
miner_fractions = miner_stats / total_blocks

# Get top miners (you can adjust the number)
top_miners = miner_stats.head(top_value)
top_miner_fractions = miner_fractions.head(top_value)

# Add new parameter for sliding window size
lambda_size = 10

# Modify the individual plots for each miner to include both distributions
for miner in top_miners.index:
    miner_data = df[df['miner_address'] == miner]
    
    # Calculate rolling average of time gaps
    rolling_gaps = miner_data['time_gap'].rolling(window=lambda_size).mean()
    rolling_gaps = rolling_gaps.dropna()  # Remove NaN values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    
    # Plot 1: Original distribution
    bins = np.linspace(global_min, global_max, 50)
    ax1.hist(miner_data['time_gap'], bins=bins, alpha=0.6, density=True)
    ax1.set_title('Individual Block Timestamp Gap Distribution')
    ax1.set_xlabel('Timestamp Gap')
    ax1.set_ylabel('Density')
    ax1.set_xlim(global_min, global_max)
    sns.kdeplot(data=miner_data['time_gap'], color='red', label='KDE', ax=ax1)
    ax1.legend()
    
    # Plot 2: Rolling average distribution
    ax2.hist(rolling_gaps, bins=bins, alpha=0.6, density=True)
    ax2.set_title(f'Rolling Average ({lambda_size} blocks) Timestamp Gap Distribution')
    ax2.set_xlabel('Average Timestamp Gap')
    ax2.set_ylabel('Density')
    ax2.set_xlim(global_min, global_max)
    sns.kdeplot(data=rolling_gaps, color='red', label='KDE', ax=ax2)
    ax2.legend()
    
    # Add statistics as text
    miner_blocks = len(miner_data)
    miner_fraction = miner_blocks / total_blocks
    stats_text = f'Total Blocks: {miner_blocks}\nFraction: {miner_fraction:.2%}\n'
    
    plt.figtext(0.15, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    miner_short = miner[:10]
    plt.savefig(f'./plot/zcash/miner_{miner_short}_statistics.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

# Modify the combined KDE plot to include both distributions
plt.figure(figsize=(12, 12))
colors = plt.cm.rainbow(np.linspace(0, 1, top_value))

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

# Plot original KDEs
for i, (miner, color) in enumerate(zip(top_miners.index[:top_value], colors)):
    miner_data = df[df['miner_address'] == miner]
    sns.kdeplot(data=miner_data['time_gap'], 
                color=color, 
                label=f'Miner {miner[:10]}... ({miner_stats[miner]} blocks)',
                alpha=0.7,
                ax=ax1)

ax1.set_title('Individual Block Timestamp Gap Distribution Comparison')
ax1.set_xlabel('Timestamp Gap')
ax1.set_ylabel('Density')
ax1.set_xlim(global_min, global_max)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot rolling average KDEs
for i, (miner, color) in enumerate(zip(top_miners.index[:top_value], colors)):
    miner_data = df[df['miner_address'] == miner]
    rolling_gaps = miner_data['time_gap'].rolling(window=lambda_size).mean().dropna()
    sns.kdeplot(data=rolling_gaps, 
                color=color, 
                label=f'Miner {miner[:10]}... ({miner_stats[miner]} blocks)',
                alpha=0.7,
                ax=ax2)

ax2.set_title(f'Rolling Average ({lambda_size} blocks) Timestamp Gap Distribution Comparison')
ax2.set_xlabel('Average Timestamp Gap')
ax2.set_ylabel('Density')
ax2.set_xlim(global_min, global_max)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('./plot/zcash/combined_distribution.png', 
            dpi=300, 
            bbox_inches='tight')
plt.close()

# Create an overall summary plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(top_miner_fractions)), top_miner_fractions.values)
plt.title('Block Share of Top Miners')  
plt.xlabel('Miner')
plt.ylabel('Fraction of Total Blocks')
plt.xticks(range(len(top_miner_fractions)),
          [f'{addr[:10]}...\n{fraction:.2%}' 
           for addr, fraction in zip(top_miner_fractions.index, top_miner_fractions.values)],
          rotation=45)
plt.tight_layout()
plt.savefig('./plot/zcash/overall_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Save overall statistics to a text file
with open('./plot/zcash/overall_statistics.txt', 'w') as f:
    f.write("Overall Mining Statistics:\n")
    f.write("-" * 50 + "\n")
    f.write(f"Total number of blocks: {total_blocks}\n")
    f.write("\nTop miners by block count:\n")
    for miner, count in top_miners.items():
        f.write(f"\nMiner: {miner}\n")
        f.write(f"Blocks mined: {count}\n")
        f.write(f"Percentage: {count/total_blocks:.2%}\n")
        miner_data = df[df['miner_address'] == miner]
        f.write(f"Mean gap: {miner_data['time_gap'].mean():.2f}\n")
        f.write(f"Median gap: {miner_data['time_gap'].median():.2f}\n")
        f.write(f"Std gap: {miner_data['time_gap'].std():.2f}\n")