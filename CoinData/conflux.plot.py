import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Create directory if it doesn't exist
os.makedirs('./plot/conflux', exist_ok=True)

# Read the CSV file
df = pd.read_csv('conflux_mining_data.csv')

# Calculate miner statistics 
miner_stats = df['miner_address'].value_counts()
total_blocks = len(df)
miner_fractions = miner_stats / total_blocks

# Get top miners (you can adjust the number)
top_miners = miner_stats.head(20)
top_miner_fractions = miner_fractions.head(20)

# Calculate global min and max for time gaps (excluding extreme outliers)
global_min = df['time_gap'].min()
global_max = df['time_gap'].quantile(0.99)  # Using 99th percentile to exclude extreme outliers

# Generate individual plots for each miner
for miner in top_miners.index:
    miner_data = df[df['miner_address'] == miner]
    
    # Create figure
    plt.figure(figsize=(6, 6))
    
    # Plot histogram of timestamp gaps using global range
    bins = np.linspace(global_min, global_max, 50)
    plt.hist(miner_data['time_gap'], bins=bins, alpha=0.6, density=True)
    plt.title('Timestamp Gap Distribution')
    plt.xlabel('Timestamp Gap')
    plt.ylabel('Density')
    
    # Set x-axis limits using global range
    plt.xlim(global_min, global_max)
    
    # Add KDE plot
    sns.kdeplot(data=miner_data['time_gap'], color='red', label='KDE')
    plt.legend()
    
    # Add statistics as text
    miner_blocks = len(miner_data)
    miner_fraction = miner_blocks / total_blocks
    stats_text = f'Total Blocks: {miner_blocks}\nFraction: {miner_fraction:.2%}\n'
    
    plt.figtext(0.15, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    # the address format is: CFX:TYPE.USER:<address>
    # miner_short = miner[:10]  # Use first 10 characters of miner address for filename
    miner_short = miner.split(':')[2]
    plt.savefig(f'./plot/conflux/miner_{miner_short}_statistics.png', 
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
plt.savefig('./plot/conflux/overall_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Create combined KDE plot for top N miners
N = 5  # Number of top miners to include in combined plot
plt.figure(figsize=(12, 8))

# Create color palette
colors = plt.cm.rainbow(np.linspace(0, 1, N))

# Plot KDE for each top miner
for i, (miner, color) in enumerate(zip(top_miners.index[:N], colors)):
    miner_data = df[df['miner_address'] == miner]
    sns.kdeplot(data=miner_data['time_gap'], 
                color=color, 
                label=f'Miner {miner[:10]}... ({miner_stats[miner]} blocks)',
                alpha=0.7)

plt.title('Timestamp Gap Distribution Comparison (Top Miners)')
plt.xlabel('Timestamp Gap')
plt.ylabel('Density')
plt.xlim(global_min, global_max)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('./plot/conflux/combined_distribution.png', 
            dpi=300, 
            bbox_inches='tight')
plt.close()

# Save overall statistics to a text file
with open('./plot/conflux/overall_statistics.txt', 'w') as f:
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