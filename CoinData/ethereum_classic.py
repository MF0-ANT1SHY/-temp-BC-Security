import requests
import csv
import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np

def fetch_block(block_number):
    url = 'http://127.0.0.1:8545'
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex(block_number), True],
        "id": 1
    }
    response = requests.post(url, json=payload)
    block = response.json().get('result')

    if block:
        timestamp = int(block['timestamp'], 16)
        block_time = datetime.datetime.fromtimestamp(timestamp)
        return {
            "Number": int(block['number'], 16),
            "Timestamp": timestamp,
            "Miner": block['miner'],
            "Difficulty": int(block['difficulty'], 16),
            "totDifficulty": int(block['totalDifficulty'], 16)
        }
    return None

def fetch_blocks(start_block, end_block, batch_size=100):
    blocks_data = []
    for i in range(start_block, end_block + 1, batch_size):
        batch_end = min(i + batch_size - 1, end_block)
        # print(f"Fetching blocks from {i} to {batch_end}...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_block, block_number): block_number for block_number in range(i, batch_end + 1)}

            for future in futures:
                block_info = future.result()
                if block_info:
                    blocks_data.append(block_info)
                    # print(f"Fetched block number: {block_info['Number']}")
        
        write_to_csv(blocks_data)
        blocks_data = []  # 清空当前批次的数据以准备下一批

def write_to_csv(data):
    with open('eth_classic_blocks_raw.csv', mode='a', newline='') as file:
        if file.tell() == 0: 
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()  # 写入表头
        else:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writerows(data)
        print(f"Data written to CSV, highest block number: {data[-1]['Number']}")

def miner_frequency():
    with open('eth_classic_blocks_raw.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        miner_counter = Counter(row['Miner'] for row in reader)

    total_blocks = 9373673 
    sorted_miners = sorted(miner_counter.items(), key=lambda x: x[1], reverse=True)

    with open('Miner_Frequency.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Miner', 'Frequency', 'Ratio'])  
        
        for miner, count in sorted_miners:
            frequency_ratio = count / total_blocks
            writer.writerow([miner, count, frequency_ratio])

def add_timestamp_diff(input_file="eth_classic_blocks_raw.csv", output_file="eth_classic_blocks_processed.csv"):
    df = pd.read_csv(input_file)
    df['diff'] = df['Timestamp'].diff()
    df.to_csv(output_file, index=False)

def draw_graph():
    original_file = 'eth_classic_blocks_processed.csv'
    frequency_file = 'Miner_Frequency.csv'
    output_folder = 'image'     

    os.makedirs(output_folder, exist_ok=True)

    original_df = pd.read_csv(original_file)
    frequency_df = pd.read_csv(frequency_file)

    for _, row in frequency_df.iterrows():
        miner_address = row['Miner']
        count = row['Frequency']
        frequency_ratio = row['Ratio']

        if count < 100:
            continue

        miner_data = original_df[original_df['Miner'] == miner_address].copy()
        miner_data = miner_data.dropna(subset=['diff'])
        time_diff_counts = miner_data['diff'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(time_diff_counts.index, time_diff_counts.values, width=1, edgecolor='black', align='center')
        plt.xlabel('Difference (second)')
        plt.ylabel('Frequency')
        plt.title(f'Miner Address: {miner_address}\n Frequency: {count}, Ratio: {frequency_ratio:.5f}')
        
        sanitized_address = miner_address.replace("/", "_").replace("\\", "_")  
        plt.savefig(os.path.join(output_folder, f'{count}.{sanitized_address}.png'))
        plt.close()

def plot_combined_time_diff_histogram(address_list, original_file='eth_classic_blocks_processed.csv'):
    original_df = pd.read_csv(original_file)
    filtered_data = original_df[original_df['Miner'].isin(address_list)].copy()
    total_count = len(filtered_data)
    total_frequency = total_count / 4777393
    filtered_data = filtered_data.dropna(subset=['diff'])
    time_diff_counts = filtered_data['diff'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(time_diff_counts.index, time_diff_counts.values, width=1.0, edgecolor='black', align='center')
    plt.xlabel('Difference (second)')
    plt.ylabel('Frequency')
    plt.title(f'Total Frequency: {total_count}, Total Ratio: {total_frequency:.5f}')
    plt.tight_layout()

    output_folder = 'image'
    plt.savefig(os.path.join(output_folder, f'{total_count}.png'))
    plt.close()

def get_uncle_block_count(block_number, node_url='http://127.0.0.1:8545'):
    try:
        print(block_number)
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getUncleCountByBlockNumber",
            "params": [hex(block_number)],
            "id": 1
        }

        headers = {
            'Content-Type': 'application/json',
        }

        # 发送 HTTP 请求到节点
        response = requests.post(node_url, data=json.dumps(payload), headers=headers)

        # 解析返回的 JSON 数据
        result = response.json()

        if 'result' in result:
            return int(result['result'], 16)  # 返回十六进制的叔块数量，转换为整数
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 0  # 如果请求失败，返回 0
    except Exception as e:
        print(f"Error in get_uncle_block_count for block {block_number}: {e}")
        return 0  # 如果发生异常，返回 0

def extract_and_check_uncle(address, original_file='eth_classic_blocks_processed.csv', output_file='output_with_uncle_info.csv', node_url='http://127.0.0.1:8545'):
    try:
        print(f"加载原始文件 {original_file}...")
        # 读取 CSV 文件
        original_df = pd.read_csv(original_file)
        print(f"文件加载完成，共 {len(original_df)} 行数据。")
        
        # 预处理地址并过滤
        address = address.strip().lower()
        original_df['Miner'] = original_df['Miner'].str.strip().str.lower()
        filtered_data = original_df[original_df['Miner'] == address]
        print(f"匹配完成，找到 {len(filtered_data)} 行数据。")
        
        if len(filtered_data) == 0:
            print("未找到符合条件的行，退出程序。")
            return
        
        # 获取每个区块的叔块数量
        print(f"查询每个区块的叔块数量...")
        filtered_data['UncleBlockCount'] = filtered_data['Number'].apply(
            lambda x: get_uncle_block_count(x, node_url=node_url)
        )
        
        # 保存结果到文件
        filtered_data.to_csv(output_file, index=False)
        print(f"结果已保存到 {output_file}")

    except Exception as e:
        print(f"发生错误：{e}")

def calculate_uncle_count_percentage(input_file='output_with_uncle_info.csv', output_file='uncle_count_percentage.csv'):
    try:
        # 读取数据
        df = pd.read_csv(input_file)
        
        # 检查 diff 和 UncleBlockCount 列是否存在
        if 'diff' not in df.columns or 'UncleBlockCount' not in df.columns:
            print("Error: 'diff' or 'UncleBlockCount' column not found in the data.")
            return
        
        # 计算每个 diff 值中 UncleBlockCount 为 1 的数量
        uncle_count_1 = df[df['UncleBlockCount'] == 1].groupby('diff').size()
        
        # 计算每个 diff 值的总行数
        total_count = df.groupby('diff').size()
        
        # 计算占比
        percentage = (uncle_count_1 / total_count).fillna(0) * 100
        
        # 将结果合并为一个 DataFrame
        result = pd.DataFrame({
            'diff': percentage.index,
            'UncleBlockCount_1_Percentage': percentage.values
        })
        
        # 将结果保存到 CSV 文件
        result.to_csv(output_file, index=False)
        
        print(f"结果已保存到 {output_file}")
    
    except Exception as e:
        print(f"发生错误: {e}")

def filter_and_plot_histogram(input_file='uncle_count_percentage.csv', output_image='uncle_count_percentage_histogram.png'):
    try:
        # 读取数据
        df = pd.read_csv(input_file)
        
        # 检查 diff 和 UncleBlockCount_1_Percentage 列是否存在
        if 'diff' not in df.columns or 'UncleBlockCount_1_Percentage' not in df.columns:
            print("Error: 'diff' or 'UncleBlockCount_1_Percentage' column not found in the data.")
            return
        
        # 提取 diff 和 UncleBlockCount_1_Percentage 列
        diff_values = df['diff']
        percentage_values = df['UncleBlockCount_1_Percentage']
        
        # 创建图形
        plt.figure(figsize=(40, 6))
        
        # 绘制条形图
        plt.bar(diff_values, percentage_values, width=1.0, edgecolor='black', alpha=0.7)
        
        # 设置标题和标签
        plt.title('UncleBlockCount为1占比 vs diff')
        plt.xlabel('diff')
        plt.ylabel('UncleBlockCount_1_Percentage')
        
        # 调整 X 轴刻度
        plt.xticks(np.arange(min(diff_values), max(diff_values) + 1, 1))  # 确保间隔为1
        
        # 保存图片
        plt.savefig(output_image)
        plt.show()
        
        print(f"直方图已保存为 {output_image}")
    
    except Exception as e:
        print(f"发生错误: {e}")


# fetch_blocks(11700000, 21073673)
# miner_frequency()
# add_timestamp_diff()
# draw_graph()
# address_list = [
#     '0xf20b338752976878754518183873602902360704',
#     '0xbcc817f057950b0df41206c5d7125e6225cae18e',
#     '0x99c85bb64564d9ef9a99621301f22c9993cb89e3',
#     '0x829bd824b016326a401d083b33d092293333a830'
# ]
# plot_combined_time_diff_histogram(address_list)

# extract_and_check_uncle(
#     address="0x60f814acce2b2129707228c61065696bcc3e8b9f",
#     original_file="eth_classic_blocks_processed.csv",
#     output_file="output_with_uncle_info.csv",
#     node_url="http://127.0.0.1:8545"  # 本地节点地址
# )

# calculate_uncle_count_percentage(
#     input_file="output_with_uncle_info.csv",
#     output_file="uncle_count_percentage.csv"
# )

filter_and_plot_histogram(
    input_file="uncle_count_percentage.csv",
    output_image="0x60f814acce2b2129707228c61065696bcc3e8b9f.png"
)
