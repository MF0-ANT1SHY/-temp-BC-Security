import requests
import csv
from concurrent.futures import ThreadPoolExecutor


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
        return {
            "Number": int(block['number'], 16),
            "Timestamp": int(block['timestamp'], 16),
            "Miner": block['miner'],
            "Difficulty": int(block['difficulty'], 16),
            "totDifficulty": int(block['totalDifficulty'], 16)
        }
    return None


def fetch_blocks(start_block, end_block):
    blocks_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_block, block_number): block_number for block_number in
                   range(start_block, end_block + 1)}

        for future in futures:
            block_info = future.result()
            if block_info:
                blocks_data.append(block_info)
                # 每次添加数据时检查当前最高区块
                print(f"Fetched block number: {block_info['Number']}")

    # 根据 Number 排序
    blocks_data.sort(key=lambda x: x["Number"])
    write_to_csv(blocks_data)


def write_to_csv(data):
    with open('eth_blocks_raw_2.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        print(f"All data written to CSV, highest block number: {data[-1]['Number']}")  # 输出最高区块


# 示例用法
fetch_blocks(0, 15537393)
