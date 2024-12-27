import requests
import csv

def get_block_by_number(session, block_number):
    url = "http://127.0.0.1:8114/"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "get_block_by_number",
        "params": [hex(block_number)]
    }
    headers = {'content-type': 'application/json'}

    try:
        response = session.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get('result', None)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching block at height {block_number}: {e}")
        return None

def get_miner_address(block):
    try:
        if block and block.get('transactions'):
            # Get the cellbase (first transaction)
            cellbase = block['transactions'][0]
            # Get the lock script of first output
            if cellbase.get('outputs') and len(cellbase['outputs']) > 0:
                lock_script = cellbase['outputs'][0]['lock']
                # Return the args as miner identifier
                return lock_script.get('args', 'unknown')
        return 'unknown'
    except Exception as e:
        print(f"Error extracting miner address: {e}")
        return 'unknown'

def get_block_data(session, block_number):
    current_block = get_block_by_number(session, block_number)
    if current_block:
        # Get parent block for timestamp comparison
        parent_block = None
        if block_number > 0:
            parent_block = get_block_by_number(session, block_number - 1)

        current_timestamp = int(current_block['header']['timestamp'], 16)
        parent_timestamp = int(parent_block['header']['timestamp'], 16) if parent_block else current_timestamp
        time_gap = current_timestamp - parent_timestamp

        block_info = {
            'height': block_number,
            'timestamp': current_timestamp,
            'time_gap': time_gap,
            'miner_address': get_miner_address(current_block)
        }
        return block_info
    return None

def fetch_blocks(start_height, end_height):
    blocks_data = []

    with requests.Session() as session:
        # Get the current tip if end_height is not specified
        if end_height is None:
            tip_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "get_tip_block_number",
                "params": []
            }
            try:
                response = session.post("http://127.0.0.1:8114/", json=tip_payload, headers={'content-type': 'application/json'})
                response.raise_for_status()
                end_height = int(response.json().get('result', '0x0'), 16)
            except requests.exceptions.RequestException as e:
                print(f"Error fetching tip block number: {e}")
                return

        for height in range(start_height, end_height + 1):
            block_info = get_block_data(session, height)
            if block_info:
                blocks_data.append(block_info)

            # Write to CSV every 100 blocks
            if len(blocks_data) % 10000 == 0:
                print(f"Processing block number {height}")
                write_to_csv(blocks_data)
                blocks_data = []

        # Write remaining blocks
        if blocks_data:
            write_to_csv(blocks_data)

def write_to_csv(data):
    filename = 'ckb_blocks_analysis.csv'
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['height', 'timestamp', 'time_gap', 'miner_address'])
        if file.tell() == 0:  # Write header if file is empty
            writer.writeheader()
        writer.writerows(data)

# Fetch from genesis block to current tip
fetch_blocks(0, None)