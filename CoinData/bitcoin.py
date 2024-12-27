import requests
import csv


def get_block_hash(session, block_height):
    url = "http://127.0.0.1:8332/"
    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "getblockhash",
        "params": [block_height]
    }
    headers = {'content-type': 'text/plain;'}

    try:
        response = session.post(url, json=payload, headers=headers, auth=('ubuntu', 'ubuntu'))
        response.raise_for_status()
        return response.json().get('result', None)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching block hash for height {block_height}: {e}")
        return None


def get_block_data(session, block_height):
    block_hash = get_block_hash(session, block_height)
    if block_hash:
        url = "http://127.0.0.1:8332/"
        payload = {
            "jsonrpc": "1.0",
            "id": "curltest",
            "method": "getblock",
            "params": [block_hash, 1]  # verbosity set to 1
        }
        headers = {'content-type': 'text/plain;'}

        try:
            response = session.post(url, json=payload, headers=headers, auth=('ubuntu', 'ubuntu'))
            response.raise_for_status()
            return response.json().get('result', {})
        except requests.exceptions.RequestException as e:
            print(f"Error fetching block {block_hash}: {e}")
            return None
    return None


def fetch_blocks(start_height, end_height):
    blocks_data = []

    with requests.Session() as session:
        for height in range(start_height, end_height + 1):
            block = get_block_data(session, height)
            if block:
                block_info = {
                    'height': block.get('height'),
                    'time': block.get('time'),
                    'difficulty': block.get('difficulty'),
                }
                blocks_data.append(block_info)

            if len(blocks_data) % 100 == 0:
                print(f"Write to block number {height}")
                write_to_csv(blocks_data)
                blocks_data = []

        if blocks_data:
            write_to_csv(blocks_data)


def write_to_csv(data):
    with open('bitcoin_blocks_raw.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(data)


fetch_blocks(0, 862803)
