import subprocess
import json
import csv
from typing import Optional, Dict, List

def get_block_data(block_height: int) -> Optional[Dict]:
    """Get block data using zcash-cli command."""
    try:
        # First get block hash
        hash_result = subprocess.run(
            ['/data1/sureyang/Blockchains/zcash/src/zcash-cli', '-datadir=/data1/sureyang/Blockchains/.zcash', 
             '-conf=/data1/sureyang/Blockchains/.zcash/zcash.conf', 'getblockhash', str(block_height)],
            capture_output=True,
            text=True,
            check=True
        )
        block_hash = hash_result.stdout.strip()

        # Then get block data using the hash
        block_result = subprocess.run(
            ['/data1/sureyang/Blockchains/zcash/src/zcash-cli', '-datadir=/data1/sureyang/Blockchains/.zcash', 
             '-conf=/data1/sureyang/Blockchains/.zcash/zcash.conf', 'getblock', block_hash, '2'],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(block_result.stdout)
    except Exception as e:
        print(f"Error getting block {block_height}: {e}")
        return None

def get_miner_address(block_data: Dict) -> Optional[str]:
    """Get miner address from coinbase transaction."""
    try:
        # Get the coinbase transaction (first transaction)
        coinbase_tx = block_data.get('tx', [])[0]
        if not coinbase_tx:
            return None

        # Look for the miner address in the vout
        for vout in coinbase_tx.get('vout', []):
            addresses = vout.get('scriptPubKey', {}).get('addresses', [])
            if addresses:
                return addresses[0]  # Return the first address
        return None
    except Exception as e:
        print(f"Error getting miner address: {e}")
        return None

def fetch_blocks(start_height: int, end_height: int) -> None:
    """Fetch timestamp gaps and miner addresses for blocks."""
    blocks_data: List[Dict] = []
    previous_timestamp = None

    for height in range(start_height, end_height + 1):
        block = get_block_data(height)
        if block:
            current_timestamp = block.get('time', 0)
            
            # Calculate timestamp gap (will be None for first block)
            timestamp_gap = None
            if previous_timestamp is not None:
                timestamp_gap = current_timestamp - previous_timestamp
            
            # Get miner address from coinbase transaction
            miner_address = get_miner_address(block)
            
            block_info = {
                'height': height,
                'timestamp_gap': timestamp_gap,
                'miner_address': miner_address
            }
            blocks_data.append(block_info)
            
            # Update previous timestamp for next iteration
            previous_timestamp = current_timestamp

        # Write to CSV every 100 blocks
        if len(blocks_data) % 10000 == 0:
            print(f"Processing block {height}")
            write_to_csv(blocks_data)
            blocks_data = []

    # Write any remaining blocks
    if blocks_data:
        write_to_csv(blocks_data)

def write_to_csv(data: List[Dict]) -> None:
    """Write block data to CSV file."""
    with open('zcash_mining_data.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['height', 'timestamp_gap', 'miner_address'])
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(data)

def main():
    """Main function to execute the script."""
    try:
        # Test if zcash-cli is available
        subprocess.run(['/data1/sureyang/Blockchains/zcash/src/zcash-cli', '-datadir=/data1/sureyang/Blockchains/.zcash', 
                       '-conf=/data1/sureyang/Blockchains/.zcash/zcash.conf', 'getinfo'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Please make sure Zcash is installed and the daemon is running.")
        return

    # Adjust end_height based on current blockchain height
    fetch_blocks(254299, 2736297)

if __name__ == "__main__":
    main()

# /data1/sureyang/Blockchains/zcash/src/zcash-cli -datadir=/data1/sureyang/Blockchains/.zcash -conf=/data1/sureyang/Blockchains/.zcash/zcash.conf getrawtransaction "5f2702708af1d8727ad3f0da3ba74de14019232499c0324ddce236cf97e32548" 1
# /data1/sureyang/Blockchains/zcash/src/zcash-cli -datadir=/data1/sureyang/Blockchains/.zcash -conf=/data1/sureyang/Blockchains/.zcash/zcash.conf getinfo