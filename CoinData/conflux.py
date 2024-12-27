import subprocess
import json
import csv
from typing import Optional, Dict, List
import time

def get_block_data(epoch: int) -> Optional[Dict]:
    """Get block data using conflux rpc command."""
    try:
        hex_epoch = hex(epoch)  # Convert integer to hex string
        result = subprocess.run(
            ['/data2/sureyang/Blockchain/conflux-rust/target/release/conflux', 'rpc', 'block-by-epoch', '--epoch', hex_epoch],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting block for epoch {hex_epoch}: {e}")
        return None

def fetch_blocks(start_epoch: int, end_epoch: int) -> None:
    """Fetch timestamp gaps and miner addresses for blocks."""
    blocks_data: List[Dict] = []
    previous_timestamp = None

    for epoch in range(start_epoch, end_epoch + 1):
        block = get_block_data(epoch)
        if block:
            # Convert hex timestamp to decimal
            current_timestamp = int(block.get('timestamp', '0x0'), 16)
            
            # Calculate timestamp gap (will be None for first block)
            timestamp_gap = None
            if previous_timestamp is not None:
                timestamp_gap = current_timestamp - previous_timestamp
            
            block_info = {
                'epoch': epoch,
                'block_number': int(block.get('blockNumber', '0x0'), 16),
                'timestamp': current_timestamp,
                'timestamp_gap': timestamp_gap,
                'miner_address': block.get('miner', '')
            }
            blocks_data.append(block_info)
            
            # Update previous timestamp for next iteration
            previous_timestamp = current_timestamp

        # Write to CSV every 100 blocks
        if len(blocks_data) % 1000 == 0:
            print(f"Processing epoch {epoch}")
            write_to_csv(blocks_data)
            blocks_data = []

    # Write any remaining blocks
    if blocks_data:
        write_to_csv(blocks_data)

def write_to_csv(data: List[Dict]) -> None:
    """Write block data to CSV file."""
    with open('conflux_mining_data.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, 
                              fieldnames=['epoch', 'block_number', 'timestamp', 
                                        'timestamp_gap', 'miner_address'])
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(data)

def main():
    """Main function to execute the script."""
    try:
        # Test if conflux rpc is available
        subprocess.run(['/data2/sureyang/Blockchain/conflux-rust/target/release/conflux', 'rpc', '--version'], 
                     capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Please make sure Conflux is installed and the command is being run from the correct directory.")
        return

    # You can adjust these values as needed
    start_epoch = 15000
    end_epoch = 109920000

    fetch_blocks(start_epoch, end_epoch)

if __name__ == "__main__":
    main()