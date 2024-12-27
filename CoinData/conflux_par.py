import subprocess
import json
import csv
from typing import Optional, Dict, List
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import os

def get_block_data(epoch: int) -> Optional[Dict]:
    """Get block data using conflux rpc command."""
    try:
        hex_epoch = hex(epoch)
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

def process_chunk(chunk_data: tuple) -> List[Dict]:
    """Process a chunk of epochs and return block data."""
    start, end = chunk_data
    blocks_data = []
    previous_timestamp = None

    for epoch in range(start, end):
        block = get_block_data(epoch)
        if block:
            current_timestamp = int(block.get('timestamp', '0x0'), 16)
            
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
            previous_timestamp = current_timestamp

    return blocks_data

def write_to_csv(data: List[Dict], file_path: str) -> None:
    """Write block data to CSV file with file locking."""
    if not data:
        return
        
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, mode=mode, newline='') as file:
        writer = csv.DictWriter(file, 
                              fieldnames=['epoch', 'block_number', 'timestamp', 
                                        'timestamp_gap', 'miner_address'])
        if mode == 'w':
            writer.writeheader()
        writer.writerows(data)

def fetch_blocks_parallel(start_epoch: int, end_epoch: int, chunk_size: int = 1000) -> None:
    """Fetch blocks data in parallel."""
    # Create chunks of epochs to process
    chunks = [(i, min(i + chunk_size, end_epoch)) 
             for i in range(start_epoch, end_epoch, chunk_size)]
    
    # Use number of CPU cores minus 1 to avoid overloading
    num_processes = max(1, cpu_count() - 1)
    
    output_file = 'conflux_mining_data_.csv'
    
    print(f"Starting parallel processing with {num_processes} processes")
    
    with Pool(processes=num_processes) as pool:
        for i, blocks_data in enumerate(pool.imap_unordered(process_chunk, chunks)):
            if blocks_data:
                write_to_csv(blocks_data, output_file)
                print(f"Processed chunk {i+1}/{len(chunks)}")

def main():
    """Main function to execute the script."""
    try:
        # Test if conflux rpc is available
        subprocess.run(['/data2/sureyang/Blockchain/conflux-rust/target/release/conflux', 'rpc', '--version'], 
                     capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Please make sure Conflux is installed and the command is being run from the correct directory.")
        return

    start_epoch = 0
    end_epoch = 109920000
    chunk_size = 1000  # Adjust this value based on your system's capabilities

    start_time = time.time()
    fetch_blocks_parallel(start_epoch, end_epoch, chunk_size)
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()