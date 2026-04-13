import os
from tqdm import tqdm

def create_partial_split(split_name):
    original_path = f"datasets/shapenet_split/{split_name}.txt"
    partial_dir = "datasets/shapenet_split_partial"
    partial_path = f"{partial_dir}/{split_name}.txt"
    
    data_path = "gs_data/shapesplat/shapesplat_ply"
    
    if not os.path.exists(original_path):
        print(f"Original split {original_path} not found.")
        return

    os.makedirs(partial_dir, exist_ok=True)
    
    print(f"Processing {split_name}...")
    valid_entries = []
    
    with open(original_path, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        filename = line.strip()
        if not filename:
            continue
            
        # Check if file exists in data_path
        full_path = os.path.join(data_path, filename)
        if os.path.exists(full_path):
            valid_entries.append(filename)
            
    with open(partial_path, 'w') as f:
        for entry in valid_entries:
            f.write(f"{entry}\n")
            
    print(f"Created {partial_path} with {len(valid_entries)}/{len(lines)} entries.")

if __name__ == "__main__":
    create_partial_split("train")
    create_partial_split("test")
