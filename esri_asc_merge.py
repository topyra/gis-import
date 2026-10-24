import os
import sys
import glob
import numpy as np

def read_asc_header(filepath):
    with open(filepath, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip()
            if not line: continue
            parts = line.split()
            key, value = parts[0], parts[1]
            header[key] = int(value) if value.lstrip('-').isdigit() else float(value)
        return header

def main(input_dir, output_file):
    asc_files = sorted(glob.glob(os.path.join(input_dir, '*.asc')))
    if not asc_files:
        print("No .asc files found.")
        return

    # 1. Read all headers to find global bounds
    print(f"Analyzing {len(asc_files)} files...")
    headers = []
    for f in asc_files:
        headers.append(read_asc_header(f))
        
    cellsize = headers[0]['cellsize']
    nodata = headers[0]['nodata_value']
    
    min_x_center = min(h['xllcenter'] for h in headers)
    max_x_center = max(h['xllcenter'] + (h['ncols'] - 1) * cellsize for h in headers)
    min_y_center = min(h['yllcenter'] for h in headers)
    max_y_center = max(h['yllcenter'] + (h['nrows'] - 1) * cellsize for h in headers)
    
    global_ncols = int(round((max_x_center - min_x_center) / cellsize)) + 1
    global_nrows = int(round((max_y_center - min_y_center) / cellsize)) + 1
    
    print(f"Global Grid Size: {global_ncols} cols x {global_nrows} rows")
    
    # 2. Initialize global canvas
    # We use float64 to safely accumulate sums for averaging overlapping areas
    global_sum = np.zeros((global_nrows, global_ncols), dtype=np.float64)
    global_count = np.zeros((global_nrows, global_ncols), dtype=np.int32)
    
    # 3. Load and paste each tile
    for f, h in zip(asc_files, headers):
        print(f"Processing {os.path.basename(f)}...")
        
        # Read data
        with open(f, 'r') as fp:
            for _ in range(6): fp.readline()
            data = []
            for line in fp:
                if line.strip():
                    data.extend(map(float, line.split()))
        
        tile_data = np.array(data, dtype=np.float32).reshape(h['nrows'], h['ncols'])
        
        # Calculate tile position on global canvas
        col_start = int(round((h['xllcenter'] - min_x_center) / cellsize))
        row_start = int(round((max_y_center - (h['yllcenter'] + (h['nrows'] - 1) * cellsize)) / cellsize))
        
        # Create mask of valid data
        mask = tile_data != nodata
        
        # Add valid data to global sum and increment count
        global_sum[row_start:row_start+h['nrows'], col_start:col_start+h['ncols']][mask] += tile_data[mask]
        global_count[row_start:row_start+h['nrows'], col_start:col_start+h['ncols']][mask] += 1
        
    # 4. Calculate average for overlapping areas
    print("Calculating averages for overlapping regions...")
    merged_data = np.full((global_nrows, global_ncols), nodata, dtype=np.float32)
    
    valid_mask = global_count > 0
    merged_data[valid_mask] = global_sum[valid_mask] / global_count[valid_mask]
    
    overlap_pixels = np.sum(global_count > 1)
    print(f"Found {overlap_pixels} overlapping pixels that were seamlessly averaged.")
    
    # 5. Write output
    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(f"ncols {global_ncols}\n")
        f.write(f"nrows {global_nrows}\n")
        f.write(f"xllcenter {min_x_center}\n")
        f.write(f"yllcenter {min_y_center}\n")
        f.write(f"cellsize {cellsize}\n")
        f.write(f"nodata_value {int(nodata) if nodata == int(nodata) else nodata}\n")
        
        for r in range(global_nrows):
            row_str = []
            for val in merged_data[r]:
                if val == nodata:
                    row_str.append(str(int(nodata)) if nodata == int(nodata) else str(nodata))
                else:
                    row_str.append(f"{val:.2f}")
            f.write(" ".join(row_str) + "\n")
            
    print("Done!")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python esri_asc_merge.py <input_dir> <output_file.asc>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])