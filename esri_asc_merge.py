import os
import sys
import glob
import numpy as np

def read_asc(filepath):
    """Read an ESRI ASCII grid file and return header and data."""
    with open(filepath, 'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid header line in {filepath}: {line}")
            key, value = parts
            try:
                header[key] = int(value)
            except ValueError:
                header[key] = float(value)
        
        # Read data
        data = []
        for line in f:
            if line.strip():
                data.extend(map(float, line.split()))
        
        nrows = int(header['nrows'])
        ncols = int(header['ncols'])
        if len(data) != nrows * ncols:
            raise ValueError(f"Data size mismatch in {filepath}: expected {nrows * ncols}, got {len(data)}")
        
        data = np.array(data, dtype=np.float32).reshape(nrows, ncols)
        return header, data

def main(input_dir, output_file):
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory.")
        sys.exit(1)
    
    asc_files = glob.glob(os.path.join(input_dir, '*.asc'))
    if not asc_files:
        print(f"No .asc files found in {input_dir}.")
        sys.exit(1)
    
    print(f"Found {len(asc_files)} .asc files.")
    
    # Read all files
    tiles = {}
    first_header = None
    for f in asc_files:
        print(f"Reading {os.path.basename(f)}...")
        header, data = read_asc(f)
        if first_header is None:
            first_header = header
        
        # Round coordinates to avoid floating-point matching issues
        xll = round(header['xllcenter'], 4)
        yll = round(header['yllcenter'], 4)
        tiles[(xll, yll)] = (header, data)
    
    # Get parameters from first header
    ncols = int(first_header['ncols'])
    nrows = int(first_header['nrows'])
    cellsize = first_header['cellsize']
    nodata = first_header['nodata_value']
    
    tile_width = ncols * cellsize
    tile_height = nrows * cellsize
    
    # Find bounding box
    xlls = [coord[0] for coord in tiles.keys()]
    ylls = [coord[1] for coord in tiles.keys()]
    
    min_xll = min(xlls)
    max_xll = max(xlls)
    min_yll = min(ylls)
    max_yll = max(ylls)
    
    # Calculate grid dimensions
    num_tiles_x = int(round((max_xll - min_xll) / tile_width)) + 1
    num_tiles_y = int(round((max_yll - min_yll) / tile_height)) + 1
    
    print(f"Grid size: {num_tiles_x} x {num_tiles_y} tiles")
    print(f"Total size: {num_tiles_x * ncols} x {num_tiles_y * nrows} cells")
    
    # Generate expected coordinates
    expected_xlls = [round(min_xll + i * tile_width, 4) for i in range(num_tiles_x)]
    expected_ylls = [round(min_yll + i * tile_height, 4) for i in range(num_tiles_y)]
    
    # Create merged grid filled with nodata
    total_ncols = num_tiles_x * ncols
    total_nrows = num_tiles_y * nrows
    merged_data = np.full((total_nrows, total_ncols), nodata, dtype=np.float32)
    
    # Fill in tiles
    filled_count = 0
    for i, xll in enumerate(expected_xlls):
        for j, yll in enumerate(expected_ylls):
            # j=0 is min_yll (bottom), map to bottom rows in output
            row_idx = (num_tiles_y - 1 - j) * nrows
            col_idx = i * ncols
            
            if (xll, yll) in tiles:
                _, data = tiles[(xll, yll)]
                merged_data[row_idx:row_idx + nrows, col_idx:col_idx + ncols] = data
                filled_count += 1
    
    print(f"Filled {filled_count} out of {num_tiles_x * num_tiles_y} tiles.")
    
    # Write output
    print(f"Writing output to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(f"ncols {total_ncols}\n")
        f.write(f"nrows {total_nrows}\n")
        f.write(f"xllcenter {min_xll}\n")
        f.write(f"yllcenter {min_yll}\n")
        f.write(f"cellsize {cellsize}\n")
        f.write(f"nodata_value {nodata}\n")
        
        # Write data with appropriate formatting
        for row in merged_data:
            formatted = []
            for val in row:
                if val == nodata:
                    formatted.append(str(int(nodata)))
                else:
                    formatted.append(f"{val:.2f}")
            f.write(' '.join(formatted) + '\n')
    
    print("Done.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python merge_asc.py <input_dir> <output_file>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    main(input_dir, output_file)