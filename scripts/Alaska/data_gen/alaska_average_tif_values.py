import os
import rasterio
import pandas as pd
import numpy as np

def calculate_tif_average(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first band
        valid_data = data[data != src.nodata]  # Exclude nodata values

        if valid_data.size > 0:
            avg_value = valid_data.mean()
            print(f"Calculated average for {file_path}: {avg_value}")
        else:
            avg_value = np.nan  # Assign NaN if no valid data
            print(f"No valid data found in {file_path}. Assigned NaN for average.")
            
    return avg_value

def process_tif_files(root_folder):
    results = []
    print("Starting to process directories...")

    for subdir, dirs, files in os.walk(root_folder):
        for directory in dirs:
            huc_code = directory
            data_dir = os.path.join(subdir, directory)
            print(f"Processing directory: {data_dir}")

            if "AK_50_Dataset" in data_dir:
                file_names = [
                    f"curvature_{huc_code}.tif",
                    f"swm1_{huc_code}.tif",
                    f"swm2_{huc_code}.tif",
                    f"ori_{huc_code}.tif",
                    f"dsm_{huc_code}.tif",
                    f"geomorph_{huc_code}.tif",
                    f"pos_openness_{huc_code}.tif",
                    f"tpi_11_{huc_code}.tif",
                    f"twi_{huc_code}.tif"
                ]
            else:
                file_names = [
                    f"curvature_{huc_code}.tif",
                    f"swm1_{huc_code}.tif",
                    f"swm2_{huc_code}.tif",
                    f"ori_ave_{huc_code}.tif",
                    f"dsm_{huc_code}.tif",
                    f"geomorph_{huc_code}.tif",
                    f"pos_openness_{huc_code}.tif",
                    f"tpi_11_{huc_code}.tif",
                    f"twi_{huc_code}.tif"
                ]

            row = {'huc_code': huc_code}
            for file_name in file_names:
                prefix = file_name.split('_')[0]  # Extract prefix before "_"
                file_path = os.path.join(data_dir, file_name)
                if os.path.exists(file_path):
                    avg_value = calculate_tif_average(file_path)
                    row[f'avg_{prefix}'] = avg_value  # Use prefix for column name
                else:
                    row[f'avg_{prefix}'] = None
                    print(f"File not found: {file_path}")
            results.append(row)

    print("Finished processing all directories.")
    return results

def main():
    root_folder = '/projects/bcrm/nathanj/TIFF_data/Alaska'
    print("Starting the script...")
    results = process_tif_files(root_folder)
    df = pd.DataFrame(results)
    df.to_csv('average_values.csv', index=False)
    print("CSV file has been created successfully.")

if __name__ == '__main__':
    main()
