import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def process_nc_file(file_path, output_dir, time_step=0, component_index=0):
    """
    Process a single .nc file to extract the 'VEL' variable and save it as an image.
    """
    try:
        # Load the NetCDF file
        nc_data = netCDF4.Dataset(file_path)
        
        # Check if 'VEL' is present in the variables
        if 'VEL' not in nc_data.variables:
            print(f"Variable 'VEL' not found in {file_path}")
            return
        
        # Extract the VEL variable
        vel_data = nc_data.variables['VEL'][:]
        
        # Check the shape of VEL (should be 4D with the last dimension of size 2)
        if vel_data.ndim != 4 or vel_data.shape[-1] != 2:
            print(f"Unsupported data shape for 'VEL' in {file_path}: {vel_data.shape}")
            return
        
        # Extract a 2D slice (time step and component)
        image_data = vel_data[time_step, :, :, component_index]
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the output file path with .jpg extension
        filename = os.path.splitext(os.path.basename(file_path))[0] + ".jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Plot and save the image
        plt.figure(figsize=(8, 6))
        plt.imshow(image_data, cmap='viridis')
        plt.colorbar()
        plt.title(f"VEL (Time step: {time_step}, Component: {component_index})")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved image: {output_path}")
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    finally:
        nc_data.close()

def select_random_files(root_dir, category):
    """
    Select 50% of the available .nc files from a given category ('Test' or 'Train')
    within subdirectories labeled '2022'.
    """
    files_list = []
    
    # Walk through the directory to find .nc files in '2022' folders
    for subdir, _, files in os.walk(root_dir):
        if category in subdir and '2022' in subdir:
            for file in files:
                if file.endswith('.nc'):
                    files_list.append(os.path.join(subdir, file))
    
    # Randomly select 50% of the files
    selected_count = max(1, len(files_list) // 2)
    print(f"Selecting {selected_count} files out of {len(files_list)} for category '{category}'")
    selected_files = random.sample(files_list, selected_count) if files_list else []
    
    return selected_files

def process_directory(root_dir):
    """
    Process the Test and Train datasets by randomly selecting 50% of files 
    from the '2022' subdirectories and converting them to images.
    """
    # Select 50% of random files from Test/2022 and Train/2022
    test_files = select_random_files(root_dir, 'Test')
    train_files = select_random_files(root_dir, 'Train')
    
    # Process the selected Test files
    for file_path in test_files:
        output_dir = os.path.join("Tornet_Dataset_Images", "Test")
        process_nc_file(file_path, output_dir)
    
    # Process the selected Train files
    for file_path in train_files:
        output_dir = os.path.join("Tornet_Dataset_Images", "Train")
        process_nc_file(file_path, output_dir)

# Directory containing the .nc files
root_directory = "Tornet_Dataset"

# Run the processing function
process_directory(root_directory)
