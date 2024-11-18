import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def display_frame_labels(file_path):
    """
    Display and return the 'frame_labels' variable for a .nc file if it exists.
    """
    try:
        nc_data = netCDF4.Dataset(file_path)
        
        if 'frame_labels' in nc_data.variables:
            frame_labels = nc_data.variables['frame_labels'][:]
            print(f"{frame_labels}")
            return frame_labels
        else:
            print(f"'frame_labels' not found in {file_path}")
            return None

    except Exception as e:
        print(f"Error displaying 'frame_labels' in {file_path}: {e}")
        return None
    
    finally:
        nc_data.close()


def process_nc_file(file_path, base_output_dir, time_step=0, component_index=0):

    try:
        nc_data = netCDF4.Dataset(file_path)
        
        frame_labels = display_frame_labels(file_path)
        if frame_labels is None:
            return
        
        label_str = ''.join(str(int(label)) for label in frame_labels)
        
        if 'VEL' not in nc_data.variables:
            print(f"Variable 'VEL' not found in {file_path}")
            return
        
        VEL_data = nc_data.variables['VEL'][:]
        
        if VEL_data.ndim != 4 or VEL_data.shape[-1] != 2:
            print(f"Unsupported data shape for 'VEL' in {file_path}: {VEL_data.shape}")
            return
        
        image_data = VEL_data[time_step, :, :, component_index]
        
        output_dir = os.path.join(base_output_dir, label_str)
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.splitext(os.path.basename(file_path))[0] + ".jpg"
        output_path = os.path.join(output_dir, filename)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(
            image_data,
            cmap='bwr', # Color map
            origin='lower',   
            aspect='auto' # dont mess with this, unless you want to 
        )
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=240)
        plt.close()
        
        print(f"Saved image: {output_path}")
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    finally:
        nc_data.close()


def select_random_files(root_dir, category):

    files_list = []
    
    for subdir, _, files in os.walk(root_dir):
        if category in subdir and '2013' in subdir:
            for file in files:
                if file.endswith('.nc'):
                    files_list.append(os.path.join(subdir, file))
    
    selected_count = max(1, len(files_list)) #add // number to get a percentage of the total data
    print(f"Selecting {selected_count} files out of {len(files_list)} for category '{category}'")
    selected_files = random.sample(files_list, selected_count) if files_list else []
    
    return selected_files


def process_directory(root_dir):
    test_files = select_random_files(root_dir, 'Test')
    train_files = select_random_files(root_dir, 'Train')
    
    for file_path in test_files:
        base_output_dir = os.path.join("Tornet_Dataset_Images", "Test")
        process_nc_file(file_path, base_output_dir)
    
    for file_path in train_files:
        base_output_dir = os.path.join("Tornet_Dataset_Images", "Train")
        process_nc_file(file_path, base_output_dir)


# Directory containing the .nc files
root_directory = "Tornet_Dataset"

process_directory(root_directory)
