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
        # Load the NetCDF file
        nc_data = netCDF4.Dataset(file_path)
        
        # Check if 'frame_labels' is present in the variables
        if 'frame_labels' in nc_data.variables:
            frame_labels = nc_data.variables['frame_labels'][:]
            print(f"Frame labels in {file_path}: {frame_labels}")
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
    """
    Process a single .nc file to extract the 'WIDTH' variable and save it as an image
    with a directory structure based on the label.
    """
    try:
        # Load the NetCDF file
        nc_data = netCDF4.Dataset(file_path)
        
        # Extract the frame labels
        frame_labels = display_frame_labels(file_path)
        if frame_labels is None:
            return
        
        # Convert frame labels to a string representation (e.g., "0101")
        label_str = ''.join(str(int(label)) for label in frame_labels)
        
        # Check if 'WIDTH' is present in the variables
        if 'WIDTH' not in nc_data.variables:
            print(f"Variable 'WIDTH' not found in {file_path}")
            return
        
        # Extract the WIDTH variable
        WIDTH_data = nc_data.variables['WIDTH'][:]
        
        # Check the shape of WIDTH (should be 4D with the last dimension of size 2)
        if WIDTH_data.ndim != 4 or WIDTH_data.shape[-1] != 2:
            print(f"Unsupported data shape for 'WIDTH' in {file_path}: {WIDTH_data.shape}")
            return
        
        # Extract a 2D slice (time step and component)
        image_data = WIDTH_data[time_step, :, :, component_index]
        
        # Create the output directory based on the label
        output_dir = os.path.join(base_output_dir, label_str)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the output file path with .jpg extension
        filename = os.path.splitext(os.path.basename(file_path))[0] + ".jpg"
        output_path = os.path.join(output_dir, filename)
        
        # Plot the image
        plt.figure(figsize=(4, 4))
        plt.imshow(
            image_data,
            cmap='viridis',  # Color map
            origin='lower',   # Flip vertically if needed
            aspect='auto'     # Maintain aspect ratio
        )
        #plt.colorbar(label='WIDTH')
        #plt.title(f'WIDTH - Label: {label_str}')
        #plt.xlabel("X-axis")
        #plt.ylabel("Y-axis")
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
        plt.close()
        
        print(f"Saved image: {output_path}")
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    finally:
        nc_data.close()


def select_random_files(root_dir, category):
    """
    Select 25% of the available .nc files from a given category ('Test' or 'Train')
    within subdirectories labeled '2013'.
    """
    files_list = []
    
    for subdir, _, files in os.walk(root_dir):
        if category in subdir and '2013' in subdir:
            for file in files:
                if file.endswith('.nc'):
                    files_list.append(os.path.join(subdir, file))
    
    # Randomly select 10% of the files
    selected_count = max(1, len(files_list) // 4)
    print(f"Selecting {selected_count} files out of {len(files_list)} for category '{category}'")
    selected_files = random.sample(files_list, selected_count) if files_list else []
    
    return selected_files


def process_directory(root_dir):
    """
    Process the Test and Train datasets by randomly selecting 10% of files 
    from the '2013' subdirectories and converting them to images.
    """
    # Select 10% of random files from Test/YEAR and Train/YEAR
    test_files = select_random_files(root_dir, 'Test')
    train_files = select_random_files(root_dir, 'Train')
    
    # Process the selected Test files
    for file_path in test_files:
        base_output_dir = os.path.join("Tornet_Dataset_Images", "Test")
        process_nc_file(file_path, base_output_dir)
    
    # Process the selected Train files
    for file_path in train_files:
        base_output_dir = os.path.join("Tornet_Dataset_Images", "Train")
        process_nc_file(file_path, base_output_dir)


# Directory containing the .nc files
root_directory = "Tornet_Dataset"

# Run the processing function
process_directory(root_directory)
