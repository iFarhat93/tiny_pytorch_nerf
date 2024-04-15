from PIL import Image
import os

def resize_images(source_folder, target_folder, size=(100, 100)):
    """Resize images in the source_folder and save them to the target_folder with new names."""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Initialize a counter for the filenames
    file_counter = 0
    
    for filename in os.listdir(source_folder):
        if filename.endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            
            new_filename = f"frame_{file_counter}.png" 
            target_path = os.path.join(target_folder, new_filename)
            img_resized.save(target_path)
            print(f"Resized and saved {filename} as {new_filename} to {target_folder}")
            
            file_counter += 1

source_folder = "/home/nerfstudio/Desktop/EVT_CAPTURES/TEST_002/capture_take26/capture/"
target_folder = os.path.join("data/")
resize_images(source_folder, target_folder)
