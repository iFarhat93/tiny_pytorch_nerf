import numpy as np
import os
import json
from matplotlib import image
import argparse
from PIL import Image
import os
import shutil
import time

def resize_images(source_folder, target_folder, size=(100, 100)):
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

def extract_number(filename):
    part = filename.split('_')[1]  
    number = int(part.split('.')[0]) 
    return number

def load_images_and_poses(image_directory, json_file_path):
    images = []

    # Load each image
    filenames = sorted(os.listdir(image_directory), key=extract_number)
    print("Starting Image stacking.")
    for filename in filenames:
        img_path = os.path.join(image_directory, filename)
        # Make sure the path points to a file
        #print(img_path)
        if os.path.isfile(img_path):
            img = image.imread(img_path)
            images.append(img)

    # Stack images into a single numpy array
    image_stack = np.array(images)
    print("Image stacking done.")
    print("Starting parsing json parsing.")
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Sort the frames based on the filename
    sorted_frames = sorted(data['frames'], key=lambda x: int(x["file_path"].split('_')[-1].split('.')[0]))
    
    # Extract and adjust the "transform_matrix" for each frame after sorting, excluding the last row
    poses = np.array([frame["transform_matrix"][:] for frame in sorted_frames])
    print("Poses stacking done.")
    return image_stack, poses



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load images and poses and save them as an NPZ file.')
    parser.add_argument('--image_directory', type=str, required=True,
                        help='Directory where your images are stored')
    parser.add_argument('--json_file_path', type=str, required=True,
                        help='Path to the JSON file with transformations')
    parser.add_argument('--output_name', type=str, required=True,
                        help='Name of the output NPZ file')
    

    args = parser.parse_args()

    target_directory = 'tmp'

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    resize_images(args.image_directory, target_directory)
    time.sleep(10)

    images, poses = load_images_and_poses(target_directory, args.json_file_path)

    time.sleep(10)
    np.savez_compressed(args.output_name+'.npz', images=images, poses=poses)

    time.sleep(10)
    data = np.load(args.output_name+'.npz')
    #print(data['images'])
    shutil.rmtree(target_directory)
    print("NPZ file has been saved with images and poses.")