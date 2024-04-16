import shutil
import os
import torch
import gc


def remove_pycache(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        if '__pycache__' in dirnames:
            shutil.rmtree(os.path.join(dirpath, '__pycache__'))

            
def clear_all():
    torch.cuda.empty_cache()  # Clear unused memory
    gc.collect()

    remove_pycache(os.getcwd())  
