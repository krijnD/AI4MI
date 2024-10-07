import os

def create_new_folders():
    # Go up one level to reach the AI4MI/data directory
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    new_folders = ['segthor_train', 'segthor_affine', 'segthor_elastic', 'segthor_noise']
    
    # Loop through and create the new folders
    for folder in new_folders:
        folder_path = os.path.join(base_data_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

if __name__ == "__main__":
    create_new_folders()