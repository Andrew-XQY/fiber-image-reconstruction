import os
from pathlib import Path
os.chdir(Path.cwd().parent)   # go one level up
print(os.getcwd())         

from config_utils import detect_machine
machine = detect_machine() 

# if win, windows test in machine then auto switch to windows path, otherwise use mac path
if any(win in machine for win in ["win", "windows"]): # windows
    dirs = {
        "merged_db_path": "C:/Users/qiyuanxu/Desktop/clear_2025_dataset.db",
        "merged_processed_db_path": "C:/Users/qiyuanxu/Desktop/clear_2025_processed_dataset.db",
        "raw_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/datasets",
        "processed_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed",
        "final_merged_db_path": "C:/Users/qiyuanxu/Desktop/dataset_meta.db",
        
        "globel_save_to": "C:/Users/qiyuanxu/Desktop/",
        
        "DMD_only": {
            "dataset_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_DMD/db/dataset_meta.db",
            "dataset_extracted_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_DMD/",
            
            "output_dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_DMD/dataset/",
            "dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_DMD/",
            "output_db_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_DMD/db/dataset_meta.db",
        },
        
        "Chromox_only": {
            "dataset_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_Chromox/db/dataset_meta.db",
            "dataset_extracted_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_Chromox/",
            
            "output_dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Chromox/dataset/",
            "dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Chromox/",
            "output_db_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Chromox/db/dataset_meta.db",
        },
        
        "Yag_only": {
            "dataset_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_Yag/db/dataset_meta.db",
            "dataset_extracted_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_Yag/",
            
            "output_dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Yag/dataset/",
            "dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Yag/",
            "output_db_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Yag/db/dataset_meta.db",
        },
        
        "Chromox_Laser_only": {
            "output_dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Chromox_Laser/dataset/",
            "output_db_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Chromox_Laser/db/dataset_meta.db",
        },
        
        "YAG_Laser_only": {
            "output_dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Yag_Laser/dataset/",
            "output_db_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Yag_Laser/db/dataset_meta.db",
        }   
    }

elif any(mac in machine for mac in ["mac", "darwin"]): # Mac
    dirs = {
        "merged_db_path": "/Users/andrewxu/Desktop/clear_2025_dataset.db",
        "merged_processed_db_path": "/Users/andrewxu/Desktop/clear_2025_processed_dataset.db",
        "raw_db_dir": "/Users/andrewxu/Documents/DataHub/datasets",
        "processed_db_dir": "/Users/andrewxu/Documents/DataHub/processed",
        "final_merged_db_path": "/Users/andrewxu/Desktop/dataset_meta.db",
        
        "globel_save_to": "/Users/andrewxu/Desktop/",
        
        "DMD_lab": {
            "dataset_db_dir": "/Users/andrewxu/Documents/DataHub/processed/2025-11-06/db/dataset_meta.db",
            "dataset_extracted_dir": "/Users/andrewxu/Documents/DataHub/processed/2025-11-06/"
        },
        "DMD_only": {
            "dataset_db_dir": "/Users/andrewxu/Documents/DataHub/processed/CLEAR25_DMD/db/dataset_meta.db",
            "dataset_extracted_dir": "/Users/andrewxu/Documents/DataHub/processed/CLEAR25_DMD/",
            
            "output_dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_DMD/dataset/",
            "dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_DMD/",
            "output_db_dir": "/Users/andrewxu/Desktop/CLEAR25_DMD/db/dataset_meta.db",
        },
        "Chromox_only": {
            "dataset_db_dir": "/Users/andrewxu/Documents/DataHub/processed/CLEAR25_Chromox/db/dataset_meta.db",
            "dataset_extracted_dir": "/Users/andrewxu/Documents/DataHub/processed/CLEAR25_Chromox/",
            
            "output_dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_Chromox/dataset/",
            "dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_Chromox/",
            "output_db_dir": "/Users/andrewxu/Desktop/CLEAR25_Chromox/db/dataset_meta.db",
        },
        "Yag_only": {
            "dataset_db_dir": "/Users/andrewxu/Documents/DataHub/processed/CLEAR25_Yag/db/dataset_meta.db",
            "dataset_extracted_dir": "/Users/andrewxu/Documents/DataHub/processed/CLEAR25_Yag/",
            
            "output_dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_Yag/dataset/",
            "dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_Yag/",
            "output_db_dir": "/Users/andrewxu/Desktop/CLEAR25_Yag/db/dataset_meta.db",
        },
        "Chromox_Laser_only": {
            "output_dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_Chromox_Laser/dataset/",
            "output_db_dir": "/Users/andrewxu/Desktop/CLEAR25_Chromox_Laser/db/dataset_meta.db",
        },
        "YAG_Laser_only": {
            "output_dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_Yag_Laser/dataset/",
            "output_db_dir": "/Users/andrewxu/Desktop/CLEAR25_Yag_Laser/db/dataset_meta.db",
        }
    }

else:
    raise ValueError(f"Unsupported machine: {machine}")
