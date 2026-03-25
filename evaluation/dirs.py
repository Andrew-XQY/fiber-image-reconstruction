import os
from pathlib import Path
os.chdir(Path.cwd().parent)   # go one level up
print(os.getcwd())         

from config_utils import detect_machine
machine = detect_machine() 

# if win, windows test in machine then auto switch to windows path, otherwise use mac path
if any(win in machine for win in ["win", "windows"]): # windows
    dirs = {
        "Dataset_root_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/",
        "merged_db_path": "C:/Users/qiyuanxu/Desktop/clear_2025_dataset.db",
        "raw_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/datasets",
        "final_merged_db_path": "C:/Users/qiyuanxu/Desktop/dataset_meta.db",
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
            "output_dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Yag/dataset/",
            "dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Yag/",
            "output_db_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_Yag/db/dataset_meta.db",
        },
    }

elif any(mac in machine for mac in ["mac", "darwin"]): # Mac
    dirs = {
        "merged_db_path": "/Users/andrewxu/Desktop/clear_2025_dataset.db",
        "raw_db_dir": "/Users/andrewxu/Documents/DataHub/datasets",
        "final_merged_db_path": "/Users/andrewxu/Desktop/dataset_meta.db",
        "DMD_only": {
            "output_dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_DMD/dataset/",
            "dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_DMD/",
            "output_db_dir": "/Users/andrewxu/Desktop/CLEAR25_DMD/db/dataset_meta.db",
        },
        "Chromox_only": {
            "output_dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_Chromox/dataset/",
            "dataset_dir": "/Users/andrewxu/Desktop/CLEAR25_Chromox/",
            "output_db_dir": "/Users/andrewxu/Desktop/CLEAR25_Chromox/db/dataset_meta.db",
        }
    }

else:
    raise ValueError(f"Unsupported machine: {machine}")
