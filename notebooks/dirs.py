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
        
        "processed_chromox_ipac": "C:/Users/qiyuanxu/Desktop/CLEAR25_Chromox/",

        "DMD_cockcroft": {
            "dataset_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/local_images/backup/2024-08-15/db/dataset_meta.db",
            "dataset_extracted_dir": "C:/Users/qiyuanxu/Documents/DataHub/local_images/backup/2024-08-15/"
        },
        "DMD_lab": {
            "dataset_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/local_images/backup/2025-11-06/db/dataset_meta.db",
            "dataset_extracted_dir": "C:/Users/qiyuanxu/Documents/DataHub/local_images/backup/2025-11-06/"
        },
        
        "DMD_only": {
            "dataset_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_DMD/db/dataset_meta.db",
            "dataset_extracted_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_DMD/",
            
            "output_dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_DMD/dataset/",
            "dataset_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_DMD/",
            "output_db_dir": "C:/Users/qiyuanxu/Desktop/CLEAR25_DMD/db/dataset_meta.db",
        },
        
        "Chromox_cropped_only": {
            "dataset_db_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_Chromox_Cropped/db/dataset_meta.db",
            "dataset_extracted_dir": "C:/Users/qiyuanxu/Documents/DataHub/processed/CLEAR25_Chromox_Cropped/",
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
        },
        
        "models": {
            "root": "C:/Users/qiyuanxu/Desktop/Models/",
            "chromox_cropped_cae_line_scan_sgm": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_sgm/model.pt",
            "chromox_cropped_cae_random_scan": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_random_scan/model.pt",
            "chromox_cropped_cae_random_scan_leakless": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_random_scan_leakless/model.pt",
            "chromox_cae_random_scan": "C:/Users/qiyuanxu/Desktop/Models/chromox_cae_random_scan/model.pt",
            "clear_dmd_cae_line_scan_sgm": "C:/Users/qiyuanxu/Desktop/Models/clear_dmd_cae_line_scan_sgm/model.pt",
            "chromox_cropped_cae_line_scan_real_beam_image": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_real_beam_image/model.pt",
            "chromox_cropped_cae_line_scan_mixture": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_mixture/model.pt",
            "chromox_cropped_cae_line_scan_mixture_prab_model": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_mixture_prab_model/model.pt",
            "chromox_cropped_TM_line_scan": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_TM_line_scan/model.pt",
            "chromox_cropped_cae_line_scan": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan/model.pt",
            "chromox_cropped_cae_line_scan_single_sample": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_single_sample/model.pt",
            
            "Model_trained_on_CHROMOX_only": "C:/Users/qiyuanxu/Desktop/Models/Scope 1 cross field/Model_trained_on_CHROMOX_only/model.pt",
            "Model_trained_on_CHROMOX_LASER_only": "C:/Users/qiyuanxu/Desktop/Models/Scope 1 cross field/Model_trained_on_CHROMOX_LASER_only/model.pt",
    
            "Model_trained_on_DMD_orth_SGM_inLab_150MB": "C:/Users/qiyuanxu/Desktop/Models/Model_trained_on_DMD_orth_SGM_inLab_150MB/model.pt",
            
            "baseline_random_predict": "C:/Users/qiyuanxu/Desktop/Models/baseline_random_predict/model.pt",
        },
        
        "save": {
            "model_inference": {
                "chromox_cropped_cae_line_scan_sgm": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_sgm/inference/",
                "chromox_cropped_cae_random_scan": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_random_scan/inference/",
                "chromox_cropped_cae_random_scan_leakless": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_random_scan_leakless/inference/",
                "chromox_cae_random_scan": "C:/Users/qiyuanxu/Desktop/Models/chromox_cae_random_scan/inference/",
                "clear_dmd_cae_line_scan_sgm": "C:/Users/qiyuanxu/Desktop/Models/clear_dmd_cae_line_scan_sgm/inference/",
                "chromox_cropped_cae_line_scan_real_beam_image": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_real_beam_image/inference/",
                "chromox_cropped_cae_line_scan_mixture": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_mixture/inference/",
                "chromox_cropped_cae_line_scan_mixture_prab_model": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_mixture_prab_model/inference/",
                "chromox_cropped_TM_line_scan": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_TM_line_scan/inference/",
                "chromox_cropped_cae_line_scan": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan/inference/",
                "chromox_cropped_cae_line_scan_single_sample": "C:/Users/qiyuanxu/Desktop/Models/chromox_cropped_cae_line_scan_single_sample/inference/",
                
                "Model_trained_on_CHROMOX_only": "C:/Users/qiyuanxu/Desktop/Models/Scope 1 cross field/Model_trained_on_CHROMOX_only/inference/",
                "Model_trained_on_CHROMOX_LASER_only": "C:/Users/qiyuanxu/Desktop/Models/Scope 1 cross field/Model_trained_on_CHROMOX_LASER_only/inference/",
                
                "Model_trained_on_DMD_orth_SGM_inLab_150MB": "C:/Users/qiyuanxu/Desktop/Models/Model_trained_on_DMD_orth_SGM_inLab_150MB/inference/",
                
                "baseline_random_predict": "C:/Users/qiyuanxu/Desktop/Models/baseline_random_predict/inference/",
                },
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
        "temp_save_to": "/Users/andrewxu/Desktop/Output/",
        
        "DMD_cockcroft": {
            "dataset_db_dir": "/Users/andrewxu/Documents/DataHub/local_images/backup/2024-08-15/db/dataset_meta.db",
            "dataset_extracted_dir": "/Users/andrewxu/Documents/DataHub/local_images/backup/2024-08-15/"
        },
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
