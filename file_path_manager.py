#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 15:39:36 2025

@author: amarkov
"""
# %%
#Import Libraries
import numpy as np
import re
import pandas as pd
from nilearn import image, plotting, decoding
import openpyxl # dependency for pd.read_excel
from pathlib import Path

# %% FileManager CLASS FOR MANAGING FILE PATHS
class FileManager:
    """
    Utility class to manage file paths related to subject-level neuroimaging data.
    
    This class centralizes directory structure, filename conventions, and 
    access to event files such as rotation data and scan timestamps. 
    It ensures consistent handling of subject IDs, session IDs, and 
    smoothed functional data paths.
    """

    def __init__(self, project_folder_path, SID, sequence_name, fwmh=8):
        """
        Initialize a FileManager for a specific subject and sequence.

        Parameters
        ----------
        project_folder_path : Path
            Root folder of the project.
        SID : int
            Subject ID (numeric).
        sequence_name : str
            Functional sequence name (e.g., "2Depi").
        fwmh : int, optional
            Smoothing kernel used during preprocessing (FWHM in mm).
        """
        self.project_folder_path = Path(project_folder_path)
        self.SID = SID
        self.sequence_name = sequence_name
        self.fwmh = fwmh

        # Construct subject folder name (e.g., sub-MLHD002 or sub-MLHD010)
        self.subject_folder_name = f"sub-MLHD0{self.SID:02d}"
        
        # Core directory definitions
        self.path_to_subject_folder = (self.project_folder_path / "ProcessedData" / "Experiment" / "spmprep" / self.subject_folder_name)
        self.path_to_subject_details_file = (self.project_folder_path / "RawData" / "Experiment" / "LabProtocols" / self.subject_folder_name
            / f"{self.subject_folder_name}_ProtokollMRI.xlsx")
        self.path_to_subject_event_folder = (self.project_folder_path / "RawData" / "Experiment" / "EventTimes" / self.subject_folder_name)

        # Read subject protocol sheet (contains run/session metadata)
        self.protocoll_df = pd.read_excel(self.path_to_subject_details_file,sheet_name="Info_sequence")

        # Identify functional runs that are marked as usable
        func_df = self.protocoll_df.loc[
            (self.protocoll_df["Useful"] == 1)
            & (self.protocoll_df["Scan_type"] == "func")
        ]

        # Extract session IDs and run numbers
        self.func_run_ids = func_df["ID_from_the_folder"].astype(str).str.zfill(2).to_numpy()
        self.func_session_ids = func_df["Session_ID"].astype(str).to_numpy()

        # Construct filenames for preprocessed functional runs
        self.func_run_filenames = (
            "s0"
            + str(self.fwmh)
            + "wufmapCorr_"
            + self.subject_folder_name
            + "_ses-0"
            + self.func_session_ids
            + "_task-"
            + self.sequence_name
            + "_run-"
            + self.func_run_ids
            + "_bold.nii"
        )
        
        # Construct filenames for functional run motion parameter files
        # These are created during preprocessing with SPM-12 in our case
        # rp_fmapCorr_sub-MLHD002_ses-01_task-2Depi_run-05_bold.txt
        self.motion_parameter_filenames = (
            "rp_fmapCorr_"
            + self.subject_folder_name
            + "_ses-0"
            + self.func_session_ids
            + "_task-"
            + self.sequence_name
            + "_run-"
            + self.func_run_ids
            + "_bold.txt"
        )

        # Construct full paths to functional NIfTI files
        self.func_run_paths = (self.path_to_subject_folder / ("ses-0" + self.func_session_ids) / "func" / self.func_run_filenames)
        
        # Construct full paths to corresponding motion parameter txt files
        self.motion_parameter_paths = (self.path_to_subject_folder / ("ses-0" + self.func_session_ids) / "func" / self.motion_parameter_filenames)
        
        # Get path to the preprocessed, normalised anatomical brain T1 (for mask and plotting)
        anat_useful = self.protocoll_df.loc[
            (self.protocoll_df["Useful"] == 1)
            & (self.protocoll_df["Scan_type"] == "anat")
        ]
        anat_run_id = anat_useful["ID_from_the_folder"].astype(str).str.zfill(2).to_numpy()[0]
        anat_session_id = anat_useful["Session_ID"].astype(str).to_numpy()[0]
        self.anat_path = (self.path_to_subject_folder / ("ses-0" + anat_session_id) / "anat"  / "mri" 
        / str("wm"+self.subject_folder_name+"_ses-0" + anat_session_id + "_run-" + anat_run_id + "_T1w.nii"))
        
    # ---------------------------------------------------------------------
    # EVENT DATA: ROTATION FILES
    # ---------------------------------------------------------------------
    def get_rotation_data_paths(self):
        """
        Locate all rotation_data.csv files for the subject and extract run/trial info.

        Returns
        -------
        pandas.DataFrame
            Columns: ['run', 'trial', 'path']
        """
        rows = []
        for csv_path in self.path_to_subject_event_folder.glob("*.csv"):
            name = csv_path.name

            if "rotation_data.csv" not in name:
                continue

            run_match = re.search(r"run_(\d+)", name)
            trial_match = re.search(r"trial_(\d+)", name)

            if run_match and trial_match:
                rows.append({
                    "run": int(run_match.group(1)),
                    "trial": int(trial_match.group(1)),
                    "path": csv_path
                })

        df = (pd.DataFrame(rows).sort_values(["run", "trial"]).reset_index(drop=True))
        return df

    # ---------------------------------------------------------------------
    # EVENT DATA: SCAN TIMESTAMP FILES
    # ---------------------------------------------------------------------
    def get_scan_times_data_paths(self):
        """
        Locate all scan_times.csv files for the subject and extract run info.

        Returns
        -------
        pandas.DataFrame
            Columns: ['run', 'path']
        """
        rows = []
        for csv_path in self.path_to_subject_event_folder.glob("*.csv"):
            name = csv_path.name

            if "scan_times.csv" not in name:
                continue

            run_match = re.search(r"run_(\d+)", name)
            if run_match:
                rows.append({
                    "run": int(run_match.group(1)),
                    "path": csv_path
                })
                
        df = (pd.DataFrame(rows).sort_values(["run"]).reset_index(drop=True))
        return df
    
    def get_beta_images(self):
        self.path_to_beta_image_folder = self.project_folder_path / "ProcessedData" / "Experiment" / "MVPA" / self.subject_folder_name / "Beta_images"
        rows = []
        for niigz_path in self.path_to_beta_image_folder.glob("*.nii.gz"):
            name = niigz_path.name

            run_match = re.search(r"run(\d+)", name)
            trial_match = re.search(r"_trial(\d+)", name)
            if run_match:
                rows.append({
                    "run": int(run_match.group(1)),
                    "trial": int(trial_match.group(1)),
                    "path": niigz_path
                })
                
        df = (pd.DataFrame(rows).sort_values(["run","trial"]).reset_index(drop=True))
        return df
          
