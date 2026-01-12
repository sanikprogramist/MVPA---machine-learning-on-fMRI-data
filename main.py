#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:34:07 2025

@author: amarkov
"""

#%% LIBRARIES
import numpy as np
import pandas as pd
from nilearn import image, plotting, decoding
from pathlib import Path
from file_path_manager import FileManager
from helper_functions import *
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
import pickle
warnings.filterwarnings("always")

# %% INITIALISE VARS

# Subject ID: 
SID = 2

# Sequence name:
SEQUENCE_NAME = "2Depi"

# Hemodynamic Response Function: approximate peak latency (ms)
HRF_PEAK_TIME = 6000  

# Target rotation bins for circular binning
TARGET_ANGLES = np.array([15, 75, 135, 195, 255, 315])

# For thresholding the brain mask
brain_mask_threshold = 0.2

# whether beta images have already been computed earlier:
# If you have never run this script before, set it to False, and Betas will be computed for you using LSS approach
betas_already_computed = True

# Searchlight parameters:
original_voxel_size = 2.5 # from MRI sequence paramaters
downsample_factor = 1 # eg 0.5 will reduce resolution by half
searchlight_radius_in_voxels = 2 # sphere with this radius will be considered by the algorithm

# Initialize file manager object for subject/session paths
paths = FileManager(Path(__file__).parent.parent.resolve(), SID=SID, sequence_name=SEQUENCE_NAME)

# %% LOAD ROTATION DATA
rotation_paths_df = paths.get_rotation_data_paths()

# Load all runs into a single DataFrame
df = pd.concat(
    (pd.read_csv(path, index_col=False) for path in rotation_paths_df["path"]),
    ignore_index=True
)

# Keep only encoding-phase trials
df = df[df["trial_type"] == "encode"].copy()

# Identify Stimulus Onset:
# "Stimulus onset" is defined as the first time frame when the participant reaches the rotation that they lock in 
df["stimulus_onset"] = 0
df = df.groupby(["run","trial"], group_keys = False).apply(mark_stimulus_onset_and_duration, degree_tolerance=2.5)

#%% Compute the beta images! This uses the LSS design, which loops over trials, and makes two regressors:
# The first regressor is all OTHER trials, and the second regressor is the CURRENT trial
# then, a beta image for that current trial is extracted. 
if not betas_already_computed:
    df_beta_computation = df.copy()
    df_beta_computation = df_beta_computation[df_beta_computation["stimulus_onset"]==1] 
    df_beta_computation = df_beta_computation[["run","trial","timestamp","duration"]]
    df_beta_computation = df_beta_computation.rename(columns={"timestamp":"onset"})
    df_beta_computation["onset"] = df_beta_computation["onset"].values/1000 # convert to s for nilearn's GLM fitting
    df_beta_computation["duration"] = df_beta_computation["duration"].values/1000 # convert to s for nilearn's GLM fitting
    df_beta_computation = [df_beta_computation.groupby("run").get_group(group) for group in df_beta_computation.groupby("run").groups]
    # Fetch motion parameters generated from spm
    motion_parameter_list = [pd.read_csv(path, sep="  ", header=None) for path in paths.motion_parameter_paths]
    motion_parameter_list = [df.set_axis(["x","y","z","pitch","roll","yaw"],axis=1) for df in motion_parameter_list]
    
    compute_lss_betas(run_imgs = paths.func_run_paths, 
                      run_events = df_beta_computation,
                      output_dir = (Path(paths.project_folder_path) / "ProcessedData" / "Experiment" / "MVPA" / paths.subject_folder_name / "Beta_images"),
                      confound_list = motion_parameter_list)
# Load Beta Image paths
beta_images_df = paths.get_beta_images()
beta_images = beta_images_df["path"].values
beta_images = image.concat_imgs(beta_images)
# %% BUILD LABELS DATAFRAME
# Keep one row per stimulus onset
labels_df = df.loc[df["stimulus_onset"] == 1, ["run", "trial", "rotation"]].copy()

# Apply circular binning per trial
labels_df["rotation_bin"] = labels_df["rotation"].apply(
    circular_round_to_nearest_target,
    target_angles=TARGET_ANGLES
)

# create final labels array and scan inclusion mask
labels = labels_df["rotation_bin"].astype(str).values
run_labels = labels_df["run"].astype(str).values

#Now we have run_labels for our groups, labels for our classes and beta_images as our input images. We are ready for Maschine Learning!

#%% downsample the fmri images and get brainmask
final_images_resampled = downsample_4D_nii(beta_images, downsample_factor=downsample_factor)
brain_mask_resampled = resample_and_threshold_brain_mask(paths.anat_path,final_images_resampled,thr=brain_mask_threshold)
plotting.plot_epi(image.mean_img(final_images_resampled))
plotting.plot_anat(brain_mask_resampled)
print(f"The dimensions of the 4D nii are {final_images_resampled.shape}")
print(f"The dimensions of the brain mask are {brain_mask_resampled.shape}")
#%% 

# ----------------------------------- ######################### ----------------------------------- #
# ----------------------------------- ###  RUN SEARCHLIGHT  ### ----------------------------------- #
# ----------------------------------- ######################### ----------------------------------- #
searchlight_radius_in_voxels = 2
njobs = -1
sl_rad = searchlight_radius_in_voxels * (original_voxel_size/downsample_factor) # = N * Voxel Size after downsample

#We will use the leave one group out validator, which will use 4/5 runs as training data and 1/5 runs as validation data
logo_validator = LeaveOneGroupOut()
#Create searchlight object
searchlight = decoding.SearchLight(
    brain_mask_resampled,
    process_mask_img=brain_mask_resampled, # or process_mask_img 
    radius=sl_rad, 
    n_jobs=njobs,
    scoring="accuracy",
    verbose=1,
    cv=logo_validator)

#Fit the data°
searchlight.fit(final_images_resampled,labels,groups=run_labels)

#Visualise the scores
scores_img=searchlight.scores_img_
mean_fmri = image.mean_img(final_images_resampled, copy_header=True)
plotting.plot_img(scores_img,
                  display_mode="ortho",
                  cut_coords=[-19,-10,-2],
                  bg_img=mean_fmri,
                  threshold=0.166,
                  cmap="inferno",
                  black_bg=True,
                  colorbar=True)
#%% save data
#save the image:
output_path = paths.project_folder_path / "ProcessedData" / "Experiment" / "MVPA" / paths.subject_folder_name 
output_path.mkdir(parents=True, exist_ok=True)
scores_img_path = output_path/ f"Searchlight_wbrain_v_rad{sl_rad}_betas_downsample{downsample_factor}.nii"
scores_img.to_filename(scores_img_path)
#save the searchlight object:
with open(output_path / f"Searchlight_wbrain_v_rad{sl_rad}_betas_downsample{downsample_factor}.pkl", "wb") as f:
    pickle.dump(searchlight, f)
## you can load it again like this:
#with open(output_path / f"Searchlight_wbrain_{run_type}_rad{sl_rad}_scans_4-5-6-7-8_logo-cv.pkl", "rb") as f:
#    test = pickle.load(f)

