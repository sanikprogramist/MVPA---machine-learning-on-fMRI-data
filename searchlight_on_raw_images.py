#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 17:17:13 2026

@author: amarkov
"""

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

# Searchlight parameters:
original_voxel_size = 2.5 # from MRI sequence paramaters
downsample_factor = 0.8 # eg 0.5 will reduce resolution by half
searchlight_radius_in_voxels = 1 # sphere with this radius will be considered by the algorithm

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

# Timestamp of HRF peak response relative to stimulus onset
df["hrf_peak_timestamp"] = df["timestamp"] + HRF_PEAK_TIME


# %% IDENTIFY STIMULUS ONSET
# "Stimulus onset" is defined as the first frame where the subject enters the lock phase.
df["stimulus_onset"] = 0
is_lock_phase = df["phase"].eq("lock")

# Index of the first 'lock' entry per (run, trial)
first_lock_idx = (
    df[is_lock_phase]
    .groupby(["run", "trial"])
    .head(1)
    .index
)

df.loc[first_lock_idx, "stimulus_onset"] = 1


# %% BUILD LABELS DATAFRAME
# Keep one row per stimulus onset
labels_df = df.loc[df["stimulus_onset"] == 1, ["run", "trial", "rotation", "hrf_peak_timestamp"]].copy()

# Apply circular binning per trial
labels_df["rotation_bin"] = labels_df["rotation"].apply(
    circular_round_to_nearest_target,
    target_angles=TARGET_ANGLES
)
# %% LOAD SCAN TIMES
scan_times_paths_df = paths.get_scan_times_data_paths()

# Load scan timestamps for each run
scan_times_df = pd.concat(
    (pd.read_csv(path, index_col=False).assign(run=i)
        for i, path in enumerate(scan_times_paths_df["path"], start=1)),
    ignore_index=True
)

scan_times_df["keep_scan"] = 0  # Flag to mark scans closest to each HRF peak timestamp


# %% MATCH HRF PEAK TO CLOSEST SCAN TIMESTAMP
for run_id in labels_df["run"].unique():

    # HRF peak times for this run
    hrf_run = labels_df.query("run == @run_id")["hrf_peak_timestamp"].values

    # Scan timestamps for this run
    run_idx = scan_times_df.query("run == @run_id").index
    scan_times = scan_times_df.loc[run_idx, "timestamp"].values

    # Replace first 6 scans with dummy timestamps (discard initial T1 stabilization scans)
    scan_times[:6] = 0

    # Find insertion indices such that scan_times[idx] >= hrf_time
    insertion_idxs = np.searchsorted(scan_times, hrf_run)

    closest_idxs = []
    for hrf, idx in zip(hrf_run, insertion_idxs):
        # Candidate scan indices (current and previous)
        candidates = []
        if idx < len(scan_times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        # Choose scan closest in absolute temporal distance
        best_idx = min(candidates, key=lambda i: abs(scan_times[i] - hrf))
        closest_idxs.append(best_idx)

    # Map back to full scan_times_df index
    chosen_df_idxs = run_idx[closest_idxs]

    # Mark selected scans
    scan_times_df.loc[chosen_df_idxs, "keep_scan"] = 1
    
# create final labels array and scan inclusion mask
labels = labels_df["rotation_bin"].astype(str).values
run_labels = labels_df["run"].astype(str).values

#%% Load in the fmri data, shave off the useless scans at the end and mask
run_lengths_from_logs = scan_times_df.groupby("run").size().values
func_runs_container = []
for i in np.arange(0,len(paths.func_run_paths)):
    func_run_img = image.index_img(paths.func_run_paths[i], slice(0,run_lengths_from_logs[i],1)) # shave off extra scans from the end
    scan_inclusion_mask_one_run = scan_times_df.loc[scan_times_df["run"] == (i+1), "keep_scan"].astype(bool).values
    func_run_img = image.index_img(func_run_img, scan_inclusion_mask_one_run) # keep selected scans from peak HRF 
    func_runs_container.append(func_run_img)
    
final_scans = image.concat_imgs(func_runs_container) # this contains only the scans we want now

#%% downsample the fmri images and get brainmask
final_scans_resampled = downsample_4D_nii(final_scans, downsample_factor=downsample_factor)
brain_mask_resampled = resample_and_threshold_brain_mask(paths.anat_path,final_scans_resampled,thr=brain_mask_threshold)
#%% 

# ----------------------------------- ######################### ----------------------------------- #
# ----------------------------------- ###  RUN SEARCHLIGHT  ### ----------------------------------- #
# ----------------------------------- ######################### ----------------------------------- #

njobs = -1
sl_rad = searchlight_radius_in_voxels * (original_voxel_size/downsample_factor) # = N * Voxel Size after downsample

logo_validator = LeaveOneGroupOut()
searchlight = decoding.SearchLight(
    brain_mask_resampled,
    process_mask_img=brain_mask_resampled, # or process_mask_img 
    radius=sl_rad, 
    n_jobs=njobs,
    scoring="accuracy",
    verbose=1,
    cv=logo_validator)

searchlight.fit(final_scans_resampled,labels,groups=run_labels)

# visualise scores image
scores_img=searchlight.scores_img_
mean_fmri = image.mean_img(final_scans_resampled, copy_header=True)
plotting.plot_img(scores_img,
                  display_mode="ortho",
                  #cut_coords=[-19,-10,-2],
                  bg_img=mean_fmri,
                  #threshold=0.2,
                  cmap="inferno",
                  black_bg=True,
                  colorbar=True)
#save the image:
output_path = paths.project_folder_path / "ProcessedData" / "Experiment" / "MVPA" / paths.subject_folder_name 
output_path.mkdir(parents=True, exist_ok=True)
scores_img_path = output_path/ f"Searchlight_wbrain_v_rad_{sl_rad}_HRF{HRF_PEAK_TIME}.nii"
scores_img.to_filename(scores_img_path)
#save the searchlight object:
with open(output_path / f"Searchlight_wbrain_v_rad_{sl_rad}_HRF{HRF_PEAK_TIME}.pkl", "wb") as f:
    pickle.dump(searchlight, f)
## you can load it again like this:
#with open(output_path / f"Searchlight_wbrain_{run_type}_rad{sl_rad}_scans_4-5-6-7-8_logo-cv.pkl", "rb") as f:
#    test = pickle.load(f)

#%%
plotting.plot_anat(brain_mask_resampled) # just testing