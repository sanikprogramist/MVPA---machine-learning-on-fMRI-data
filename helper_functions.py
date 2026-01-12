from nilearn import image
import numpy as np

def downsample_4D_nii(nii_img_4D, downsample_factor=0.5):
    target_shape = tuple(int(s*downsample_factor) for s in nii_img_4D.shape[:3])
    target_affine = nii_img_4D.affine.copy()
    target_affine[:3,:3] *= (nii_img_4D.shape[0]/target_shape[0])
    return image.resample_img(nii_img_4D, target_shape = target_shape, target_affine = target_affine,interpolation="continuous")
    
def resample_and_threshold_brain_mask(brain_mask_img, nii_img_4D, thr=0.2):
    # Resample the anatomical image to match functional data
    bm_resampled = image.resample_to_img(brain_mask_img,
                                         image.mean_img(nii_img_4D),
                                         force_resample=True,
                                         interpolation="linear")
    mask_data = bm_resampled.get_fdata()
    
    binary_mask_data = (mask_data > thr).astype(int)
    mask_data = np.clip(mask_data, 0, 1)  # Remove any negative values and above 1
    return image.new_img_like(bm_resampled, binary_mask_data)

def circular_round_to_nearest_target(angle, target_angles):
    """
    Map an angle to the nearest target bin using circular distance.
    """
    # Compute minimal signed angular distance in circular space
    distances = np.abs((angle - target_angles + 180) % 360 - 180)
    return target_angles[np.argmin(distances)]

def mark_stimulus_onset_and_duration(group, degree_tolerance):
    """
    To be applied on grouped data (run, trial)

    Adds:
    - stimulus_onset == 1 at the detected onset row
    - duration_ms for the whole trial (same value for all rows in group)
    """

    group = group.copy()
    # --------------------------------------------------
    # 1. Find first lock
    # --------------------------------------------------
    lock_rows = group[group["phase"] == "lock"]
    if lock_rows.empty:
        return group  # safety: no lock found

    first_lock_idx = lock_rows.index[0] - 1

    # rotation just before lock
    target_rotation = group.loc[first_lock_idx, "rotation"]
    # --------------------------------------------------
    # 2. Find stimulus onset
    # --------------------------------------------------
    before_lock = group.loc[:first_lock_idx]

    within_tolerance = (
        (before_lock["rotation"] - target_rotation).abs()
        <= degree_tolerance
    )

    last_false_idx = within_tolerance[~within_tolerance].last_valid_index()

    # onset = first True AFTER last False
    onset_idx = within_tolerance.loc[last_false_idx:].index[1]

    group.loc[onset_idx, "stimulus_onset"] = 1
    # --------------------------------------------------
    # 3. Compute duration
    # --------------------------------------------------
    onset_time = group.loc[onset_idx, "timestamp"]

    last_lock_time = lock_rows["timestamp"].iloc[-1]

    duration_ms = (last_lock_time + 100) - onset_time

    # store duration (same value for all rows in trial)
    group["duration"] = duration_ms

    return group
