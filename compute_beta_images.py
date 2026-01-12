#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 12:36:15 2026

@author: amarkov
"""

from nilearn.glm.first_level import FirstLevelModel
import pandas as pd
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt




def compute_lss_betas(run_imgs, run_events, output_dir, brain_mask=None, 
                      t_r=0.7, hrf_model="spm", confound_list=None):
    """
    

    Parameters
    ----------
    run_imgs : list of 4D nii
        list of 4D images corresponding to each run.
    run_events : list of dfs
        list of df with onset, duration, trial for each run
    confound_list : list of dfs
        motion parameters and other additional things, upload as csv with columns, where nrows == n scans
    others self explanatory
    Returns
    -------
    None.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for run_idx, func_img in enumerate(run_imgs, start=0):
        events_df = run_events[run_idx].copy().reset_index()
        confound = confound_list[run_idx] if confound_list is not None else None
        
        n_trials = len(events_df)
        
        print(f"Processing run {run_idx+1}, {n_trials} trials")
        
        #FirstLevelModel per run
        fmri_glm = FirstLevelModel(t_r=t_r, hrf_model = hrf_model, mask_img=brain_mask)
        
        #Loop over trials
        for trial_idx in range(n_trials):
            print("-----------------------------------------------")
            print(f"PROCESSING TRIAL {trial_idx+1} / {n_trials}")
            #Regressor 1 is the current trial
            target_trial = events_df.iloc[[trial_idx]].copy()
            target_trial["trial_type"] = "target"
            
            #Regressor 2 is all the others
            other_trials = events_df.drop(trial_idx, axis=0).copy()
            other_trials["trial_type"] = "other"
            
            design_events = pd.concat([target_trial, other_trials], ignore_index = True)
            
            #Fit GLM
            fmri_glm = fmri_glm.fit(func_img, events = design_events, confounds=confound)
            
            
            
            #extract beta for current trial
            beta_img = fmri_glm.compute_contrast("target", output_type="effect_size")
            
            #save_beta_img
            trial_label = target_trial["trial"].values[0]
            plt.savefig((output_dir / f"run{run_idx+1}_trial{trial_label}_design_matrix.png"))
            beta_fname = output_dir / f"run{run_idx+1}_trial{trial_label}_BETA.nii.gz"
            beta_img.to_filename(beta_fname)
            
            #design matrix
            design_matrix = fmri_glm.design_matrices_[0]
            plot_design_matrix(design_matrix, output_file = (output_dir / f"run{run_idx+1}_trial{trial_label}_design_matrix.png"))
            plt.title(f"Run {run_idx+1}, Trial {trial_idx+1}")
            #plt.show() # you will see a white empty graph anyway
            print(f"File output to {output_dir}")


