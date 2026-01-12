#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 13:55:15 2025

@author: amarkov
"""

# in this file are old decoders

#%% decode and cross validate using Leave One group Out validator object

logo_validator = LeaveOneGroupOut()
correct = np.zeros(logo_validator.get_n_splits(groups=trials))

for i, (train_index, test_index) in enumerate(logo_validator.split(hd_labels,hd_labels,trials)):
    #run decoder
    print(f"STARTING FOLD {i} out of {logo_validator.get_n_splits(groups=trials)}")
    print(test_index)
    decoder = decoding.Decoder(
        estimator = "svc",
        mask = mask_path,
        cv = logo_validator,
        standardize = "zscore_sample",
        screening_percentile=100
        )
    training_data = image.index_img(scans_masked, train_index)
    training_labels = hd_labels[train_index]
    
    validation_data = image.index_img(scans_masked, test_index)
    validation_labels = hd_labels[test_index]
    print(validation_labels)
    
    decoder.fit(training_data,training_labels, groups=trials[train_index])
    print("COMPLETED")
    prediction = decoder.predict(validation_data)
    print("------------------------------------------------------")
    print(f"CV fold {i} | Prediction accuracy = {np.sum(prediction == validation_labels)/len(prediction)}")
    print("------------------------------------------------------")
    correct[i-1] = np.sum(prediction == validation_labels)/len(prediction)
print(f"MEAN ACCURACY: {np.mean(correct)}")


#%% decode and cross validate using KFold validator object

n_splits = 6
kf_validator = KFold(n_splits=n_splits, shuffle=True, random_state=9)
accuracies = np.zeros(n_splits)

for i, (train_index, test_index) in enumerate(kf_validator.split(hd_labels)):
    #run decoder
    print(f"STARTING FOLD {i} out of {kf_validator.get_n_splits()}")
    print(test_index)
    decoder = decoding.Decoder(
        estimator = "svc",
        mask = mask_path,
        cv = kf_validator,
        standardize = "zscore_sample",
        screening_percentile=100
        )
    training_data = image.index_img(scans_masked, train_index)
    training_labels = hd_labels[train_index]
    
    validation_data = image.index_img(scans_masked, test_index)
    validation_labels = hd_labels[test_index]
    print(validation_labels)
    
    decoder.fit(training_data,training_labels)
    print("COMPLETED")
    prediction = decoder.predict(validation_data)
    print("--------------------------------------------------------------------")
    print(f"CV fold {i} | Prediction accuracy = {np.sum(prediction == validation_labels)/len(prediction)}")
    print("--------------------------------------------------------------------")
    accuracies[i-1] = np.sum(prediction == validation_labels)/len(prediction)
print(f"MEAN ACCURACY: {np.mean(accuracies)}")

#%% decode using mean images for each trial
# we must remake hd labels and scans object

hd_labels = conditions_df[include_mask].groupby("trial").mean("target_angle")["target_angle"].to_numpy().astype(str)
n_trials = 54
trial_imgs = []
for i in range(n_trials):
    beginning_index = scans_per_trial*i
    end_index = scans_per_trial*(i+1)
    vols = image.index_img(scans_masked, slice(beginning_index,end_index,1))
    trial_imgs.append(image.mean_img(vols))
print("done averaging per trial")
scans_masked = image.concat_imgs(trial_imgs)

n_splits = 6
kf_validator = KFold(n_splits=n_splits, shuffle=True, random_state=9)
accuracies = np.zeros(n_splits)
y_true = np.zeros(len(hd_labels))
y_pred = np.zeros(len(hd_labels))


for i, (train_index, test_index) in enumerate(kf_validator.split(hd_labels)):
    #run decoder
    print(f"STARTING FOLD {i+1} out of {kf_validator.get_n_splits()}")
    
    print(test_index)
    decoder = decoding.Decoder(
        estimator = "svc",
        mask = mask_path,
        cv = kf_validator,
        standardize = "zscore_sample",
        screening_percentile=100
        )
    training_data = image.index_img(scans_masked, train_index)
    training_labels = hd_labels[train_index]
    
    validation_data = image.index_img(scans_masked, test_index)
    validation_labels = hd_labels[test_index]
    print(validation_labels)
    
    decoder.fit(training_data,training_labels)
    print("COMPLETED")
    prediction = decoder.predict(validation_data)
    y_true[test_index] = validation_labels
    y_pred[test_index] = prediction
    print("-------------------")
    print(f"CV fold {i+1} | Prediction accuracy = {np.sum(prediction == validation_labels)/len(prediction)}")
    print("--------------------------------------------------------------------")
    accuracies[i] = np.sum(prediction == validation_labels)/len(prediction)
print(f"MEAN ACCURACY: {np.mean(accuracies)}")
    
# for anterior thalamus, sub16: 0.3, sub17: 0.09, sub18: 0.17, sub19: 0.13 sub 20: 0.19 sub21: 0.15 sub22: 0.17, sub24: 0.2, sub25: 0.24
### average: 0.182
#i lost one at some point
