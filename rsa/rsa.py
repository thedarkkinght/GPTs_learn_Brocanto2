import numpy as np
import pandas as pd
import os
import re 
import matplotlib.pyplot as plt
from nilearn.image import new_img_like, load_img, get_data, resample_to_img
from scipy.ndimage import binary_closing
import nibabel as nib
from mne_rsa_ysg.source_level import rsa_nifti
import mne_rsa
from joblib import Parallel, delayed

# Load subject information
read_file = pd.read_excel('data/fmri_map_id.xlsx')
subjects = list(read_file['BIDS_ID'])

# Load material information and create mappings
df_ma = pd.read_excel('data/MATERIAL.xlsx')
id_2_dur = dict(zip(df_ma['TrialID'], df_ma['soundDur']))

# Create sentence list by concatenating words
trail_list = [str(df_ma['word1'][i]) + ' ' + str(df_ma['word2'][i]) + ' ' + str(df_ma['word3'][i]) + ' ' + str(df_ma['word4'][i]) + ' '   
              + str(df_ma['word5'][i]) + ' ' + str(df_ma['word6'][i]) + ' ' + str(df_ma['word7'][i]) + ' ' + str(df_ma['word8'][i])
               for i in range(len(df_ma))]
trail_list = [sentence.replace("nan", "").strip() for sentence in trail_list]

def remove_duplicates(sentence):
    """Remove duplicate consecutive words from the end of a sentence."""
    words = sentence.split()
    for i in range(7):
        if len(words) >= 2 and words[-1] == words[-2]:
            words.pop(-1)
    return ' '.join(words)

# Process sentences to remove duplicates
new_list = [remove_duplicates(sentence) for sentence in trail_list]
id_2_trail_type = dict(zip(list(df_ma['TrialID']), new_list))

# Read corpus file
file_path = 'data/corpus.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
item_listxt = [line.strip() for line in lines]

# Separate sentences into corpus and non-corpus categories
new_list_co = []
new_list_unc = []
for co in new_list:
    if co in item_listxt:
        new_list_co.append(co)
    else:
        new_list_unc.append(co)

new_list = new_list_co + new_list_unc

# Create count-based sentence identifiers
count_dict = {}
count_new_list = []
for sentence in new_list:
    if sentence in count_dict:
        count_dict[sentence] += 1
    else:
        count_dict[sentence] = 0
    count_new_list.append(f"{sentence} {count_dict[sentence]}")

# Load GPT model representations and compute RDMs for generative classification model
gc_model_rdm_layers = []
for i in range(3):
    gpt_array = np.load(f'model_representations/gpt_pc/layer{i}_representations.npy')
    print(gpt_array.shape)
    model_rdm_layer = mne_rsa.compute_rdm(gpt_array, metric='cosine')
    mne_rsa.plot_rdms(model_rdm_layer, f"GPT-gc_layer{i}")
    gc_model_rdm_layers.append(model_rdm_layer)
    print("RDM shape:", model_rdm_layer.shape)

# Load and compute RDMs for classification model
cla_model_rdm_layers = []
for i in range(3):
    gpt_array = np.load(f'model_representations/gpt_classification/layer{i}_representations.npy')
    print(gpt_array.shape)
    model_rdm_layer = mne_rsa.compute_rdm(gpt_array, metric='cosine')
    mne_rsa.plot_rdms(model_rdm_layer, f"GPT-c_layer{i}")
    cla_model_rdm_layers.append(model_rdm_layer)
    print("RDM shape:", model_rdm_layer.shape)

# Load and compute RDMs for prediction model
pre_model_rdm_layers = []
for i in range(3):
    gpt_array = np.load(f'model_representations/gpt_prediction/layer{i}_representations.npy')
    model_rdm_layer = mne_rsa.compute_rdm(gpt_array, metric='cosine')
    mne_rsa.plot_rdms(model_rdm_layer, f"GPT-g_layer{i}")
    pre_model_rdm_layers.append(model_rdm_layer)
    print("RDM shape:", model_rdm_layer.shape)

# Load and compute RDMs for blank model
blank_model_rdm_layers = []
for i in range(3):
    gpt_array = np.load(f'model_representations/gpt_blank/layer{i}_representations.npy')
    model_rdm_layer = mne_rsa.compute_rdm(gpt_array, metric='cosine')
    mne_rsa.plot_rdms(model_rdm_layer, f"GPT-blank_layer{i}")
    blank_model_rdm_layers.append(model_rdm_layer)
    print("RDM shape:", model_rdm_layer.shape)

# Create sentence length RDM
length_new_list = []
for iii in range(len(new_list)):
    length_new_list.append(len(new_list[iii].split()))
length_new_list_array = np.array(length_new_list)
np.save('length.npy', length_new_list_array)
length_rdm = mne_rsa.compute_rdm(length_new_list_array, metric='euclidean')
mne_rsa.plot_rdms(length_rdm)
print("RDM shape:", length_rdm.shape)

# Set output path for RSA results
rsa_map_path = 'results/rsa_analysis/'

def process_subject(subject, layer_index, gc_model_rdm_layers, pre_model_rdm_layers, cla_model_rdm_layers, blank_model_rdm_layers, count_new_list):
    """Process RSA analysis for a single subject at a specific layer."""
    
    # Prepare model RDMs for analysis
    new_rdm_used = []
    new_rdm_used.append(gc_model_rdm_layers[layer_index])
    new_rdm_used.append(pre_model_rdm_layers[layer_index])
    new_rdm_used.append(cla_model_rdm_layers[layer_index])
    new_rdm_used.append(blank_model_rdm_layers[layer_index])
    new_rdm_used.append(length_rdm)

    # Load contrast files for the subject
    folder_path = f'data/fmri_data/sub-{subject}/contrasts/'
    file_names = os.listdir(folder_path)

    # Extract contrast names from file names
    contrast_name = []
    for file_name in file_names:
        if file_name.endswith('.nii.gz'):
            start_index = file_name.find('con-') + len('con-')
            end_index = file_name.find('.nii.gz')
            content = file_name[start_index:end_index]
            contrast_name.append(content)

    # Check for duplicate contrasts
    if len(contrast_name) != len(set(contrast_name)):
        print(f"Subject {subject}: Duplicate elements found in contrast list")
    else:
        print(f"Subject {subject}: No duplicate elements in contrast list")

    # Sort contrasts according to sentence order
    sorted_contrast_name = [item for item in count_new_list if item in contrast_name]
    if len(sorted_contrast_name) != len(new_list):
        print(f"Subject {subject}: Missing contrast items")

    # Create image file list
    img_list = []
    for con in sorted_contrast_name:
        pattern = re.compile(f'sub-{subject}_(.*?)_con-{con}.nii.gz')
        matched_files = [file_name for file_name in file_names if pattern.match(file_name)]
        img_list.extend(matched_files)

    # Load brain mask
    gray_img = load_img(f"data/fmri_data/sub-{subject}/anat/sub-{subject}_desc-brain_mask.nii.gz")

    # Process gray matter mask
    gm_target_data = get_data(gray_img)
    gm_target_mask = (gm_target_data > 0.2).astype("int8")
    gm_target_mask = binary_closing(gm_target_mask, iterations=2)
    gray_mask = new_img_like(gray_img, gm_target_mask)

    # Load functional reference image
    resample_target_img = load_img(f"data/fmri_data/sub-{subject}/func/sub-{subject}_task-bctnpvp_run-01_space-T1w_desc-preproc_bold.nii.gz")

    # Resample gray matter mask to functional space
    gray_mask = resample_to_img(gray_mask, resample_target_img, interpolation="nearest")
    x, y, z = gray_mask.shape
    data = np.zeros((len(img_list), x, y, z))
    
    # Load all contrast images
    for i, im in enumerate(img_list):
        data[i] = load_img(folder_path + im).get_fdata()

    data_final = new_img_like(resample_target_img, np.transpose(data, (1, 2, 3, 0)))

    # Perform RSA analysis
    result = rsa_nifti(data_final,
                       rdm_model=new_rdm_used,
                       spatial_radius=0.01,
                       image_rdm_metric='euclidean',
                       rsa_metric='partial-regression',
                       brain_mask=gray_mask,
                       verbose=True,
                       n_jobs=7,
                       ignore_nan=True)

    # Save results for each model type
    model_names = ['pc', 'pre', 'cla', 'blank', 'length']
    r2_model_names = ['r2_pc', 'r2_pre', 'r2_cla', 'r2_blank', 'r2_length', 'r2']
    
    for i, model_name in enumerate(model_names):
        output_folder = rsa_map_path + f'accountnum_{model_name}_reg_layer_{layer_index}/sub-{subject}/'
        os.makedirs(output_folder, exist_ok=True)
        result[i].to_filename(f'{output_folder}/sub-{subject}_space-T1w_RSA.nii.gz')
    
    for i, model_name in enumerate(r2_model_names):
        output_folder = rsa_map_path + f'accountnum_{model_name}_reg_layer_{layer_index}/sub-{subject}/'
        os.makedirs(output_folder, exist_ok=True)
        result[i + 5].to_filename(f'{output_folder}/sub-{subject}_space-T1w_RSA.nii.gz')

# Execute parallel processing for all layers
for layer_index in range(3):
    Parallel(n_jobs=17)(
        delayed(process_subject)(subject, layer_index, gc_model_rdm_layers, pre_model_rdm_layers, cla_model_rdm_layers, blank_model_rdm_layers, count_new_list)
        for subject in subjects
    )