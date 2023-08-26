import nibabel
import scipy.io as sio
from brainsmash.mapgen.base import Base
from nilearn.maskers import NiftiLabelsMasker
from atlasTransform.atlasTransform.utils.atlas import load_shen_268
from pathlib import Path
import numpy as np
from nilearn.plotting import find_parcellation_cut_coords
import glob
import pandas as pd
from fc_analysis import loadings, nan_all_for_spearman_r

atlas = load_shen_268(1)
n=10000
n1=1000


def distance_mat_generator(nan_columns, file_name=None):
    coords = find_parcellation_cut_coords(atlas)
    # delete rows in nan_columns
    coords = np.delete(coords, nan_columns, axis=0)
    dm = []
    for i in range(coords.shape[0]):
        d = []
        for j in range(coords.shape[0]):
            d.append(np.sqrt(((coords[i] - coords[j]) ** 2).sum()))
        dm.append(d)
    distance_mat = np.vstack(dm)
    np.save(file_name, distance_mat)
    print(f'atlas distance matrix file done: {file_name}')


# distance_mat_generator(nan_all_for_spearman_r, 'shen_distance_fc.npy')


def surrogate_generator(target_data, distance_mat, surrogate_name):
    term_surrogates = []
    gen = Base(target_data, distance_mat)
    surrogate_maps = gen(n=n)
    term_surrogates.append(surrogate_maps)
    term_surrogates = np.vstack(term_surrogates)
    np.save(surrogate_name, term_surrogates)

    print(f'{surrogate_name} file done: {surrogate_name}')


# load csv file
window_0 = pd.read_csv('./data_generated/windows_pls/window_0.csv', header=None, index_col=None)
nan_columns = [50, 59, 108, 111, 114, 117, 128, 134, 135, 188, 238, 239, 242, 248, 249, 265]
for i in nan_columns:
    window_0.insert(i, f'NaN_{i}', np.nan)
print(window_0)

window_0_mild = pd.read_csv('./data_generated/windows_pls_mild/window_0.csv', header=None, index_col=None)
nan_columns_mild = [50, 59, 96, 99, 104, 107, 108, 111, 114, 115, 117, 128, 134, 135, 188, 235, 238, 239, 241, 242, 245, 248, 249, 251, 265]
for i in nan_columns_mild:
    window_0_mild.insert(i, f'NaN_{i}', np.nan)
print(window_0_mild)

lv_vals_mild_map = sio.loadmat('./data_generated/u1_df_mild.mat')
lv_vals_mild_map = lv_vals_mild_map['u1_df_mild']
lv_vals_mild_first = lv_vals_mild_map[:, 0]
for i in nan_columns_mild:
    lv_vals_mild_map = np.insert(lv_vals_mild_map, i, np.nan, axis=0)

lv_vals_deep_map = sio.loadmat('./data_generated/u1_df_deep.mat')
lv_vals_deep_map = lv_vals_deep_map['u1_df_deep']
lv_vals_deep_first = lv_vals_deep_map[:, 0]
for i in nan_columns:
    lv_vals_deep_map = np.insert(lv_vals_deep_map, i, np.nan, axis=0)



atlas_distance_mat_file = './data_generated/shen_distance.npy'
if not Path(atlas_distance_mat_file).exists():
    coords = find_parcellation_cut_coords(atlas)
    dm = []
    for i in range(coords.shape[0]):
        d = []
        for j in range(coords.shape[0]):
            d.append(np.sqrt(((coords[i] - coords[j]) ** 2).sum()))
        dm.append(d)
    distance_mat = np.vstack(dm)
    np.save(atlas_distance_mat_file,distance_mat)
else:
    distance_mat = np.load(atlas_distance_mat_file,allow_pickle=True)

print(f'atlas distance matrix file done: {atlas_distance_mat_file}')


atlas_distance_mat_file_mild_pls = './data_generated/shen_distance_mild_pls.npy'
if not Path(atlas_distance_mat_file_mild_pls).exists():
    coords = find_parcellation_cut_coords(atlas)
    # delete rows in nan_columns_mild
    coords = np.delete(coords, nan_columns_mild, axis=0)
    dm = []
    for i in range(coords.shape[0]):
        d = []
        for j in range(coords.shape[0]):
            d.append(np.sqrt(((coords[i] - coords[j]) ** 2).sum()))
        dm.append(d)
    distance_mat = np.vstack(dm)
    np.save(atlas_distance_mat_file_mild_pls, distance_mat)
else:
    distance_mat_mild_pls = np.load(atlas_distance_mat_file_mild_pls,allow_pickle=True)

print(f'atlas distance matrix file done: {atlas_distance_mat_file_mild_pls}')

atlas_distance_mat_file_deep_pls = './data_generated/shen_distance_deep_pls.npy'
if not Path(atlas_distance_mat_file_deep_pls).exists():
    coords = find_parcellation_cut_coords(atlas)
    # delete rows in nan_columns
    coords = np.delete(coords, nan_columns, axis=0)
    dm = []
    for i in range(coords.shape[0]):
        d = []
        for j in range(coords.shape[0]):
            d.append(np.sqrt(((coords[i] - coords[j]) ** 2).sum()))
        dm.append(d)
    distance_mat = np.vstack(dm)
    np.save(atlas_distance_mat_file_deep_pls, distance_mat)
else:
    distance_mat_deep_pls = np.load(atlas_distance_mat_file_deep_pls, allow_pickle=True)

print(f'atlas distance matrix file done: {atlas_distance_mat_file_deep_pls}')


terms = [
'action',
'adaptation',
'addiction',
'anticipation',
'anxiety',
'arousal',
'association',
'attention',
'autobiographical_memory',
'balance',
'belief',
'categorization',
'cognitive_control',
'communication',
'competition',
'concept',
'consciousness',
'consolidation',
'context',
'coordination',
'decision',
'decision_making',
'detection',
'discrimination',
'distraction',
'eating',
'efficiency',
'effort',
'emotion_regulation',
'emotions',
'empathy',
'encoding',
'episodic_memory',
'expectancy',
'expertise',
'extinction',
'face_recognition',
'facial_expression',
'familiarity',
'fear',
'fixation',
'focus',
'gaze',
'goal',
'hyperactivity',
'imagery',
'impulsivity',
'induction',
'inference',
'inhibition',
'insight',
'integration',
'intelligence',
'intention',
'interference',
'judgment',
'knowledge',
'language',
'language_comprehension',
'learning',
'listening',
'localization',
'loss',
'maintenance',
'manipulation',
'meaning',
'memory',
'memory_retrieval',
'mental_imagery',
'monitoring',
'mood',
'morphology',
'motor_control',
'movement',
'multisensory',
'naming',
'navigation',
'object_recognition',
'pain',
'perception',
'planning',
'priming',
'psychosis',
'reading',
'reasoning',
'recall',
'recognition',
'rehearsal',
'reinforcement_learning',
'response_inhibition',
'response_selection',
'retention',
'retrieval',
'reward_anticipation',
'rhythm',
'risk',
'rule',
'salience',
'search',
'selective_attention',
'semantic_memory',
'sentence_comprehension',
'skill',
'sleep',
'social_cognition',
'spatial_attention',
'speech_perception',
'speech_production',
'strategy',
'strength',
'stress',
'sustained_attention',
'task_difficulty',
'thought',
'uncertainty',
'updating',
'utility',
'valence',
'verbal_fluency',
'visual_attention',
'visual_perception',
'word_recognition',
'working_memory']


kept_terms_file = './data_generated/neurosynth_terms.npy'
kept_terms_map_file = './data_generated/neurosynth_terms_parcel_maps.npy'
term_files = [kept_terms_file, kept_terms_map_file]
files_exist = sum([Path(x).exists() for x in term_files])
if files_exist <2:
    masker = NiftiLabelsMasker(atlas,resampling_target='labels')
    files = glob.glob('Neurosynth/*') # HERE CHANGE PATH TO YOUR FILES DOWNLOADED FROM NEUROSYNTH
    kept_terms = [Path(x).stem.split('_')[0] for x in files]
    maps = np.vstack([masker.fit_transform(nibabel.load(f)).flatten() for f in files])
    np.save(kept_terms_file,kept_terms)
    np.save(kept_terms_map_file,maps)
else:
    kept_terms = np.load(kept_terms_file,allow_pickle=True)
    kept_terms_maps = np.load(kept_terms_map_file, allow_pickle=True)

print(f'kept terms file done: {kept_terms_file}')

term_sur_file ='./data_generated/term_surrogates.npy'
if not Path(term_sur_file).exists():
    term_surrogates = []
    for i in range(len(kept_terms)):
        print(i)
        gen = Base(kept_terms_maps[i,:], distance_mat) # create a surrogate generator based on the original map and the atlas distance matrix
        surrogate_maps = gen(n=n)
        term_surrogates.append(surrogate_maps)
    term_surrogates = np.vstack(term_surrogates)
    np.save(term_sur_file,term_surrogates)
else:
    term_surrogates = np.load(term_sur_file, allow_pickle=True)

print(f'term surrogates file done: {term_sur_file}')


term_sur_file_deep ='./data_generated/term_surrogates_deep.npy'
if not Path(term_sur_file_deep).exists():
    term_surrogates_deep = []
    # convert to numpy array
    window_0 = window_0.to_numpy()
    for i in range(len(window_0)):
        print(i)
        gen = Base(window_0[i,:], distance_mat)
        surrogate_maps = gen(n=n1)
        term_surrogates_deep.append(surrogate_maps)
    term_surrogates_deep = np.vstack(term_surrogates_deep)
    np.save(term_sur_file_deep,term_surrogates_deep)
else:
    term_surrogates_deep = np.load(term_sur_file_deep, allow_pickle=True)

print(f'term surrogates deep file done: {term_sur_file_deep}')

term_sur_file_mild ='./data_generated/term_surrogates_mild.npy'
if not Path(term_sur_file_mild).exists():
    term_surrogates_mild = []
    # convert to numpy array
    window_0_mild = window_0_mild.to_numpy()
    for i in range(len(window_0_mild)):
        print(i)
        gen = Base(window_0_mild[i,:], distance_mat)
        surrogate_maps = gen(n=n1)
        term_surrogates_mild.append(surrogate_maps)
    term_surrogates_mild = np.vstack(term_surrogates_mild)
    np.save(term_sur_file_mild,term_surrogates_mild)
else:
    term_surrogates_mild = np.load(term_sur_file_mild, allow_pickle=True)

print(f'term surrogates mild file done: {term_sur_file_mild}')

term_sur_file_mild_pls ='./data_generated/term_surrogates_mild_pls.npy'
if not Path(term_sur_file_mild_pls).exists():
    term_surrogates_mild_pls = []
    gen = Base(lv_vals_mild_first, distance_mat_mild_pls)
    surrogate_maps = gen(n=n)
    term_surrogates_mild_pls.append(surrogate_maps)
    term_surrogates_mild_pls = np.vstack(term_surrogates_mild_pls)
    np.save(term_sur_file_mild_pls,term_surrogates_mild_pls)
else:
    term_surrogates_mild_pls = np.load(term_sur_file_mild_pls, allow_pickle=True)

print(f'term surrogates mild pls file done: {term_sur_file_mild_pls}')

term_sur_file_deep_pls ='./data_generated/term_surrogates_deep_pls.npy'
if not Path(term_sur_file_deep_pls).exists():
    term_surrogates_deep_pls = []
    gen = Base(lv_vals_deep_first, distance_mat_deep_pls)
    surrogate_maps = gen(n=n)
    term_surrogates_deep_pls.append(surrogate_maps)
    term_surrogates_deep_pls = np.vstack(term_surrogates_deep_pls)
    np.save(term_sur_file_deep_pls,term_surrogates_deep_pls)
else:
    term_surrogates_deep_pls = np.load(term_sur_file_deep_pls, allow_pickle=True)

print(f'term surrogates deep pls file done: {term_sur_file_deep_pls}')


distance_mat_fc = np.load('./data_generated/shen_distance_fc.npy')
lv_val_movie = loadings[:, 4]
lv_val_rest = loadings[:, 5]
lv_val_effect_of_movie = loadings[:, 6]
lv_val_combined = loadings[:, 7]
# surrogate_generator(lv_val_movie, distance_mat_fc, 'fc_surrogates_movie.npy')
# surrogate_generator(lv_val_rest, distance_mat_fc, 'fc_surrogates_rest.npy')
# surrogate_generator(lv_val_effect_of_movie, distance_mat_fc, 'fc_surrogates_effect_of_movie.npy')
surrogate_generator(lv_val_combined, distance_mat_fc, 'fc_surrogates_combined.npy')