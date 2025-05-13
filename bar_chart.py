import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
import numpy as np
from plotting_function import contrast_names


# read the data
measurement = 'Edges'
# '8_condition', 'Effects of Narrative-Listening', 'Effects of Propofol', 'Effects of Propofol on Narrative-Listening'
contrast = 'Effects of Propofol on Narrative-Listening'
flip = True
contrast_name = contrast_names[contrast]
LV = 1
LV_loc = LV - 1

x_ticks_map = {
    '8_condition': ['Rest Awake', 'Rest Mild', 'Rest Deep', 'Rest Recovery', 'NL Awake', 'NL Mild', 'NL Deep', 'NL Recovery'],
    'Effects of Narrative-Listening': ['Rest Awake', 'NL Awake'],
    'Effects of Propofol': ['Rest Awake', 'Rest Mild', 'Rest Deep', 'Rest Recovery'],
    'Effects of Propofol on Narrative-Listening': ['NL Awake', 'NL Mild', 'NL Deep', 'NL Recovery']
}
x_ticks = x_ticks_map[contrast]

base = './data_generated'
lv_vals_path = f'{base}/PLS_results/PLS_outputTaskPLS{measurement}_{contrast_name}_lv_vals.mat'
boot_ratio_path = f'{base}/PLS_results/PLS_outputTaskPLS{measurement}_{contrast_name}.mat'
missing_path = f'{base}/Contrasts/{contrast}/{measurement}/missing_columns.csv'

lv_vals = sio.loadmat(lv_vals_path)
design = sio.loadmat(boot_ratio_path)['result'][0]['boot_result'][0][0]['orig_usc'][0][:, LV_loc]
ll = sio.loadmat(boot_ratio_path)['result'][0]['boot_result'][0][0]['llusc'][0][:, LV_loc]
ul = sio.loadmat(boot_ratio_path)['result'][0]['boot_result'][0][0]['ulusc'][0][:, LV_loc]

# flip the sign of the design matrix
if flip:
    bar_heights = -design
    lower_err = abs(-ul + design)
    upper_err = abs(-design + ll)
else:
    bar_heights = design
    lower_err = abs(ll - design)
    upper_err = abs(design - ul)

# flip the order
if contrast == '8_condition':
    k = 4
    bar_heights = np.concatenate([bar_heights[-k:], bar_heights[:-k]])
    lower_err = np.concatenate([lower_err[-k:], lower_err[:-k]])
    upper_err = np.concatenate([upper_err[-k:], upper_err[:-k]])
elif contrast == 'Effects of Narrative-Listening':
    k = 1
    bar_heights = np.concatenate([bar_heights[-k:], bar_heights[:-k]])
    lower_err = np.concatenate([lower_err[-k:], lower_err[:-k]])
    upper_err = np.concatenate([upper_err[-k:], upper_err[:-k]])

colors = ['#1f77b4' if h < 0 else '#d62728' for h in bar_heights]

# start plotting
sns.set_theme(style='white', font='Arial')
fig, ax = plt.subplots(figsize=(6, 6))
ax.bar(x_ticks, bar_heights, yerr=[lower_err, upper_err], capsize=5, color=colors, edgecolor='black')
ax.set_ylabel('Design Salience', fontsize=20)
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.axhline(0, color='black', linewidth=0.8)
sns.despine()
plt.tight_layout()
plt.savefig(f'./graphs/{measurement}_{contrast}_LV{LV}.png', dpi=600)
plt.show()


