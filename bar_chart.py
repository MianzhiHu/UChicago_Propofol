import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
import numpy as np


# read the data
measurement = 'FC'
contrast = '8_condition'
LV = 1
LV_loc = LV - 1

base = './data_generated'
lv_vals_path = f'{base}/PLS_results/PLS_outputTaskPLS{measurement}_{contrast}_lv_vals.mat'
boot_ratio_path = f'{base}/PLS_results/PLS_outputTaskPLS{measurement}_{contrast}.mat'
missing_path = f'{base}/Contrasts/{contrast}/{measurement}/missing_columns.csv'

lv_vals = sio.loadmat(lv_vals_path)
design = sio.loadmat(boot_ratio_path)['result'][0]['boot_result'][0][0]['orig_usc'][0][:, LV_loc]
ll = sio.loadmat(boot_ratio_path)['result'][0]['boot_result'][0][0]['llusc'][0][:, LV_loc]
ul = sio.loadmat(boot_ratio_path)['result'][0]['boot_result'][0][0]['ulusc'][0][:, LV_loc]
lower_err = abs(-ul + design)
upper_err = abs(-design + ll)

# flip the sign of the design matrix
bar_heights = -design

# flip the order
k = 4
bar_heights = np.concatenate([bar_heights[-k:], bar_heights[:-k]])
lower_err = np.concatenate([lower_err[-k:], lower_err[:-k]])
upper_err = np.concatenate([upper_err[-k:], upper_err[:-k]])

colors = ['#1f77b4' if h < 0 else '#d62728' for h in bar_heights]
x_ticks = ['Rest Awake', 'Rest Mild', 'Rest Deep', 'Rest Recovery', 'NL Awake', 'NL Mild', 'NL Deep', 'NL Recovery']

# start plotting
sns.set_theme(style='white', font='Arial')
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x_ticks, bar_heights, yerr=[lower_err, upper_err], capsize=5, color=colors, edgecolor='black')
ax.set_ylabel('Design Salience', fontsize=14)
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
ax.axhline(0, color='black', linewidth=0.8)
sns.despine()
plt.tight_layout()
plt.savefig(f'./graphs/{measurement}_{contrast}_LV{LV}.png', dpi=300)
plt.show()


