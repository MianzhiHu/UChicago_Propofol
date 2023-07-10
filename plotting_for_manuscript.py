import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

SL_DFA = plt.imread('./graphs/SL-DFA.png')
NL_DFA = plt.imread('./graphs/NL-DFA.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True, constrained_layout=True)
ax[0].imshow(SL_DFA)
ax[0].set_title('Main Effect of Sedation Level - DFA')
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(NL_DFA)
ax[1].set_title('Main Effect of Narrative Listening - DFA')
# remove the x and y ticks
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()

SL_wavelet = plt.imread('./graphs/SL-wavelet.png')
NL_wavelet = plt.imread('./graphs/NL-wavelet.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True, constrained_layout=True)
ax[0].imshow(SL_wavelet)
ax[0].set_title('Main Effect of Sedation Level - Wavelet')
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(NL_wavelet)
ax[1].set_title('Main Effect of Narrative Listening - Wavelet')
# remove the x and y ticks
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()

ave_hurst = plt.imread('./graphs/ave_hurst_movie.png')
brain_loadings = plt.imread('./graphs/brain loadings.png')
brain_loadings_left = plt.imread('./graphs/brain loadings_left.png')
pls_movie = plt.imread('./graphs/pls_movie.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
# remove the borders
for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0)
    ax[1].spines[axis].set_linewidth(0)
# make the two subplots of the same height
ax[0].set_position([0.05, 0.05, 0.45, 0.9])
ax[1].set_position([0.50, 0.05, 0.45, 0.9])
ax[0].imshow(pls_movie)
ax[0].set_title('The Latent Variable of Propofol - Narrative Listening', y = 1.0)
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(ave_hurst)
ax[1].set_title('Hurst values for the Significant Brain Nodes in PLS', y = 1.0)
# remove the x and y ticks
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, dpi=650)
# remove the borders
for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0)
    ax[1].spines[axis].set_linewidth(0)
# make the two subplots of the same height
ax[0].set_position([0.05, 0.05, 0.45, 0.9])
ax[1].set_position([0.50, 0.05, 0.45, 0.9])
ax[0].imshow(brain_loadings_left)
ax[0].set_title('Left Hemisphere', y = 1.0)
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(brain_loadings)
ax[1].set_title('Right Hemisphere', y = 1.0)
# remove the x and y ticks
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()

pls_all = plt.imread('./graphs/pls_all.png')
brain_loadings_all = plt.imread('./graphs/brain loadings - all.png')

fig, ax = plt.subplots(2, 1, figsize=(4, 6), constrained_layout=True, dpi=300)
# remove the borders
for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0)
    ax[1].spines[axis].set_linewidth(0)
# # make the two subplots of the same height
ax[0].set_position([0.05, 0.55, 0.9, 0.45])
ax[1].set_position([0.05, 0.05, 0.9, 0.45])
ax[0].imshow(pls_all)
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(brain_loadings_all)
# remove the x and y ticks
ax[1].set_xticks([])
ax[1].set_yticks([])
# set a left-aligned title
ax[0].set_title('(a)', loc='left')
ax[1].set_title('(b)', loc='left')
plt.show()



hurst_averages = pd.read_csv('./data_generated/hurst_averages.csv')
n, bins, patches = plt.hist(hurst_averages['r_squared'], bins=30, density=True)
density = gaussian_kde(hurst_averages['r_squared'])
x = np.linspace(min(bins), max(bins), 1000)
plt.plot(x, density(x), color='red')
# reverse the x-axis
plt.gca().invert_xaxis()
# make the y-axis only contain integers
plt.yticks(np.arange(0, 20, 5))
plt.title('R-Squared Distribution - DFA')
plt.xlabel('R-Squared')
plt.ylabel('Freuency')
plt.show()


p_mild = plt.imread('./graphs/p-value.png')
p_deep = plt.imread('./graphs/p-values - deep.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
# remove the borders
for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0)
    ax[1].spines[axis].set_linewidth(0)
# make the two subplots of the same height
ax[0].set_position([0.05, 0.05, 0.45, 0.9])
ax[1].set_position([0.50, 0.05, 0.45, 0.9])
ax[0].imshow(p_mild)
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(p_deep)
# remove the x and y ticks
ax[1].set_xticks([])
ax[1].set_yticks([])
plt.show()

corr_mild = plt.imread('./graphs/PLS_mild_correlation.png')
corr_deep = plt.imread('./graphs/PLS_deep_correlation.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
# remove the borders
for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0)
    ax[1].spines[axis].set_linewidth(0)
# make the two subplots of the same height
ax[0].set_position([0.05, 0.05, 0.45, 0.9])
ax[1].set_position([0.50, 0.05, 0.45, 0.9])
ax[0].imshow(corr_mild)
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Mild Sedation - Narrative Listening')
ax[1].imshow(corr_deep)
# remove the x and y ticks
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Deep Sedation - Narrative Listening')
plt.show()


mild_el = plt.imread('./graphs/mild-el.png')
deep_el = plt.imread('./graphs/deep-el.png')
brain_loadings_mild = plt.imread('./graphs/brain loadings - mild_left.png')
brain_loadings_deep = plt.imread('./graphs/brain loadings - deep_left.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, dpi=300)
# remove the borders
for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0)
    ax[1].spines[axis].set_linewidth(0)
# make the two subplots of the same height
ax[0].set_position([0.05, 0.05, 0.45, 0.9])
ax[1].set_position([0.50, 0.05, 0.45, 0.9])
ax[0].imshow(brain_loadings_mild)
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Mild Sedation - Early Versus Late')
ax[1].imshow(brain_loadings_deep)
# remove the x and y ticks
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Deep Sedation - Early Versus Late')
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
# remove the borders
for axis in ['top','bottom','left','right']:
    ax[0][0].spines[axis].set_linewidth(0)
    ax[0][1].spines[axis].set_linewidth(0)
    ax[1][0].spines[axis].set_linewidth(0)
    ax[1][1].spines[axis].set_linewidth(0)
# make the two subplots of the same height
ax[0][0].set_position([0.05, 0.50, 0.45, 0.45])
ax[0][1].set_position([0.50, 0.50, 0.45, 0.45])
ax[1][0].set_position([0.05, 0.05, 0.45, 0.45])
ax[1][1].set_position([0.50, 0.05, 0.45, 0.45])
ax[0][0].imshow(mild_el)
# remove the x and y ticks
ax[0][0].set_xticks([])
ax[0][0].set_yticks([])
ax[0][0].set_title('Mild Sedation - Early Versus Late')
# set the title font
ax[0][0].title.set_fontsize(16)
# make it bold
ax[0][0].title.set_fontweight('bold')
ax[0][1].imshow(deep_el)
ax[0][1].set_title('Deep Sedation - Early Versus Late')
ax[0][1].title.set_fontsize(16)
ax[0][1].title.set_fontweight('bold')
# remove the x and y ticks
ax[0][1].set_xticks([])
ax[0][1].set_yticks([])
ax[1][0].imshow(brain_loadings_mild)
ax[1][0].set_title('Mild Sedation - Brain Loadings')
ax[1][0].title.set_fontsize(16)
ax[1][0].title.set_fontweight('bold')
# remove the x and y ticks
ax[1][0].set_xticks([])
ax[1][0].set_yticks([])
ax[1][1].imshow(brain_loadings_deep)
ax[1][1].set_title('Deep Sedation - Brain Loadings')
ax[1][1].title.set_fontsize(16)
ax[1][1].title.set_fontweight('bold')
# remove the x and y ticks
ax[1][1].set_xticks([])
ax[1][1].set_yticks([])
plt.show()

mild_network = plt.imread('./graphs/brain_loading_Mild Sedation.png')
deep_network = plt.imread('./graphs/brain_loading_Deep Sedation.png')

fig, ax = plt.subplots(2, 1, figsize=(5, 8), constrained_layout=True, dpi=500)
# remove the borders
for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0)
    ax[1].spines[axis].set_linewidth(0)
# make the two subplots of the same height
ax[0].set_position([0.05, 0.50, 0.9, 0.45])
ax[1].set_position([0.05, 0.10, 0.9, 0.45])
ax[0].imshow(mild_network)
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Mild Sedation - Brain Network')
ax[1].imshow(deep_network)
# remove the x and y ticks
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Deep Sedation - Brain Network')
plt.show()

