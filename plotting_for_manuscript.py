import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from PIL import Image


def side_by_side_plot_generator(img1, img2, figure_length, figure_width, title1, title2, title, orientation, loc1=None,
                                loc2=None, dpi=None):
    if orientation == 'horizontal':
        fig, ax = plt.subplots(1, 2, figsize=(figure_length, figure_width), constrained_layout=True, dpi=dpi)
        ax[0].set_position([0, 0, 0.5, 1])
        ax[1].set_position([0.5, 0, 0.5, 1])
    if orientation == 'vertical':
        fig, ax = plt.subplots(2, 1, figsize=(figure_length, figure_width), constrained_layout=True, dpi=dpi)
        ax[0].set_position([0, 0.5, 1, 0.5])
        ax[1].set_position([0, 0, 1, 0.5])
    # remove the borders
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[0].spines[axis].set_linewidth(0)
        ax[1].spines[axis].set_linewidth(0)
    ax[0].imshow(img1)
    ax[0].set_title(title1, loc=loc1)
    # remove the x and y ticks
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].imshow(img2)
    ax[1].set_title(title2, loc=loc2)
    # remove the x and y ticks
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    for i in range(2):
        ax[i].title.set_fontsize(8)
    # save the figure
    plt_name = title + '.png'
    plt_dir = './graphs/'
    plt_path = plt_dir + plt_name
    plt.savefig(plt_path)
    plt.show()


side_by_side_plot_generator(img1=plt.imread('./graphs/brain loadings - fc.png'),
                            img2=plt.imread('./graphs/brain loadings - fc_left.png'),
                            figure_length=5,
                            figure_width=3,
                            title1='Right Hemisphere',
                            title2='Left Hemisphere',
                            title='FC_Movie_brain',
                            orientation='horizontal',
                            dpi=650)

side_by_side_plot_generator(img1=plt.imread('./graphs/edges_movie_abs.png'),
                            img2=plt.imread('./graphs/edges_movie_plot.png'),
                            figure_length=3,
                            figure_width=5,
                            title1='(a) Combined Effects PLS - Edges',
                            title2='(b) Significant Edges in PLS - Edges',
                            title='Combined Effects - Edges',
                            orientation='vertical',
                            dpi=600)

side_by_side_plot_generator(img1=plt.imread('./graphs/PLS_neurosynth_hurst.png'),
                            img2=plt.imread('./graphs/Neurosynth_2_terms.png'),
                            figure_length=5,
                            figure_width=3,
                            title1='(a) Combined Effects - Neurosynth',
                            title2='(b) Neurosynth Templates',
                            title='Combined Effects - Neurosynth',
                            orientation='horizontal',
                            dpi=600)


def four_consecutive_plot_generator(img1, img2, img3, img4, figure_length, figure_width,
                                    title1, title2, title3, title4, title, orientation,
                                    loc=None, dpi=None):
    if orientation == 'horizontal':
        fig, ax = plt.subplots(1, 4, figsize=(figure_length, figure_width), constrained_layout=True, dpi=dpi)
        ax[0].set_position([0, 0, 0.25, 1])
        ax[1].set_position([0.25, 0, 0.25, 1])
        ax[2].set_position([0.5, 0, 0.25, 1])
        ax[3].set_position([0.75, 0, 0.25, 1])
    if orientation == 'vertical':
        fig, ax = plt.subplots(4, 1, figsize=(figure_length, figure_width), constrained_layout=True, dpi=dpi)
        ax[0].set_position([0, 0.72, 1, 0.24])
        ax[1].set_position([0, 0.48, 1, 0.24])
        ax[2].set_position([0, 0.24, 1, 0.24])
        ax[3].set_position([0, 0, 1, 0.24])
    # remove the borders
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[0].spines[axis].set_linewidth(0)
        ax[1].spines[axis].set_linewidth(0)
        ax[2].spines[axis].set_linewidth(0)
        ax[3].spines[axis].set_linewidth(0)
    ax[0].imshow(img1)
    ax[0].set_title(title1, loc=loc)
    ax[1].imshow(img2)
    ax[1].set_title(title2, loc=loc)
    ax[2].imshow(img3)
    ax[2].set_title(title3, loc=loc)
    ax[3].imshow(img4)
    ax[3].set_title(title4, loc=loc)
    # remove the x and y ticks
    for i in range(4):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].title.set_fontsize(20)
    # save the figure
    plt_name = title + '.png'
    plt_dir = './graphs/'
    plt_path = plt_dir + plt_name
    plt.savefig(plt_path)
    plt.show()


four_consecutive_plot_generator(img1=plt.imread('./graphs/pls_movie.png'),
                                img2=plt.imread('./graphs/Hurst_Movie.png'),
                                img3=plt.imread('./graphs/FC_Movie.png'),
                                img4=plt.imread('./graphs/FC_Movie_brain.png'),
                                figure_length=15,
                                figure_width=30,
                                title1='(a) Combined Effects PLS - Hurst',
                                title2='(b) Significant Brain Nodes in PLS - Hurst',
                                title3='(c) Combined Effects PLS - General FC',
                                title4='(d) Significant Brain Nodes in PLS - General FC',
                                title='Combined Effects',
                                orientation='vertical',
                                dpi=650)






SL_DFA = plt.imread('./graphs/SL-DFA.png')
NL_DFA = plt.imread('./graphs/NL-DFA.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True, constrained_layout=True)
ax[0].set_position([0.05, 0.025, 0.45, 0.95])
ax[1].set_position([0.50, 0.025, 0.45, 0.95])
ax[0].imshow(SL_DFA)
ax[0].set_title('Main Effect of Sedation Level')
# remove the x and y ticks
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(NL_DFA)
ax[1].set_title('Main Effect of Narrative Listening')
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

