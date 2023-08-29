import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


def side_by_side_plot_generator(img1, img2, figure_length, figure_width, title, orientation, title1=None, title2=None, loc1=None,
                                loc2=None, dpi=None):
    if orientation == 'horizontal':
        fig, ax = plt.subplots(1, 2, figsize=(figure_length, figure_width), constrained_layout=True, dpi=dpi)
        ax[0].set_position([0, 0, 0.5, 1])
        ax[1].set_position([0.5, 0, 0.5, 1])
    if orientation == 'vertical':
        fig, ax = plt.subplots(2, 1, figsize=(figure_length, figure_width), constrained_layout=True, dpi=dpi)
        ax[0].set_position([0, 0.5, 1, 0.5])
        ax[1].set_position([0, 0, 1, 0.5])
        # ax[0].set_position([0, 0.33, 1, 0.66])
        # ax[1].set_position([0, 0, 1, 0.33])
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


# side_by_side_plot_generator(img1=plt.imread('./graphs/brain loadings - fc.png'),
#                             img2=plt.imread('./graphs/brain loadings - fc_left.png'),
#                             figure_length=5,
#                             figure_width=3,
#                             title1='Right Hemisphere',
#                             title2='Left Hemisphere',
#                             title='FC_Movie_brain',
#                             orientation='horizontal',
#                             dpi=650)

side_by_side_plot_generator(img1=plt.imread('./graphs/PLS summary.png'),
                            img2=plt.imread('./graphs/edges_effect_of_propofol_plot.png'),
                            figure_length=3,
                            figure_width=5,
                            # title1='(a) Effect of Propofol - Edges',
                            title2='(c) Significant Edges With Full Data',
                            title='PLS summary',
                            orientation='vertical',
                            dpi=600)

# side_by_side_plot_generator(img1=plt.imread('./graphs/PLS_neurosynth_hurst.png'),
#                             img2=plt.imread('./graphs/Neurosynth_2_terms.png'),
#                             figure_length=5,
#                             figure_width=3,
#                             title1='(a) Combined Effects - Neurosynth',
#                             title2='(b) Neurosynth Templates',
#                             title='Combined Effects - Neurosynth',
#                             orientation='horizontal',
#                             dpi=600)


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


# four_consecutive_plot_generator(img1=plt.imread('./graphs/pls_movie.png'),
#                                 img2=plt.imread('./graphs/Hurst_Movie.png'),
#                                 img3=plt.imread('./graphs/FC_Movie.png'),
#                                 img4=plt.imread('./graphs/FC_Movie_brain.png'),
#                                 figure_length=15,
#                                 figure_width=30,
#                                 title1='(a) Combined Effects PLS - Hurst',
#                                 title2='(b) Significant Brain Nodes in PLS - Hurst',
#                                 title3='(c) Combined Effects PLS - General FC',
#                                 title4='(d) Significant Brain Nodes in PLS - General FC',
#                                 title='Combined Effects',
#                                 orientation='vertical',
#                                 dpi=650)

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


movie_propofol_h = plt.imread('./graphs/brain_loading_Narrative Listening + Propofol - H.png')
movie_propofol_fc = plt.imread('./graphs/brain_loading_Narrative Listening + Propofol - FC.png')
movie_h = plt.imread('./graphs/brain_loading_Narrative Listening - H.png')
movie_fc = plt.imread('./graphs/brain_loading_Narrative Listening - FC.png')
propofol_h = plt.imread('./graphs/brain_loading_Propofol - H.png')
propofol_fc = plt.imread('./graphs/brain_loading_Propofol - FC.png')

fig, ax = plt.subplots(3, 2, figsize=(30, 20), constrained_layout=True)
for i in range(3):
    for j in range(2):
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
        ax[i][j].spines['top'].set_linewidth(0)
        ax[i][j].spines['bottom'].set_linewidth(0)
        ax[i][j].spines['left'].set_linewidth(0)
        ax[i][j].spines['right'].set_linewidth(0)
ax[0][0].imshow(movie_propofol_h)
ax[1][0].imshow(movie_h)
ax[2][0].imshow(propofol_h)
ax[0][1].imshow(movie_propofol_fc)
ax[1][1].imshow(movie_fc)
ax[2][1].imshow(propofol_fc)
# set position
ax[0][0].set_position([0, 0.66, 0.5, 0.33])
ax[1][0].set_position([0, 0.33, 0.5, 0.33])
ax[2][0].set_position([0, 0, 0.5, 0.33])
ax[0][1].set_position([0.5, 0.66, 0.5, 0.33])
ax[1][1].set_position([0.5, 0.33, 0.5, 0.33])
ax[2][1].set_position([0.5, 0, 0.5, 0.33])
plt.savefig('./graphs/network_summary.png', dpi=2100)
plt.show()




