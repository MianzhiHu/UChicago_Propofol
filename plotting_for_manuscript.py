import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, pearsonr
import turtle
import pickle
from scipy.signal import welch
from nilearn import plotting
from atlasTransform.atlasTransform.utils.atlas import load_shen_268


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


# side_by_side_plot_generator(img1=plt.imread('./graphs/Combined Effects - FC.png'),
#                             img2=plt.imread('./graphs/Combined Effects - FC_left.png'),
#                             figure_length=5,
#                             figure_width=3,
#                             title1='Right Hemisphere',
#                             title2='Left Hemisphere',
#                             title='Combined Effects - FC',
#                             orientation='horizontal',
#                             dpi=650)
#
# side_by_side_plot_generator(img1=plt.imread('./graphs/edges_combined.png'),
#                             img2=plt.imread('./graphs/edges_combined_effects_plot.png'),
#                             figure_length=3,
#                             figure_width=5,
#                             title1='(a) Combined Effects (5-condition) PLS - Edges',
#                             title2='(b) Significant Edges in PLS - Combined Effects',
#                             title='Combined Effects - Edges',
#                             orientation='vertical',
#                             dpi=650)
#
# side_by_side_plot_generator(img1=plt.imread('./graphs/PLS_neurosynth_hurst.png'),
#                             img2=plt.imread('./graphs/Neurosynth_2_terms.png'),
#                             figure_length=5,
#                             figure_width=3,
#                             title1='(c) Decoupled Effects PLS - Neurosynth',
#                             title2='(d) Neurosynth Templates',
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


# four_consecutive_plot_generator(img1=plt.imread('./graphs/FC_Movie.png'),
#                                 img2=plt.imread('./graphs/FC_Movie_brain.png'),
#                                 img3=plt.imread('./graphs/edges_movie_abs.png'),
#                                 img4=plt.imread('./graphs/edges_movie_plot.png'),
#                                 figure_length=15,
#                                 figure_width=30,
#                                 title1='(a) Combined Effects Decoupled PLS - General FC',
#                                 title2='(b) Significant Brain Nodes in PLS - General FC',
#                                 title3='(c) Combined Effects Decoupled PLS - Edges',
#                                 title4='(d) Significant Brain Nodes in PLS - Edges',
#                                 title='Combined Effects - Supplementary',
#                                 orientation='vertical',
#                                 dpi=650)

# # plot the histogram of r-squared
# hurst_averages = pd.read_csv('./data_generated/hurst_averages.csv')
# n, bins, patches = plt.hist(hurst_averages['r_squared'], bins=30, density=True)
# density = gaussian_kde(hurst_averages['r_squared'])
# x = np.linspace(min(bins), max(bins), 1000)
# plt.plot(x, density(x), color='red')
# # reverse the x-axis
# plt.gca().invert_xaxis()
# # make the y-axis only contain integers
# plt.yticks(np.arange(0, 20, 5))
# plt.title('R-Squared Distribution - DFA')
# plt.xlabel('R-Squared')
# plt.ylabel('Frequency')
# plt.show()

# # create a three consecutive plot
# fig, ax = plt.subplots(3, 1, figsize=(20, 30), constrained_layout=True)
# for i in range(3):
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
#     ax[i].spines['top'].set_linewidth(0)
#     ax[i].spines['bottom'].set_linewidth(0)
#     ax[i].spines['left'].set_linewidth(0)
#     ax[i].spines['right'].set_linewidth(0)
# ax[0].imshow(plt.imread('./graphs/pls_movie.png'))
# ax[0].set_title('(a) Combined Effects Decoupled - Hurst', fontsize=30)
# ax[1].imshow(plt.imread('./graphs/Hurst_Movie.png'))
# ax[1].set_title('(b) Significant Brain Nodes in PLS - Hurst', fontsize=30)
# ax[2].imshow(plt.imread('./graphs/Combined Effects - Neurosynth.png'))
# # set position
# ax[0].set_position([0, 0.64, 1, 0.33])
# ax[1].set_position([0, 0.33, 1, 0.30])
# ax[2].set_position([0, 0, 1, 0.36])
# plt.savefig('./graphs/Combined Effects Decoupled', dpi=650)
# plt.show()

# # create a five consecutive plot
# fig, ax = plt.subplots(5, 1, figsize=(20, 45), constrained_layout=True)
# for i in range(5):
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
#     ax[i].spines['top'].set_linewidth(0)
#     ax[i].spines['bottom'].set_linewidth(0)
#     ax[i].spines['left'].set_linewidth(0)
#     ax[i].spines['right'].set_linewidth(0)
# ax[0].imshow(plt.imread('./graphs/hurst_combined.png'))
# ax[0].set_title('(a) Combined Effects (5-condition) PLS - Hurst', fontsize=30)
# ax[1].imshow(plt.imread('./graphs/average_hurst_combined.png'))
# ax[1].set_title('(b) Average H for Significant Brain Nodes in PLS', fontsize=30)
# ax[2].imshow(plt.imread('./graphs/Combined Effects - Hurst.png'))
# ax[2].set_title('(c) Significant Brain Nodes in PLS - Hurst', fontsize=30)
# ax[3].imshow(plt.imread('./graphs/fc_combined.png'))
# ax[3].set_title('(d) Combined Effects PLS (5-condition) - General FC', fontsize=30)
# ax[4].imshow(plt.imread('./graphs/Combined Effects - FC.png'))
# ax[4].set_title('(e) Significant Brain Nodes in PLS - General FC', fontsize=30)
# # set position
# ax[0].set_position([0, 0.79, 1, 0.19])
# ax[1].set_position([0, 0.58, 1, 0.19])
# ax[2].set_position([0, 0.38, 1, 0.18])
# ax[3].set_position([0, 0.19, 1, 0.19])
# ax[4].set_position([0, 0, 1, 0.18])
# # plt.savefig('./graphs/Combined Effects - 5', dpi=650)
# plt.show()

# # network summary
# movie_propofol_h = plt.imread('./graphs/brain_loading_Narrative Listening + Propofol - H.png')
# movie_propofol_fc = plt.imread('./graphs/brain_loading_Narrative Listening + Propofol - FC.png')
# movie_h = plt.imread('./graphs/brain_loading_Narrative Listening - H.png')
# movie_fc = plt.imread('./graphs/brain_loading_Narrative Listening - FC.png')
# propofol_h = plt.imread('./graphs/brain_loading_Propofol - H.png')
# propofol_fc = plt.imread('./graphs/brain_loading_Propofol - FC.png')
#
# fig, ax = plt.subplots(3, 2, figsize=(30, 20), constrained_layout=True)
# for i in range(3):
#     for j in range(2):
#         ax[i][j].set_xticks([])
#         ax[i][j].set_yticks([])
#         ax[i][j].spines['top'].set_linewidth(0)
#         ax[i][j].spines['bottom'].set_linewidth(0)
#         ax[i][j].spines['left'].set_linewidth(0)
#         ax[i][j].spines['right'].set_linewidth(0)
# ax[0][0].imshow(movie_propofol_h)
# ax[1][0].imshow(movie_h)
# ax[2][0].imshow(propofol_h)
# ax[0][1].imshow(movie_propofol_fc)
# ax[1][1].imshow(movie_fc)
# ax[2][1].imshow(propofol_fc)
# # set position
# ax[0][0].set_position([0, 0.66, 0.5, 0.33])
# ax[1][0].set_position([0, 0.33, 0.5, 0.33])
# ax[2][0].set_position([0, 0, 0.5, 0.33])
# ax[0][1].set_position([0.5, 0.66, 0.5, 0.33])
# ax[1][1].set_position([0.5, 0.33, 0.5, 0.33])
# ax[2][1].set_position([0.5, 0, 0.5, 0.33])
# plt.savefig('./graphs/network_summary.png', dpi=2100)
# plt.show()

# # make a huge plot
# fig, ax = plt.subplots(4, 2, figsize=(20, 30), constrained_layout=True)
# for i in range(4):
#     for j in range(2):
#         ax[i][j].set_xticks([])
#         ax[i][j].set_yticks([])
#         ax[i][j].spines['top'].set_linewidth(0)
#         ax[i][j].spines['bottom'].set_linewidth(0)
#         ax[i][j].spines['left'].set_linewidth(0)
#         ax[i][j].spines['right'].set_linewidth(0)
#         ax[i][j].title.set_fontsize(20)
# ax[0][0].imshow(plt.imread('./graphs/hurst_effect_of_movie.png'))
# ax[0][0].set_title('(a) Effect of Narrative Listening PLS - Hurst')
# ax[1][0].imshow(plt.imread('./graphs/Effect of Narrative Listening - Hurst.png'))
# ax[1][0].set_title('(b) Significant Brain Nodes for Narrative Listening - Hurst')
# ax[2][0].imshow(plt.imread('./graphs/fc_effect_of_movie.png'))
# ax[2][0].set_title('(c) Effect of Narrative Listening PLS - General FC')
# ax[3][0].imshow(plt.imread('./graphs/Effect of Narrative Listening - FC.png'))
# ax[3][0].set_title('(d) Significant Brain Nodes for Narrative Listening - General FC')
# ax[0][1].imshow(plt.imread('./graphs/rest_last.png'))
# ax[0][1].set_title('(e) Effect of Propofol PLS - Hurst')
# ax[1][1].imshow(plt.imread('./graphs/Effect of Propofol - Hurst.png'))
# ax[1][1].set_title('(f) Significant Brain Nodes for Propofol - Hurst')
# ax[2][1].imshow(plt.imread('./graphs/fc_rest_60.png'))
# ax[2][1].set_title('(g) Effect of Propofol PLS - General FC')
# ax[3][1].imshow(plt.imread('./graphs/Effect of Propofol - FC.png'))
# ax[3][1].set_title('(h) Significant Brain Nodes for Propofol - General FC')
# # set position
# ax[0][0].set_position([0, 0.75, 0.5, 0.25])
# ax[1][0].set_position([0, 0.5, 0.5, 0.25])
# ax[2][0].set_position([0, 0.25, 0.5, 0.25])
# ax[3][0].set_position([0, 0, 0.5, 0.25])
# ax[0][1].set_position([0.5, 0.75, 0.5, 0.25])
# ax[1][1].set_position([0.5, 0.5, 0.5, 0.25])
# ax[2][1].set_position([0.5, 0.25, 0.5, 0.25])
# ax[3][1].set_position([0.5, 0, 0.5, 0.25])
# plt.savefig('./graphs/individual_effects.png', dpi=2100)
# plt.show()

# plotting for figure 1
with open('./pickles/outcome_268.pickle', 'rb') as f:
    results_dict = pickle.load(f)
    counter = 0
    for key, value in results_dict.items():
        keys = list(results_dict.keys())
        values = [value['hurst'] for value in results_dict.values()]
        # vertically stack the hurst values
        hurst_values = np.vstack(values)
        r_squared = [value['r_squared'] for value in results_dict.values()]
        # vertically stack the r_squared values
        r_squared_values = np.vstack(r_squared)

# # load the data
example_high = np.load('./data_clean/02CB_01_rest_01_LPI_000.npy')
example_high1 = example_high[0,:]
example_high2 = example_high[4,:]

# print(pearsonr(example_high1, example_high2))


example_low = np.load('./data_clean/02CB_01_rest_03_LPI_000.npy')
example_low = example_low[2,:]

example_low1 = example_high[263, :]
example_low2 = example_high[243, :]

print(pearsonr(example_low2, example_low1))


# Function to plot Welch's power spectral density estimate
def plot_welch_spectrum(time_series, title):
    # Use different window sizes for spectral estimation
    for nperseg in [len(time_series) // 16, len(time_series) // 8, len(time_series) // 4, len(time_series) // 2]:
        freqs, powers = welch(time_series, nperseg=nperseg)
        plt.plot(freqs, powers, label=f'Window size: {nperseg}s')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.show()

# # plot the Welch's power spectral density estimate for the high and low Hurst time series
plot_welch_spectrum(example_high1, 'Power Spectral Density Estimate for High Hurst Time Series')
# plot_welch_spectrum(example_low, 'Power Spectral Density Estimate for Low Hurst Time Series')
plot_welch_spectrum(example_high2, 'Power Spectral Density Estimate for High Hurst Time Series')
plot_welch_spectrum(example_low1, 'Power Spectral Density Estimate for Low Hurst Time Series')
plot_welch_spectrum(example_low2, 'Power Spectral Density Estimate for Low Hurst Time Series')

#
# # plot example_high and example_low as subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), dpi=160)
# plt.xlim(-2, 255)
# plt.xlabel('Time', fontsize=13)
# ax1.plot(example_high, color='red')
# ax1.set_title('Example High Hurst Time Series')
# ax2.plot(example_low, color='blue')
# ax2.set_title('Example Low Hurst Time Series')
# # in the upper subplot, add a text
# ax1.text(0.05, 0.05, 'Hurst = 0.96', fontsize=11, color='black', transform=ax1.transAxes)
# # in the lower subplot, add a text
# ax2.text(0.05, 0.05, 'Hurst = 0.50', fontsize=11, color='black', transform=ax2.transAxes)
# plt.show()

# plot example_high and example_high1 in the same plot
plt.plot(example_high1, color='red', label='High Hurst Time Series')
plt.plot(example_high2, color='blue', label='High Hurst Time Series 1')
plt.xlabel('Time', fontsize=13)
plt.ylabel('Amplitude', fontsize=13)
plt.title('Example High Hurst Time Series')
plt.legend()
plt.savefig('./graphs/h-fc.png', dpi=650)
plt.show()

plt.plot(example_low1, color='red', label='Low Hurst Time Series')
plt.plot(example_low2, color='blue', label='Low Hurst Time Series 1')
plt.xlabel('Time', fontsize=13)
plt.ylabel('Amplitude', fontsize=13)
plt.title('Example Low Hurst Time Series')
plt.legend()
plt.savefig('./graphs/l-fc.png', dpi=650)
plt.show()


# draw the Sierpinski triangle example
def draw_sierpinski_triangle_specific_section(vertices, depth, ax, target_depth, current_depth=0):
    if current_depth == target_depth:
        # At the target depth, fill the triangle with a specific color
        ax.fill(*zip(*vertices), edgecolor="grey", facecolor="black", linewidth=0.5)
    elif current_depth < target_depth:
        # Calculate the midpoints of the triangle's sides
        midpoint01 = (vertices[0] + vertices[1]) / 2
        midpoint12 = (vertices[1] + vertices[2]) / 2
        midpoint20 = (vertices[2] + vertices[0]) / 2

        # Recursively draw smaller triangles
        draw_sierpinski_triangle_specific_section([vertices[0], midpoint01, midpoint20], depth, ax, target_depth, current_depth + 1)
        draw_sierpinski_triangle_specific_section([vertices[1], midpoint12, midpoint01], depth, ax, target_depth, current_depth + 1)
        draw_sierpinski_triangle_specific_section([vertices[2], midpoint20, midpoint12], depth, ax, target_depth, current_depth + 1)
    else:
        # For depths greater than the target, do not draw to leave the section empty
        return


# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.axis('off')
# plt.gca().set_facecolor('white')
#
# # remove x and y axis
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
#
#
# # Define the vertices of the initial triangle
# vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
#
# # Draw the specific 1/3 section of the Sierpinski triangle by targeting a specific depth
# total_depth = 5  # Total depth of the whole figure
# target_depth = total_depth - 1  # Depth of the section to highlight (1 level above the total to represent 1/3 of the figure)
#
# # Draw the entire Sierpinski triangle but highlight a specific section
# draw_sierpinski_triangle_specific_section(vertices, total_depth, ax, target_depth)
#
# plt.show()


def draw_sierpinski_line_improved_light_gray(vertices, depth, ax, level=0):
    if depth == 0:
        # Use a gradient of gray shades based on the recursion depth
        gray_value = 0.8 - (level * 0.1) % 0.55  # Cycle through lighter shades of gray
        ax.fill(*zip(*vertices), edgecolor="grey", facecolor="grey", linewidth=0.5)
    else:
        # Calculate the midpoints of the triangle's sides
        midpoint01 = (vertices[0] + vertices[1]) / 2
        midpoint12 = (vertices[1] + vertices[2]) / 2
        midpoint20 = (vertices[2] + vertices[0]) / 2

        # Recursively draw smaller triangles
        draw_sierpinski_line_improved_light_gray([vertices[0], midpoint01, midpoint20], depth - 1, ax, level + 1)
        draw_sierpinski_line_improved_light_gray([vertices[1], midpoint12, midpoint01], depth - 1, ax, level + 1)
        draw_sierpinski_line_improved_light_gray([vertices[2], midpoint20, midpoint12], depth - 1, ax, level + 1)


# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.axis('off')
# plt.gca().set_facecolor('white')
#
# # Define the vertices of the initial triangle
# vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
#
# # Draw the Sierpinski triangle using filled triangles with a specified depth of recursion
# draw_sierpinski_line_improved_light_gray(vertices, 5, ax)
#
# plt.show()


# # plotting for figure 2
# # Plot a blank glass brain
# plotting.plot_glass_brain(None, display_mode='r')
#
# # Show the plot
# plotting.show()

# to illustrate our point, I need to simulate a highly stereotyped, repetitive, and predictable time series
# I will use a sine wave to illustrate this point
time = np.arange(0, 100, 1)

# generate 500 sine waves with random phase shifts
sine_waves = [np.sin(time + np.random.uniform(0, 2 * np.pi)) for _ in range(1000)]

import random
# randomly select 500 sine waves
random_sine_waves = random.sample(sine_waves, 500)
mean_sine_wave = np.mean(random_sine_waves, axis=0)

# pick the other half of the sine waves
random_sine_waves_2 = random.sample(sine_waves, 500)
mean_sine_wave_2 = np.mean(random_sine_waves_2, axis=0)

# plot the sine waves together with the mean sine wave
plt.plot(time, mean_sine_wave, label='Mean Sine Wave')
plt.plot(time, mean_sine_wave_2, label='Mean Sine Wave 2')
plt.legend()
plt.show()


