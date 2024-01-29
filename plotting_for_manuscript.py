import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import turtle


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
#
#

def draw_triangle(points, color, my_turtle):
    my_turtle.fillcolor(color)
    my_turtle.up()
    my_turtle.goto(points[0][0], points[0][1])
    my_turtle.down()
    my_turtle.begin_fill()
    my_turtle.goto(points[1][0], points[1][1])
    my_turtle.goto(points[2][0], points[2][1])
    my_turtle.goto(points[0][0], points[0][1])
    my_turtle.end_fill()


def midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)


def sierpinski(points, degree, my_turtle):
    colormap = ['red', 'gold', 'green', 'skyblue']
    # colormap = ['black', 'white', 'black', 'black']
    draw_triangle(points, colormap[degree], my_turtle)
    if degree > 0:
        sierpinski([points[0], midpoint(points[0], points[1]), midpoint(points[0], points[2])], degree-1, my_turtle)
        sierpinski([points[1], midpoint(points[0], points[1]), midpoint(points[1], points[2])], degree-1, my_turtle)
        sierpinski([points[2], midpoint(points[2], points[1]), midpoint(points[0], points[2])], degree-1, my_turtle)


my_turtle = turtle.Turtle()
screen = turtle.Screen()
points = [[-200, -150], [0, 200], [200, -150]]
sierpinski(points, 3, my_turtle)

canvas = turtle.getcanvas()
canvas.postscript(file="sierpinski_triangle.ps", colormode='color')
screen.exitonclick()



