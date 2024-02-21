import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from dfa import dfa

target_dir = 'data_clean'


def read_files(directory: str):
    for file in os.listdir(directory):
        if file.endswith(".netts"):
            # load data as numpy 2d array
            array_2d = np.loadtxt(os.path.join(target_dir, file), delimiter='\t')
            yield array_2d, file

# for array_2d, file in read_files(directory=target_dir):
#     print(array_2d.shape)


def preprocess():
    tr: int = 2  # 2000 ms
    max_frequency = 0.1
    min_frequency = 0.01
    mn = int(np.ceil(1 / (tr * max_frequency)))
    mx = int(np.floor(1 / (tr * min_frequency)))
    counter = 0
    results_dict = {}
    for array_2d, file in read_files(directory=target_dir):
        dfa_results = dfa(x=array_2d.T, max_window_size=mx, min_window_size=mn, return_confidence_interval=True,
                          return_windows=False)
        split_h = len(dfa_results[0])
        # turn results into a 2d array, where each column is a key
        hurst = np.hsplit(np.array(np.array(dfa_results[0])), split_h)
        print(dfa_results[0])
        print(dfa_results[1])
        print(f'length of dfa_results[0]: {len(dfa_results[0])}')
        print(f'length of dfa_results[1]: {len(dfa_results[1])}')
        cis0 = np.array([item[0] for item in dfa_results[1]])
        cis1 = np.array([item[1] for item in dfa_results[1]])
        cis0 = np.hsplit(cis0, split_h)
        cis1 = np.hsplit(cis1, split_h)
        r_squared = np.hsplit(np.array(np.array(dfa_results[2])), split_h)

        array_new = np.hstack((hurst, cis0, cis1, r_squared))
        df = pd.DataFrame(array_new, columns=['hurst', 'cis0', 'cis1', 'r_squared'])
        results_dict[file] = df
        counter += 1

        print(f'processing done for {file}')

    print('finished processing all files, saving to pickle')
    with open('outcome.pickle', 'wb') as outfile:
        pickle.dump(results_dict, outfile)


#preprocess()

#     print('show results from pickle')
#
#     # Average across movie and rest condition
#     print(f'# Averaging across movie and rest condition')
#     movie_average: list = []
#     rest_average: list = []
#
#     with open('outcome_268.pickle', 'rb') as f:
#         results_dict = pickle.load(f)
#         counter = 0
#         for key, value in results_dict.items():
#             if 'movie' in key:
#                 counter += 1
#                 movie_average.append(value['hurst'].mean())
#                 movie_total_average = mean(movie_average)
#             elif 'rest' in key:
#                 counter += 1
#                 rest_average.append(value['hurst'].mean())
#                 rest_total_average = mean(rest_average)
#         print(f'movie_average is {movie_average}')
#         print(f'rest_average is {rest_average}')
#         print(f'movie_total_average is {movie_total_average}')
#         print(f'rest_total_average is {rest_total_average}')
#         print(stats.ttest_ind(movie_average, rest_average))
#
#         df1 = [movie_average, rest_average]
#         plt.boxplot(df1)
#         plt.xticks([1, 2], ['movie_average', 'rest_average'])
#         plt.show()
#
#     # Average across sedation levels
#     print(f'# Averaging across sedation levels')
#     awake_average: list = []
#     mild_average: list = []
#     deep_average: list = []
#     recovery_average: list = []
#
#     with open('outcome_268.pickle', 'rb') as f:
#         results_dict = pickle.load(f)
#         counter = 0
#         for key, value in results_dict.items():
#             if '_01_LPI' in key:
#                 counter += 1
#                 awake_average.append(value['hurst'].mean())
#                 awake_total_average = mean(awake_average)
#             elif '_02_LPI' in key:
#                 counter += 1
#                 mild_average.append(value['hurst'].mean())
#                 mild_total_average = mean(mild_average)
#             elif '_03_LPI' in key:
#                 counter += 1
#                 deep_average.append(value['hurst'].mean())
#                 deep_total_average = mean(deep_average)
#             elif '_04_LPI' in key:
#                 counter += 1
#                 recovery_average.append(value['hurst'].mean())
#                 recovery_total_average = mean(recovery_average)
#         print(f'awake_average is {awake_average}')
#         print(f'mild_average is {mild_average}')
#         print(f'deep_average is {deep_average}')
#         print(f'recovery_average is {recovery_average}')
#         print(f'awake_total_average is {awake_total_average}')
#         print(f'mild_total_average is {mild_total_average}')
#         print(f'deep_total_average is {deep_total_average}')
#         print(f'recovery_total_average is {recovery_total_average}')
#         print(stats.f_oneway(awake_average, mild_average, deep_average, recovery_average))
#         print(stats.ttest_ind(deep_average, awake_average))
#         print(stats.kruskal(awake_average, mild_average, deep_average, recovery_average))
#
#         df2 = [awake_average, mild_average, deep_average, recovery_average]
#         plt.boxplot(df2)
#         plt.xticks([1, 2, 3, 4], ['awake', 'mild', 'deep', 'recovery'])
#         plt.show()
#
#     # Average across sedation levels for movie condition
#     print(f'# Averaging across sedation levels for movie condition')
#     movie_awake_average: list = []
#     movie_mild_average: list = []
#     movie_deep_average: list = []
#     movie_recovery_average: list = []
#
#     with open('outcome_268.pickle', 'rb') as f:
#         results_dict = pickle.load(f)
#         counter = 0
#         for key, value in results_dict.items():
#             if 'movie_01_LPI' in key:
#                 counter += 1
#                 movie_awake_average.append(value['hurst'].mean())
#                 movie_awake_total_average = mean(movie_awake_average)
#             elif 'movie_02_LPI' in key:
#                 counter += 1
#                 movie_mild_average.append(value['hurst'].mean())
#                 movie_mild_total_average = mean(movie_mild_average)
#             elif 'movie_03_LPI' in key:
#                 counter += 1
#                 movie_deep_average.append(value['hurst'].mean())
#                 movie_deep_total_average = mean(movie_deep_average)
#             elif 'movie_04_LPI' in key:
#                 counter += 1
#                 movie_recovery_average.append(value['hurst'].mean())
#                 movie_recovery_total_average = mean(movie_recovery_average)
#         print(f'movie_awake_average is {movie_awake_average}')
#         print(f'movie_mild_average is {movie_mild_average}')
#         print(f'movie_deep_average is {movie_deep_average}')
#         print(f'movie_recovery_average is {movie_recovery_average}')
#         print(f'movie_awake_total_average is {movie_awake_total_average}')
#         print(f'movie_mild_total_average is {movie_mild_total_average}')
#         print(f'movie_deep_total_average is {movie_deep_total_average}')
#         print(f'movie_recovery_total_average is {movie_recovery_total_average}')
#         print(stats.f_oneway(movie_awake_average, movie_mild_average, movie_deep_average, movie_recovery_average))
#         print(stats.ttest_ind(movie_deep_average, movie_awake_average))
#         print(stats.kruskal(movie_awake_average, movie_mild_average, movie_deep_average, movie_recovery_average))
#
#         df3 = [movie_awake_average, movie_mild_average, movie_deep_average, movie_recovery_average]
#         plt.boxplot(df3)
#         plt.xticks([1, 2, 3, 4], ['movie_awake', 'movie_mild', 'movie_deep', 'movie_recovery'])
#         plt.show()
#
#     # Average across sedation levels for rest condition
#     print(f'# Averaging across sedation levels for rest condition')
#     rest_awake_average: list = []
#     rest_mild_average: list = []
#     rest_deep_average: list = []
#     rest_recovery_average: list = []
#
#     with open('outcome_268.pickle', 'rb') as f:
#         results_dict = pickle.load(f)
#         counter = 0
#         for key, value in results_dict.items():
#             if 'rest_01_LPI' in key:
#                 counter += 1
#                 rest_awake_average.append(value['hurst'].mean())
#                 rest_awake_total_average = mean(rest_awake_average)
#             elif 'rest_02_LPI' in key:
#                 counter += 1
#                 rest_mild_average.append(value['hurst'].mean())
#                 rest_mild_total_average = mean(rest_mild_average)
#             elif 'rest_03_LPI' in key:
#                 counter += 1
#                 rest_deep_average.append(value['hurst'].mean())
#                 rest_deep_total_average = mean(rest_deep_average)
#             elif 'rest_04_LPI' in key:
#                 counter += 1
#                 rest_recovery_average.append(value['hurst'].mean())
#                 rest_recovery_total_average = mean(rest_recovery_average)
#         print(f'rest_awake_average is {rest_awake_average}')
#         print(f'rest_mild_average is {rest_mild_average}')
#         print(f'rest_deep_average is {rest_deep_average}')
#         print(f'rest_recovery_average is {rest_recovery_average}')
#         print(f'rest_awake_total_average is {rest_awake_total_average}')
#         print(f'rest_mild_total_average is {rest_mild_total_average}')
#         print(f'rest_deep_total_average is {rest_deep_total_average}')
#         print(f'rest_recovery_total_average is {rest_recovery_total_average}')
#         print(stats.f_oneway(rest_awake_average, rest_mild_average, rest_deep_average, rest_recovery_average))
#         print(stats.ttest_ind(rest_deep_average, rest_awake_average))
#         print(stats.kruskal(rest_awake_average, rest_mild_average, rest_deep_average, rest_recovery_average))
#
#         df4 = [rest_awake_average, rest_mild_average, rest_deep_average, rest_recovery_average]
#         plt.boxplot(df4)
#         plt.xticks([1, 2, 3, 4], ['rest_awake', 'rest_mild', 'rest_deep', 'rest_recovery'])
#         plt.show()

    # with open('outcome_268.pickle', 'rb') as f:
    #     results_dict = pickle.load(f)
    #     counter = 0
    #     for key, value in results_dict.items():
    #         keys = list(results_dict.keys())
    #         values = [value['hurst'].mean() for value in results_dict.values()]
    #         r_squared = [value['r_squared'].mean() for value in results_dict.values()]
    #         # create a DataFrame from the keys and values
    #         df = pd.DataFrame({'key': keys, 'hurst_average': values, 'r_squared': r_squared})
    #
    #     # print the DataFrame
    #     print(df)
    #     df.to_csv('hurst_averages.csv', index=False)


with open('./pickles/outcome_268.pickle', 'rb') as f:
    results_dict = pickle.load(f)
    for key, value in results_dict.items():
        # check how many files contain the string 'rest_01_LPI'
        key_1 = [key for key in results_dict.keys() if 'rest_01_LPI' in key]
        key_2 = [key for key in results_dict.keys() if 'rest_02_LPI' in key]
        key_3 = [key for key in results_dict.keys() if 'rest_03_LPI' in key]
        key_4 = [key for key in results_dict.keys() if 'rest_04_LPI' in key]
        key_5 = [key for key in results_dict.keys() if 'movie_01_LPI' in key]
        key_6 = [key for key in results_dict.keys() if 'movie_02_LPI' in key]
        key_7 = [key for key in results_dict.keys() if 'movie_03_LPI' in key]
        key_8 = [key for key in results_dict.keys() if 'movie_04_LPI' in key]
    # print the number of files
    print(f'Number of files containing rest_01_LPI is {len(key_1)}')
    print(f'Number of files containing rest_02_LPI is {len(key_2)}')
    print(f'Number of files containing rest_03_LPI is {len(key_3)}')
    print(f'Number of files containing rest_04_LPI is {len(key_4)}')
    print(f'Number of files containing movie_01_LPI is {len(key_5)}')
    print(f'Number of files containing movie_02_LPI is {len(key_6)}')
    print(f'Number of files containing movie_03_LPI is {len(key_7)}')
    print(f'Number of files containing movie_04_LPI is {len(key_8)}')







