import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from load_node_labels import load_node_labels
from Propofol.dfa import dfa

target_dir = 'data_clean'


def load_ts(netts_file_path):
    # we have to grab the node labels from the netcc path
    netcc_file_path = netts_file_path.replace('netts', 'netcc')
    node_labels = load_node_labels(netcc_file_path)

    with open(netts_file_path) as f:
        lines = f.readlines()
        ts = [i.split('\t') for i in lines]
        ts = np.array(ts)
        sub_ts_final = np.full([268, ts.shape[1]], np.nan)
        for i, node_label in enumerate(node_labels):
            node_label = int(node_label) - 1
            sub_ts_final[node_label] = ts[i, :]
        return sub_ts_final


# for file in os.listdir('data_clean'):
#     if file.endswith(".netts"):
#         ts = load_ts(os.path.join('data_clean', file))
#         # save each ts as a .csv file in the MATLAB folder
#         np.savetxt(os.path.join(file.replace('.netts', '.npy')), ts, delimiter=',')


# for file in os.listdir('data_clean'):
#     if file.endswith(".npy"):
#         ts = np.load(os.path.join('data_clean', file))
#         print(ts.shape)

def read_files_268(directory: str):
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            # load data as numpy 2d array
            array_2d = np.load(os.path.join(target_dir, file))
            yield array_2d, file

def preprocess_268():
    tr: int = 2  # 2000 ms
    max_frequency = 0.1
    min_frequency = 0.01
    mn = int(np.ceil(1 / (tr * max_frequency)))
    mx = int(np.floor(1 / (tr * min_frequency)))
    counter = 0
    results_dict = {}
    for array_2d, file in read_files_268(directory=target_dir):
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
    with open('outcome_268.pickle', 'wb') as outfile:
        pickle.dump(results_dict, outfile)

# preprocess_268()


# if __name__ == '__main__':
#     rsquared_average: list = []
#     rsquared_min: list = []
#     print('showing results from pickle')
#
#     with open('outcome_268.pickle', 'rb') as f:
#         results_dict = pickle.load(f)
#         for key, value in results_dict.items():
#             # print(key)
#             # print(value)
#             rsquared_average.append(value['r_squared'].mean())
#             rsquared_min.append(value['r_squared'].min())
#             # print(f'average r_squared for {key}: {value["r_squared"].mean()}')
#             # print(f'min r_squared for {key}: {value["r_squared"].min()}')
#     plt.plot(rsquared_average, color='red')
#     plt.ylabel('average r_squared', color='red')
#     plt.xlabel('subject')
#     ax = plt.twinx()
#     ax.plot(rsquared_min, color='blue')
#     ax.set_ylabel('min r_squared', color='blue')
#     plt.show()
#
#     plt.plot(rsquared_average, color='red')
#     plt.axhline(y=0.95, color='black', linestyle='--')
#     plt.axhline(y=0.9, color='black', linestyle='--')
#     plt.ylabel('average r_squared', color='red')
#     plt.xlabel('subject')
#     plt.show()
#
#     plt.plot(rsquared_min, color='blue')
#     plt.axhline(y=0.8, color='black', linestyle='--')
#     plt.axhline(y=0.65, color='black', linestyle='--')
#     plt.ylabel('min r_squared', color='blue')
#     plt.xlabel('subject')
#     plt.show()