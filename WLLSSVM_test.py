import Input_Data
from WLLSSVM import WLLSVM_train_and_evaluate
from WLLSSVM import loadDataSet
import hydra
from omegaconf import DictConfig
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np

np.random.seed(42)
@hydra.main(config_path='conf',config_name='regression')
def main(clg:DictConfig):
    C, k1, kernel = clg.C, clg.k1, clg.kernel
    clg.noise_level = 0.3
    n_split = clg.n_split

    working_dir = str(Path.cwd())
    file = open(working_dir + '\WLLSSVM_test.txt','w')

    KF = KFold(n_splits=n_split, shuffle=True)

    data_list = [Input_Data.regression_data1, Input_Data.regression_data5, Input_Data.regression_data8,
                 Input_Data.regression_data9, Input_Data.regression_data13, Input_Data.regression_data17]
    data_index = [1, 5, 8, 9, 13, 17]
    noise_level = clg.noise_level

    for data, index in zip(data_list, data_index):
        X, y = data()
        file.writelines(f'data:{index}\n')
        for C in np.arange(0.1,2,0.1):
            for k1 in np.arange(0.1,1,0.1):
                for noise_level_ in np.arange(0.1,0.7,0.1):
                    noise_level = noise_level_
                    mae_score = []
                    for train_index, test_index in KF.split(X):
                        X_train = X[train_index]
                        X_test = X[test_index]
                        y_train = y[train_index]
                        y_test = y[test_index]

                        if noise_level != 0:
                            X_train, y_train = Input_Data.add_regression_noise_model1(X_train, y_train, noise_level)

                        mae_score.append(WLLSVM_train_and_evaluate(X_train, y_train, C, k1, kernel))

                    mae_score = np.array(mae_score)
                    ave = np.average(mae_score)
                    print(ave)
                    std = np.sqrt(np.sum((mae_score - ave) ** 2) / (mae_score.size - 1))
                    print(std)
                    file.writelines(f'k1:{k1},C:{C},noise_level:{noise_level}\n')
                    file.writelines(f'ave:{ave}\n')
                    file.writelines(f'std:{std}\n')

    file.close()



if __name__ == '__main__':
    main()
