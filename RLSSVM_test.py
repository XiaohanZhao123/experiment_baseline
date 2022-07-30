import numpy as np

import Input_Data
from RLSSVM import RLSSVM
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from pathlib import Path
np.random.seed(42)

import warnings
warnings.filterwarnings("ignore")
@hydra.main(config_path='conf',config_name='regression')
def main(clg:DictConfig):
    C, sigma, tol, p = clg.C, clg.sigma, clg.tol, clg.p
    n_split = clg.n_split

    work_dir = str(Path.cwd())
    file = open(work_dir + '\RLSSVM.txt','w')

    data_list = [Input_Data.regression_data1, Input_Data.regression_data5, Input_Data.regression_data8,
                 Input_Data.regression_data9, Input_Data.regression_data13, Input_Data.regression_data17]
    data_index = [1, 5, 8, 9, 13, 17]
    noise_level = clg.noise_level


    for data, index in zip(data_list, data_index):
        X, y = data()
        file.writelines(f'data:{index}\n')
        model = RLSSVM(sigma=sigma, C=C, tol=tol, p=p)

        for sigma in np.arange(0.1,2.0,0.1):
            for C in np.arange(0.1,2.0,0.1):
                model.sigma = sigma
                model.C = C
            for noise_level_ in np.arange(0.1,0.7,0.1):
                noise_level = noise_level_

                mae_score = []
                KF = KFold(n_splits=n_split, shuffle=True)
                for train_index, test_index in KF.split(X):
                    X_train = X[train_index]
                    X_test = X[test_index]
                    y_train = y[train_index]
                    y_test = y[test_index]

                    if noise_level != 0:
                        X_train, y_train = Input_Data.add_regression_noise_model1(X_train, y_train, noise_level)

                    model.fit(X_train,y_train)
                    y_pre = model.predict(X_test)
                    mae = np.sum(np.abs(y_pre - y_test)) / y_pre.size
                    mae_score.append(mae)

                mae_score = np.array(mae_score)
                ave = np.average(mae_score)
                print(ave)
                std = np.sqrt(np.sum((mae_score - ave) ** 2) / (mae_score.size - 1))
                print(std)
                file.writelines(f'sigma:{sigma},C:{C},noise_level:{noise_level}\n')
                file.writelines(f'ave:{ave}\n')
                file.writelines(f'std:{std}\n')

    file.close()

if __name__ == '__main__':
    main()
