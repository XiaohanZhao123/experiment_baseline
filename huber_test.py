import sklearn
from sklearn.linear_model import SGDRegressor
import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import KFold
import Input_Data
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)


@hydra.main(config_path='conf', config_name='huber')
def main(clg: DictConfig):
    epsilon = clg.epsilon
    alpha = clg.alpha
    loss = clg.loss
    penalty = clg.penalty
    tol = clg.tol
    noise_level = clg.noise_level
    n_split = clg.n_split
    work_dir = str(Path.cwd())
    file = open(work_dir + '/huber.txt', 'w')

    data_list = [Input_Data.regression_data1, Input_Data.regression_data5, Input_Data.regression_data8,
                 Input_Data.regression_data9, Input_Data.regression_data13, Input_Data.regression_data17]
    data_index = [1, 5, 8, 9, 13, 17]

    for data, index in zip(data_list, data_index):
        X, y = data()
        file.writelines(f'data:{index}\n')
        KF = KFold(n_splits=n_split, shuffle=True)

        huber_lasso = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, tol=tol, epsilon=epsilon, random_state=42)
        for alpha in np.arange(0.0001, 0.0020, 0.0001):
            huber_lasso.alpha = alpha
            for epsilon in np.arange(0.1, 0.9, 0.1):
                huber_lasso.epsilon = epsilon
                for noise_level_ in np.arange(0.1, 0.7, 0.1):
                    noise_level = noise_level_
                    mae_score = []
                    for train_index, test_index in KF.split(X):
                        X_train = X[train_index]
                        X_test = X[test_index]
                        y_train = y[train_index]
                        y_test = y[test_index]

                        if noise_level != 0:
                            X_train, y_train = Input_Data.add_regression_noise_model1(X_train, y_train, noise_level)

                        huber_lasso.fit(X_train, y_train)
                        y_pre = huber_lasso.predict(X_test)
                        mae = np.sum(np.abs(y_pre - y_test)) / y.size
                        mae_score.append(float(mae))

                    mae_score = np.array(mae_score)
                    ave = np.average(mae_score)
                    print(ave)
                    std = np.sqrt(np.sum((mae_score - ave) ** 2) / (mae_score.size - 1))
                    print(std)
                    file.writelines(f'epsilon:{epsilon},alpha:{alpha},noise_level:{noise_level}\n')
                    file.writelines(f'ave:{ave}\n')
                    file.writelines(f'std:{std}\n')

    file.close()


if __name__ == '__main__':
    main()
