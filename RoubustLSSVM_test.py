import logging, numpy as np, matplotlib.pyplot as plt
logging.root.setLevel(logging.ERROR)
import RobustLSSVM as svm, data
import hydra
from omegaconf import DictConfig
import Input_Data
from sklearn.model_selection import KFold
from pathlib import Path
from sklearn.model_selection import train_test_split

np.random.seed(42)
@hydra.main(config_path='conf',config_name='classfication')
def main(clg:DictConfig):
    kernel = clg.kernel
    C = clg.C
    mu = clg.mu
    noise_level = clg.noise_level
    work_dir = str(Path.cwd())
    file = open(work_dir + '/RoubutSSVM.txt','w')
    n_split = clg.n_split

    data_list = [Input_Data.handwritten_digits1,Input_Data.handwritten_digits2,Input_Data.handwritten_digits3,Input_Data.handwritten_digits7]
    data_index = [1,2,3,7]

    for data, index in zip(data_list,data_index):
        file.writelines(f'data:{index}\n')


        for C in np.arange(0.1,2,0.1):
            for mu in np.arange(0.1,2,0.1):
                for noise_level_ in np.arange(0.1,0.7,0.1):
                    noise_level = noise_level_

                    X, y_p = data(noise_level=noise_level)

                    acc_score = []
                    for i in range(0,n_split):
                        X_train, X_test, y_train, y_test = train_test_split(X, y_p, random_state=42, stratify=y_p)
                        trainer = svm.SVMTrainer(kernel, C, ln_robust=True, mu=mu)
                        predictor = trainer.train(X_train, y_train, remove_zero=True)

                        y_pre = predictor.predict(X_test)
                        y_test = np.array(y_test)
                        y_pre = np.array(y_pre)

                        y_test.reshape(y_test.size)
                        y_pre.reshape(y_pre.size)

                        count = 0
                        for y1, y2 in zip(y_test, y_pre):
                            if y1 == y2:
                                count += 1

                        acc = count / y_pre.size
                        acc_score.append(acc)

                    acc_score = np.array(acc_score)
                    ave = np.average(acc_score)
                    std = np.sqrt(np.sum((acc_score - ave)**2) / (acc_score.size - 1))

                    print(ave,std)
                    file.writelines(f"C:{C},mu:{mu},noise_level:{noise_level}\n")
                    file.writelines(f'ave:{ave}\n')
                    file.writelines(f'std:{std}\n')
    file.close()


if __name__ == '__main__':
    main()

