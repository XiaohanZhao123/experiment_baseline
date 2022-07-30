import matplotlib.pyplot as plt
import numpy as np
from LSSVMlib.LSSVMRegression  import LSSVMRegression
from sklearn.datasets import load_boston, load_diabetes
from sklearn.metrics.pairwise import rbf_kernel
from numpy.linalg import pinv, norm
from scipy.spatial.distance import cdist

class RLSSVM:
    def __init__(self, sigma, C, tol, p=1):
        self.sigma = sigma
        self.C = C
        self.p = p
        self.tol = tol
        self.alpha = []
        self.b = 0
        self.X = []
    def fit(self, X, y):
        K = rbf_kernel(X,gamma=1/2*self.sigma**2)
        m = X.shape[0]
        n = X.shape[1]

        model = LSSVMRegression(gamma=self.C, sigma=self.sigma)
        model.fit(X, y)
        alpha = model.coef_
        b = model.intercept_
        t = 0
        while True:
            C = self.C
            z = model.predict(X)
            p_sqr = np.sqrt(self.p)
            h = self.p-(y-z)**2
            h[np.logical_and(z>=y-p_sqr, z<=y+p_sqr)] = 0
            lam = np.zeros(m)
            s1 = y-p_sqr-h
            s2 = y-p_sqr+h
            s3 = y+p_sqr-h
            s4 = y+p_sqr+h
            i1 = np.where((z<s1) | (z>s4))[0]
            i2 = np.where((z>=s1) & (z<=s2))[0]
            i4 = np.where((z>=s3) & (z<=s4))[0]
            lam[i1] = C*(y-z)[i1] if len(i1)!=0 else 0
            lam[i2] = C*(h+2*p_sqr)*(y+h-p_sqr-z)[i2]/(4*h[i2]) if len(i2)!=0 else 0
            lam[i4] = C*(h+2*p_sqr)*(y-h+p_sqr-z)[i2]/(4*h[i4]) if len(i4)!=0 else 0

            e = np.ones((m, 1))
            update = -pinv(np.block([[0, e.T],
                                 [e, np.identity(m)+self.C*K]]))@np.block([[0],
                                                                    [(lam-self.C*y).reshape(-1,1)]])
            b_new = update[0]
            alpha_new = update[1:].reshape(-1,)
            d = np.sqrt(norm(alpha-alpha_new)**2+(b_new-b)**2)
            print(d)
            if d<self.tol:
                break
            else:
                alpha = alpha_new
                b = b_new
            t += 1
        self.alpha = alpha
        self.b = b
        self.X = X

    def predict(self, X_test): #X_test为一个二维数组
        K_test = np.exp(-(cdist(self.X, X_test, metric='sqeuclidean'))/(2*(self.sigma**2)))
        return K_test.T@self.alpha+self.b


if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    X = (X-X.mean(axis=0))/X.std(axis=0)
    y = (y-y.mean())/y.std()

    C = 1
    sigma = 1.0
    tol = 1e-5
    p = 1

    model = RLSSVM(sigma=sigma, C=C, tol=tol, p=p)
    model.fit(X, y)
    print(model.predict(X[:50]))
