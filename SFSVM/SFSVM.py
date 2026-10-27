import numpy as np

import solver_old as solver_old

class SFSVM_old:
    def __init__(self, mu, C = 10):
        self.C = C
        self.mu = mu


    def fit(self, x_train, y_train):
        kernels = solver_old.build_kernels(x_train, x_train)
        kadd = np.eye(len(y_train)) + np.outer(y_train, y_train)
        train_kernels = kernels * kadd[:, :, None] 

        # n - objects, k - features
        n, k = x_train.shape
        empty_task = solver_old.kernel_task_SKM_MSK(k, n, self.C)

        self.fit_result = solver_old.kernel_SKM_MSK_origin(train_kernels, y_train,
                                                      empty_task, self.mu)
        self.x_train = x_train
        self.y_train = y_train
    

    def decision_function(self, x_test):
        kernels = solver_old.build_kernels(self.x_train, x_test)

        K = np.zeros((self.x_train.shape[0], x_test.shape[0]))
        for i in range(kernels.shape[2]):
            K += kernels[:, :, i] * self.fit_result["r"][i]

        weights = self.fit_result["lambda"] * self.y_train
        scores = np.sum(weights[:, None] * K, axis=0) + self.fit_result["b"]

        return scores


    # def predict(self, x_test):
    #     return np.sign(self.decision_function(x_test))





import solver_new as solver_new

class SFSVM_new:
    def __init__(self, mu, C = 10):
        self.C = C
        self.mu = mu


    def fit(self, x_train, y_train):
        self.dual = solver_new.solve_dual(x_train, y_train, self.C, self.mu)
        self.lp = solver_new.solve_final_lp(x_train, y_train, self.C, self.mu, self.dual["lambda"])
    

    def decision_function(self, x_test):
        return np.dot(x_test, self.lp["a"]) + self.lp["b"]


    def predict(self, x_test):
        return np.where(self.decision_function(x_test) >= 0, 1, -1)