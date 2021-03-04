import numpy as np
import params

class ZCA(object):

    def __init__(self, D, n_components):
        self.n_components = n_components
        self.U, self.S, self.m = self.fit(D, n_components)

    def fit(self, D, n_components):
        m = np.mean(D, axis=1)[:, np.newaxis]
        D -= m
        D *= 1.0 / np.sqrt(D.shape[1] - 1)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        D *= np.sqrt(D.shape[1] - 1)
        D += m
        return U[:, :n_components], S[:n_components], m

    def transform(self, D, whiten=True, ZCA=True,
                  regularizer= params.whiten_r):

        if whiten:
            Sinv = 1.0 / (self.S + regularizer)

            if ZCA:
                W = np.dot(self.U,
                           np.dot(np.diag(Sinv),
                                  self.U.T))
            else:
                W = np.dot(np.diag(Sinv), self.U.T)

        else:
            W = self.U.T

        # Transform
        return np.dot(W, D - self.m)

