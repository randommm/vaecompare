#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

from __future__ import division
from scipy.special import logsumexp
from .estimate import VAE
import numpy as np
import time

class Compare():
    def __init__(self,
                 *args,
                 distribution="gaussian",
                 **kwargs,
                 ):

        self.vae0 = VAE(*args, distribution=distribution, **kwargs)
        self.vae1 = VAE(*args, distribution=distribution, **kwargs)
        self.samples = np.empty(0)
        self.fitted = False

    def fit(self, y_train0, y_train1, nsamples=10000):
        self.vae0.fit(y_train0)
        self.vae1.fit(y_train1)
        self.fitted = True
        self.improve_comparison(nsamples)

        return self

    def improve_comparison(self, nsamples=10000):
        if not self.fitted:
            raise Exception("Call method fit first")

        s0 = self.vae0.sample_parameters(nsamples)
        s1 = self.vae1.sample_parameters(nsamples)

        samples = np.empty(nsamples)
        for i in range(nsamples):
            if self.distribution == "gaussian":
                if i % 2:
                    mua = s0[0][i]
                    mub = s1[0][i]
                    logvara = s0[1][i]
                    logvarb = s1[1][i]
                else:
                    mua = s1[0][i]
                    mub = s0[0][i]
                    logvara = s1[1][i]
                    logvarb = s0[1][i]

                distance = logvarb.sum() - logvara.sum()
                distance -= len(logvara)
                distance += np.exp(logsumexp(logvara - logvarb))

                distance += ((mub - mua)**2 * np.exp(logvara)).sum()
            elif self.distribution == "bernoulli":


            samples[i] = 0.5*distance

        self.samples = np.hstack([self.samples, samples])

        return self

