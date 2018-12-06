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
from .compare import Compare
import numpy as np
import time

class HTest():
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        self.args = args
        self.kwargs = kwargs

    def fit(self, y_train0, y_train1, nsamples=10000, ncomparisons=100):
        len0 = len(y_train0)
        y_train01 = np.vstack((y_train0, y_train1))

        pvalues = np.empty(ncomparisons)
        for i in range(ncomparisons):
            samples = Compare(*self.args, **self.kwargs)
            samples = samples.fit(y_train0, y_train1, nsamples).samples
            pvalues[i] = samples.mean()

            y_train01 = np.random.permutation(y_train01)
            y_train0 = y_train01[:len0]
            y_train1 = y_train01[len0:]
            print("Made comparison", i+1, "out of", ncomparisons)

        self.divergence_unpermuted = pvalues[0]
        self.divergence_permuted = pvalues[1:]

        n1 = (self.divergence_unpermuted <=
            self.divergence_permuted).sum() / (ncomparisons - 1)
        n2 = (self.divergence_unpermuted <
            self.divergence_permuted).sum() / (ncomparisons - 1)

        self.pvalue = (n1 + n2) / 2

        return self
