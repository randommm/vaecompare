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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.distributions.normal import Normal
from scipy.special import logsumexp


import numpy as np
import time
import itertools
from scipy import stats
from copy import deepcopy

from .utils import (loss_function, Collate)

class Compare():
    def __init__(self,
                 vae0,
                 vae1
                 ):

        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

        self.samples = np.empty(0)

    def sample(self, nsamples=100):
        s0 = self.vae0.sample_mean_logvar(nsamples)
        s1 = self.vae1.sample_mean_logvar(nsamples)

        samples = np.empty(nsamples)
        for i in range(nsamples):
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

            samples[i] = 0.5*distance

        self.samples = np.hstack([self.samples, samples])

        return self

