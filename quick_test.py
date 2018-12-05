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

import numpy as np
import pandas as pd
from scipy import stats

from vaecompare import VAE, Compare

def data_gen(size, dim=10):
    res = np.linspace(0.2, 0.9, dim)
    res = stats.lognorm.rvs(res, scale=2, size=(size, dim))
    res = res - stats.lognorm.rvs(0.5, scale=2, size=(size, 1))
    return res

y_train = data_gen(10_000)
vae = VAE(dataloader_workers=1, verbose=2)
vae.fit(y_train)

y_rep = vae.sample_y(10000)
y_test = data_gen(1_000_000)

print("------------------------------------")
print("means:")
print(y_rep.mean(0))
print(y_test.mean(0))

print("------------------------------------")
print("stds:")
print(y_rep.std(0))
print(y_test.std(0))

print("------------------------------------")
print("thirds moments:")
print((y_rep**3).mean(0))
print((y_test**3).mean(0))

print("------------------------------------")
print("fourth moments:")
print((y_rep**4).mean(0))
print((y_test**4).mean(0))

print("------------------------------------")
print("difference of correlation matrices:")
corr = np.corrcoef(y_rep, rowvar=False)
corr -= np.corrcoef(y_test, rowvar=False)
corr = pd.DataFrame(corr)
print(corr)

compare = Compare(vae, vae)
compare.sample()
print(compare.samples)
