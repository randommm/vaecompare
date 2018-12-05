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
import time
import pickle
from scipy import stats

from db_structure import Result, db
from vaecompare import VAE, Compare
from sstudy_storage import do_simulation_study

to_sample = dict(
    distribution = range(1),
    no_instances = [10_000],
    random_seed = range(10),
    dissimilarity = range(10)
)

def func(distribution, no_instances, random_seed, dissimilarity):
    def data_gen(size, dim, mu, random_state):
        res = np.linspace(0.2, 0.9, dim)
        res = stats.lognorm.rvs(res, scale=2, size=(size, dim),
            random_state=random_state)
        res -= stats.lognorm.rvs(0.5, scale=2, size=(size, 1),
            random_state=random_state)
        res += stats.norm.rvs(loc=mu, scale=2, size=(size, 1),
            random_state=random_state)
        return res

    random_state = np.random.RandomState(
        random_seed + no_instances * 1000 + distribution * 1000)

    start_time = time.time()
    y_train = data_gen(no_instances, 10, 0, random_state)
    vae0 = VAE(dataloader_workers=1, verbose=2)
    vae0.fit(y_train)

    y_train = data_gen(no_instances, 10, dissimilarity, random_state)
    vae1 = VAE(dataloader_workers=1, verbose=2)
    vae1.fit(y_train)

    compare = Compare(vae0, vae1)
    compare.sample(10000)
    elapsed_time = time.time() - start_time

    return dict(
        samples=pickle.dumps(compare.samples),
        elapsed_time=elapsed_time,
        )

do_simulation_study(to_sample, func, db, Result)
