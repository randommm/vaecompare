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

from compare_db_structure import ResultVAECompare, db
from vaecompare import VAE, Compare
from sstudy_storage import do_simulation_study

to_sample = dict(
    distribution = range(1),
    no_instances = [10_000],
    dissimilarity = np.hstack([range(1,9), np.geomspace(1,10)-1])
)

def func(distribution,
    no_instances,
    dissimilarity):

    def data_gen(size, dim, mu):
        res = np.linspace(0.2, 0.9, dim)
        res = stats.lognorm.rvs(res, scale=2, size=(size, dim))
        res -= stats.lognorm.rvs(0.5, scale=2, size=(size, 1))
        res += stats.norm.rvs(loc=mu, scale=2, size=(size, 1))
        return res

    start_time = time.time()
    y_train0 = data_gen(no_instances, 10, 0)
    y_train1 = data_gen(no_instances, 10, dissimilarity)
    compare = Compare(dataloader_workers=1, verbose=2)
    compare.fit(y_train0, y_train1, 10000)
    elapsed_time = time.time() - start_time

    return dict(
        samples=pickle.dumps(compare.samples),
        elapsed_time=elapsed_time,
        )

do_simulation_study(to_sample, func, db, ResultVAECompare, max_count=100)
