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

from cifar_compare_db_structure import CIFARResult, db
from vaecompare import VAE, Compare
from sstudy import do_simulation_study

to_sample = dict(
    category1 = range(10),
    category2 = range(10),
)

def unpickle(f):
    with open(f, 'rb') as fo:
        obj = pickle.load(fo, encoding='bytes')
    return obj

def get_category(category):
    datasets = []
    for i in range(1, 6):
        obj = unpickle('data_batch_' + str(i))
        idx = np.array(obj[b'labels']) == category
        obj = obj[b'data'][idx]
        datasets.append(obj)
    x = np.array(np.vstack(datasets), dtype=np.float32)
    x = x / 255
    return x

def get_categories(category1, category2):
    if category1 != category2:
        db1 = get_category(category1)
        db2 = get_category(category2)
        db1 = db1[np.random.choice(range(5000), 2500, False)]
        db2 = db2[np.random.choice(range(5000), 2500, False)]
    else:
        db = get_category(category1)
        idx1 = np.random.choice(range(5000), 2500, False)
        idx2 = list(set(range(5000)).difference(idx1))
        db1 = np.ascontiguousarray(db[idx1])
        db2 = np.ascontiguousarray(db[idx2])

    return db1, db2

def func(category1, category2):
    start_time = time.time()
    y_train1, y_train2 = get_categories(category1, category2)
    compare = Compare(dataloader_workers=1, verbose=2,
        distribution="bernoulli")
    compare.fit(y_train1, y_train2, 10000)
    elapsed_time = time.time() - start_time

    return dict(
        samples=pickle.dumps(compare.samples),
        elapsed_time=elapsed_time,
        )

do_simulation_study(to_sample, func, db, CIFARResult)
