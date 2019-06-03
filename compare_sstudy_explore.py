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
import pickle
from scipy import stats

from compare_db_structure import ResultVAECompare, db

df = pd.DataFrame(list(ResultVAECompare.select().dicts()))
df["samples"] = [pickle.loads(x).mean() for x in df["samples"]]

del(df["id"])
del(df["random_seed"])
df.rename(columns={'samples': 'mean_kl_divergence'}, inplace=1)
df.rename(columns={'elapsed_time': 'mean_elapsed_time'}, inplace=1)

groups = ["dissimilarity", "distribution", "no_instances"]

del(df["mean_elapsed_time"])
del(df["no_instances"])
del(df["distribution"])
groups = ["dissimilarity"]

res = df.groupby(groups).mean()
print(res)
