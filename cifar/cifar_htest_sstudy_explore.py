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

from cifar_htest_db_structure import ResultVAECIFARHTest, db

df = pd.DataFrame(list(ResultVAECIFARHTest.select().dicts()))
del(df["id"])

datap = dict()
datae = dict()
for cat1 in range(10):
    valsp = []
    valse = []
    for cat2 in range(10):
        try:
            valp = df[df['category1'] == max(cat1, cat2)]
            valp = valp[valp['category2'] == min(cat1, cat2)]
            valp = valp.pvalue.item()
        except ValueError:
            valp = np.nan
        if cat1 == cat2:
            vale = "G" if valp > .05 else "E1"
        else:
            vale = "G" if valp <= .05 else "E2"

        if cat1 >= cat2:
            valsp.append("{0:.2f}".format(valp))
            valse.append(vale)
        else:
            valsp.append("-")
            valse.append("-")

    datap['c' + str(cat1)] = valsp
    datae['c' + str(cat1)] = valse

dfp = pd.DataFrame.from_dict(datap)
dfe = pd.DataFrame.from_dict(datae)

dfp.index = ['c'+str(cat) for cat in range(10)]
dfe.index = dfp.index

print(dfp)
print(dfe)

print(dfp.to_latex())
print(dfe.to_latex())
