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

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

from htest_db_structure import Result, db

df = pd.DataFrame(list(Result.select().dicts()))
del(df["id"])

cls = [":", "-", "-.", "--", "-", "-."]
clw = [2.2, 2.2, 2.2, 2.2, 1.0, 1.0]
for ncomparisons in [1, 100]:
    fig, ax = plt.subplots()
    for i, dissimilarity in enumerate([0, 0.01, 0.1, 0.2]):
        vals = df
        vals = vals[vals['dissimilarity'] == dissimilarity]
        vals = vals[vals['ncomparisons'] == ncomparisons]
        vals = vals.pvalue
        name = "dissimilarity " + str(dissimilarity)
        ecdf = ECDF(vals)
        ax.plot(ecdf.x, ecdf.y, label=name,
            linestyle=cls[i], lw=clw[i])

    legend = ax.legend(shadow=True, frameon=True, loc='best',
        fancybox=True, borderaxespad=.5)

    ax.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), c='black')

    filename = "plots/gen_data_compare_ncomparisons_"
    filename += str(ncomparisons) + ".pdf"
    with PdfPages(filename) as ps:
        ps.savefig(fig, bbox_inches='tight')
    plt.close(ax.get_figure())
