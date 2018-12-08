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
#along with this program.    If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.backends.backend_pdf import PdfPages
from htest_db_structure import Result

cls = ["-", ":", "-.", "--"]
clw = [1.0, 2.0, 1.5, 3.0, 0.5, 4.0]
clws = list(itertools.product(clw, cls))

df = pd.DataFrame(list(Result.select().dicts()))

#for db_size in np.sort(db_size_sample):

def plotcdfs(distribution, no_instances):
    dissimilarity_sample = [0, 0.0001, 0.001, 0.01, 0.1]

    ax = plt.figure(figsize=[8.4, 5.8]).add_subplot(111)
    ax.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000))

    i = 0
    for dissimilarity in np.sort(dissimilarity_sample):
        label = "dissimilarity = " + str(dissimilarity)

        idx1 = df['dissimilarity'] == dissimilarity
        idx2 = df['distribution'] == distribution
        idx3 = df['no_instances'] == no_instances
        idxs = np.logical_and(idx1, idx2)
        idxs = np.logical_and(idxs, idx3)
        pvals = np.sort(df[idxs]['pvalue'])

        ecdf = ECDF(pvals)
        plt.plot(ecdf.x, ecdf.y,
            label=label, linestyle=clws[i][1], lw=clws[i][0])
        i += 1

    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    filename = "plots/"
    filename += "distribution_" + str(distribution)
    filename += "_and_no_instances_" + str(no_instances)
    filename += ".pdf"
    with PdfPages(filename) as ps:
        ps.savefig(ax.get_figure(), bbox_inches='tight')
    plt.close(ax.get_figure())

for distribution in range(1):
    for no_instances in [10000]:
        plotcdfs(distribution, no_instances)
