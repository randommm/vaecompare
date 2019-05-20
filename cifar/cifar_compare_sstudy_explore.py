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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from vaecompare.utils import kld_bernoullis

from cifar_compare_db_structure import CIFARResult, db

df = pd.DataFrame(list(CIFARResult.select().dicts()))
df = df.sort_values(['category1','category2'])
df["samples"] = [pickle.loads(x) for x in df["samples"] ]
df["mean_kl_divergence"] = [x.mean() for x in df["samples"]]
df["std_kl_divergence"] = [x.std() for x in df["samples"]]
del(df["id"])

assert((df["mean_kl_divergence"] > 0).all())

# Dot and box plots prepare data

for mtype1 in range(10):
    names = list()
    machine_names = np.empty(0)
    dotvals = np.empty((0,2))
    boxvals = list()
    metricobjlist = list()
    for i, (mtype2) in enumerate(range(10)):
        rawvals = df.loc[df["category1"] == min(mtype1, mtype2)]
        rawvals = rawvals.loc[df["category2"] == max(mtype1, mtype2)]
        rawvals = rawvals["samples"].item()

        names.append("{} vs {}".format(mtype1, mtype2))

        machine_name = i+1
        machine_names = np.hstack((machine_names, machine_name))
        repmn = np.repeat(machine_name, rawvals.size)
        dotval = np.column_stack((repmn, rawvals))
        dotvals = np.vstack((dotvals, dotval))

        boxvals.append(rawvals)

    fig, ax = plt.subplots()
    fig.set_size_inches(7.3, 5)

    #Dot plot
    #ax.scatter(dotvals[:, 0], dotvals[:, 1], marker='o', s=15.0,
    #                      color="green")
    #ax.set_xlabel("Models compared")
    #ax.set_ylabel("Probability")
    #ax.set_ylim(0.9, 1.05)

    #Box plot
    bplot = ax.boxplot(boxvals, whis='range', showmeans=True,
                        meanline=True)

    for mean in bplot['means']:
        mean.set(color='#FF6600')

    for median in bplot['medians']:
        median.set(color='blue')

    ax.set_xlabel("Models compared")
    ax.set_ylabel("Divergence")
    ax.set_xticklabels(names)

    for i, v in enumerate([.35, .4, .45, .49]):
        ax.axhline(
            y=kld_bernoullis(np.repeat(v, 3072), np.repeat(1-v, 3072)),
            xmin=0.0, xmax=1.0,
            color=["red", "blue", "green", "yellow"][i],
            label="KL($P(.;\\theta="+str(v)+")$, $P(.;\\theta="+str(1-v)+"$))")

    if mtype1 <= 1:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=2, fancybox=True, shadow=True)

    ps = PdfPages("dotplot_"+ str(mtype1) +".pdf")
    ps.savefig(fig, bbox_inches='tight')
    ps.close()
