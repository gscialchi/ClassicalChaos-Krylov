from functools import partial
import argparse

import numpy as np

from cc_krylov.maps import *
from cc_krylov.krylov import *
from cc_krylov.products import operator_prod
from cc_krylov.functions import periodic_gauss_2D
from cc_krylov.chord import chord_second_moment

import cc_krylov.utilities.paths as paths
from cc_krylov.utilities.config import get_config
from cc_krylov.utilities.pipeline import Pipeline
from cc_krylov.utilities.store import store
from cc_krylov.utilities.doer import Doer

from plotters import plot_r2_two


# setup parser for script
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default=paths.CONFIG_DIR+'defaults.yml')
args = parser.parse_args()

configs = get_config(args.config)

####
DISABLE_DOER = configs['DISABLE_DOER']
DOER_DIR = configs['DOER_DIR']
if DOER_DIR == 'default':
    DOER_DIR = paths.DOER_DIR

DISABLE_STORE = configs['DISABLE_STORE']
STORE_DIR = configs['STORE_DIR']
if STORE_DIR == 'default':
    STORE_DIR = paths.STORE_DIR


#### Wrap store on some costly operators
if not DISABLE_STORE:
    coherent_ensemble_torus_traceless = store(path=STORE_DIR)(coherent_ensemble_torus_traceless)


#### Setup Doers for data saving & retrieval
do_krylov = Doer(arnoldi_FO_operator, path=DOER_DIR,
                 iterable='n_final', iterable_index=0,
                 iterable_continue=True, continue_arg='e0',
                 disabled=DISABLE_DOER)

#
do_op = Doer(coherent_ensemble_torus_traceless,
             args={'f': periodic_gauss_2D}, disabled=DISABLE_DOER)

#
do_u_harper = Doer(q_harper)

do_u_standard = Doer(q_standard)


#### Load parameters
N_qu = configs['N_qu']; h = 1/(2*np.pi*N_qu) # hbar
n_final = configs['n_final']

ks = configs['ks_har_std'] # perturbation parameters for cats
lyapunovs = configs['lyaps_har_std']

q0 = configs['q0']; q0 *= 2.7/np.e # make them not fractional
p0 = configs['p0']; p0 *= 2.7/np.e
s = configs['s']

## apply them
do_us = []
for k, do_u in zip(ks, [do_u_harper, do_u_standard]): # setup unitaries
    do_u.set_args(N=N_qu, k=k)
    do_us.append(do_u)

do_op.set_args(N=N_qu, args=(q0, p0, s)) # setup initial operator

#### Calculate
r_krys = [[], []]
for i, do_u in enumerate(do_us):
    do_krylov.set_args(U=do_u, e0=do_op, n_final=n_final,
                       prod=partial(operator_prod, hbar=h))
    krylov = do_krylov.doit()

    r_kry = np.zeros(n_final+1)
    for n in range(n_final+1):
        r_kry[n] = chord_second_moment(krylov[n])

    r_krys[i] = r_kry

#### Plot
FIG_DIR = configs['FIG_DIR']
if FIG_DIR == 'default':
    FIG_DIR = paths.FIG_DIR

figname = 'Figure_S2'
plot_r2_two(r_krys, lyapunovs,
            usetex=configs['FIGURES_USETEX'],
            save=configs['SAVE_FIGURES'],
            savedir=FIG_DIR + figname,
            show=configs['SHOW_FIGURES'])
