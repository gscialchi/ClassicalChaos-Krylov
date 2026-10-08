from functools import partial
import argparse

import numpy as np

from cc_krylov.maps import *
from cc_krylov.krylov import *
from cc_krylov.products import operator_prod
from cc_krylov.functions import periodic_gauss_2D

import cc_krylov.utilities.paths as paths
from cc_krylov.utilities.config import get_config
from cc_krylov.utilities.pipeline import Pipeline
from cc_krylov.utilities.store import store
from cc_krylov.utilities.doer import Doer

from plotters import plot_verblunsky


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

do_seqs = Doer(arnoldi_sequences_verblunsky, path=DOER_DIR,
               ignore_args=['krylov'], disabled=DISABLE_DOER)

#
do_rho = Doer(coherent_ensemble_torus_traceless,
              args={'f': periodic_gauss_2D}, disabled=DISABLE_DOER)

#
do_u_cat = Doer(q_cat_perturbed)

do_u_harper = Doer(q_harper)

do_u_standard = Doer(q_standard)


#### Load parameters
N_qu = configs['N_qu']; h = 1/(2*np.pi*N_qu) # hbar
n_final = configs['n_final']

ks_cats = configs['ks_cats'] # perturbation parameters for cats
a1s_cats = configs['a1s_cats']
ks_hs = configs['ks_har_std']
a1s_hs = configs['a1s_har_std']

ks = ks_cats[:1] + ks_hs
a1s = a1s_cats[:1] + a1s_hs

q0 = configs['q0']; q0 *= 2.7/np.e # make them not fractional
p0 = configs['p0']; p0 *= 2.7/np.e
ss = [configs['s'], configs['s_har'], configs['s']]

## apply them
do_us = []
do_ops = []
for i, do_u in enumerate([do_u_cat, do_u_harper, do_u_standard]):
    do_u.set_args(N=N_qu, k=ks[i])
    do_us.append(do_u)

    do_op = do_rho.copy()
    do_op.set_args(N=N_qu, args=(q0, p0, ss[i]))
    do_ops.append(do_op)

#### Calculate
seqs = [[]]*3
for i, (do_u, do_op) in enumerate(zip(do_us, do_ops)):
    do_krylov.set_args(U=do_u, e0=do_op, n_final=n_final,
                       prod=partial(operator_prod, hbar=h))
    do_krylov.provides = ['krylov']

    do_seqs.set_args(u=do_u, hbar=h)
    do_seqs.set_fakeargs(e0=do_op, n_final=n_final)
    do_seqs.provides = ['seqs']

    pipe = Pipeline(doers=[do_krylov, do_seqs])
    pipe.doit()

    _, _, _, vn = pipe.results['seqs']
    seqs[i] = vn[1:] # ignore v_{-1} = -1

#### Plot
FIG_DIR = configs['FIG_DIR']
if FIG_DIR == 'default':
    FIG_DIR = paths.FIG_DIR

figname = 'Figure_S3'
plot_verblunsky(seqs,
                resonances=a1s,
                res_ranges=[slice(0, 14, 1)]+[slice(0, 11, 1)]*2,
                usetex=configs['FIGURES_USETEX'],
                save=configs['SAVE_FIGURES'],
                savedir=FIG_DIR + figname,
                show=configs['SHOW_FIGURES'])
