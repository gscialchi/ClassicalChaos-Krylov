"""
Calculate Lyapunov exponent for the non-linear maps.
"""
from functools import partial
import argparse

from cc_krylov.maps import *
from cc_krylov.lyapunov import get_average_lyapunov

import cc_krylov.utilities.paths as paths
from cc_krylov.utilities.config import get_config


# setup parser for script
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default=paths.CONFIG_DIR+'defaults.yml')
args = parser.parse_args()

configs = get_config(args.config)


###
ks = configs['ks_har_std']
maps = [c_harper, c_standard]
jacs = [jacobian_harper, jacobian_standard]

for k, map, jac in zip(ks, maps, jacs):
    map_k = partial(map, k)
    jac_k = partial(jac, k)

    avg_lyap = get_average_lyapunov(jac_k, map_k, Nqp=1000)
    print(f'{map.__name__} - Average Lyapunov: {avg_lyap}')
