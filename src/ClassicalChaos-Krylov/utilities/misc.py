import os

import matplotlib as mpl


###
def latex_config(usetex=True, usephysics=False):
    if not usetex:
        usephysics = False
    if usetex:
        from matplotlib import rc
        rc('text', usetex=True)
        if usephysics:
            rc('text.latex', preamble=r'\usepackage{physics}')


def savefig(fig, savedir, saveformat, *args, **kwargs):
    """
    Create path if non existent.
    """
    dir_split = savedir.split('/')
    path = '/'.join(dir_split[:-1])+'/'
    if not os.path.isdir(path): # make path before saving
        os.makedirs(path)
    savedir += f'.{saveformat}'
    fig.savefig(savedir, *args, **kwargs)
    print(f'Saved: {savedir}')
