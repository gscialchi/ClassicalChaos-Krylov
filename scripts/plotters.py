import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from cc_krylov.utilities.misc import latex_config, savefig


## main
def plot_sequences_manyk(seqs,
                         resonances=None, res_ranges=None,
                         figsize=(5, 7), size=14, labelsize=11,
                         linewidth=1.25, nyticks=4, show_legend=False,
                         usetex=True,
                         save=False, savedir=None, show=True):
    latex_config(usetex, False)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.1, wspace=0)

    axes[-1].set_xlabel(r'$n$', size=size)

    axes[0].set_ylabel(r'$|a_n|$', size=size)
    axes[1].set_ylabel(r'$1-b_{n+1}$', size=size)
    axes[2].set_ylabel(r'$|c_n|$', size=size)

    for ax in axes:
        ax.set_yscale('log')
        ax.tick_params(labelsize=labelsize)
        ax.minorticks_off()

    axes[0].set_yticks([1e-2, 1e-6, 1e-10])
    axes[1].set_yticks([1e-2, 1e-6, 1e-10])
    axes[2].set_yticks([1e-1, 1e-3, 1e-5])

    markers = ['-s', '-^', '-o']

    for i in range(3)[::-1]:
        an, bn, cn = seqs[i]

        # sequences
        an_ = np.abs(an)
        bn_ = 1 - bn
        cn_ = np.abs(cn)

        cmap = mpl.colormaps['tab20b']
        n = np.arange(len(an_))
        axes[0].plot(an_, markers[i], c=cmap(4*0+2-i), linewidth=linewidth)
        axes[1].plot(bn_, markers[i], c=cmap(4*3+2-i), linewidth=linewidth)
        axes[2].plot(cn_, markers[i], c=cmap(4*4+2-i), linewidth=linewidth)

        # plot resonances on top
        res_range = res_ranges[i]
        n_res = n[res_range]
        resonance = resonances[i]

        m = -1

        a2 = resonance**(2*n_res)
        a1 = resonance**n_res

        # chose factor that minimizes mean square deviations in logarithmic scale
        A = np.exp(np.mean((np.log(an_[res_range][:-1]) - 2*np.log(resonance)*n_res[:-1])))
        B = np.exp(np.mean((np.log(bn_[res_range][:m]) - 2*np.log(resonance)*n_res[:m])))
        C = np.exp(np.mean((np.log(cn_[res_range][:-1]) - np.log(resonance)*n_res[:-1])))

        if i == 0:
            label1=r'$|\lambda_1|^{n}$'
            label2=r'$|\lambda_1|^{2n}$'
        else:
            label1=None
            label2=None

        axes[0].plot(n_res, A*a2, c='white',
                     linewidth=2*linewidth, linestyle='solid')
        axes[0].plot(n_res, A*a2, c=cmap(4*0+2-i),
                     linewidth=1.25*linewidth, linestyle='solid',
                     label=label2)

        axes[1].plot(n_res, B*a2, c='white',
                     linewidth=2*linewidth, linestyle='solid')
        axes[1].plot(n_res, B*a2, c=cmap(4*3+2-i),
                     linewidth=1.25*linewidth, linestyle='solid')

        axes[2].plot(n_res, C*a1, c='white',
                     linewidth=2*linewidth, linestyle='solid')
        axes[2].plot(n_res, C*a1, c=cmap(4*4+2-i),
                     linewidth=1.5*linewidth, linestyle='dashed',
                     label=label1)

    if show_legend:
        for ax in [axes[0], axes[2]]:
            ax.legend(frameon=False, fontsize=size)

    if save:
        savefig(fig, savedir, saveformat='pdf',
                bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_r2(r2, lyapunov,
            figsize=(5, 4), size=14, labelsize=11, linewidth=1.5,
            usetex=True,
            save=False, savedir=None, show=True):
    latex_config(usetex, False)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.tick_params(labelsize=labelsize)

    r2_uniform = (1/6)**0.5 # approx value for uniform op in chord rep

    n = np.arange(len(r2))

    cmap = mpl.colormaps['tab20b']

    ax.semilogy(n, r2, '-o', c=cmap(4*0+2), linewidth=linewidth)

    ax.semilogy(n, 3*r2[0]*np.exp(lyapunov*n),
                c='k', linestyle='--', linewidth=1.25*linewidth,
                label=r'$e^{\Lambda n}$')

    ax.axhline(y=r2_uniform, linestyle=(0,(1, 1)), color='tab:gray',
               label=r'$1/\sqrt{6}$', linewidth=1.5*linewidth)

    ax.set_ylabel(r'$r_n$', size=size)
    ax.set_xlabel(r'$n$', size=size)
    ax.set_ylim(1e-3, 1)
    plt.legend(frameon=False, fontsize=size)
    if save:
        savefig(fig, savedir, saveformat='pdf',
                bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


## supplemental
def plot_sequences_two(seqs,
                       resonances=None, res_ranges=None,
                       figsize=(10, 5), size=14, labelsize=11,
                       linewidth=1.5, nyticks=4,
                       usetex=True,
                       save=False, savedir=None, show=True):
    latex_config(usetex, False)

    fig, axes = plt.subplots(3, 2, figsize=figsize,
                             sharex='col', sharey='row')
    fig.subplots_adjust(hspace=0.1, wspace=0.05)

    for ax in axes[-1, :]: ax.set_xlabel(r'$n$', size=size)

    axes[0, 0].set_ylabel(r'$|a_n|$', size=size)
    axes[1, 0].set_ylabel(r'$1-b_{n+1}$', size=size)
    axes[2, 0].set_ylabel(r'$|c_n|$', size=size)

    for ax in axes.flatten():
        ax.set_yscale('log')
        ax.tick_params(labelsize=labelsize)

    axes[0, 0].set_yticks([1e-1, 1e-5, 1e-9])
    axes[1, 0].set_yticks([1e-1, 1e-5, 1e-9])
    axes[2, 0].set_yticks([1e-1, 1e-3, 1e-5])
    axes[2, 0].set_ylim(0.8*1e-5, 2.5*1e-1)

    for i in range(2):
        an, bn, cn = seqs[i]

        # sequences
        an_ = np.abs(an)
        bn_ = 1 - bn
        cn_ = np.abs(cn)

        cmap = mpl.colormaps['tab20b']
        n = np.arange(len(an_))
        axes[0, i].plot(an_, '-s', c=cmap(4*0+2), linewidth=linewidth)
        axes[1, i].plot(bn_, '-s', c=cmap(4*3+2), linewidth=linewidth)
        axes[2, i].plot(cn_, '-s', c=cmap(4*4+2), linewidth=linewidth)

        # plot resonances on top
        res_range = res_ranges[i]
        n_res = n[res_range]
        resonance = resonances[i]

        m = -1

        a2 = resonance**(2*n_res)
        a1 = resonance**n_res

        # chose factor that minimizes mean square deviations in logarithmic scale
        A = np.exp(np.mean((np.log(an_[res_range][:-1]) - 2*np.log(resonance)*n_res[:-1])))
        B = np.exp(np.mean((np.log(bn_[res_range][:m]) - 2*np.log(resonance)*n_res[:m])))
        C = np.exp(np.mean((np.log(cn_[res_range][:-1]) - np.log(resonance)*n_res[:-1])))

        axes[0, i].plot(n_res, A*a2, c='k',
                        linewidth=1.25*linewidth, linestyle='solid',
                        label=r'$|\lambda_1|^{2n}$')

        axes[1, i].plot(n_res, B*a2, c='k',
                        linewidth=1.25*linewidth, linestyle='solid')

        axes[2, i].plot(n_res, C*a1, c='k',
                        linewidth=1.5*linewidth, linestyle='dashed',
                        label=r'$|\lambda_1|^{n}$')

    for ax in [axes[0, 0], axes[2, 0]]:
        ax.legend(frameon=False, fontsize=size)

    if save:
        savefig(fig, savedir, saveformat='pdf',
                bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_verblunsky(vns, resonances=None, res_ranges=None,
                    figsize=(5, 5), size=14, labelsize=11,
                    linewidth=1.5, nyticks=4,
                    usetex=True,
                    save=False, savedir=None, show=True):
    latex_config(usetex, False)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.1, wspace=0)

    axes[-1].set_xlabel(r'$n$', size=size)
    axes[1].set_ylabel(r'Verblunsky coefficients $|\alpha_{n}|$', size=size)

    for ax in axes:
        ax.set_yscale('log')
        ax.tick_params(labelsize=labelsize)
        ax.set_yticks([1e-1, 1e-3, 1e-5])
        ax.set_ylim(0.8*1e-5, 2.5*1e-1)

    # sequences
    vns_ = np.abs(vns)

    cmap = mpl.colormaps['tab20b']
    n = np.arange(len(vns_[0]))
    for i in range(len(vns)):
        axes[i].plot(vns_[i], '-s', c=cmap(4*1+1), linewidth=linewidth)

    # plot resonances on top
    for i in range(3):
        vn_ = vns_[i]
        n_res = n[res_ranges[i]]
        a1 = resonances[i]**n_res

        m = -1

        # chose factor that minimizes mean square deviations in logarithmic scale
        V = np.exp(np.mean((np.log(vn_[res_ranges[i]][:m]) - np.log(resonances[i])*n_res[:m])))

        #  print(i, V)

        axes[i].plot(n_res, V*a1, c='k',
                     linewidth=1.5*linewidth, linestyle='dashed',
                     label=r'$|\lambda_1|^{n}$')

    axes[0].legend(frameon=False, fontsize=size)

    #  fig.tight_layout()
    if save:
        savefig(fig, savedir, saveformat='pdf',
                bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_r2_two(r2s, lyapunovs,
                figsize=(9, 4), size=14, labelsize=11, linewidth=1.5,
                usetex=True,
                save=False, savedir=None, show=True):
    latex_config(usetex, False)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.05)

    for ax in axes: ax.tick_params(labelsize=labelsize)

    r2_uniform = (1/6)**0.5 # approx value for uniform op in chord rep

    cmap = mpl.colormaps['tab20b']

    for i in range(2):
        r2 = r2s[i][:10]
        n = np.arange(len(r2))[:10]

        axes[i].semilogy(n, r2, '-o', c=cmap(4*0+2), linewidth=linewidth)
        axes[i].semilogy(n, 8*r2[0]*np.exp(lyapunovs[i]*n),
                         c='k', linestyle='--', linewidth=1.25*linewidth,
                         label=r'$e^{\Lambda n}$')
        axes[i].axhline(y=r2_uniform, linestyle=(0,(1, 1)), color='tab:gray',
                        label=r'$1/\sqrt{6}$', linewidth=1.5*linewidth)

    axes[0].set_ylabel(r'$r_n$', size=size)
    for ax in axes: ax.set_xlabel(r'$n$', size=size)
    ax.set_ylim(1e-3, 1)
    plt.legend(frameon=False, fontsize=size)
    if save:
        savefig(fig, savedir, saveformat='pdf',
                bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_traceful(cts, vns, names, ct_model, vn_model, y0,
                  figsize=(10, 4), size=14, labelsize=11, linewidth=1.5,
                  usetex=True,
                  save=False, savedir=None, show=True):
    latex_config(usetex, False)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    fig.subplots_adjust(wspace=0.225)

    for ax in axes: ax.tick_params(labelsize=labelsize)

    cmap = mpl.colormaps['tab20b']
    colors = [cmap(4*0+2), cmap(4*3+2), cmap(4*4+2)]
    markers = ['-^', '-s', '-o']

    n = np.arange(len(vn_model))
    for i in range(3):
        axes[0].plot(cts[i], markers[i], c=colors[i], linewidth=linewidth)
        axes[1].plot(n, vns[i], markers[i], c=colors[i], label=names[i],
                     linewidth=linewidth)

    axes[0].axhline(y=y0, label=r'$1/||\hat{\rho}_0||^2$',
                    linestyle=(0,(1, 1)), color='tab:gray',
                    linewidth=linewidth)

    axes[0].plot(ct_model, '.-', c='k')
    axes[1].plot(n, vn_model, '.-', c='k', label='Model')

    axes[0].set_ylabel(r'$C(t)$', size=size)
    axes[0].set_xlabel(r'$t$', size=size)

    axes[1].set_ylabel(r'$|\alpha_n|$', size=size)
    axes[1].set_xlabel(r'$n$', size=size)

    for ax in axes: ax.legend(frameon=False, fontsize=size)
    if save:
        savefig(fig, savedir, saveformat='pdf',
                bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
