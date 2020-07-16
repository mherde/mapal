import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from num2tex import num2tex

from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size, AxesGrid

from scipy.spatial import ConvexHull

from src.utils.statistic_functions import compute_statistics


def plot_aulc_ranks(aulc_ranks, test_results, xlabels, ylabels, plot_type='standard', filename=None, filetype='pdf'):
    # creat grid plot
    fontsize = 50 if plot_type == 'standard' else 55
    max_len = np.max([len(l) for l in xlabels])
    fig, ax = plt.subplots(figsize=((aulc_ranks.shape[1] * 2 + 3), (aulc_ranks.shape[0]) * 2 + 2 + 0.125*max_len))
    aspect = 20
    pad_fraction = 0.5
    im = ax.imshow(aulc_ranks, cmap='YlGn_r', vmin=1, vmax=len(ylabels), alpha=0.75)
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1. / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    ax.set_xticks(np.arange(aulc_ranks.shape[1]))
    ax.set_yticks(np.arange(aulc_ranks.shape[0]))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.xaxis.set_tick_params(labelsize=fontsize+3, color='k')
    ax.yaxis.set_tick_params(labelsize=fontsize, color='k')
    ax.set_ylim(len(ylabels) - 0.5, -0.5)
    ax.axhline(0.5, color='k', lw=5)
    plt.sca(ax)
    plt.grid()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # loop over data dimensions and create text annotations of mean ranks
    best_ids = np.argmin(aulc_ranks, axis=0)
    if plot_type == 'standard':
        for i in range(aulc_ranks.shape[0]):
            for j in range(aulc_ranks.shape[1]):
                offset = 0.18 if i != 0 else 0
                sig_str = ''
                if best_ids[j] == i:
                    rank = '$\mathbf{{{:#1.3}}}$'.format(num2tex(aulc_ranks[i, j], precision=3))
                    if test_results[i, j] == -3:
                        sig_str = '$\dagger$' #* abs(int(test_results[i, j]))
                else:
                    rank = '${:#1.3}$'.format(num2tex(aulc_ranks[i, j], precision=3))
                    if test_results[i, j] == 3:
                        sig_str = '*' #* int(test_results[i, j])
                    elif test_results[i, j] == -3:
                        sig_str = '$\dagger$' #* int(test_results[i, j])
                ax.text(j, i + offset, rank, ha="center", va="center", color='black', fontsize=fontsize)
                ax.text(j, i + offset - 0.42, sig_str, ha="center", va="center", color='k', fontsize=fontsize)
    elif plot_type == 'mean':
        for i in range(aulc_ranks.shape[0]):
            for j in range(aulc_ranks.shape[1]):
                offset = -0.15 if i != 0 else 0
                if best_ids[j] == i:
                    rank = '$\mathbf{{{:#1.3}}}$'.format(num2tex(aulc_ranks[i, j], precision=3))
                else:
                    rank = '${:#1.3}$'.format(num2tex(aulc_ranks[i, j], precision=3))
                    wins = np.sum(test_results[i, :] == 3)
                    losses = np.sum(test_results[i, :] == -3)
                    draws = len(test_results[i, :]) - wins - losses
                    result_str = '${}/{}/{}$'.format(wins, draws, losses)
                    ax.text(0, i + offset + 0.35, result_str, ha="center", va="center", color='k',
                            fontsize=fontsize - 18)
                ax.text(j, i + offset, rank, ha="center", va="center", color='black', fontsize=fontsize)

    # setup colorbar
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.invert_yaxis()
    cbar.ax.tick_params(labelsize=fontsize, color='k')
    cbar.ax.set_ylabel('mean rank', rotation=-90, va="bottom", fontsize=fontsize, color='k')
    if filename is not None:
        fig.savefig('{}-{}-ranks.{}'.format(filename, plot_type, filetype), bbox_inches="tight")
    plt.show()


def plot_scores_2d(figsize, X, y_true, y, X_1_mesh, X_2_mesh, labeled_indices, unlabeled_indices, scores, P=None,
                   title=None, vmin=0, vmax=1, filename=None, filetype='pdf', fontsize=15, cbar_label=None):
    n_annotators = y.shape[1]

    # setup figure
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, n_annotators),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )
    for a, ax in enumerate(grid):
        ax.set_xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
        ax.set_ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
        ax.set_title(r'annotator $a_{}$'.format(a+1), fontsize=fontsize)
        ax.set_xlabel(r'feature $x_1$', fontsize=fontsize, color='k')
        ax.set_ylabel(r'feature $x_2$', fontsize=fontsize, color='k')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False)  # labels along the bottom edge are off

        ax.scatter(X[labeled_indices[a], 0], X[labeled_indices[a], 1], c=[[.2, .2, .2]], s=180, marker='o', zorder=3.8)
        ax.scatter(X[labeled_indices[a], 0], X[labeled_indices[a], 1], c=[[.8, .8, .8]], s=120, marker='o', zorder=4)
        is_false = y_true != y[:, a]
        for cl, color in zip([0, 1], ['r', 'b']):
            cl_labeled_idx_false = labeled_indices[a][np.logical_and(y[labeled_indices[a], a] == cl,
                                                                     is_false[labeled_indices[a]])]
            cl_labeled_idx_correct = labeled_indices[a][np.logical_and(y[labeled_indices[a], a] == cl,
                                                                       ~is_false[labeled_indices[a]])]
            cl_unlabeled_idx_false = unlabeled_indices[a][np.logical_and(y[unlabeled_indices[a], a] == cl,
                                                                         is_false[unlabeled_indices[a]])]
            cl_unlabeled_idx_correct = unlabeled_indices[a][np.logical_and(y[unlabeled_indices[a], a] == cl,
                                                                           ~is_false[unlabeled_indices[a]])]
            ax.scatter(X[cl_labeled_idx_false, 0], X[cl_labeled_idx_false, 1], color=color, marker='x',
                       vmin=-0.2, vmax=1.2, cmap='coolwarm', s=40, zorder=5)
            ax.scatter(X[cl_labeled_idx_correct, 0], X[cl_labeled_idx_correct, 1], color=color, marker='s',
                       vmin=-0.2, vmax=1.2, cmap='coolwarm', s=40, zorder=5)
            ax.scatter(X[cl_unlabeled_idx_false, 0], X[cl_unlabeled_idx_false, 1], c=color, marker='x',
                       vmin=-0.2, vmax=1.2, cmap='coolwarm', s=40, zorder=3)
            ax.scatter(X[cl_unlabeled_idx_correct, 0], X[cl_unlabeled_idx_correct, 1], c=color, marker='s',
                       vmin=-0.2, vmax=1.2, cmap='coolwarm', s=40, zorder=3)
        im = ax.contourf(X_1_mesh, X_2_mesh, scores[a], np.linspace(vmin, vmax, 10),
                         cmap='Greens', alpha=.75, vmin=vmin, vmax=vmax)
        if P is not None:
            ax.contour(X_1_mesh, X_2_mesh, P[a], [.49, .51], cmap='coolwarm', linewidths=[4, 4],
                        zorder=1, alpha=.8, vmin=.488, vmax=.512)
            ax.contour(X_1_mesh, X_2_mesh, P[a], [.5], colors='k', linewidths=[2], zorder=1)

        handles, labels = ax.get_legend_handles_labels()

    line = Line2D([0], [0], color='k', linewidth=2)
    patch = Line2D([0], [0], marker='o', markerfacecolor='grey', markeredgecolor='k',
                   markersize=20, alpha=0.8, color='w')
    true_patches = (Line2D([0], [0], marker='s', markerfacecolor='b', markersize=15, color='w'),
                    Line2D([0], [0], marker='s', markerfacecolor='r', markersize=15, color='w'))
    false_patches = (Line2D([0], [0], marker='x', markerfacecolor='b', markeredgecolor='b', markersize=15, color='w'),
                     Line2D([0], [0], marker='x', markerfacecolor='r', markeredgecolor='r', markersize=15, color='w'))
    handles = [patch, true_patches, false_patches, line]
    labels = ['acquired annotation', 'true annotation', 'false annotation', 'decision boundary']

    vdiff = vmax - vmin
    cbar = ax.cax.colorbar(im, ticks=[vmin, vmax])
    cbar = grid.cbar_axes[0].colorbar(im, ticks=[vmin, vmax])
    cbar.ax.set_yticklabels(['low', 'high'], fontsize=fontsize)
    if cbar_label is not None:
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", color='k', fontsize=fontsize, labelpad=-40)
    if filename is not None:
        fig.savefig('{}.{}'.format(filename, filetype), bbox_inches="tight")
        figlegend = plt.figure(figsize=figsize)
        figlegend.legend(handles, labels, ncol=len(handles), fontsize=fontsize,
                         loc='center', handler_map={tuple: HandlerTuple(ndivide=None)})
        figlegend.savefig('{}_legend.{}'.format(filename, filetype), bbox_inches="tight")

    return fig


def plot_simulation(figsize, X, y_true, y, y_cluster=None, title=None, filename=None, filetype='pdf', fontsize=15):
    n_annotators = y.shape[1]

    # setup figure
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, n_annotators),
                    axes_pad=0.05,
                    )

    if y_cluster is not None:
        hulls = []
        X_cluster = []
        for cluster in np.unique(y_cluster):
            X_cluster.append(X[y_cluster == cluster])
            hulls.append(ConvexHull(X_cluster[-1]))

    for a, ax in enumerate(grid):
        ax.set_xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
        ax.set_ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
        ax.set_title(r'annotator $a_{}$'.format(a+1), fontsize=fontsize)
        ax.set_xlabel(r'feature $x_1$', fontsize=fontsize, color='k')
        ax.set_ylabel(r'feature $x_2$', fontsize=fontsize, color='k')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False)  # labels along the bottom edge are off

        is_false = y[:, a] != y_true
        for cl, color in zip([0, 1], ['r', 'b']):
            cl_idx_false = np.logical_and(y[:, a] == cl, is_false)
            cl_idx_correct = np.logical_and(y[:, a] == cl, ~is_false)
            ax.scatter(X[cl_idx_false, 0], X[cl_idx_false, 1], color=color, marker='x',
                       vmin=-0.2, vmax=1.2, cmap='coolwarm', s=40, zorder=5)
            ax.scatter(X[cl_idx_correct, 0], X[cl_idx_correct, 1], color=color, marker='s',
                       vmin=-0.2, vmax=1.2, cmap='coolwarm', s=40, zorder=5)

        if y_cluster is not None:
            for c_idx in np.unique(y_cluster):
                for simplex in hulls[c_idx].simplices:
                    ax.plot(X_cluster[c_idx][simplex, 0], X_cluster[c_idx][simplex, 1], 'k-')


    true_patches = (Line2D([0], [0], marker='s', markerfacecolor='b', markersize=15, color='w'),
                    Line2D([0], [0], marker='s', markerfacecolor='r', markersize=15, color='w'))
    false_patches = (Line2D([0], [0], marker='x', markerfacecolor='b', markeredgecolor='b', markersize=15, color='w'),
                     Line2D([0], [0], marker='x', markerfacecolor='r', markeredgecolor='r', markersize=15, color='w'))
    handles = [true_patches, false_patches]
    labels = ['true annotation', 'false annotation']
    if y_cluster is not None:
        line = Line2D([0], [0], color='k', linewidth=2)
        handles.append(line)
        labels.append('cluster boundary')

    if filename is not None:
        fig.savefig('{}.{}'.format(filename, filetype), bbox_inches="tight")
        figlegend = plt.figure(figsize=figsize)
        figlegend.legend(handles, labels, ncol=len(handles), fontsize=fontsize,
                         loc='center', handler_map={tuple: HandlerTuple(ndivide=None)})
        figlegend.savefig('{}-legend.{}'.format(filename, filetype), bbox_inches="tight")

    return fig


def plot_learning_curves(results_dict, strategy_df, save=False, filename='learning_curves', fontsize=15,
                         filetype='pdf'):
    for d in results_dict.keys():
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        #fig.suptitle('{}'.format(d), fontsize=fontsize)
        for q in results_dict[d].keys():
            stats_dict = compute_statistics(results_dict[d][q])
            lw = 3 if 'mapal' in q else 1.5
            zorder = 2 if 'mapal' in q else 1
            ax[0, 0].plot(stats_dict['test-misclf-rate-mean'], label=strategy_df.loc[q]['name'], zorder=zorder,
                          color=strategy_df.loc[q]['color'], ls=strategy_df.loc[q]['line'], lw=lw)
            ax[0, 1].plot(stats_dict['train-misclf-rate-mean'], zorder=zorder,
                          color=strategy_df.loc[q]['color'], ls=strategy_df.loc[q]['line'], lw=lw)
            ax[1, 0].plot(stats_dict['n-labeled-samples-mean'], zorder=zorder,
                          color=strategy_df.loc[q]['color'], ls=strategy_df.loc[q]['line'], lw=lw)
            ax[1, 1].plot(stats_dict['n-false-labels-mean'], zorder=zorder,
                          color=strategy_df.loc[q]['color'], ls=strategy_df.loc[q]['line'], lw=lw)
        for i, a in enumerate(ax.flat):
            if i == 0:
                y_label = r'test misclf-rate'
            elif i == 1:
                y_label = r'training misclf-rate'
            elif i == 2:
                y_label = r'no. of annotated instances'
            else:
                y_label = r'no. of false annotations'
            a.set_ylabel(y_label, fontsize=fontsize, color='k')
            a.set_xlabel(r'total no. of annotations', fontsize=fontsize, color='k')
        leg = fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=False,
                         ncol=4, fontsize=fontsize)
        for line in leg.get_lines():
            line.set_linewidth(1.5)
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        if save:
            fig.savefig('{}-{}.{}'.format(filename, d, filetype))
        plt.show()


def create_latex_table(aulc_mean, aulc_std, test_results, data_set_names, strategy_names):
    df = pd.DataFrame(index=data_set_names, columns=strategy_names)
    best = np.argmin(aulc_mean, axis=1)
    for d_idx, d in enumerate(data_set_names):
        for s_idx, s in enumerate(strategy_names):
            test_string = ''
            if test_results[s_idx, d_idx] > 0:
                test_string = '*' * int(test_results[s_idx, d_idx])
            elif test_results[s_idx, d_idx] < 0:
                test_string = '\dagger' * int(abs(test_results[s_idx, d_idx]))
            if best[d_idx] == s_idx:
                df.loc[d][s] = '$\mathbf{{{:#1.2}}} \pm {:#1.2} {}$'.format(aulc_mean[d_idx, s_idx],
                                                                            aulc_std[d_idx, s_idx], test_string)
            else:
                df.loc[d][s] = '${:#1.2} \pm {:#1.2} {}$'.format(aulc_mean[d_idx, s_idx], aulc_std[d_idx, s_idx],
                                                                 test_string)
    return df.to_latex(escape=False)
