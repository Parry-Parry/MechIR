import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.io as pio

# TODO: results loading helper functions

# TODO: token segmenting function


"""
Plot results from patching overall blocks (i.e., pre-residual stream, attention block outputs, MLP outputs)
"""


def plot_blocks(
    data,  # shape: (3, num_layers, num_token_labels)
    labels,
    save_path,
    title="Activation Patching Per Block",
):

    fig = sp.make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Residual Stream", "Attn Output", "MLP Output"],
        shared_yaxes=True,
        horizontal_spacing=0.1,
    )

    # Create heatmaps for each experiment
    for i in range(3):
        heatmap_data = data[i, :, :]
        heatmap = go.Heatmap(z=heatmap_data, colorscale="RdBu", zmin=-1, zmax=1)
        fig.add_trace(heatmap, row=1, col=i + 1)

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Position",
            showline=True,
            showgrid=False,
            tickvals=np.arange(len(labels)),
            ticktext=labels,
        ),
        yaxis=dict(title="Layer", showline=True, showgrid=False),
        xaxis2=dict(
            title="Position",
            showline=True,
            showgrid=False,
            tickvals=np.arange(len(labels)),
            ticktext=labels,
        ),
        xaxis3=dict(
            title="Position",
            showline=True,
            showgrid=False,
            tickvals=np.arange(len(labels)),
            ticktext=labels,
        ),
        width=1000,
        height=400,
    )

    if save_path:
        pio.write_image(fig, save_path)

    return fig


"""
Plot results for individual attention heads, with the option to include MLPs for each layer.
"""


def plot_components(
    data,  # shape: (num_layers, num_heads) or (num_layers, num_heads + 1) if include_mlp=True
    save_path=None,
    title="Component Patching Results",
    include_mlp=False,
):

    data = data.astype(float)
    plt.figure(figsize=(10, 6))

    ax = sns.heatmap(
        data,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        fmt=".2f",
        annot=True,
        annot_kws={"size": 8},
    )

    if include_mlp:
        new_labels = list(map(str, range(data.shape[1] - 1)))
        new_labels.append("MLP")
        ax.set_xticklabels(new_labels)

    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Layer")

    if save_path:
        plt.savefig(save_path)
        plt.close()

    else:
        plt.show()


__all__ = ["plot_blocks", "plot_components"]
