import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils import fitting


# Function to plot pain as function of current with fitting results
def pain_current_plot(
    xdata, ydata, expfitparams, linfitparams, p, fig, plot_lin=True, plot_exp=True
):
    # min and max x values
    xmin = min(xdata)
    xmax = max(xdata)

    # build figure
    fig.add_trace(
        go.Scatter(
            x=xdata,
            y=ydata,
            mode="markers",
            name=str(p),
            marker=dict(size=10, color=px.colors.qualitative.Plotly[p]),
            marker_opacity=0.3,
        )
    )

    if plot_exp:
        fig.add_trace(
            go.Scatter(
                x=np.linspace(xmin, xmax, xdata.shape[0]),
                y=fitting.exponential(
                    np.linspace(xmin, xmax, xdata.shape[0]), *expfitparams[0]
                ),
                mode="lines",
                name=str(p) + "exp",
                line=dict(dash="solid", width=3, color=px.colors.qualitative.Plotly[p]),
            )
        )

    if plot_lin:
        fig.add_trace(
            go.Scatter(
                x=np.linspace(xmin, xmax, xdata.shape[0]),
                y=fitting.linear(
                    np.linspace(xmin, xmax, xdata.shape[0]), *linfitparams[0]
                ),
                mode="lines",
                name=str(p) + "linear",
                line=dict(dash="solid", width=3, color=px.colors.qualitative.Plotly[p]),
            )
        )


def pain_current_subplot(
    xdata,
    ydata,
    expfitparams,
    linfitparams,
    p,
    fig,
    plot_lin=True,
    plot_exp=True,
    row=1,
    col=1,
):
    # min and max x values
    xmin = min(xdata)
    xmax = max(xdata)

    # build figure
    fig.add_trace(
        go.Scatter(
            x=xdata,
            y=ydata,
            mode="markers",
            name=str(p),
            marker=dict(size=10, color=px.colors.qualitative.Plotly[p]),
            marker_opacity=0.3,
        ),
        row=row,
        col=col,
    )

    if plot_exp:
        fig.add_trace(
            go.Scatter(
                x=np.linspace(xmin, xmax, xdata.shape[0]),
                y=fitting.exponential(
                    np.linspace(xmin, xmax, xdata.shape[0]), *expfitparams[0]
                ),
                mode="lines",
                name=str(p) + "exp",
                line=dict(dash="solid", width=3, color=px.colors.qualitative.Plotly[p]),
            ),
            row=row,
            col=col,
        )

    if plot_lin:
        fig.add_trace(
            go.Scatter(
                x=np.linspace(xmin, xmax, xdata.shape[0]),
                y=fitting.linear(
                    np.linspace(xmin, xmax, xdata.shape[0]), *linfitparams[0]
                ),
                mode="lines",
                name=str(p) + "linear",
                line=dict(dash="dash", width=3, color=px.colors.qualitative.Plotly[p]),
            ),
            row=row,
            col=col,
        )
