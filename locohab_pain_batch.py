# load libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from utils import fitting
import utils.render as render


def main():
    # Initializations
    p_colours = px.colors.qualitative.Vivid
    timepoints = np.arange(30, 630, 30).reshape(1, 20)  # up to 630 so 600 is included.
    tonic_fits = {
        "overall": {
            "linear": {"coeffs": {}, "cov": {}, "info": {}, "ypred": {}},
            "exp": {"coeffs": {}, "cov": {}, "info": {}, "ypred": {}},
        },
        "participant": {
            "linear": {"coeffs": {}, "cov": {}, "info": {}, "ypred": {}},
            "exp": {"coeffs": {}, "cov": {}, "info": {}, "ypred": {}},
        },
    }

    # Load data
    d = pd.read_excel("../locohab data.xlsx", sheet_name="painratings")

    # remove any remaining participant place holders
    d = d.dropna()

    # set column headers to strings
    d.columns = d.columns.astype(str)

    # combine trial and condition columns
    pid = d.loc[:, "pid"].to_numpy()

    # Change to longform and remove any remaining participant place holders
    tonic_data = pd.melt(
        d,
        id_vars=["pid", "session", "trial", "paincond"],
        value_vars=d.columns[3:].to_numpy(),
        value_name="painrating",
        var_name="timepoint",
    ).reset_index(drop=True)

    tonic_data["timepoint"] = tonic_data["timepoint"].astype(int)

    # Fit linear and exponential decay models to the mean tonic pain data
    (
        tonic_fits["overall"]["linear"]["coeffs"],
        tonic_fits["overall"]["linear"]["cov"],
        tonic_fits["overall"]["linear"]["info"],
        _,
    ) = fitting.fit_lin(tonic_data["timepoint"], tonic_data["painrating"])
    tonic_fits["overall"]["linear"]["ypred"] = fitting.linear(
        timepoints, *tonic_fits["overall"]["linear"]["coeffs"]
    )

    # Mean exponential decay fit
    (
        tonic_fits["overall"]["exp"]["coeffs"],
        tonic_fits["overall"]["exp"]["cov"],
        tonic_fits["overall"]["exp"]["info"],
        _,
    ) = fitting.fit_exp(tonic_data["timepoint"], tonic_data["painrating"])
    tonic_fits["overall"]["exp"]["ypred"] = fitting.exponential(
        timepoints, *tonic_fits["overall"]["exp"]["coeffs"]
    )

    # For each participant
    for i in np.unique(tonic_data["pid"]):
        # fit linear
        tonic_data_filt = tonic_data.loc[tonic_data["pid"] == i, :]
        (
            tonic_fits["participant"]["linear"]["coeffs"][i],
            tonic_fits["participant"]["linear"]["cov"][i],
            tonic_fits["participant"]["linear"]["info"][i],
            _,
        ) = fitting.fit_lin(tonic_data_filt["timepoint"], tonic_data_filt["painrating"])
        tonic_fits["participant"]["linear"]["ypred"][i] = fitting.linear(
            timepoints, *tonic_fits["participant"]["linear"]["coeffs"][i]
        )
        # Fit Exponential Decay
        (
            tonic_fits["participant"]["exp"]["coeffs"][i],
            tonic_fits["participant"]["exp"]["cov"][i],
            tonic_fits["participant"]["exp"]["info"][i],
            _,
        ) = fitting.fit_exp(tonic_data_filt["timepoint"], tonic_data_filt["painrating"])
        tonic_fits["participant"]["exp"]["ypred"][i] = fitting.exponential(
            timepoints, *tonic_fits["participant"]["exp"]["coeffs"][i]
        )

    # save fit data
    with open(
        os.path.join("../data/locohab/", "habituation_fit_results.pkl"), "wb"
    ) as f:
        pickle.dump(tonic_fits, f)

    # Plot data and fits
    tonic_data["pid"] = tonic_data["pid"].astype(str)
    participant_numbers = np.unique(tonic_data["pid"])
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Linear Fit", "Exponential Fit"),
        x_title="Time (s)",
        y_title="Pain rating (0-10)",
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
    )

    for i, p in enumerate(participant_numbers):
        fig = render.pain_time_plot(
            fig, p, i, tonic_data, tonic_fits, timepoints, p_colours
        )

    # Add overall fits
    fig.add_trace(
        go.Scatter(
            x=timepoints.flatten(),
            y=tonic_fits["overall"]["linear"]["ypred"].flatten(),
            mode="lines",
            name="Overall Fit",
            line=dict(dash="dash", color="white", width=4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timepoints.flatten(),
            y=tonic_fits["overall"]["exp"]["ypred"].flatten(),
            mode="lines",
            name="Overall Fit",
            line=dict(dash="dash", color="white", width=4),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_traces(
        marker_size=5,
        marker_opacity=0.8,
    )

    fig.update_layout(
        width=1200,
        height=600,
        template="plotly_dark",
        title="Pain ratings over time for each participant and overall fit",
        legend=dict(orientation="v"),
    )
    fig.update_xaxes(range=[-5, 605])

    fig.show()


if __name__ == "__main__":
    main()
