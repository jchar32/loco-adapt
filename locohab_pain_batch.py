# %%load libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from utils import fitting


def fit_to_data(
    data, timepoints, fit_data, grouping_name="overall", index=None, trial=None
):
    # Fit linear decay models to the tonic pain data
    (lin_coefs, lin_cov, lin_info, _) = fitting.fit_lin(
        data["timepoint"], data["painrating"]
    )

    lin_ypred = fitting.linear(timepoints, *lin_coefs)
    lin_r2 = fitting.rsquared(
        data["painrating"], fitting.linear(data["timepoint"], *lin_coefs)
    )

    # exponential decay fit
    (exp_coefs, exp_cov, exp_info, _) = fitting.fit_exp(
        data["timepoint"], data["painrating"]
    )

    exp_ypred = fitting.exponential(timepoints, *exp_coefs)
    exp_r2 = fitting.rsquared(
        data["painrating"], fitting.exponential(data["timepoint"], *exp_coefs)
    )

    if index is None:
        index = 0

    # if trial is None:
    fit_data["linear"]["coeffs"][index] = lin_coefs
    fit_data["linear"]["cov"][index] = lin_cov
    fit_data["linear"]["info"][index] = lin_info
    fit_data["linear"]["ypred"][index] = lin_ypred
    fit_data["linear"]["r2"][index] = lin_r2

    fit_data["exp"]["coeffs"][index] = exp_coefs
    fit_data["exp"]["cov"][index] = exp_cov
    fit_data["exp"]["info"][index] = exp_info
    fit_data["exp"]["ypred"][index] = exp_ypred
    fit_data["exp"]["r2"][index] = exp_r2

    return fit_data


def process_habituation_data(p_colours, timepoints):
    # !! Really should be a Class but this is simpler for exporting stuff elsewhere, for now...
    fit_dict = {
        "linear": {"coeffs": {}, "cov": {}, "info": {}, "ypred": {}, "r2": {}},
        "exp": {"coeffs": {}, "cov": {}, "info": {}, "ypred": {}, "r2": {}},
    }
    tonic_fits = {
        "overall": fit_dict,
        "participant": fit_dict,
        "trial1": fit_dict,
        "trial2": fit_dict,
        "trial3": fit_dict,
    }
    # !! -----------------------

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

    # # Fit linear and exponential decay models to the mean tonic pain data
    tonic_fits["overall"] = fit_to_data(
        tonic_data, timepoints, tonic_fits["overall"], grouping_name="overall"
    )

    # For each participant
    for i in np.unique(tonic_data["pid"]):
        # fit linear
        tonic_data_filt = tonic_data.loc[tonic_data["pid"] == i, :]
        tonic_fits["participant"] = fit_to_data(
            tonic_data_filt,
            timepoints,
            tonic_fits["participant"],
            grouping_name="participant",
            index=i,
        )

        # fit for each trial (n=3)
        for t in np.unique(tonic_data_filt["trial"]):
            tonic_fits[f"trial{t}"] = fit_to_data(
                tonic_data_filt.loc[tonic_data_filt["trial"] == t, :],
                timepoints,
                tonic_fits[f"trial{t}"],
                grouping_name="participant_trial",
                index=i,
            )
    return tonic_data, tonic_fits


def save_habituation_data(tonic_data, tonic_fits) -> None:
    # save fit data
    with open(
        os.path.join("../data/locohab/", "habituation_fit_results.pkl"), "wb"
    ) as f:
        pickle.dump(tonic_fits, f)

    # save raw pain data data
    with open(os.path.join("../data/locohab/", "pain_time_data.pkl"), "wb") as f2:
        pickle.dump(tonic_data, f2)

    # put into csv for transfer to R stats
    tonic_data.to_csv("../data/locohab/pain_time_data.csv")


# %%
def plot_alltrials(tonic_data, tonic_fits, timepoints, p_colours):
    # Plot data and fits
    tonic_data["pid"] = tonic_data["pid"].astype(str)
    participant_numbers = np.unique(tonic_data["pid"])

    # All trials together
    fig = go.Figure()
    for i, p in enumerate(participant_numbers):
        p = int(p)
        fig.add_trace(
            go.Scatter(
                x=tonic_data["timepoint"][tonic_data["pid"] == str(p)],
                y=tonic_data["painrating"][tonic_data["pid"] == str(p)]
                .to_numpy()
                .flatten(),
                mode="markers",
                name=f"P{p}",
                marker=dict(size=10, opacity=0.8, color=p_colours[i]),
                showlegend=False,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=timepoints.flatten(),
                y=tonic_fits["participant"]["exp"]["ypred"][p].flatten(),
                mode="lines",
                name=f"P{p}",
                line=dict(dash="solid", color=p_colours[i]),
                showlegend=False,
            ),
        )

    fig.add_trace(
        go.Scatter(
            x=timepoints.flatten(),
            y=tonic_fits["overall"]["exp"]["ypred"][0].flatten(),
            mode="lines",
            name="Overall Fit",
            line=dict(dash="dash", color="white", width=4),
            showlegend=True,
        ),
    )

    fig.update_traces(
        marker_size=7, marker_opacity=0.4, marker_line_color="white", showlegend=True
    )

    fig.update_layout(
        width=600,
        height=600,
        template="simple_white",
        title="Pain ratings over time for each participant and overall fit",
        legend=dict(orientation="v"),
    )
    fig.update_xaxes(range=[-5, 605])

    fig.show()
    return fig


# %%
def plot_separatetrials(tonic_data, tonic_fits, timepoints, p_colours):
    tonic_data["pid"] = tonic_data["pid"].astype(str)
    participant_numbers = np.unique(tonic_data["pid"])
    # Trials separated: pain over time, per trial
    subp = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=("Trial 1", "Trial 2", "Trial 3"),
        x_title="Time (s)",
        y_title="Pain rating (0-10)",
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
    )
    tplot = {}
    for i, p in enumerate(participant_numbers):
        for t in range(1, 4):
            tplot[t] = subp.add_trace(
                go.Scatter(
                    x=tonic_data["timepoint"][
                        (tonic_data["pid"] == str(p)) & (tonic_data["trial"] == t)
                    ],
                    y=tonic_data["painrating"][
                        (tonic_data["pid"] == str(p)) & (tonic_data["trial"] == t)
                    ]
                    .to_numpy()
                    .flatten(),
                    mode="lines+markers",
                    name=f"P{p}",
                    marker=dict(size=10, opacity=0.5, color=p_colours[i]),
                    line=dict(dash="solid", color=p_colours[i], width=2),
                    showlegend=False,
                ),
                row=1,
                col=t,
            )
            if i == len(participant_numbers) - 1:
                subp.add_trace(
                    go.Scatter(
                        x=timepoints.flatten(),
                        y=tonic_fits[f"trial{t}"]["exp"]["ypred"][0].flatten(),
                        mode="lines",
                        name=f"Mean Habituation Trial{t}",
                        line=dict(dash="solid", color="black", width=4),
                        showlegend=False,
                    ),
                    row=1,
                    col=t,
                )
    subp.update_layout(
        width=1200,
        height=600,
        template="simple_white",
        title="Pain Ratings Over Time",
        legend=dict(orientation="v"),
        font_family="Arial, sans-serif",
        font_size=16,
    )
    subp.update_xaxes(range=[-5, 605])

    subp.show()
    return subp


# %%
if __name__ == "__main__":
    # Initializations
    p_colours = px.colors.qualitative.Vivid
    timepoints = np.arange(30, 630, 30).reshape(1, 20)  # up to 630 so 600 is included.
    tonic_data, tonic_fits = process_habituation_data(p_colours, timepoints)
    # save_habituation_data(tonic_data, tonic_fits)
    alltrials_plot = plot_alltrials(tonic_data, tonic_fits, timepoints, p_colours)
    separatetrials_plot = plot_separatetrials(
        tonic_data, tonic_fits, timepoints, p_colours
    )
    separatetrials_plot.write_html("../data/locohab/pain_habituation_plot-3panel.html")
