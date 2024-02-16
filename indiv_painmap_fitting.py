# %%
if __name__ == "__main__":
    import numpy as np
    from utils import fitting
    from utils import ui
    import pandas as pd
    # import matplotlib.pyplot as plt
    import os
    import plotly.graph_objects as go
    import pickle
    import config

    configs = config.painmap()

    (datadir, filenames) = ui.get_path()

    p_data = {"direc": datadir, "files": filenames}
    pain_current_data = []
    for f in filenames:
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(p_data["direc"], f), header=0, index_col=None)
            pain_current_data.append(df)

    #  Concatenate all the data files (there will usually be >=2)
    pain_current_data = pd.concat(pain_current_data, axis=0, ignore_index=True)

    #  replace NANs with 0 as that is what they mean.
    pain_current_data = pain_current_data.fillna(0)

    # add to dictionary
    p_data["data"] = pain_current_data

    # Prepare data for fitting

    xdata = p_data["data"]["mA"].to_numpy()
    ydata = p_data["data"]["painreport"].to_numpy()

    xmin = p_data["data"]["mA"].min()
    xmax = p_data["data"]["mA"].max()
    ymin = p_data["data"]["painreport"].min()
    ymax = p_data["data"]["painreport"].max()

    # construct constant x interval spacing for normalization
    xintervals = np.linspace(xmin, xmax, xdata.shape[0])

    # normalized x axis
    p_data["data"]["mA_norm"] = (xdata - xmin) / (xmax - xmin)
    xdata_norm = p_data["data"]["mA_norm"].to_numpy()

    # Data above a priori pain cutoff
    yfiltered = p_data["data"][ydata >= configs["pain_cutoff_for_linearfit"]]

    # Fit data to exponential and linear functions
    # Exponential fit
    p_data["efit"] = fitting.fit_exp(xdata, ydata)

    p_data["efit_y"] = fitting.exponential(xdata, *p_data["efit"][0])
    p_data["data"]["efit_y"] = p_data["efit_y"]

    p_data["efit_r2_all"] = fitting.rsquared(ydata, p_data["efit_y"])

    p_data["efit_r2_yfiltered"] = fitting.rsquared(
        yfiltered["painreport"],
        fitting.exponential(yfiltered["mA"], *p_data["efit"][0]),
    )

    # exponential fit on normalized x axis
    p_data["efit_norm"] = fitting.fit_exp(xdata_norm, ydata)
    p_data["efit_norm_y"] = fitting.exponential(xdata_norm, *p_data["efit_norm"][0])
    p_data["data"]["efit_norm_y"] = p_data["efit_norm_y"]

    # Basic linear fit
    # Fit linear to unnormalized x axis
    p_data["lfit"] = fitting.fit_lin(yfiltered["mA"], yfiltered["painreport"])
    p_data["lfit_y"] = fitting.linear(yfiltered["mA"], *p_data["lfit"][0])

    p_data["lfit_r2"] = fitting.rsquared(
        yfiltered["painreport"], fitting.linear(yfiltered["mA"], *p_data["lfit"][0])
    )

    p_data["lfit_norm"] = fitting.fit_lin(yfiltered["mA_norm"], yfiltered["painreport"])
    p_data["lfit_norm_y"] = fitting.linear(
        yfiltered["mA_norm"], *p_data["lfit_norm"][0]
    )

    p_data["radius"] = fitting.rad_of_curve(
        xintervals, fitting.exponential(xintervals, *p_data["efit"][0])
    )

    #  Print Data to console
    print(
        f'Exponential fit parameters:\n a={p_data["efit"][0][0]} \n b={p_data["efit"][0][1]} \n a={p_data["efit"][0][2]}'
    )
    print(f'Exponential fit R\u00b2: {p_data["efit_r2_all"]}')
    print("\n")
    print(
        f'Linear fit parameters:\n m={p_data["lfit"][0][0]} \n b={p_data["lfit"][0][1]}'
    )
    print(f'Linear fit R\u00b2: {p_data["lfit_r2"]}')

    # Gather current values at specific pain rates for convenience and print to screen
    x_at_i = []
    for i in [1, 2, 3, 4, 5]:
        print("mA at pain intervals fit with Exponential: ------")
        x_at_i.append(fitting.inverse_exponential(i, *p_data["efit"][0]))
        print(f"mA at {i} pain: {x_at_i[i-1]}")

    # Visualize participant's data

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xdata,
            y=ydata,
            mode="markers",
            name="trial",
            marker=dict(size=10, color="black"),
            marker_opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(xmin, xmax, xdata.shape[0]),
            y=fitting.exponential(
                np.linspace(xmin, xmax, xdata.shape[0]), *p_data["efit"][0]
            ),
            mode="lines",
            name="exp fit",
            line=dict(dash="solid", width=3, color="red"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=yfiltered["mA"],
            y=fitting.linear(yfiltered["mA"], *p_data["lfit"][0]),
            mode="lines",
            name="lin fit",
            line=dict(dash="solid", width=3, color="blue"),
        )
    )
    fig.show()

    figt = go.Figure()
    figt.add_trace(
        go.Table(
            header=dict(values=["Pain", "mA"]),
            cells=dict(values=[[1, 2, 3, 4, 5], np.round(x_at_i, 2)]),
        )
    )
    figt.show()

    # Specify the file path and name
    file_path = os.path.join(datadir, "fit_data.pkl")

    # Save p_data as a .pkl file
    try:
        with open(file_path, "wb") as file:
            pickle.dump(p_data, file)
        print("File saved successfully.")
    except Exception as e:
        print(f"Error saving file: {str(e)}")
