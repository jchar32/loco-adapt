# %%
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import utils.processing as processing
import numpy as np
import utils.event_detectors as gaitevents
import pickle
import utils.file_io as io

# obtain all participant folders for analysis
base_path = "../data/painmap/"
proc_data_path = "../data/processed/painmap/"

p_fld = os.listdir(
    base_path,
)
folders = [
    entry
    for entry in p_fld
    if os.path.isdir(os.path.join(base_path, entry)) & ("p" in entry)
]


# %% Calibration Functions


def apply_s2bcal(data_in, cal):
    """
    Apply sensor-to-body calibration to the input data.

    Parameters
    ----------
    data_in : pandas.DataFrame
        Input data to be calibrated.
    cal : dict
        Calibration rotation matrix for each sensor.

    Returns
    -------
    pandas.DataFrame
        Calibrated data.

    Notes
    -----
    This function applies the sensor-to-body calibration to the input data.
    It multiplies the accelerometer and gyroscope data of each sensor by the
    corresponding calibration matrix.

    The calibration dictionary `cal` should have the following structure:
    {
        'sensor1': calibration_matrix1,
        'sensor2': calibration_matrix2,
        ...
    }

    The input data should have columns named as follows:
    - Accelerometer: 'sensor1_ax', 'sensor1_ay', 'sensor1_az', ...
    - Gyroscope: 'sensor1_gx', 'sensor1_gy', 'sensor1_gz', ...

    The calibrated data will replace the original data in the input DataFrame.

    Examples
    --------
    >>> cal = {
    ...     'sensor1': calibration_matrix1,
    ...     'sensor2': calibration_matrix2
    ... }
    >>> calibrated_data = apply_s2bcal(input_data, cal)
    """

    data = data_in.copy(deep=True)

    for c in cal.keys():
        if c == "info":
            continue
        data.loc[:, f"{c}_ax" : f"{c}_az"] = (
            data.loc[:, f"{c}_ax" : f"{c}_az"].to_numpy() @ cal[c]
        )
        data.loc[:, f"{c}_gx" : f"{c}_gz"] = (
            data.loc[:, f"{c}_gx" : f"{c}_gz"].to_numpy() @ cal[c]
        )

    return data


def get_imu_calibration(imuname):
    """
    Get the calibration data for a specific IMU.

    Parameters
    ----------
    imuname : str
        The name of the IMU.

    Returns
    -------
    r : numpy.ndarray
        Orientation correction data.
    acc_off : numpy.ndarray
        Accelerometer offset data.
    gyr_off : numpy.ndarray
        Gyroscope offset data.
    """
    r = pd.read_csv(
        imu_cal_path + imuname + "/cal0/ori_corr.txt", header=None
    ).to_numpy()
    acc_off = pd.read_csv(
        imu_cal_path + imuname + "/cal0/accel_offset.txt", header=None
    ).to_numpy()
    gyr_off = pd.read_csv(
        imu_cal_path + imuname + "/cal0/gyro_offset.txt", header=None
    ).to_numpy()
    return r, acc_off, gyr_off


def apply_imu_calibration(data_in, imucal):
    """
    Apply calibration parameters to IMU data.

    Parameters
    ----------
    data_in : pandas.DataFrame
        Input IMU data.
    imucal : dict
        Calibration parameters for each IMU sensor.

    Returns
    -------
    pandas.DataFrame
        Calibrated IMU data.
    """
    data = data_in.copy(deep=True)

    for c in imucal.keys():
        if c == "info":
            continue

        data.loc[:, f"{c}_ax" : f"{c}_az"] += imucal[c]["a_off"]
        data.loc[:, f"{c}_gx" : f"{c}_gz"] -= imucal[c]["g_off"]
        data.loc[:, f"{c}_ax" : f"{c}_az"] = (
            data.loc[:, f"{c}_ax" : f"{c}_az"].to_numpy() @ imucal[c]["r"]
        )
        data.loc[:, f"{c}_gx" : f"{c}_gz"] = (
            data.loc[:, f"{c}_gx" : f"{c}_gz"].to_numpy() @ imucal[c]["r"]
        )

    return data


# %% Compile imu data
# Iterate over folders representing participants
for pid, p in enumerate(folders):
    # Define the base path to the participant's data
    path = f"{base_path}{p}/walkmaps/"
    cal_path = f"{base_path}{p}/calibration/"
    imu_cal_path = "../data/calibrations/mpu/"

    # If the data path does not exist, skip this participant
    if not os.path.exists(path):
        continue

    # Get a list of all data folders in the participant's directory
    fld_available = os.listdir(path)

    # Dictionary to store calibration matrices for sensor to body calibration
    s2bcal = {
        "info": "s2b r matrix. It can be right side multiplied. It is transposed already.",
        "a68": None,
        "a69": None,
        "b68": None,
        "b69": None,
    }

    # Load sensor to body calibration matrices from calibration files
    s2bcal["a68"] = pd.read_csv(cal_path + "/a68.txt", header=None).to_numpy().T
    s2bcal["a69"] = pd.read_csv(cal_path + "/a69.txt", header=None).to_numpy().T
    s2bcal["b68"] = pd.read_csv(cal_path + "/b68.txt", header=None).to_numpy().T
    s2bcal["b69"] = pd.read_csv(cal_path + "/b69.txt", header=None).to_numpy().T

    # List to accumulate raw IMU data across folders
    imu_data_raw = []
    temp = []  # Temporary storage for data before concatenation
    load_header_flag = True  # Flag to indicate if the header should be loaded

    # Iterate over available folders to process IMU data
    for f, fld in enumerate(fld_available):
        # Skip if it's not a directory
        if not os.path.isdir(path + fld):
            continue

        # Load the headers from the first folder (only once)
        if load_header_flag:
            headernames = (
                pd.read_csv(path + fld + "/headers.txt", header=None).iloc[0].to_list()
            )
            load_header_flag = False  # Headers are loaded, no need to load again

        # Get the list of available files in the folder and sort them
        files_available = os.listdir(path + fld)
        sorted_file_list = io.sort_file_names(files_available)

        # Iterate over sorted files and append data to temporary storage
        for i, file in enumerate(sorted_file_list):
            # Skip header files and other non-data files
            if ("header" in file) or ("walkmaps" in file):
                continue

            # Read the data file and append to temporary storage
            dfile = pd.read_csv(path + fld + "/" + file, header=None)
            temp.append(dfile)

    # Concatenate all data and reset index
    imu_data_raw = pd.concat(temp, axis=0).reset_index(drop=True)

    # Filter out 'nan' columns and set the column names
    imu_data_raw.columns = list(filter(lambda x: x != "nan", np.array(headernames)))

    # Dictionary to store IMU calibration data
    imucal = {}
    imu_name_map = {
        "a68": "9250A",
        "a69": "6050A",
        "b68": "9250B",
        "b69": "6050B",
    }

    # If the participant is not in the list with calibrated data
    if p not in ["p01", "p04"]:  # p01 and p04 are already calibrated
        # Load calibration data for each IMU type
        for imu in imu_name_map.keys():
            imucal[imu] = {}
            (
                imucal[imu]["r"],  # Rotation matrix
                imucal[imu]["a_off"],  # Acceleration offset
                imucal[imu]["g_off"],  # Gyroscope offset
            ) = get_imu_calibration(imu_name_map[imu])

        # Apply the IMU calibration to the raw data
        imu_data_corrected = apply_imu_calibration(imu_data_raw, imucal)

        # Apply the sensor-to-body calibration
        imu_data = apply_s2bcal(imu_data_corrected, s2bcal)
    else:
        # If already calibrated, use the raw data directly
        imu_data = imu_data_raw.copy(deep=True)

    # Save the raw IMU data to a CSV file
    imu_data_raw.to_csv(f"{base_path}{p}/walkmaps/walkmap_data_raw.csv")
    imu_data.to_csv(f"{base_path}{p}/walkmaps/walkmap_data.csv")

    # Save calibration data as a pickle file
    # Convert IMU calibration data to a DataFrame for easy saving
    imu_cals = pd.DataFrame.from_dict(imucal)
    with open(f"{base_path}{p}/walkmaps/imu_cals.pkl", "wb") as f:
        pickle.dump(imucal, f)
    with open(f"{base_path}{p}/walkmaps/s2b_cals.pkl", "wb") as f:
        pickle.dump(s2bcal, f)

    # Print a confirmation message indicating processing completion
    print(f"Processed data for {p}")


# %% Load all data and pull out pain intensity ratings

# %% Load all data and pull out pain intensity ratings

# Dictionary to store all traces for pain intensity analysis
all_traces = {}

# Iterate over participant folders to extract data
for pid, p in enumerate(folders):
    # Define the path to the walkmaps data for the current participant
    path = f"{base_path}{p}/walkmaps/"

    # If the path doesn't exist, skip to the next participant
    if not os.path.exists(path):
        continue

    # Attempt to read the walkmap data; if it fails, skip this participant
    try:
        imu_data_raw = pd.read_csv(path + "walkmap_data.csv")
    except:
        continue

    # Fill missing values with zeros (could also use interpolation)
    imu_data = imu_data_raw.fillna(value=0)

    # Calculate the difference between consecutive time values to identify irregularities
    time_diff = np.diff(imu_data["time"])
    time_reversals = np.where((time_diff > 0.01) | (time_diff < -0.015))

    # Filter the IMU gyroscope data with a low-pass filter (6 Hz cutoff, 2nd order)
    gyr_filt = processing.filter_signal(
        imu_data.loc[:, ["b69_gx", "b69_gy", "b69_gz"]], 6, 200, "lowpass", 2
    )

    # Filter the IMU accelerometer data with a low-pass filter (10 Hz cutoff, 2nd order)
    acc_filt = processing.filter_signal(
        imu_data[["b69_ax", "b69_ay", "b69_az"]], 10, 200, "lowpass", 2
    )

    # Detect negative peaks in gyroscope data, indicating possible heel strike events
    negpeak_idx, negpks_thresh, frames_between_pks = gaitevents.index_gyro_negpeaks(
        gyr_filt, mlaxis=1
    )

    # Detect midswing peaks in gyroscope data
    midswing, midswing_pk_props = gaitevents.midswing_peak(
        gyroml=gyr_filt[:, 1],
        negpeak_idx=negpeak_idx,
        min_peak_dist=frames_between_pks,
    )

    # Identify gait events such as heel strike, toe-off, etc.
    events = gaitevents.mariani(acc_filt, gyr_filt, midswing, negpeak_idx)

    # Remove any duplicate events that may have been detected
    events.hs = np.unique(events.hs)
    events.to = np.unique(events.to)
    events.midswing = np.unique(events.midswing)
    events.midstance = np.unique(events.midstance)

    # Apply a low-pass filter to the potentiometer data to remove noise
    paintrace_filt = processing.filter_signal(
        imu_data["potent_out"], 5, 200, "lowpass", 2
    )

    # Initialize a list to store the normalized pain traces and peak ranges
    pain_traces = []
    peak_range = []
    trace = {}

    # Keep a maximum of 75 strides for analysis
    n_strides_2keep = 75

    # Iterate over heel strike events to extract pain intensity traces
    for hs in range(0, events.hs.shape[0] - 1):
        # Ignore if the difference between consecutive heel strikes is greater than -600
        if events.hs[hs] - events.hs[hs + 1] < -600:
            continue

        # Extract and time-normalize the pain trace between consecutive heel strikes
        y = paintrace_filt[events.hs[hs] : events.hs[hs + 1]]
        t = np.linspace(0, y.shape[0] * (1 / 120), y.shape[0])
        t_norm = np.linspace(t[0], t[-1], 101)

        # Interpolate to normalize the signal over a standardized time range
        signal_normalized = np.interp(t_norm, t, y)
        pain_traces.append(signal_normalized)

        # Calculate the peak range of the signal
        peak_range.append(signal_normalized.max() - signal_normalized.min())

    # Compile data for the current participant into a dictionary
    trace["peak_range"] = np.array(peak_range[:n_strides_2keep]).mean()
    trace["nd"] = pain_traces
    trace["mean"] = np.array(pain_traces[:n_strides_2keep]).mean(axis=0)
    trace["sd"] = np.array(pain_traces[:n_strides_2keep]).std(axis=0)
    trace["nstride"] = [len(pain_traces[:n_strides_2keep])]
    trace["max"] = np.array(pain_traces[:n_strides_2keep]).max(axis=0)
    trace["min"] = np.array(pain_traces[:n_strides_2keep]).min(axis=0)
    trace["full"] = paintrace_filt
    trace["events"] = events
    trace["time"] = imu_data["time"]

    # Indicate that the data processing is complete for the current participant
    print(f"Processed data for {p}")

    # Store the trace data in the all_traces dictionary
    all_traces[p] = trace

    # Delete intermediate variables to free up memory
    del (
        imu_data_raw,
        imu_data,
        gyr_filt,
        acc_filt,
        negpeak_idx,
        negpks_thresh,
        frames_between_pks,
        midswing,
        midswing_pk_props,
        events,
    )

# Save all traces to a pickle file for future analysis
with open(f"{proc_data_path}pain_traces_50spm.pkl", "wb") as f:
    pickle.dump(all_traces, f)


# %% Create plots

# Load data for the plots
with open(f"{proc_data_path}pain_traces_50spm.pkl", "rb") as f:
    all_traces = pickle.load(f)

# // 3 Panel Plot
p_colours = px.colors.qualitative.Vivid + px.colors.qualitative.Plotly

paintrace_fig = make_subplots(
    rows=1,
    cols=3,
    shared_yaxes=True,
    subplot_titles=(
        "Representative data",
        "Segmented and Normalized",
        "All Participant Ensembles",
    ),
    horizontal_spacing=0.03,
    vertical_spacing=0.1,
)

paintrace_fig.update_yaxes(range=[0, 5.5])

# // Left Panel -  representative data
trace1_partic = "p07"
trace1_colour_id = p_colours[5]
trace1 = all_traces[trace1_partic]["full"][
    all_traces[trace1_partic]["events"].hs[0] : all_traces[trace1_partic]["events"].hs[
        -1
    ]
]
paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            trace1.shape[0] * (1 / 200),
            trace1.shape[0],
        ),
        y=trace1,
        name=trace1_partic,
        mode="lines",
        line=dict(dash="solid", color=trace1_colour_id, width=2),
        showlegend=False,
    ),
    row=1,
    col=1,
)

trace2_partic = "p11"
trace2_colour_id = p_colours[9]
trace2 = all_traces[trace2_partic]["full"][
    all_traces[trace2_partic]["events"].hs[0] : all_traces[trace2_partic]["events"].hs[
        -1
    ]
]
paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            trace2.shape[0] * (1 / 200),
            trace2.shape[0],
        ),
        y=trace2,
        name=trace2_partic,
        mode="lines",
        line=dict(dash="solid", color=trace2_colour_id, width=2),
        showlegend=False,
    ),
    row=1,
    col=1,
)
paintrace_fig.update_xaxes(title_text="Time (s)", row=1, col=1)
paintrace_fig.update_yaxes(title_text="Pain Intensity (0-10)", row=1, col=1)


# // Middle Panel - Gait Cycle Normalized
for i in all_traces[trace1_partic]["nd"]:
    paintrace_fig.add_trace(
        go.Scatter(
            x=np.linspace(
                0,
                100,
                101,
            ),
            y=i,
            name=trace1_partic,
            mode="lines",
            line=dict(dash="solid", color=trace1_colour_id, width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
for i in all_traces[trace2_partic]["nd"]:
    paintrace_fig.add_trace(
        go.Scatter(
            x=np.linspace(
                0,
                100,
                101,
            ),
            y=i,
            name=trace2_partic,
            mode="lines",
            line=dict(dash="solid", color=trace2_colour_id, width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
paintrace_fig.update_xaxes(title_text="Gait cycle (%)", row=1, col=2)


# // Right Panel - All Participant Ensembles
for i, p in enumerate(all_traces.keys()):
    paintrace_fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 100, 101),
            y=all_traces[p]["mean"],
            name=p,
            mode="lines",
            line=dict(dash="solid", color=p_colours[i], width=2),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
paintrace_fig.update_xaxes(title_text="Gait cycle (%)", row=1, col=3)


paintrace_fig.update_layout(
    width=1200,
    height=600,
    template="simple_white",
    title="Pain Intensity by Gait Cycle",
    # legend=dict(orientation="v"),
    font_family="Arial, sans-serif",
    font_size=20,
)
paintrace_fig.update_annotations(font=dict(size=20, family="Arial, sans-serif"))

paintrace_fig.show()
paintrace_fig.write_html(f"{proc_data_path}pain_hallway_3panel.html")
paintrace_fig.write_image(f"{proc_data_path}pain_hallway_3panel.svg")
paintrace_fig.write_image(f"{proc_data_path}pain_hallway_3panel.png")
