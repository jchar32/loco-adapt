# %% Library imports
import numpy as np

# import matplotlib.pyplot as plt
import os
import cv2  # type: ignore
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

"""
3rd Party Requirements:
numpy, plotly, opencv-python, pandas
"""

# Note: set image naming convention
""" Currently this program assumes files are named as follows:

    [study name]_[condition number]_[view].[file extension]
    e.g. kneemap_1_Page 1.png or kneemap_1_frontal.png

    study name: not used can be anything
    condition number: [int] the condition number 
    view: [str] the view of the image (frontal, transverse or some other code you want)

    Currently, it is assumed there are two [views] per [condition number] (e.g., frontal and transverse view images).

    I have not implemented large checks to handle edge case naming conventions, so it is preferred you stick with this convention for now. If you have a large set of images already, you can download PowerToys (a Microsoft program) and use the PowerRename tool to rename all the files in a folder to a consistent naming convention.
"""


def render_maps(map, moments=None, cmap="hot"):
    # # sum across maps to get counts per pixel
    # sum_map = np.sum(map, axis=0)

    # # mask array to remove the background
    # masked_map = np.ma.masked_array(sum_map, mask=sum_map == 0)

    # fig, ax = plt.subplots(figsize=(8, 8))
    # # plot centroids if provided
    # if moments is not None:
    #     for moment in moments:
    #         plt.scatter(moment[0], moment[1], c="w", s=100, zorder=10)

    # # plot summed and masked map
    # ci = plt.pcolormesh(masked_map, cmap=cmap, vmin=0, alpha=0.5)
    # plt.colorbar(ci)
    # plt.show()
    # ax.set_aspect("equal", "box")
    pass


def extract_contours(image):
    # Convert to grayscale and binarize
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_bin = cv2.threshold(image_gray, 150, 1, cv2.THRESH_BINARY_INV)

    # Identify contours (external only)
    contours, _ = cv2.findContours(
        image_bin, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    return contours, image_bin


def map_metrics(contour, total_area: int):
    moments = cv2.moments(contour)
    centroidx = moments["m10"] / moments["m00"]
    centroidy = moments["m01"] / moments["m00"]
    area_px = cv2.contourArea(contour)
    area_perc = area_px / np.prod(total_area)
    return centroidx, centroidy, area_px, area_perc


def find_painmap_files(abs_path, p, image_type=".png"):
    # Find images to process
    image_file_names = [
        fname
        for fname in os.listdir(os.path.join(abs_path, p))
        if fname.endswith(image_type)
    ]
    return image_file_names


def parse_filenames(names, delimiter="_"):
    conditions, map_views, img_file_types = [], [], []
    for name in names:
        name.strip(".png")
        split_fnames = name.split(delimiter)
        conditions.append(split_fnames[1])
        map_views.append(split_fnames[2].split(".")[0])
        img_file_types.append(split_fnames[-1].split(".")[-1])

    return conditions, map_views, img_file_types


def main() -> tuple:
    datadir = relative_data_dir
    abs_path = os.path.abspath(datadir)

    all_fr_img_stack, all_tr_img_stack = [], []
    all_moment_dfs = []
    for j, p in enumerate(next(os.walk(abs_path))[1]):
        img_file_names = find_painmap_files(abs_path, p)

        if len(img_file_names) == 0:
            print(f"No image files found for participant {p}.")
            continue
        num_img_files = len(img_file_names)

        # first dimension size is left as initializing with zero wont change the summation result
        fr_img_stack = np.full(
            (int(num_img_files / 2), image_dimensions[0], image_dimensions[1]), 0
        )
        tr_img_stack = np.full(
            (int(num_img_files / 2), image_dimensions[0], image_dimensions[1]), 0
        )

        # // File naming assumption:
        # Assumes file names are in the following format:
        # "kneemap_1_Page 1.png"
        # [study name]_[condition number]_[view].[file extension]
        # Study name is not used so can be whatever you wish.
        conditions, map_views, img_file_type = parse_filenames(
            img_file_names, delimiter="_"
        )
        # //

        moments = []
        tr_idx, fr_idx = 0, 0
        for i, img_file_name in enumerate(img_file_names):
            # read in image
            image = cv2.imread(os.path.join(abs_path, p, img_file_name))

            # Convert to grayscale and binarize
            contours, image_bin = extract_contours(image)

            if map_views[0] in img_file_name:
                fr_img_stack[fr_idx, :, :] = image_bin
                fr_idx += 1
            elif map_views[1] in img_file_name:
                tr_img_stack[tr_idx, :, :] = image_bin
                tr_idx += 1
            else:
                raise NameError("Image naming convention not recognized")

            for k, contour in enumerate(contours):
                # a catch for oddly small contours which are likely just errors in the drawing
                if contour.size < 100:
                    continue

                totalarea = image_bin.shape
                d = {
                    "x": map_metrics(contour, totalarea)[0],
                    "y": map_metrics(contour, totalarea)[1],
                    "area_px": map_metrics(contour, totalarea)[2],
                    "area_perc": map_metrics(contour, totalarea)[3],
                    "contour_num": k,
                    "fname": img_file_name,
                    "cond": conditions[i],
                    "view": map_views[i],
                }
                moments.append(d)

        moments_df = pd.DataFrame(moments)

        moments_df.to_csv(os.path.join(abs_path, p, "moments.csv"))

        with open(os.path.join(abs_path, p, "maps_frontal.pkl"), "wb") as f:
            pickle.dump(fr_img_stack, f)

        with open(os.path.join(abs_path, p, "maps_transverse.pkl"), "wb") as f:
            pickle.dump(tr_img_stack, f)

        all_fr_img_stack.append([fr_img_stack])
        all_tr_img_stack.append([tr_img_stack])

        moments_df["participant"] = p
        all_moment_dfs.append(moments_df)
        print(f"Processed participant {p}: img size {image_bin.shape}")
    # stack the kneemap moment dataframes by participant on first axis
    all_moment_dfs = pd.concat(all_moment_dfs)
    all_moment_dfs.to_csv(os.path.join(abs_path, "all_maps_moments.csv"))

    # Stack knee map matrices by participant on first axis
    all_fr_img_stack = np.stack(all_fr_img_stack).squeeze()
    all_tr_img_stack = np.stack(all_tr_img_stack).squeeze()

    return all_fr_img_stack, all_tr_img_stack, all_moment_dfs


if __name__ == "__main__":
    # SETUP
    # 1. set relative data directory from this script to your data folders
    relative_data_dir = "../data/locohab/"
    proc_data_path = "../data/processed/painmap"
    # 2. set expected image dimensions
    image_dimensions = (1875, 1875)
    all_fr_img_stack, all_tr_img_stack, all_moment_dfs = main()

    # Save the stacked images for all participants
    with open(f"{proc_data_path}all_maps_frontal.pkl", "wb") as f:
        pickle.dump(all_fr_img_stack, f)
    with open(f"{proc_data_path}all_maps_transverse.pkl", "wb") as f:
        pickle.dump(all_tr_img_stack, f)

    # render maps to figure
    num_conditions = all_fr_img_stack.shape[
        1
    ]  # assumed dimension containing conditions
    # plt.style.use("default")
    # for c in range(num_conditions):
    # render_maps(all_fr_img_stack[:, c, :, :], cmap="hot")
    # render_maps(all_tr_img_stack[:, c, :, :], cmap="hot")

# %%
# Knee map plots for manuscript
frontal_map = make_subplots(
    rows=1,
    cols=3,
    shared_yaxes=True,
    subplot_titles=("Pain 1/10", "Pain 3/10", "Pain 5/10"),
    x_title="",
    y_title="",
    horizontal_spacing=0.02,
    vertical_spacing=0.1,
)
# sum across maps to get counts per pixel
for c in range(num_conditions):
    # render_maps(all_fr_img_stack[:, c, :, :], cmap="hot")
    # render_maps(all_tr_img_stack[:, c, :, :], cmap="hot")
    sum_map = np.sum(all_fr_img_stack[:, c, :, :], axis=0)

    # mask array to remove the background
    masked_map = np.ma.masked_array(sum_map, mask=sum_map == 0)
    frontal_map.add_trace(
        go.Heatmap(z=masked_map, colorscale="hot", showscale=False),
        row=1,
        col=c + 1,
    )

frontal_map.update_layout(
    autosize=True,
    width=800,
    height=400,
    coloraxis=dict(colorscale="hot_r"),
    showlegend=False,
    title_text="Frontal View Pain Maps",
)
frontal_map.show()


# %% Frontal view pain maps
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# frontal_map = make_subplots(
#         rows=1,
#         cols=3,
#         shared_yaxes=True,
#         # subplot_titles=(
#         #     "Trial 1: tau=" + str(np.round(1 / overall_fits[1][1], 2)),
#         #     "Trial 2: tau=" + str(np.round(1 / overall_fits[2][1], 2)),
#         #     "Trial 3: tau=" + str(np.round(1 / overall_fits[3][1], 2)),
#         )
for c in range(num_conditions):
    sum_map = np.sum(all_fr_img_stack[:, c, :, :], axis=0)
    # mask array to remove the background
    # masked_map = np.ma.masked_array(sum_map, mask=sum_map == 0)
    sum_map = sum_map.astype(float)
    sum_map[sum_map == 0] = np.nan
    masked_map = sum_map.copy()
    frontal_map = go.Figure()
    frontal_map.add_trace(
        go.Heatmap(
            z=masked_map,
            colorscale="hot",
            reversescale=True,
            showlegend=False,
            showscale=True,
            opacity=0.9,
        ),
        # row=1, col=c+1
    )
    frontal_map.update_xaxes(
        showline=False, showgrid=False, zeroline=False, showticklabels=False
    )
    frontal_map.update_yaxes(
        showline=False, showgrid=False, zeroline=False, showticklabels=False
    )
    frontal_map.update_layout(
        autosize=True,
        width=600,
        height=600,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    frontal_map.show()
    frontal_map.write_html(f"{relative_data_dir}kneemap_fr_plot_pain{c+1}.html")
    frontal_map.write_image(f"{relative_data_dir}kneemap_fr_plot_pain{c+1}.svg")
    del frontal_map

# %%
for c in range(num_conditions):
    sum_map = np.sum(all_tr_img_stack[:, c, :, :], axis=0)
    # mask array to remove the background
    # masked_map = np.ma.masked_array(sum_map, mask=sum_map == 0)
    sum_map = sum_map.astype(float)
    sum_map[sum_map == 0] = np.nan
    masked_map = sum_map.copy()
    trans_map = go.Figure()
    trans_map.add_trace(
        go.Heatmap(
            z=masked_map,
            colorscale="hot",
            reversescale=True,
            showlegend=False,
            showscale=True,
            opacity=0.9,
        )
    )
    trans_map.update_xaxes(
        showline=False, showgrid=False, zeroline=False, showticklabels=False
    )
    trans_map.update_yaxes(
        showline=False, showgrid=False, zeroline=False, showticklabels=False
    )
    trans_map.update_layout(
        autosize=True,
        width=600,
        height=600,
        showlegend=False,
    )
    trans_map.write_html(f"{relative_data_dir}kneemap_tr_plot_pain{c+1}.html")
    trans_map.write_image(f"{relative_data_dir}kneemap_tr_plot_pain{c+1}.svg")
    del trans_map

# %%
c = 1
sum_map = np.sum(all_fr_img_stack[:, c, :, :], axis=0)
# mask array to remove the background
# masked_map = np.ma.masked_array(sum_map, mask=sum_map == 0)
sum_map = sum_map.astype(float)
sum_map[sum_map == 0] = np.nan
masked_map = sum_map.copy()
trans_map = go.Figure()
trans_map.add_trace(
    go.Heatmap(
        z=masked_map,
        colorscale="hot",
        reversescale=False,
        showlegend=False,
        showscale=True,
        opacity=0.5,
    )
)
trans_map.update_xaxes(
    showline=False, showgrid=False, zeroline=False, showticklabels=False
)

trans_map.update_layout(
    autosize=True,
    width=600,
    height=600,
    showlegend=False,
)
trans_map.show()
# trans_map.write_image(f"../data/painmap/kneemap_legend_scalebar_plot_pain{c+1}.svg")
