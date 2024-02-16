# Library imports
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  # type: ignore
import pandas as pd
import pickle

# Built a virtual environtment:
# use conda to set up with numpy, matplotlib, ipython, ipympl
# use pip to install opencv-python

# SETUP
# 1. set relative data directory
relative_data_dir = "../data/painmap/"
# 2. set participant naming code (folders holding data)
participant_code = "p"
# 3. set expected image dimensions
image_dimensions = (1875, 1875)

# Note: set image naming convention
""" Currently this program assums files are named as follows:

    [study name]_[condition number]_[view].[file extension]
    e.g. kneemap_1_Page 1.png or kneemap_1_frontal.png

    study name: not used can be anything
    condition number: [int] the condition number 
    view: [str] the view of the image (frontal, transverse or some other code you want)

    I have not implemented large checks to handle edge case naming conventions, so it is preferred you stick with this convention for now. If you have a large set of images already, you can download PowerToys (a Microsoft program) and use the PowerRename tool to rename all the files in a folder to a consistent naming convention.
    """

# // currently assumes that there are 2 images per X number of conditions (e.g., frontal and transverse view images)


def render_maps(map, moments=None, cmap="hot"):
    # sum across maps to get counts per pixel
    sum_map = np.sum(map, axis=0)

    # mask array to remove the background
    masked_map = np.ma.masked_array(sum_map, mask=sum_map == 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    # plot centroids if provided
    if moments is not None:
        for moment in moments:
            plt.scatter(moment[0], moment[1], c="w", s=100, zorder=10)

    # plot summed and masked map
    ci = plt.pcolormesh(masked_map, cmap=cmap, vmin=0, alpha=0.5)
    plt.colorbar(ci)
    plt.show()
    ax.set_aspect("equal", "box")


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
        f for f in os.listdir(os.path.join(abs_path, p)) if f.endswith(image_type)
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


def batch_process():
    datadir = relative_data_dir
    abs_path = os.path.abspath(datadir)

    for j, p in enumerate(os.listdir(abs_path)):
        img_file_names = find_painmap_files(abs_path, p)

        if len(img_file_names) == 0:
            print(f"No image files found for participant {p}.")
            continue
        num_img_files = len(img_file_names)

        # first dimension size is left as initializing with zero wont change the summation result
        fr_img_stack = np.full(
            (int(num_img_files), image_dimensions[0], image_dimensions[1]), 0
        )
        tr_img_stack = np.full(
            (int(num_img_files), image_dimensions[0], image_dimensions[1]), 0
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
        for i, img_file_name in enumerate(img_file_names):
            # read in image
            image = cv2.imread(os.path.join(abs_path, p, img_file_name))

            # Convert to grayscale and binarize
            contours, image_bin = extract_contours(image)

            if map_views[0] in img_file_name:
                fr_img_stack[i, :, :] = image_bin
            elif map_views[1] in img_file_name:
                tr_img_stack[i, :, :] = image_bin
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

        # render_maps(fr_img_stack, moments=None, cmap="hot_r")
        # render_maps(tr_img_stack, moments=None, cmap="hot_r")

        pd.DataFrame(moments_df).to_csv(os.path.join(abs_path, p, "moments.csv"))

        with open(os.path.join(abs_path, p, "maps_frontal.pkl"), "wb") as f:
            pickle.dump(fr_img_stack, f)

        with open(os.path.join(abs_path, p, "maps_transverse.pkl"), "wb") as f:
            pickle.dump(tr_img_stack, f)


if __name__ == "__main__":
    batch_process()
