import os
import wx


def get_path(base_dir=None, wildcard: str = ""):
    """Simply gui to get file path(s). If no file is selected, returns

    Parameters
    ----------
    wildcard : str, optional
        Value of file extension to filter browser for, by default ""

    Returns
    -------
    direc : str | List[str]
        directory of files selected
    file : str | List[str]
        file name(s)
        _description_
    """

    # option to specify a base directory to start from.
    # Helps when a project uses data from one particular directory
    if base_dir is None:
        base_dir = os.getcwd()
    else:
        base_dir = os.path.abspath(base_dir)

    app = wx.App(redirect=False)

    style = (
        wx.FD_OPEN
        | wx.FD_FILE_MUST_EXIST
        | wx.STAY_ON_TOP
        | wx.FD_MULTIPLE
        | wx.FD_NO_FOLLOW
    )
    dialog = wx.FileDialog(
        None,
        "Select file(s) you want to load",
        defaultDir=base_dir,
        wildcard=wildcard,
        style=style,
    )

    direc = ""
    file = ""

    with wx.FileDialog(
        None,
        "Select file(s) you want to load",
        defaultDir=base_dir,
        wildcard=wildcard,
        style=style,
    ) as dialog:
        if dialog.ShowModal() == wx.ID_CANCEL:
            direc = ""
            file = ""
            wx.LogError("No file selected")
            return direc, file

        paths = dialog.GetPaths()
        direc = os.path.dirname(paths[0])
        if len(paths) == 1:
            file = os.path.basename(paths[0])
        elif len(paths) > 1:
            file = []
            for p in paths:
                file.append(os.path.basename(p))
        else:
            wx.LogError("No file selected")

    app.Destroy()
    return direc, file


if __name__ == "__main__":
    d, p = get_path(base_dir="../../data/")
    print(f"\n Directory Selected: {d} --> \n")
    for j in p:
        print(j)
