import sys

if sys.platform == "win32":
    import torch  # must load before Qt/Napari on Windows