
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from .conv_paint import ConvPaintWidget
from napari_plugin_engine import napari_hook_implementation

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [ConvPaintWidget]
