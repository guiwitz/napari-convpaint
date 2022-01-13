# napari-convpaint

[![License](https://img.shields.io/pypi/l/napari-convpaint.svg?color=green)](https://github.com/guiwitz/napari-convpaint/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-convpaint.svg?color=green)](https://pypi.org/project/napari-convpaint)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-convpaint.svg?color=green)](https://python.org)
[![tests](https://github.com/guiwitz/napari-convpaint/workflows/tests/badge.svg)](https://github.com/guiwitz/napari-convpaint/actions)
[![codecov](https://codecov.io/gh/guiwitz/napari-convpaint/branch/main/graph/badge.svg)](https://codecov.io/gh/guiwitz/napari-convpaint)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-convpaint)](https://napari-hub.org/plugins/napari-convpaint)

This napari plugin can be used the segment images into multiple classes based on a pixel classifier trained with sparse manual annotations. The features used for classification are extracted using the filters from the first layer of a VGG16 Neural Network trained on natural images. To include features at multiple scales, the image as well as multiple downscaled versions of it are used to generate features (for example using the raw image as well as one downscaled 2x, one obtains 2*64=128 features per pixel). The features are then classified using scikit-learn's Random Forest classifier. The trained classifier is saved and can be reused either directly in the plugin on other images or in a separate pipeline.

**The idea behind the plugin comes directly from the work of Lucien Hinderling (University of Bern) and can be found here: https://github.com/hinderling/napari_pixel_classifier).**

## Installation

**------Not on pypi yet------**
You can install `napari-convpaint` via [pip]:

    pip install napari-convpaint
**------Not on pypi yet------**

To install latest development version :

    pip install git+https://github.com/guiwitz/napari-convpaint.git

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-convpaint" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/guiwitz/napari-convpaint/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
