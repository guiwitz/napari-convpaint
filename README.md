# napari-convpaint

[![License](https://img.shields.io/pypi/l/napari-convpaint.svg?color=green)](https://github.com/guiwitz/napari-convpaint/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-convpaint.svg?color=green)](https://pypi.org/project/napari-convpaint)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-convpaint.svg?color=green)](https://python.org)
[![tests](https://github.com/guiwitz/napari-convpaint/workflows/tests/badge.svg)](https://github.com/guiwitz/napari-convpaint/actions)
[![codecov](https://codecov.io/gh/guiwitz/napari-convpaint/branch/main/graph/badge.svg)](https://codecov.io/gh/guiwitz/napari-convpaint)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-convpaint)](https://napari-hub.org/plugins/napari-convpaint)

This napari plugin can be used to segment objects in images based on a few brush strokes providing examples of foreground and background. Based on the same idea as other tools like ilastik, its main strength is that it provides good results without adjusting any parameters. The filters used to generate features for classification are indeed taken from layers of trained deep learning networks and don't need to be chosen manually. Find more information in the [docs](https://guiwitz.github.io/napari-convpaint/).

**The idea behind the plugin comes directly from the work of Lucien Hinderling (University of Bern) and can be found here: https://github.com/hinderling/napari_pixel_classifier.**

## Installation

You can install `napari-convpaint` via [pip]:

    pip install napari-convpaint

To install latest development version :

    pip install git+https://github.com/guiwitz/napari-convpaint.git

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-convpaint" is free and open source software

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

# Authors

The idea behind this napari plugin was first developed by Lucien Hinderling in the group of [Olivier Pertz](https://www.pertzlab.net/), at the Institute of Cell Biology, University of Bern. The code has first been shared as open source resource in form of a [Jupyter Notebook](https://github.com/hinderling/napari_pixel_classifier). With the desire to make this resource accessible to a broader public in the scientific community, the Pertz lab obtained a CZI napari plugin development grant with the title ["Democratizing Image Analysis with an Easy-to-Train Classifier"](https://chanzuckerberg.com/science/programs-resources/imaging/napari/democratizing-image-analysis-with-an-easy-to-train-classifier/) which supported the adaptation of the initial concept as a napari plugin called napari-convpaint. The plugin has been developed by Guillaume Witz, Mykhailo Vladymyrov and Ana Stojiljkovic at the [Data Science Lab](https://www.dsl.unibe.ch/), University of Bern, in tight collaboration with the Pertz lab.

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
