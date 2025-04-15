
[![License](https://img.shields.io/pypi/l/napari-convpaint.svg?color=green)](https://github.com/guiwitz/napari-convpaint/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-convpaint.svg?color=green)](https://pypi.org/project/napari-convpaint)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-convpaint.svg?color=green)](https://python.org)
[![tests](https://github.com/guiwitz/napari-convpaint/workflows/tests/badge.svg)](https://github.com/guiwitz/napari-convpaint/actions)
[![codecov](https://codecov.io/gh/guiwitz/napari-convpaint/branch/main/graph/badge.svg)](https://codecov.io/gh/guiwitz/napari-convpaint)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-convpaint)](https://napari-hub.org/plugins/napari-convpaint)



![overview conv-paint](/images/overview_github.png)
This napari plugin can be used to segment objects or structures in images based on a few brush strokes providing examples of the classes. Based on the same idea as other tools like ilastik, its main strength is that it can use features from pretrained neural networks like VGG16 or DINOV2, enabling the segmentation of more complex images.

**Find more information and tutorials in the [docs](https://guiwitz.github.io/napari-convpaint/) or read the [preprint](https://doi.org/10.1101/2024.09.12.610926).**


![overview conv-paint](/images/network_github.png)

## Installation

You can install `napari-convpaint` via [pip]

    pip install napari-convpaint

To install latest development version :

    pip install git+https://github.com/guiwitz/napari-convpaint.git


## Example use case: Tracking shark body parts in a movie
These are the scribble annotations provided for training:
![](./images/shark_annot.png)

And this is the resulting Convpaint segmentation:
<video src="https://github.com/user-attachments/assets/6a2be1fe-25cc-4af1-9f50-aab9bc4123d9"></video>

Check out the documentation or the paper for more usecases!

## API

You can now use the API in a fashion very similar to the napari plugin. The ConvpaintModel class combines a feature extractor and a classifier model, and holds all the parameters defining the model. Initialize a ConvpaintModel object, train its classifier and use it to segment an image:

```Python
cp_model = ConvpaintModel("dino") # alternatively use vgg, cellpose or gaussian
cp_model.train(image, annotations)
segmentation = cp_model.segment(image)
```

There are many other options, such as predicting all class probabilities (see below) and we will update the documentation and notebook examples soon. In the meantime feel free to test it yourself.

```Python
probas = cp_model.predict_probas(image)
```

## License

Distributed under the terms of the [BSD-3] license,
"napari-convpaint" is free and open source software

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

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

## Authors

The idea behind this napari plugin was first developed by [Lucien Hinderling](https://hinderling.github.io) in the group of [Olivier Pertz](https://www.pertzlab.net/), at the Institute of Cell Biology, University of Bern. Pertz lab obtained a CZI napari plugin development grant with the title ["Democratizing Image Analysis with an Easy-to-Train Classifier"](https://chanzuckerberg.com/science/programs-resources/imaging/napari/democratizing-image-analysis-with-an-easy-to-train-classifier/) which supported the adaptation of the initial concept as a napari plugin called napari-convpaint. The plugin has been developed by [Guillaume Witz<sup>1</sup>](https://guiwitz.github.io/blog/about/), [Roman Schwob<sup>1,2</sup>](https://github.com/quasar1357) and Lucien Hinderling<sup>2</sup> with much appreciated assistance of [Benjamin Grädel<sup>2</sup>](https://x.com/benigraedel), [Maciej Dobrzyński<sup>2</sup>](https://macdobry.net), Mykhailo Vladymyrov<sup>1</sup> and Ana Stojiljković<sup>1</sup>.

<sup>1</sup>[Data Science Lab](https://www.dsl.unibe.ch/), University of Bern \
<sup>2</sup>[Pertz Lab](https://www.pertzlab.net/), Institute of Cell Biology, University of Bern 

## Cite Convpaint

If you find Convpaint useful in your research, please consider citing our work. Please also cite any Feature Extractor you have used in Convpaint, such as [ilastik](https://github.com/ilastik/ilastik-napari), [cellpose](https://cellpose.readthedocs.io/en/latest/) or [DINOv2](https://github.com/facebookresearch/dinov2).

Convpaint:
```
@article {Hinderling2024,
	author = {Hinderling, Lucien and Witz, Guillaume and Schwob, Roman and Stojiljković, Ana and Dobrzyński, Maciej and Vladymyrov, Mykhailo and Frei, Joël and Grädel, Benjamin and Frismantiene, Agne and Pertz, Olivier},
	title = {Convpaint - Interactive pixel classification using pretrained neural networks},
	elocation-id = {2024.09.12.610926},
	doi = {10.1101/2024.09.12.610926},
	journal = {bioRxiv},
	publisher = {Cold Spring Harbor Laboratory},
	year = {2024},
}
```
Suggested citations for feature extractors:
```
@article {Berg2019,
	author = {Berg, Stuart and Kutra, Dominik and Kroeger, Thorben and Straehle, Christoph N. and Kausler, Bernhard X. and Haubold, Carsten and Schiegg, Martin and Ales, Janez and Beier, Thorsten and Rudy, Markus and Eren, Kemal and Cervantes, Jaime I. and Xu, Buote and Beuttenmueller, Fynn and Wolny, Adrian and Zhang, Chong and Koethe, Ullrich and Hamprecht, Fred A. and Kreshuk, Anna},
	title = {ilastik: interactive machine learning for (bio)image analysis.},
	issn = {1548-7105},
	url = {https://doi.org/10.1038/s41592-019-0582-9},
	doi = {10.1038/s41592-019-0582-9},
	journal = {Nature Methods},
	publisher = {Springer Nature},
	year = {2019},
	journal = {Nature Methods},
}
```
```
@article {Stringer2021,
	author = {Stringer, Carsen and Wang, Tim and Michaelos, Michalis and Pachitariu Marius},
	title = {Cellpose: a generalist algorithm for cellular segmentation.},
	elocation-id = {s41592-020-01018-x},
	doi = {10.1038/s41592-020-01018-x},
	journal = {Nature Methods},
	publisher = {Springer Nature},
	year = {2021},
}
```
```
@article {oquab2024dinov2learningrobustvisual,
      title={DINOv2: Learning Robust Visual Features without Supervision}, 
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2024},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.07193}
}

```
