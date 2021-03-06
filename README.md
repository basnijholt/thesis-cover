# thesis-cover
*Parametrically designing the cover of my PhD thesis*
> Title: Towards realistic numerical simulations of Majorana devices 🎓 ([pdf](http://files.nijho.lt/thesis.pdf) | [source code](https://github.com/basnijholt/thesis))

Read [this blog post](https://quantumtinkerer.tudelft.nl/blog/thesis-cover/) for more context!

[Click here to open the color-picker app ![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/basnijholt/thesis-cover/master?filepath=color-picker.ipynb).

* [`thesis_cover.py`](thesis_cover.py) has all the relevant functions to convert a [`adaptive.Learner2D`](https://adaptive.readthedocs.io/en/latest/docs.html#examples) into a thesis cover in PDF format.
* [`generate-covers.ipynb`](generate-covers.ipynb) calls the functions in [`thesis_cover.py`](thesis_cover.py) on all data files
* [`color-picker.ipynb`](color-picker.ipynb) has an interactive widget to modify colormaps and display them on the different covers
* [`thesis-cover-selection.ipynb`](thesis-cover-selection.ipynb) was used to select between the 3671 options (this doesn't work unless you download all data from https://github.com/basnijholt/spin-orbit-nanowires-data)

![](images/close-up.jpg)
![](images/theses.jpg)
