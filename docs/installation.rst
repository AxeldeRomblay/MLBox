Installation guide
==================

|Documentation Status| |PyPI version| |Build Status| |GitHub Issues| |codecov| |License| |Downloads| |Python Versions|


Compatibilities
---------------

* *It is compatible with:* **Python 2.7, 3.5 & 3.6**. & **64-bit version only** (32-bit python is not supported)
* *Operating system:* **Linux, MacOS & Windows**.


Preparation
-----------

First, make sure you have `setuptools <https://pypi.python.org/pypi/setuptools>`__ installed.

Since MLBox package contains C++ source code (LightGBM model), check that the following requirements are installed:

* For **Windows** users, `VC runtime <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ is needed if **Visual Studio** (2015 or newer) is not installed.

* For **Linux** users, **glibc** >= 2.14 is required.

* For **macOS** users, **OpenMP** is required for running LightGBM on the system with the Apple Clang compiler. You can install it by the following command: ``brew install libomp``.

If you get any errors during preparation, please refer to `LightGBM's installation guide <https://github.com/Microsoft/LightGBM/tree/master/python-package#lightgbm-python-package>`__


Installation
------------

Install from pip
~~~~~~~~~~~~~~~~

MLBox is available on **PyPI**, so you only need to run the following command:

.. code-block:: console

    $ pip install mlbox


Install from the Github
~~~~~~~~~~~~~~~~~~~~~~~

* **The sources for MLBox can be downloaded** from the `Github repo`_.

    * You can either clone the public repository:

    .. code-block:: console

        $ git clone git://github.com/AxeldeRomblay/mlbox

    * Or download the `tarball`_:

    .. code-block:: console

        $ curl  -OL https://github.com/AxeldeRomblay/mlbox/tarball/master


* Once you have a copy of the source, **you can install it** using setup.py :

    .. code-block:: console

        $ python setup.py install


.. _Github repo: https://github.com/AxeldeRomblay/mlbox

.. _tarball: https://github.com/AxeldeRomblay/mlbox/tarball/master

.. |Documentation Status| image:: https://readthedocs.org/projects/mlbox/badge/?version=latest
   :target: http://mlbox.readthedocs.io/en/latest/?badge=latest
.. |PyPI version| image:: https://badge.fury.io/py/mlbox.svg
   :target: https://pypi.python.org/pypi/mlbox
.. |Build Status| image:: https://travis-ci.org/AxeldeRomblay/MLBox.svg?branch=master
   :target: https://travis-ci.org/AxeldeRomblay/MLBox
.. |GitHub Issues| image:: https://img.shields.io/github/issues/AxeldeRomblay/MLBox.svg
   :target: https://github.com/AxeldeRomblay/MLBox/issues
.. |codecov| image:: https://codecov.io/gh/AxeldeRomblay/MLBox/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/AxeldeRomblay/MLBox
.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/AxeldeRomblay/MLBox/blob/master/LICENSE
.. |Downloads| image:: https://pepy.tech/badge/mlbox
   :target: https://pepy.tech/project/mlbox
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/mlbox.svg
   :target: https://pypi.org/project/mlbox
