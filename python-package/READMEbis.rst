MLBox, Machine Learning Box
===========================

\|Documentation Status\| \|PyPI version\| \|Build Status\| \|GitHub
Issues\| \|codecov\| \|License\|

**MLBox is a powerful Automated Machine Learning python library.**

-  *It is compatible with:* **Python 2.7, 3.4 - 3.6**. & **64-bit
   version only** (32-bit python is not supported)
-  *Operating system:* **Linux**. (MacOS & Windows very soon...)

Preparation
-----------

First, make sure you have
`setuptools <https://pypi.python.org/pypi/setuptools>`__ installed.
Since MLBox package contains C++ source code, check that the following
requirements are installed:

-  `gcc <https://gcc.gnu.org/>`__

.. code:: console

    $ sudo apt-get install build-essential

-  `cmake <https://cmake.org/>`__

.. code:: console

    $ pip install cmake

Stable version
--------------

Install from pip
~~~~~~~~~~~~~~~~

MLBox is now available on **PyPI**, so you only need to run the
following command:

.. code:: console

    $ pip install mlbox

Install from the Github
~~~~~~~~~~~~~~~~~~~~~~~

-  **The sources for MLBox can be downloaded** from the `Github
   repo <https://github.com/AxeldeRomblay/mlbox>`__.

       -  You can either clone the public repository:

       .. code:: console

           $ git clone git://github.com/AxeldeRomblay/mlbox

       -  Or download the
          `tarball <https://github.com/AxeldeRomblay/mlbox/tarball/master>`__:

       .. code:: console

           $ curl  -OL https://github.com/AxeldeRomblay/mlbox/tarball/master

-  Once you have a copy of the source, **you can install it:**

       -  Using setup.py:

       .. code:: console

           $ cd python-package/
           $ python setup.py install

       -  Or directly, from the wheel:

       .. code:: console

           $ cd python-package/dist/
           $ pip install *.whl

Dev version
-----------

A **4.0 dev-version for MLBox is also available** on the branch
`"4.0-dev" <https://github.com/AxeldeRomblay/MLBox/tree/4.0-dev>`__ ! It
provides some interesting new features. Please refer to
`HISTORY <https://github.com/AxeldeRomblay/MLBox/blob/master/HISTORY.rst>`__.

**It depends on sklearn-0.19.dev0 which is not a stable version at the
moment.**
