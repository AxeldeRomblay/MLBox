MLBox, Machine Learning Box
===========================

__MLBox is a powerful Automated Machine Learning python library.__

_It is compatible with:_ **Python 2.7, 3.3 - 3.6**. & **64-bit version only** (32-bit python is not supported) <br/>
_Operating system:_ **Linux**. (MacOS & Windows very soon...)


## Preparation 

First, make sure you have [setuptools](https://pypi.python.org/pypi/setuptools) installed. <br/>
Since MLBox package contains C++ source code, check that the following requirements are installed, otherwise please proceed below: 

* **[gcc](https://gcc.gnu.org/)** 

.. code-block:: console

    $ sudo apt-get install build-essential
    
* **[cmake](https://cmake.org/)** : 

.. code-block:: console

    $ pip install cmake
    
    
## Stable version

* **The sources for MLBox can be downloaded** from the `Github repo`_.

    * You can either clone the public repository:

    .. code-block:: console

        $ git clone git://github.com/AxeldeRomblay/mlbox

    * Or download the `tarball`_:

    .. code-block:: console

        $ curl  -OL https://github.com/AxeldeRomblay/mlbox/tarball/master


* Once you have a copy of the source, **you can install it:**

    * Using setup.py: 
    
    .. code-block:: console

        $ cd python-package/
        $ python setup.py install

    * Or directly, from the wheel:

    .. code-block:: console

        $ cd python-package/dist/
        $ pip install *.whl


.. _Github repo: https://github.com/AxeldeRomblay/mlbox

.. _tarball: https://github.com/AxeldeRomblay/mlbox/tarball/master



## Dev version


A 3.1 dev-version for MLBox is also available on the branch [**"3.1-dev"**](https://github.com/AxeldeRomblay/MLBox/tree/3.1-dev) ! It provides some interesting new features. Please refer to [HISTORY](https://github.com/AxeldeRomblay/MLBox/blob/master/HISTORY.rst). __It depends on sklearn-0.19.dev0 which is not a stable version at the moment.__
