MLBox, Machine Learning Box
===========================

__MLBox is a powerful Automated Machine Learning python library.__

_It is compatible with:_ __Python 2.7__. (Python 3.3-3.6 very soon...)

_Operating system:_ __Linux__. (MacOS & Windows very soon...)

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

    Or directly, from the wheel:

    .. code-block:: console

        $ cd python-package/dist/
        $ pip install *.whl


.. _Github repo: https://github.com/AxeldeRomblay/mlbox

.. _tarball: https://github.com/AxeldeRomblay/mlbox/tarball/master



## Dev version


A 2.3 dev-version for MLBox is also available on the __branch "dev"__ ! It provides some interesting new features. Please refer to [HISTORY](https://github.com/AxeldeRomblay/MLBox/blob/master/HISTORY.rst). __It depends on sklearn-0.19.dev0 which is not a stable version at the moment.__

If you want, you can have a try: 

* Clone or download sklearn-0.19.dev0 from the github: https://github.com/scikit-learn/scikit-learn
* Install sklearn-0.19.dev0: 

    .. code-block:: console

        $ cd scikit-learn-master/
        $ python setup.py install 

* Clone or download MLBox-2.3.dev0 from the 'dev' branch. 
* Install MLBox-2.3.dev0: 

    .. code-block:: console

        $ cd python-package/
        $ python setup.py install 

    or:

    .. code-block:: console

        $ cd python-package/dist/
        $ pip install *.whl



