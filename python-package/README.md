MLBox, Machine Learning Box
===========================

__MLBox is a powerful Automated Machine Learning python library.__

It is compatible with: __Python 2.7-3.5.__


Stable version
==============


Extended version
----------------

MLBox relies on several models. Some can occasionally be difficult to install, so they are not included in MLBox's default installation. You are responsible for installing them yourself. 

__Keep in mind that mlbox will run fine without them installed !__
__Here is the procedure to get the extention only (MLBox will not be installed yet !):__


.. code-block:: console

    $ git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
    $ mkdir build ; cd build
    $ cmake .. 
    $ make -j 
    $ cd python-package; python setup.py install


If you get some trouble, please refer to : (https://github.com/Microsoft/LightGBM)
 
__Now you need to follow the procedure bellow to install the stable version of MLBox:__


From sources
------------

The sources for MLBox can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/AxeldeRomblay/mlbox

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/AxeldeRomblay/mlbox/tarball/master


Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ cd python-package/
    $ python setup.py install

Or directly, with the wheel:

.. code-block:: console

    $ cd python-package/dist/
    $ pip install *.whl


.. _Github repo: https://github.com/AxeldeRomblay/mlbox

.. _tarball: https://github.com/AxeldeRomblay/mlbox/tarball/master



Dev version
===========


A 2.3 dev-version for MLBox is also available on the __branch "dev"__ ! It provides some interesting new features. Please refer to [HISTORY](https://github.com/AxeldeRomblay/MLBox/blob/master/HISTORY.rst). __It depends on sklearn-0.19.dev0 which is not a stable version at the moment__: https://github.com/scikit-learn/scikit-learn

If you want, you can have a try: 

* Clone or download the dev branch
* Install sklearn-0.19.dev0: 

.. code-block:: console

    $ cd python-package/extra-dist/
    $ pip install *.whl

* Install MLBox-2.3-dev: 

.. code-block:: console

    $ cd python-package/
    $ python setup.py install 

or:

.. code-block:: console

    $ cd python-package/dist/
    $ pip install *.whl



