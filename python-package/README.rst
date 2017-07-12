MLBox, Machine Learning Box
===========================

.. -*- mode: rst -*-

|Documentation Status|_ |PyPI version|_ |Build Status|_ |GitHub Issues|_ |codecov|_ |License|_ 

.. |Documentation Status| image:: https://readthedocs.org/projects/mlbox/badge/?version=latest
.. _Documentation Status: http://mlbox.readthedocs.io/en/latest/?badge=latest

.. |PyPI version| image:: https://badge.fury.io/py/mlbox.svg
.. _PyPI version: https://pypi.org/project/mlbox/

.. |Build Status| image:: https://travis-ci.org/AxeldeRomblay/MLBox.svg?branch=master
.. _Build Status: https://travis-ci.org/AxeldeRomblay/MLBox

.. |GitHub Issues| image:: https://img.shields.io/github/issues/AxeldeRomblay/MLBox.svg
.. _GitHub Issues: https://github.com/AxeldeRomblay/MLBox/issues

.. |codecov| image:: https://codecov.io/gh/AxeldeRomblay/MLBox/branch/master/graph/badge.svg
.. _codecov: https://codecov.io/gh/AxeldeRomblay/MLBox

.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
.. _License: https://github.com/AxeldeRomblay/MLBox/blob/master/LICENSE


**MLBox is a powerful Automated Machine Learning python library.**

* *It is compatible with:* **Python 2.7, 3.4 - 3.6**. & **64-bit version only** (32-bit python is not supported)
* *Operating system:* **Linux**. (MacOS & Windows very soon...)



Preparation 
-----------

First, make sure you have `setuptools <https://pypi.python.org/pypi/setuptools>`__ installed. Since MLBox package contains C++ source code, check that the following requirements are installed: 

* `gcc <https://gcc.gnu.org/>`__ 

.. code-block:: console

    $ sudo apt-get install build-essential
    
* `cmake <https://cmake.org/>`__  

.. code-block:: console

    $ pip install cmake
    
    
    
Stable version
--------------

Install from pip 
~~~~~~~~~~~~~~~~

MLBox is now available on **PyPI**, so you only need to run the following command:

.. code-block:: console

    $ pip install mlbox


Install from the Github
~~~~~~~~~~~~~~~~~~~~~

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



Dev version
-----------

A **4.0 dev-version for MLBox is also available** on the branch `"4.0-dev" <https://github.com/AxeldeRomblay/MLBox/tree/4.0-dev>`__ ! It provides some interesting new features. Please refer to `HISTORY <https://github.com/AxeldeRomblay/MLBox/blob/master/HISTORY.rst>`__. 

**It depends on sklearn-0.19.dev0 which is not a stable version at the moment.**
