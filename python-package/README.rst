Installation guide
==================

|Documentation Status| |PyPI version| |Build Status| |Windows Build Status| |GitHub Issues| |codecov| |License|

Compatibilities 
---------------

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
    
    
    
Installation
------------

Install from pip 
~~~~~~~~~~~~~~~~

MLBox is now available on **PyPI**, so you only need to run the following command:

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

    $ cd python-package/
    $ python setup.py install


.. _Github repo: https://github.com/AxeldeRomblay/mlbox

.. _tarball: https://github.com/AxeldeRomblay/mlbox/tarball/master

.. |Documentation Status| image:: https://readthedocs.org/projects/mlbox/badge/?version=latest
   :target: http://mlbox.readthedocs.io/en/latest/?badge=latest
.. |PyPI version| image:: https://badge.fury.io/py/mlbox.svg
   :target: https://pypi.python.org/pypi/mlbox
.. |Build Status| image:: https://travis-ci.org/AxeldeRomblay/MLBox.svg?branch=master
   :target: https://travis-ci.org/AxeldeRomblay/MLBox
.. |Windows Build Status| image:: https://ci.appveyor.com/api/projects/status/5ypa8vaed6kpmli8?svg=true
   :target: https://ci.appveyor.com/project/AxeldeRomblay/mlbox
.. |GitHub Issues| image:: https://img.shields.io/github/issues/AxeldeRomblay/MLBox.svg
   :target: https://github.com/AxeldeRomblay/MLBox/issues
.. |codecov| image:: https://codecov.io/gh/AxeldeRomblay/MLBox/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/AxeldeRomblay/MLBox
.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/AxeldeRomblay/MLBox/blob/master/LICENSE
