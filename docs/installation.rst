Installation guide
==================

|Documentation Status| |PyPI version| |Build Status| |GitHub Issues| |codecov| |License| |Downloads| |Python Versions|


Compatibilities
---------------

* *Operating systems:* **Linux**, **MacOS** & **Windows**.
* *Python versions:* **3.5** - **3.6**. & **64-bit version** only (32-bit python is not supported)


Basic requirements
------------------

We suppose that `pip <https://pip.pypa.io/en/stable/installing/>`__ is already installed.

Also, please make sure you have `setuptools <https://pypi.python.org/pypi/setuptools>`__ and `wheel <https://pythonwheels.com/>`__ installed, which is usually the case if pip is installed.
If not, you can install both by running the following commands respectively: ``pip install setuptools`` and ``pip install wheel``.


Preparation (MacOS only)
------------------------

For **MacOS** users only, **OpenMP** is required. You can install it by the following command: ``brew install libomp``.


Installation
------------

You can choose to install MLBox either from pip or from the Github.


Install from pip
~~~~~~~~~~~~~~~~

Official releases of MLBox are available on **PyPI**, so you only need to run the following command:

.. code-block:: console

    $ pip install mlbox


Install from the Github
~~~~~~~~~~~~~~~~~~~~~~~

If you want to get the latest features, you can also install MLBox from the Github.

* **The sources for MLBox can be downloaded** from the `Github repo`_.

    * You can either clone the public repository:

    .. code-block:: console

        $ git clone git://github.com/AxeldeRomblay/mlbox

    * Or download the `tarball`_:

    .. code-block:: console

        $ curl  -OL https://github.com/AxeldeRomblay/mlbox/tarball/master


* Once you have a copy of the source, **you can install it**:

    .. code-block:: console

        $ cd mlbox
        $ make install


Issues
------

If you get any troubles during installation, you can refer to the `issues <https://github.com/AxeldeRomblay/MLBox/issues>`__.

**Please first check that there are no similar issues opened before opening one**.


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
