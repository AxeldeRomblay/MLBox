============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/AxeldeRomblay/mlbox/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* The smallest possible example to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

MLBox could always use more documentation, whether as part of the
official MLBox docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/AxeldeRomblay/mlbox/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `mlbox` for local development.

1. Fork the `mlbox` repo on GitHub.

2. Clone your fork::

    $ git clone git@github.com:your_name_here/mlbox.git

3. If you have virtualenv installed, skip this step. Either, run the following::

    $ pip install virtualenv
    
4. Install your local copy into a virtualenv following this commands to set up your fork for local development::

    $ cd MLBox
    $ virtualenv env
    $ source env/bin/activate
    $ make develop

If you have any troubles with the setup, please refer to the `installation guide <https://mlbox.readthedocs.io/en/latest/installation.html>`__

5. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

**Now you're set, you can make your changes locally.**

NOTE : each time you work on your branch, you will need to activate the virtualenv: ``$ source env/bin/activate``. To deactivate it, simply run: ``$ deactivate``.

6. When you're done making changes, check that your changes pass the tests.

NOTE : you need to install **pytest** before running the tests::

    $ make test

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The pull request should work for all supported Python versions and for PyPy. Check
   https://travis-ci.org/AxeldeRomblay/MLBox/pull_requests
   and make sure that the tests pass for all supported Python versions.
