"""Setup file for installing mlbox package."""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup


with open('requirements.txt', 'rt') as fh:
    requirements = fh.read().splitlines()

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('VERSION.txt') as version_file:
    version = version_file.read()


setup(
    name='mlbox',
    version=version,
    description="A powerful Automated Machine Learning python library.",
    long_description=readme,
    author="Axel ARONIO DE ROMBLAY",
    author_email='axelderomblay@gmail.com',
    url='https://github.com/AxeldeRomblay/mlbox',
    packages=['mlbox', 'mlbox.encoding', 'mlbox.model',
              'mlbox.optimisation', 'mlbox.prediction',
              'mlbox.preprocessing',
              'mlbox.model.classification',
              'mlbox.model.regression',
              'mlbox.preprocessing.drift'],
    package_dir={'mlbox': 'mlbox',
                 'mlbox.encoding': 'mlbox/encoding',
                 'mlbox.model': 'mlbox/model',
                 'mlbox.optimisation': 'mlbox/optimisation',
                 'mlbox.prediction': 'mlbox/prediction',
                 'mlbox.preprocessing': 'mlbox/preprocessing',
                 'mlbox.model.classification': 'mlbox/model/classification',
                 'mlbox.model.regression': 'mlbox/model/regression',
                 'mlbox.preprocessing.drift': 'mlbox/preprocessing/drift'
                 },
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    license='BSD-3',
    keywords='mlbox auto-ml stacking pipeline optimisation',
    classifiers=[

        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: BSD License',

        'Natural Language :: English',

        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    test_suite='tests',
    tests_require=requirements
)
