#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pip
from setuptools import setup
from setuptools.command.install import install
from mlbox.__init__ import __version__

with open('requirements.txt', 'rt') as fh:
    requirements = fh.read().splitlines()

class OverrideInstallCommand(install):
    def run(self):
        # Install all requirements
        failed = []

        for req in requirements:
            if pip.main(["install", req]) == 1:
                failed.append(req)

        if len(failed) > 0:
            print("")
            print("Error installing the following packages:")
            print(str(failed))
            print("Please install them manually")
            print("")
            raise OSError("Aborting")

        # install MlBox
        install.run(self)

with open('README.rst') as readme_file:
    readme = readme_file.read()


setup(
    name='mlbox',
    version=__version__,
    description="A powerful Automated Machine Learning python library.",
    long_description=readme,
    author="Axel ARONIO DE ROMBLAY",
    author_email='axelderomblay@gmail.com',
    url='https://github.com/AxeldeRomblay/mlbox',
    packages=[
        'mlbox','mlbox.optimisation','mlbox.prediction',
        'mlbox.preprocessing','mlbox.optimisation.encoding','mlbox.optimisation.modeling',
        'mlbox.optimisation.modeling.classification','mlbox.optimisation.modeling.regression','mlbox.preprocessing.drift'
    ],
    package_dir={'mlbox':'mlbox',
                 'mlbox.preprocessing':'mlbox/preprocessing',
                 'mlbox.optimisation':'mlbox/optimisation',
                 'mlbox.prediction':'mlbox/prediction',
                 'mlbox.optimisation.encoding':'mlbox/optimisation/encoding',
                 'mlbox.optimisation.modeling':'mlbox/optimisation/modeling',
                 'mlbox.optimisation.modeling.classification':'mlbox/optimisation/modeling/classification',
                 'mlbox.optimisation.modeling.regression': 'mlbox/optimisation/modeling/regression',
                 'mlbox.preprocessing.drift':'mlbox/preprocessing/drift'
                 },
    include_package_data=True,
    cmdclass={'install': OverrideInstallCommand},
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
        
        'Operating System :: POSIX :: Linux',
        
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    test_suite='tests',
    tests_require=requirements,
    extras_require={'extended': ["xgboost==0.6a2"]}
)
