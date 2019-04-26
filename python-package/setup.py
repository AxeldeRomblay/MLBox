#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pip
from setuptools import setup
from setuptools.command.install import install

#with open('requirements.txt', 'rt') as fh:
    #requirements = fh.read().splitlines()

requirements = [
    "numpy>=1.13.0",
    "matplotlib>=2.0.2",
    "hyperopt==0.1",
    "Keras==2.1.2",
    "pandas==0.21.0",
    "joblib==0.11",
    "scikit-learn==0.19.0",
    "Theano==1.0.1",
    "lightgbm==2.0.11",
    "networkx==1.11"
]


class OverrideInstallCommand(install):
    def run(self):
        # Install all requirements
        failed = []

        for req in requirements:
            try:
                from pip._internal import main
                if main(["install", req]) == 1:
                    failed.append(req)
            except:
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
    version="0.6.1",
    description="A powerful Automated Machine Learning python library.",
    long_description=readme,
    author="Axel ARONIO DE ROMBLAY",
    author_email='axelderomblay@gmail.com',
    url='https://github.com/AxeldeRomblay/mlbox',
    packages=[
        'mlbox','mlbox.encoding','mlbox.model','mlbox.optimisation','mlbox.prediction',
        'mlbox.preprocessing','mlbox.model.supervised','mlbox.model.supervised.classification',
        'mlbox.model.supervised.regression','mlbox.preprocessing.drift'
    ],
    package_dir={'mlbox':'mlbox',
                 'mlbox.encoding':'mlbox/encoding',
                 'mlbox.model':'mlbox/model',
                 'mlbox.optimisation':'mlbox/optimisation',
                 'mlbox.prediction':'mlbox/prediction',
                 'mlbox.preprocessing':'mlbox/preprocessing',
                 'mlbox.model.supervised':'mlbox/model/supervised',
                 'mlbox.model.supervised.classification':'mlbox/model/supervised/classification',
                 'mlbox.model.supervised.regression':'mlbox/model/supervised/regression',
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
    tests_require=requirements
)
