#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pip
from setuptools import setup
from setuptools.command.install import install
from pip.req import parse_requirements

install_reqs = parse_requirements("./requirements.txt", session=False)
reqs = [str(ir.req) for ir in install_reqs]

class OverrideInstallCommand(install):
    def run(self):
        # Install all requirements
        failed = []

        for req in reqs:
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

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "numpy>=1.13.0",
    "matplotlib>=2.0.2",
    "hyperopt==0.1",
    "ipyparallel==6.0.0",
    "Keras==2.0.4",
    "pandas>=0.18.0",
    "scikit-learn==0.18.1",
    "scipy>=0.17.0",
    "Theano==0.9.0",
    "xgboost==0.6a2",
    "lightgbm==2.0.2"
]

test_requirements = [
    "numpy>=1.13.0",
    "matplotlib>=2.0.2",
    "hyperopt==0.1",
    "ipyparallel==6.0.0",
    "Keras==2.0.4",
    "pandas>=0.18.0",
    "scikit-learn==0.18.1",
    "scipy>=0.17.0",
    "Theano==0.9.0",
    "xgboost==0.6a2",
    "lightgbm==2.0.2"
]

setup(
    name='mlbox',
    version='0.3.0',
    description="A powerful Automated Machine Learning python library  ",
    long_description=readme + '\n\n' + history,
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
    keywords=['machine learning', 'data science', 'automated machine learning','mlbox', 'auto-ml', 'deep learning', 'stacking', 'pipeline', 'optimisation'],
    classifiers=[
        
        'Development Status :: 5 - Production/Stable',
        
        'Intended Audience :: Developers',
        'Intended Audience :: Data Scientists',
        'Intended Audience :: Science/Research',
        
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        
        'License :: OSI Approved :: BSD-3 License',
        
        'Natural Language :: English',
        
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Linux',
        'Operating System :: MacOS',
        
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    test_suite='tests',
    tests_require=test_requirements
)
