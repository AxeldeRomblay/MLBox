# -*- coding: utf-8 -*-

with open('../VERSION.txt') as version_file:
    __version__ = version_file.read()

__author__ = """Axel ARONIO DE ROMBLAY"""
__email__ = 'axelderomblay@gmail.com'

from .preprocessing import *
from .encoding import *
from .optimisation import *
from .prediction import *
from .model import *