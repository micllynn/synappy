# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 21:36:07 2016

@author: michaellynn
"""

from setuptools import setup

setup(name='synappy',
      version='1.0',
      description='Automated quantification and visualization of features of synaptic events',
      url='none',
      author='Michael Lynn',
      author_email='micllynn@gmail.com',
      license='MIT',
      packages=['synappy'],
      install_requires = ['numpy', 'scipy', 'matplotlib', 'neo'],
      zip_safe = False)
