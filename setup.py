# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 21:36:07 2016

@author: michaellynn
"""

from setuptools import setup

setup(name='synappy',
      version='0.2',
      description='Automated detection and visualization of synaptic events in single-cell electrophysiology',
      url='none',
      author='Michael Lynn',
      author_email='micllynn@gmail.com',
      license='MIT',
      packages=['synappy'],
      install_requires = ['numpy', 'scipy', 'matplotlib', 'neo', 'bokeh'],
      zip_safe = False)
