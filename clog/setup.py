#!/usr/bin/env python3
from setuptools import setup

setup(
    name='clog',
    version='1.0.0',
    description='Compress and summarize logs for efficient context usage',
    py_modules=['clog'],
    entry_points={
        'console_scripts': [
            'clog=clog:main',
        ],
    },
    python_requires='>=3.8',
)
