"""Setup module for project."""

from setuptools import setup, find_packages

setup(
        name='mp18-project-skeleton',
        version='0.1',
        description='Skeleton code for Machine Perception Dynamic Gesture Recognition project.',

        author='Emre Aksan',
        author_email='eaksan@inf.ethz.ch',

        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
        install_requires=[
			# Add external libraries here.
            'numpy',
            'opencv-python',
        ],
)
