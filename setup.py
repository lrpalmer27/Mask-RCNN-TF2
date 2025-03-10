"""
The build/compilations setup

>> pip install -r requirements.txt
>> python setup.py install
"""
import pip
import logging
# import pkg_resources
import importlib
from distutils.core import setup

# try:
#     from setuptools import setup
# except ImportError:
#     from distutils.core import setup


def _parse_requirements(file_path):
    # pip_ver = pkg_resources.get_distribution('pip').version
    pip_ver = importlib.import_module('pip').__version__
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setup(
    name='mask-rcnn-tf2',
    version='1.5.0',
    url='https://github.com/lrpalmer27/Mask-RCNN-TF2',
    author='Logan Palmer',
    author_email='lrpalmer@mun.ca',
    license='MUN',
    description='Object Detecting using Mask R-CNN in TensorFlow 2.0: Applied to Ice/Ship Semantic Segmentation',
    packages=["mrcnn"],
    install_requires=install_reqs,
    include_package_data=True,
    python_requires='>=3.4',
    long_description="""This is a modified version of this project (https://github.com/matterport/Mask_RCNN), and (https://github.com/ahmedfgad/Mask-RCNN-TF2) so that the Mask R-CNN model works on TensorFlow 2.0, and be applied for the specific purpose of segmenting ice/ships in a controlled research environment. """,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MUN License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
    ],
    keywords="image instance region segmentation object detection Mask-RCNN Mask RCNN R-CNN TensorFlow 2.0 Keras",
)
