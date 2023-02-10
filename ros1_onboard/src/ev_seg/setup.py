from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['ev_seg'],
    package_dir={'': 'src'},
    scripts=['src/segmentation.py']
)

setup(**setup_args)