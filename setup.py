from distutils.core import setup
import py2exe
from skimage.filters.rank import core_cy_3d
setup(console=['synapse.py'])