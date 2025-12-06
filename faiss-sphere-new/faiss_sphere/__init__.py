"""FAISS-Sphere: Exploiting K=1 Spherical Geometry for Vector Search"""

from .index import FAISSSphere
from .core import GeodesicDistance, IntrinsicProjector, SphericalLSH, SphericalPQ

__version__ = "0.1.0"

__all__ = [
    'FAISSSphere',
    'GeodesicDistance',
    'IntrinsicProjector',
    'SphericalLSH',
    'SphericalPQ',
]
