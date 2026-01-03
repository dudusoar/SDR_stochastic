"""
Data layer for VRP toolkit.

This module contains data generation, loading, and map representation components.
"""

from .generators import OrderGenerator, DemandGenerator
from .map import RealMap, RealDataMap

__all__ = [
    'RealMap',
    'RealDataMap',
    'OrderGenerator',
    'DemandGenerator',
]