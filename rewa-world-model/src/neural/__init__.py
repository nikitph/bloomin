"""Neural encoding package"""

from .contrastive import (
    ContrastiveEncoder,
    InfoNCELoss,
    ContrastiveTrainer
)

__all__ = [
    'ContrastiveEncoder',
    'InfoNCELoss',
    'ContrastiveTrainer'
]
