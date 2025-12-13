# Ricci-REWA Continual Learning Experiment
# Testing geometric inoculation against catastrophic forgetting

from .ricci_curvature import OllivierRicci, compute_ricci_on_embeddings
from .models import SimpleMLP, ConvNet
from .continual_learning import BaselineCL, EWCCL, RicciRegCL
from .experiment import run_continual_learning_experiment
