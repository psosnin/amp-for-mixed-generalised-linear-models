import gk
from generate_data import *
from fitting.gamp import GAMP
from fitting.matrix_gamp import matrix_GAMP

"""
Configuration file for each generalised linear model and fitting algorithm.
None indicates that the algorithm is not implemented for that model.
"""


config = {
    'linear': {
        'data': generate_data_linear,
        'gk': gk.linear.gk_expect,
        'GAMP': GAMP,
        'EM': None,
        'AM': None,
    },
    'logistic': {
        'data': generate_data_logistic,
        'gk': gk.logistic.gk_expect,
        'GAMP': GAMP,
        'EM': None,
        'AM': None,
    },
    'relu': {
        'data': generate_data_relu,
        'gk': gk.relu.gk_expect,
        'GAMP': GAMP,
        'EM': None,
        'AM': None,
    },
    'mixed_linear': {
        'data': generate_data_mixed_linear,
        'gk': gk.mixed_linear.gk_expect,
        'GAMP': matrix_GAMP,
        'EM': None,
        'AM': None,
    },
    'mixed_logistic': {
        'data': generate_data_mixed_logistic,
        'gk': gk.mixed_logistic.gk_expect,
        'GAMP': matrix_GAMP,
        'EM': None,
        'AM': None,
    },
    'mixed_relu': {
        'data': generate_data_mixed_relu,
        'gk': None,
        'GAMP': None,
        'EM': None,
        'AM': None,
    },
}
