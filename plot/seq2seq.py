from plot.visualize import side_by_side
import numpy as np


def plot_userlibri_wer():
    side_by_side({
        'WER': {
            'x_axis': 'epoch',
            'metric': 'test_model-wer',
            'viz': {
                'LJ Speech': 'GossipPullAgent_55A_530E_8B_sparse(directed-3)_01-03-2024_21_03',
            },
        },
    }, axis_lim=[{'y': [0, 110], 'step': 10}])


def plot_userlibri_loss():
    side_by_side({
        'CTC loss': {
            'x_axis': 'epoch',
            'metric': ['train_model-ctc_loss', 'test_model-ctc_loss'],
            'viz': {
                'LJ Speech (train)': 'GossipPullAgent_55A_530E_8B_sparse(directed-3)_01-03-2024_21_03',
                'LJ Speech (test)': 'GossipPullAgent_55A_530E_8B_sparse(directed-3)_01-03-2024_21_03',
            },
        },
    }, axis_lim=[{'y': [0, 500], 'step': 50}], agg_fn=lambda x: np.average(x) / 100)


def plot_ljspeech_wer():
    side_by_side({
        'WER': {
            'x_axis': 'epoch',
            'metric': 'test_model-wer',
            'viz': {
                'LJ Speech': 'GossipPullAgent_55A_540E_8B_sparse(directed-3)_08-03-2024_17_58',
            },
        },
    }, axis_lim=[{'y': [0, 110], 'step': 10}])


def plot_ljspeech_loss():
    side_by_side({
        'CTC loss': {
            'x_axis': 'epoch',
            'metric': ['train_model-ctc_loss', 'test_model-ctc_loss'],
            'viz': {
                'LJ Speech (train)': 'GossipPullAgent_55A_540E_8B_sparse(directed-3)_08-03-2024_17_58',
                'LJ Speech (test)': 'GossipPullAgent_55A_540E_8B_sparse(directed-3)_08-03-2024_17_58',
            },
        },
    }, axis_lim=[{'y': [0, 500], 'step': 50}], agg_fn=lambda x: np.average(x) / 100)
