data_path = './data'
windows = {
        'CT': (40, 80),
        'CTA': (50, 350),
        'MRA': (600, 1200),
        'MRI': (40, 80),
    }

LABELS_TO_IDX = {
            'Anterior Communicating Artery': 0,
            'Basilar Tip': 1,
            'Left Anterior Cerebral Artery': 2,
            'Left Infraclinoid Internal Carotid Artery': 3,
            'Left Middle Cerebral Artery': 4,
            'Left Posterior Communicating Artery': 5,
            'Left Supraclinoid Internal Carotid Artery': 6,
            'Other Posterior Circulation': 7,
            'Right Anterior Cerebral Artery': 8,
            'Right Infraclinoid Internal Carotid Artery': 9,
            'Right Middle Cerebral Artery': 10,
            'Right Posterior Communicating Artery': 11,
            'Right Supraclinoid Internal Carotid Artery': 12
}

IMG_SIZE = 512
FACTOR = 3
SEED = 42
N_FOLDS = 5
CORES = 16
