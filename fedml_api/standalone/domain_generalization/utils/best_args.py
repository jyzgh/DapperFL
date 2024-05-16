best_args = {
    'fl_digits': {

        'fedavg': {
                'local_lr': 0.01,
                'local_batch_size': 64,
        },
        'dapperfl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'pr_ratios': ['0', '0.2', '0.4', '0.6', '0.8'],
        },
        'fedsr': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'nefl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'ps': [0.04, 0.16, 0.36, 0.64, 1],
            'method': 'W',
            'learnable_step': 1,
            'num_models': 5,
        },
        'fedprox': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'mu': 0.1,
        },

        'feddrop': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'pr_strategy': '0.6',
            'pr_ratios': ['0', '0.2', '0.4', '0.6', '0.8'],
        },

        'fedmp': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'pr_ratios': ['0', '0.2', '0.4', '0.6', '0.8'],
        },

        'moon': {
                'local_lr': 0.01,
                'local_batch_size': 64,
                'temperature': 0.5,
                'mu':5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'Note': '+ MSE'
        }
    },
    'fl_officecaltech': {

        'fedavg': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'dapperfl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'pr_ratios': ['0', '0.2', '0.4', '0.6', '0.8'],
        },
        'fedsr': {
            'local_lr': 0.01,
            'local_batch_size': 64,
        },
        'nefl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'ps': [0.04, 0.16, 0.36, 0.64, 1],
            'method': 'W',
            'learnable_step': 1,
            'num_models': 5,
        },
        'fedprox': {
            'local_lr': 0.01,
            'mu': 0.01,
            'local_batch_size': 64,
        },
        'feddrop': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'pr_strategy': '0.6',
            'pr_ratios': ['0', '0.2', '0.4', '0.6', '0.8'],
        },
        'fedmp': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'pr_ratios': ['0', '0.2', '0.4', '0.6', '0.8'],
        },
        'moon': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'temperature': 0.5,
            'mu': 5
        },

        'fpl': {
            'local_lr': 0.01,
            'local_batch_size': 64,
            'Note': '+ MSE'
        }
    }
}
