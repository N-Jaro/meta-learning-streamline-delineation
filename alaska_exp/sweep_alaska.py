import wandb
from alaska_maml_trianing_sweep import main  # Import your main function

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'mean_val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'meta_lr': {
            'value': 0.0180
        },
        'inner_lr': {
            'value': 0.0089
        },
        'inner_steps': {
            'value': 3
        },
        'num_watersheds_per_episode': {
            'value': 1
        },
        'num_samples_per_location': {
            'values': [1, 5, 10, 15, 20, 25, 30, 50, 100]
        },
        'num_episodes': {
            'value': 25
        },
        'epochs': {
            'value': 100
        },
        'patience': {
            'value': 15
        },
        'data_dir': {
            'value': '/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/huc_code_data_znorm_128/'
        },
        'training_csv': {
            'value': '/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huvc_code_clusters/huc_code_train.csv'
        },
        'testing_csv': {
            'value': '/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huvc_code_clusters/huc_code_test.csv'
        },
        'normalization_type': {
            'value': '-1'
        },
        'meta_batch_size': {
            'value': 1
        },
        'decay_steps': {
            'value': 30
        },
        'decay_rate': {
            'value': 0.96
        },
        'save_path': {
            'value': 'models'
        },
        'model_type': {
            'value': 'unet'
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="Alaska_maml_sweep")

# Run the sweep
wandb.agent(sweep_id, function=main, count=20)  # Adjust the count as needed
