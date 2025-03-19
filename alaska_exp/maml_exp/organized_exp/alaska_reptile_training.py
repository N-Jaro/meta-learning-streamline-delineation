import os
import glob
import argparse
import wandb
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from libs.alaskaNKDataloader import AlaskaNKMetaDataset
from libs.attentionUnet import AttentionUnet
from libs.unet import SimpleUNet, DeeperUnet, DeeperUnet_dropout, SimpleAttentionUNet
from libs.loss import dice_loss
from libs.reptile import ReptileTrainer  # Import the ReptileTrainer class

# --- Functions ---

def setup_model(model_type, input_shape, num_classes):
    if model_type == "unet":
        model_creator = SimpleUNet(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "deeperUnet":
        model_creator = DeeperUnet(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "deeperUnetDropout":
        model_creator = DeeperUnet_dropout(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "simpleAttentionUnet":
        model_creator = SimpleAttentionUNet(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "attentionUnet":
        model_creator = AttentionUnet(input_shape=input_shape, output_mask_channels=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model_creator.build_model()

def extract_scenario_from_filename(csv_filename):
    """Extracts scenario information from the CSV filename."""
    basename = os.path.basename(csv_filename)
    parts = basename.replace('.csv', '').split('_')
    scenario = {}
    if "random" in parts:
        scenario_type = "random"
        idx_random = parts.index("random")
        num_clusters = parts[idx_random + 1]
        cluster_id = parts[idx_random + 3]
    elif "kmean" in parts: 
        scenario_type = "kmean"
        idx_random = parts.index("kmean")
        num_clusters = parts[idx_random + 1]
        cluster_id = parts[idx_random + 3]
    else:
        scenario_type = "regular"
        idx_code = parts.index("code")
        num_clusters = parts[idx_code + 1]
        cluster_id = parts[idx_code + 3]

    scenario = {
        "type": scenario_type,
        "num_clusters": num_clusters,
        "cluster_id": cluster_id
    }
    return scenario
       
def main(args):

    config = vars(args)

    training_csv_file = glob.glob(os.path.join(args.clusters_dir, 'huc_code_*_train.csv'))[0]
    dataset = AlaskaNKMetaDataset(data_dir=args.data_dir, csv_file=training_csv_file, normalization_type=args.normalization_type, channels=args.channels, verbose=False)
    meta_train_episodes = dataset.create_multi_episodes(num_episodes=args.num_episodes, N=args.num_watersheds_per_episode, K=args.num_samples_per_location)
    episode_record = dataset.get_episode_record()
    print("Record of watersheds used in each episode:", episode_record)

    model = setup_model(args.model, input_shape=(128, 128, len(args.channels)), num_classes=1)
    model.summary()

    # Initialize Reptile Trainer
    # create run_name for the run in the initialization of the ReptileTrainer
    reptile_trainer = ReptileTrainer(model, meta_train_episodes, config, args)

    # Initialize train initial model wandb is initialized in the train function
    reptile_model, init_model_path = reptile_trainer.train()

    # Initialize WandB for adaptation runs
    adapt_cluster_run_name = f"{reptile_trainer.get_run_name()}_adapt"
    reptile_trainer.initialize_wandb(run_name=adapt_cluster_run_name)

    # Get all adapt CSV files
    adapt_csv_files = glob.glob(os.path.join(args.clusters_dir, 'huc_code_*_adapt.csv'))
    all_test_metrics = []

    for adapt_csv_path in adapt_csv_files:
        # Extract scenario information from the CSV filename
        scenario = extract_scenario_from_filename(adapt_csv_path)

        # Adapt the model
        adapted_model, adapted_model_path  = reptile_trainer.adapt_cluster(adapt_csv_path, init_model_path, scenario)
        
        #evaluate the model
        test_csv_path = adapt_csv_path.replace('_adapt.csv', '_test.csv')
        test_metrics = reptile_trainer.evaluate_clusters(adapted_model, test_csv_path, scenario, threshold=0.33) 

        if test_metrics is not None:  # Check if metrics_df is not None
            all_test_metrics.append(test_metrics)
        
    # Combine metrics from all test CSV files
    if all_test_metrics:
        combined_metrics_df = pd.concat(all_test_metrics, ignore_index=True)

        # Save the combined metrics DataFrame
        combined_metrics_save_path = os.path.join(args.run_model_save_dir, "combined_eval.csv")
        combined_metrics_df.to_csv(combined_metrics_save_path, index=False)
        print(f"Combined metrics saved at {combined_metrics_save_path}")

        # Create a WandB Table with all metrics
        table = wandb.Table(dataframe=combined_metrics_df)

        # Log the table to WandB only once
        wandb.log({'combined_evaluation_metrics': table}) 

        # Calculate average metrics
        avg_metrics = combined_metrics_df.mean(numeric_only=True)  # Calculate mean for numeric columns only

        # Log average metrics to WandB
        wandb.log({
            'avg_precision': avg_metrics['precision'],
            'avg_recall': avg_metrics['recall'],
            'avg_f1_score': avg_metrics['f1_score'],
            'avg_iou': avg_metrics['IoU'],
            'avg_kappa': avg_metrics['kappa']
        })

        # Print average metrics
        print(f"Average Precision: {avg_metrics['precision']}")
        print(f"Average Recall: {avg_metrics['recall']}")
        print(f"Average F1 Score: {avg_metrics['f1_score']}")
        print(f"Average IoU: {avg_metrics['IoU']}")
        print(f"Average Kappa: {avg_metrics['kappa']}")

    # Finish Adpatation Run
    wandb.finish()

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reptile for medical image segmentation")

    # For both training and adaptation
    parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data/data_wo_254_255/huc_code_data_znorm_128/', help='Path to data directory')
    parser.add_argument('--save_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/organized_exp/experiments/', help='Path to save trained models')
    parser.add_argument('--clusters_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/', help='Path to training CSV file')
    parser.add_argument('--wandb_project_name', type=str, default='test_reptile_exp', help='Path to save trained models')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2, 4, 6, 7, 8, 9, 10], help='Channels to use in the dataset (e.g., 0 1 2 4 6 7 8 9 10)')

    # For training part
    parser.add_argument('--num_watersheds_per_episode', type=int, default=1, help='Number of watersheds per episode')
    parser.add_argument('--num_samples_per_location', type=int, default=15, help='Number of samples per location')
    parser.add_argument('--num_episodes', type=int, default=25, help='Number of episodes')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeperUnet', 'deeperUnetDropout', 'simpleAttentionUnet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--inner_lr', type=float, default=0.00180, help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.00090, help='Meta learning rate')
    parser.add_argument('--decay_steps', type=int, default=500, help='Learning rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
    # parser.add_argument('--meta_batch_size', type=int, default=1, help='Meta batch size')                                 
    parser.add_argument('--inner_steps', type=int, default=15, help='Number of inner loop steps')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    # For adaptation part
    parser.add_argument('--adapt_steps', type=int, default=2000, help='Number of adaptation steps')
    parser.add_argument('--adapt_learning_rate', type=float, default=0.001, help='Learning rate for adaptation')

    args = parser.parse_args()
    main(args)
