import os
import argparse
import wandb
import numpy as np
import datetime
import tensorflow as tf
from libs.alaskaNKDataloader import AlaskaNKMetaDataset
from libs.attentionUnet import AttentionUnet
from libs.unet import SimpleUNet, DeeperUnet, DeeperUnet_dropout, SimpleAttentionUNet
from libs.loss import dice_loss
from libs.maml import MAMLTrainer  # Import the MAMLTrainer class

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

def main(args):

    config = vars(args)

    dataset = AlaskaNKMetaDataset(data_dir=config['data_dir'], csv_file=config['training_csv'], normalization_type=config['normalization_type'], channels=config['channels'], verbose= False)
    meta_train_episodes = dataset.create_multi_episodes(num_episodes=config['num_episodes'], N=config['num_watersheds_per_episode'], K=config['num_samples_per_location'])
    episode_record = dataset.get_episode_record()
    print("Record of watersheds used in each episode:", episode_record)

    model = setup_model(config['model_type'], input_shape=(128, 128,  len(config['channels'])), num_classes=1)
    model.summary()

    maml_trainer = MAMLTrainer(model, meta_train_episodes, config)
    maml_model, model_path = maml_trainer.train()

    print("The best MAML model saved at:", model_path)

    

    



# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAML for medical image segmentation")
    parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data/data_wo_254_255/huc_code_data_znorm_128/', help='Path to data directory')
    parser.add_argument('--save_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/wo_254_255/model', help='Path to save trained models')
    parser.add_argument('--wandb_project_name', type=str, default='alaska_wo_254_255_unet_12052024_0204', help='Path to save trained models')
    parser.add_argument('--training_csv', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/huc_code_kmean_5_train.csv', help='Path to training CSV file')
    parser.add_argument('--num_watersheds_per_episode', type=int, default=1, help='Number of watersheds per episode')
    parser.add_argument('--num_samples_per_location', type=int, default=15, help='Number of samples per location')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--num_episodes', type=int, default=25, help='Number of episodes')
    parser.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2, 4, 6, 7, 8, 9, 10], help='Channels to use in the dataset (e.g., 0 1 2 4 6 7 8 9 10)')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeperUnet', 'deeperUnetDropout', 'simpleAttentionUnet', 'attentionUnet'], help='Model architecture')
    parser.add_argument('--inner_lr', type=float, default=0.00180, help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.00090, help='Meta learning rate')
    parser.add_argument('--decay_steps', type=int, default=500, help='Learning rate decay steps')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='Learning rate decay rate')
    parser.add_argument('--meta_batch_size', type=int, default=1, help='Meta batch size')                                 
    parser.add_argument('--inner_steps', type=int, default=3, help='Number of inner loop steps')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    args = parser.parse_args()
    main(args)
