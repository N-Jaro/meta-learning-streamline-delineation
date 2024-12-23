import os
import datetime
import tensorflow as tf
import wandb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, jaccard_score
import matplotlib.pyplot as plt
from loss import dice_loss
from attentionUnet import ChannelAttention, SpatialAttention
from normalization import normalize_data


class MAMLTrainer:
    def __init__(self, base_model, episodes, config):
        self.base_model = base_model
        self.episodes = episodes
        self.config = config
        self.inner_lr = config['initial_inner_lr']
        self.meta_lr = config['initial_meta_lr']
        self.meta_optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_lr)
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.inner_lr_no_improvement_count = 0
        self.meta_lr_no_improvement_count = 0

    def train_task_model(self, model, inputs, outputs, optimizer):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = dice_loss(outputs, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return model, loss.numpy()

    def initialize_wandb(self, name):
        wandb.init(project=self.config['wandb_project_name'], name=name, config=self.config)

    def train(self):
        date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f'maml_{self.config["inner_steps"]}_{self.config["epochs"]}_{self.config["meta_batch_size"]}_{date_time}'
        model_path = os.path.join(self.config['save_path'], name)
        os.makedirs(model_path, exist_ok=True)
        print(f'Initialize the Training process: {name}')

        self.initialize_wandb(name)

        inner_lr_patience = self.config['patience'] - 5
        inner_lr_reduction_factor = 0.96
        meta_lr_patience = self.config['patience']
        meta_lr_reduction_factor = 0.96

        for epoch in range(self.config['epochs']):
            task_losses = []
            for batch_index in range(self.config['meta_batch_size']):
                task_updates = []
                for episode_index, episode in enumerate(self.episodes):
                    model_copy = tf.keras.models.clone_model(self.base_model)
                    model_copy.set_weights(self.base_model.get_weights())
                    inner_optimizer = tf.keras.optimizers.Adam(learning_rate=self.inner_lr)

                    support_data, support_labels = episode["support_set_data"], episode["support_set_labels"]
                    episode_losses = []
                    for _ in range(self.config['inner_steps']):
                        model_copy, loss = self.train_task_model(model_copy, support_data, support_labels, inner_optimizer)
                        episode_losses.append(loss)

                    query_data, query_labels = episode["query_set_data"], episode["query_set_labels"]
                    with tf.GradientTape() as meta_tape:
                        meta_tape.watch(model_copy.trainable_variables)
                        new_val_loss = dice_loss(query_labels, model_copy(query_data))
                    gradients = meta_tape.gradient(new_val_loss, model_copy.trainable_variables)
                    task_losses.append(new_val_loss.numpy())

                    wandb.log({
                        "epoch": epoch,
                        "episode": episode_index,
                        "inner_loss": tf.reduce_mean(episode_losses),
                        "outer_loss": new_val_loss.numpy(),
                        "inner_lr": self.inner_lr,
                    })

                    mapped_gradients = [tf.identity(grad) for grad in gradients]
                    task_updates.append((mapped_gradients, new_val_loss))

                if task_updates:
                    num_variables = len(self.base_model.trainable_variables)
                    mean_gradients = []
                    for i in range(num_variables):
                        grads = [update[0][i] for update in task_updates if update[0][i] is not None]
                        if grads:
                            mean_grad = tf.reduce_mean(tf.stack(grads), axis=0)
                            mean_gradients.append(mean_grad)
                        else:
                            mean_gradients.append(None)

                    gradients_to_apply = [(grad, var) for grad, var in zip(mean_gradients, self.base_model.trainable_variables) if grad is not None]
                    if gradients_to_apply:
                        self.meta_optimizer.apply_gradients(gradients_to_apply)

            mean_loss = tf.reduce_mean(task_losses)
            wandb.log({
                "epoch": epoch,
                "mean_val_loss": mean_loss,
                "meta_lr": self.meta_optimizer.learning_rate.numpy()
            })

            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.no_improvement_count = 0
                self.inner_lr_no_improvement_count = 0
                self.meta_lr_no_improvement_count = 0
                self.base_model.save(os.path.join(model_path, 'maml_model.keras'))
                print(f"Saved new best model with validation loss: {self.best_loss}")
            else:
                self.no_improvement_count += 1
                self.inner_lr_no_improvement_count += 1
                self.meta_lr_no_improvement_count += 1
                print(f"No improvement in epoch {epoch + 1}. No improvement count: {self.no_improvement_count}")

            if self.inner_lr_no_improvement_count >= inner_lr_patience:
                old_inner_lr = self.inner_lr
                self.inner_lr = old_inner_lr * inner_lr_reduction_factor
                print(f"Reducing inner learning rate from {old_inner_lr} to {self.inner_lr}")
                self.inner_lr_no_improvement_count = 0

            if self.meta_lr_no_improvement_count >= meta_lr_patience:
                old_meta_lr = self.meta_optimizer.learning_rate.numpy()
                new_meta_lr = old_meta_lr * meta_lr_reduction_factor
                self.meta_optimizer.learning_rate.assign(new_meta_lr)
                print(f"Reducing meta learning rate from {old_meta_lr} to {new_meta_lr}")
                self.meta_lr_no_improvement_count = 0

            if self.no_improvement_count >= self.config['patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs with best validation loss: {self.best_loss}")
                break

            tf.keras.backend.clear_session()
            print(f"Epoch {epoch + 1} completed, Mean Validation Loss across all episodes: {mean_loss}")

        print(f"Completed training for maximum {self.config['epochs']} epochs.")
        wandb.finish()
        return self.base_model, model_path

    def load_model(model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'ChannelAttention':ChannelAttention, 'SpatialAttention':SpatialAttention})
        print(f"Loaded model from {model_path}")
        return model


    def adapt(self, model_path, support_data, support_labels, query_data, query_labels):
        model = self.load_model(model_path)
        inner_optimizer = tf.keras.optimizers.Adam(learning_rate=self.inner_lr)
        for _ in range(self.config['inner_steps']):
            model, loss = self.train_task_model(model, support_data, support_labels, inner_optimizer)
        return model, loss
    
    def evaluate(self, model, test_data, test_labels): 
