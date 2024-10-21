import os
import numpy as np
import argparse
import csv
from tensorflow.keras.models import load_model
from skimage.io import imread, imsave
from skimage.morphology import label, disk, binary_closing
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_hist
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
import seaborn as sns
from PIL import Image
import copy
import matplotlib.pyplot as plt
from libs.loss import dice_loss, dice_coefficient

class ModelEvaluator:
    def __init__(self, model_path, test_data_path, mask_path, ref_image_path, normalization_type="-1", clean_images=True):

        self.run_name = model_path.split('/')[2]
        self.model_path = model_path
        self.normalization_type = normalization_type
        self.test_data_path = test_data_path
        self.mask_path = mask_path
        self.ref_image_path = ref_image_path
        self.clean_images_flag = clean_images
        self.output_directory = self._create_output_directory()
        os.makedirs(self.output_directory, exist_ok=True)
        self.test_data = None
        self.predictions = None
        self.model = None
        self.prediction_map = None
        self.cleaned_image = None

    def _create_output_directory(self):
        model_dir = os.path.basename(os.path.dirname(self.model_path))
        return f'predicts/{model_dir}/'

    def _load_data(self):
        self.test_data = np.load(self.test_data_path)
        print(f"Test data shape: {self.test_data.shape}")

    def _load_model(self):
        self.model = load_model(self.model_path, custom_objects={"dice_loss": dice_loss, "dice_coefficient": dice_coefficient})

    def _predict(self):
        
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded before prediction.")

        # This normalization_type was define on the top of the notebook for the dataloader
        if self.normalization_type == '0':
            data_min = 0
            data_max = 255
            data_norm = (self.test_data - data_min) / (data_max - data_min)
        elif self.normalization_type == '-1':
            data_min = 0
            data_max = 255
            data_norm = 2 * ((self.test_data - data_min) / (data_max - data_min)) - 1
        elif self.normalization_type == 'none':
            data_norm = self.test_data
        else:
            raise ValueError("Unsupported normalization type. Choose '0-1' or '-1-1'.")

        self.predictions = self.model.predict(data_norm)
        self.predictions = self.predictions.reshape(self.predictions.shape[0], 224, 224)
        print(f"Predictions shape: {self.predictions.shape}")

    def _save_predictions(self):
        if self.predictions is None:
            raise ValueError("Predictions must be generated before saving.")
        np.save(os.path.join(self.output_directory, 'predictions.npy'), self.predictions)

    def _reconstruct_and_save_npy_files(self):
        patches = self.predictions
        metadata_path = self.test_data_path.replace('bottom_half_test_data.npy', 'bottom_half_test_label_metadata.npz')
        metadata = np.load(metadata_path)

        original_shape = metadata['original_shape']
        patch_size = metadata['patch_size']
        buffer_size = metadata['buffer_size']
        num_patches_row = metadata['num_patches_row']
        num_patches_col = metadata['num_patches_col']

        effective_patch_size = patch_size - 2 * buffer_size

        # Add symmetric padding to the original shape
        padded_shape = (original_shape[0] + 2 * buffer_size, original_shape[1] + 2 * buffer_size)

        # Initialize an empty array for the reconstructed image with padding
        reconstructed_image = np.zeros(padded_shape)

        count = 0
        for i in range(num_patches_row):
            for j in range(num_patches_col):
                start_row = i * effective_patch_size
                end_row = start_row + patch_size
                start_col = j * effective_patch_size
                end_col = start_col + patch_size

                # Extract the patch and place it in the padded reconstructed image
                patch = patches[count]
                reconstructed_image[start_row:end_row, start_col:end_col] = patch
                count += 1

        # Remove padding to get back to the original shape
        self.prediction_map = reconstructed_image[buffer_size:-buffer_size, buffer_size:-buffer_size]

        # Apply mask to filter out areas outside the mask
        mask = np.load(self.mask_path)
        self.prediction_map *= mask

        # Save reconstructed image as .npy file
        output_npy_path = os.path.join(self.output_directory, 'reconstructed.npy')
        np.save(output_npy_path, self.prediction_map)
        print(f"Reconstructed image saved: {output_npy_path}")

        # Save reconstructed image as .tif file
        output_tif_path = os.path.join(self.output_directory, 'reconstructed.tif')
        imsave(output_tif_path, img_as_ubyte(self.prediction_map))
        print(f"Reconstructed image saved as TIFF: {output_tif_path}")

    def _remove_small_objects_and_connect_lines(self, image, min_size, connectivity):
        mask = np.load(self.mask_path)
        print("image shape:", image.shape, "mask shape:", mask.shape)
        image = image * mask[:image.shape[0], :image.shape[1]]
        binary = np.array(image > 0, dtype=int)
        labeled = label(binary)
        object_sizes = np.bincount(labeled.ravel())
        mask = object_sizes > min_size
        cleaned = mask[labeled]
        structure = disk(connectivity)
        cleaned = binary_closing(cleaned, structure)
        return cleaned

    def _clean_image(self, image, min_size=300, connectivity=10):
        return self._remove_small_objects_and_connect_lines(image, min_size, connectivity)

    def _process_images(self, min_size=300, connectivity=10):
        if self.clean_images_flag and self.prediction_map is not None:
            self.cleaned_image = self._clean_image(self.prediction_map, min_size, connectivity)
            self.cleaned_image = equalize_hist(self.cleaned_image)  # Enhance contrast
            output_filename = os.path.join(self.output_directory, 'reconstructed_cleaned.tif')
            imsave(output_filename, img_as_ubyte(self.cleaned_image))

    def _clip_reference_image(self, ref_image, target_image):
        return ref_image[:target_image.shape[0], :target_image.shape[1]]

    def _binarize_predictions(self, predictions, threshold=0.5):
        return (predictions >= threshold).astype(int)

    def _calculate_scores(self, reference, target):
        f1_stream = f1_score(reference, target, labels=[1], average='micro')
        precision_stream = precision_score(reference, target, labels=[1], average='micro')
        recall_stream = recall_score(reference, target, labels=[1], average='micro')
        cohen_kappa = cohen_kappa_score(reference, target)
        return precision_stream, recall_stream, f1_stream, cohen_kappa

    def _plot_confusion_matrix(self, reference, target):
        cm = confusion_matrix(reference, target)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        confusion_matrix_path = os.path.join(self.output_directory, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        print(f"Confusion Matrix saved: {confusion_matrix_path}")
        plt.close()

    def _evaluate(self):
        ref_image = np.load(self.ref_image_path)
        ref_image_clipped = self._clip_reference_image(ref_image, self.prediction_map)

        # Save reconstructed image as .tif file
        output_tif_path = os.path.join(self.output_directory, 'reference.tif')
        imsave(output_tif_path, img_as_ubyte(ref_image_clipped))
        print(f"Reconstructed image saved as TIFF: {output_tif_path}")


        binarized_prediction_map = self._binarize_predictions(self.prediction_map)

        print(f"Evaluating scores for reconstructed image (original):")
        original_precision, original_recall, original_f1, original_kappa = self._calculate_scores(ref_image_clipped.flatten(), binarized_prediction_map.flatten())
        print(f"Original - Precision: {original_precision}, Recall: {original_recall}, F1-Score: {original_f1}, Cohen Kappa: {original_kappa}")

        # Plot confusion matrix for the original prediction map
        self._plot_confusion_matrix(ref_image_clipped.flatten(), binarized_prediction_map.flatten())

        cleaned_precision, cleaned_recall, cleaned_f1, cleaned_kappa = None, None, None, None

        if self.clean_images_flag:
            binarized_cleaned_image = self._binarize_predictions(self.cleaned_image)

            print(f"Evaluating scores for reconstructed image (cleaned):")
            cleaned_precision, cleaned_recall, cleaned_f1, cleaned_kappa = self._calculate_scores(ref_image_clipped.flatten(), binarized_cleaned_image.flatten())
            print(f"Cleaned - Precision: {cleaned_precision}, Recall: {cleaned_recall}, F1-Score: {cleaned_f1}, Cohen Kappa: {cleaned_kappa}")

            # Plot confusion matrix for the cleaned prediction map
            self._plot_confusion_matrix(ref_image_clipped.flatten(), binarized_cleaned_image.flatten())

        # Save metrics to CSV
        csv_path = os.path.join(self.output_directory, 'evaluation_metrics.csv')
        csv_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not csv_exists:
                writer.writerow(['run_name', 'original_f1_score', 'original_recall', 'original_precision', 'original_kappa', 
                                 'cleaned_f1_score', 'cleaned_recall', 'cleaned_precision', 'cleaned_kappa'])
            writer.writerow([self.run_name, original_f1, original_recall, original_precision, original_kappa, 
                             cleaned_f1, cleaned_recall, cleaned_precision, cleaned_kappa])
        print(f"Metrics saved to {csv_path}")

    def _visualize_predictions(self):
        if self.prediction_map is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.prediction_map, cmap='gray')
            plt.title('Reconstructed Prediction Map')
            plt.axis('off')
            prediction_map_path = os.path.join(self.output_directory, 'reconstructed_prediction_map.png')
            plt.savefig(prediction_map_path)
            print(f"Reconstructed Prediction Map saved: {prediction_map_path}")
            plt.close()

        if self.cleaned_image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.cleaned_image, cmap='gray')
            plt.title('Cleaned Prediction Map')
            plt.axis('off')
            cleaned_image_path = os.path.join(self.output_directory, 'cleaned_prediction_map.png')
            plt.savefig(cleaned_image_path)
            print(f"Cleaned Prediction Map saved: {cleaned_image_path}")
            plt.close()

    def run_evaluation(self):
        self._load_data()
        self._load_model()
        self._predict()
        self._save_predictions()
        self._reconstruct_and_save_npy_files()
        self._process_images()
        self._evaluate()
        self._visualize_predictions()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model predictions.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file (e.g., "./model/run1/best_target_model.h5").')
    parser.add_argument('--test_data_path', default="/u/nathanj/meta-learning-streamline-delineation/samples/Covington/bottom_half_test_data.npy", type=str, help='Path to the test data .npy file.')
    parser.add_argument('--mask_path', default="/u/nathanj/meta-learning-streamline-delineation/samples/Covington/bottom_half_test_mask.npy", type=str, help='Path to the mask .npy file.')
    parser.add_argument('--normalization_type', type=str, default='-1', choices=['-1', '0', 'none'], help='Normalization range')
    parser.add_argument('--ref_image_path', default="/u/nathanj/meta-learning-streamline-delineation/samples/Covington/bottom_half_test_label.npy", type=str, help='Path to the reference image .npy file.')
    parser.add_argument('--clean_images', type=bool, default=False, help='Whether to clean the images or not.')

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        mask_path=args.mask_path,
        ref_image_path=args.ref_image_path,
        normalization_type = args.normalization_type,
        clean_images=args.clean_images
        
    )

    evaluator.run_evaluation()
