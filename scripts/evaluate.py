import os
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from skimage.io import imwrite
from skimage.morphology import label, disk, binary_closing
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
from PIL import Image
import copy

class ModelEvaluator:
    def __init__(self, test_data_path, model_path, mask_path, ref_image_path, clean_images=True):
        self.test_data_path = test_data_path
        self.model_path = model_path
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
        model_name = os.path.basename(os.path.dirname(self.model_path))
        return f'predicts/{model_name}/covington_predictions/'

    def _load_data(self):
        self.test_data = np.load(self.test_data_path)
        print(f"Test data shape: {self.test_data.shape}")

    def _load_model(self):
        self.model = load_model(self.model_path)

    def _predict(self):
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded before prediction.")
        self.predictions = self.model.predict(self.test_data)
        self.predictions = self.predictions.reshape(self.predictions.shape[0], 224, 224)
        print(f"Predictions shape: {self.predictions.shape}")

    def _save_predictions(self):
        if self.predictions is None:
            raise ValueError("Predictions must be generated before saving.")
        np.save(os.path.join(self.output_directory, 'predictions.npy'), self.predictions)

    def _reconstruct_and_save_npy_files(self):
        data = self.predictions
        print(data.shape)
        dim = data.shape
        buf = 30

        numr = 41
        numc = (dim[0] // 41) - 1
        count = -1
        for i in range(numr):
            for j in range(int(numc) - 1):
                count += 1
                temp = data[count][buf:-buf, buf:-buf]
                if j == 0:
                    rows = temp
                else:
                    rows = np.concatenate((rows, temp), axis=1)

            if i == 0:
                self.prediction_map = copy.copy(rows)
            else:
                self.prediction_map = np.concatenate((self.prediction_map, rows), axis=0)

        self.prediction_map = self.prediction_map[:, :, 0]
        print(self.prediction_map.shape)

        output_path = os.path.join(self.output_directory, 'reconstructed.tif')
        print(f"Reconstructed image saved: {output_path}")
        image = Image.fromarray(self.prediction_map)
        image.save(output_path)

    def _remove_small_objects_and_connect_lines(self, image, min_size, connectivity):
        mask = np.load(self.mask_path)
        print("image shape:", image.shape, "mask shape:", mask.shape)
        image = image * mask[:image.shape[0], :image.shape[1]]
        binary = np.array(image > 0, dtype=np.int)
        labeled, num_objects = label(binary)
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
            output_filename = os.path.join(self.output_directory, 'reconstructed_cleaned.tif')
            imwrite(output_filename, self.cleaned_image)

    def _clip_reference_image(self, ref_image, target_image):
        return ref_image[:target_image.shape[0], :target_image.shape[1]]

    def _calculate_scores(self, reference, target):
        f1_stream = f1_score(reference, target, labels=[1], average='micro')
        precision_stream = precision_score(reference, target, labels=[1], average='micro')
        recall_stream = recall_score(reference, target, labels=[1], average='micro')
        cohen_kappa = cohen_kappa_score(reference, target)
        return precision_stream, recall_stream, f1_stream, cohen_kappa

    def _evaluate(self):
        ref_image = np.load(self.ref_image_path)
        ref_image_clipped = self._clip_reference_image(ref_image, self.prediction_map)

        print(f"Evaluating scores for reconstructed image (original):")
        precision, recall, f1, cohen_kappa = self._calculate_scores(ref_image_clipped.flatten(), self.prediction_map.flatten())
        print(f"Original - Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Cohen Kappa: {cohen_kappa}")

        if self.clean_images_flag:
            cleaned_image = self.cleaned_image

            print(f"Evaluating scores for reconstructed image (cleaned):")
            precision, recall, f1, cohen_kappa = self._calculate_scores(ref_image_clipped.flatten(), cleaned_image.flatten())
            print(f"Cleaned - Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Cohen Kappa: {cohen_kappa}")

    def run_evaluation(self):
        self._load_data()
        self._load_model()
        self._predict()
        self._save_predictions()
        self._reconstruct_and_save_npy_files()
        self._process_images()
        self._evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model predictions.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data .npy file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file (e.g., models/model_name/best_model.h5).')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the mask .npy file.')
    parser.add_argument('--ref_image_path', type=str, required=True, help='Path to the reference image .npy file.')
    parser.add_argument('--clean_images', type=bool, default=True, help='Whether to clean the images or not.')

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        test_data_path=args.test_data_path,
        model_path=args.model_path,
        mask_path=args.mask_path,
        ref_image_path=args.ref_image_path,
        clean_images=args.clean_images
    )

    evaluator.run_evaluation()


# python script_name.py --test_data_path path/to/test_data.npy --model_path models/model_name/best_model.h5 --mask_path path/to/mask.npy --ref_image_path path/to/ref_image.npy --clean_images True
