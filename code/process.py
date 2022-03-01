from typing import Dict

import SimpleITK
import tqdm
import json
from pathlib import Path
import tifffile
import numpy as np

from fastai.vision.all import *
import fastai
import timm
import dill

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from evalutils.io import ImageLoader

from skimage import filters, segmentation, measure


def crop_to_shape(arr, shape, cval=0):
    """Crops a numpy array into the specified shape. If the array was larger, return centered crop. If it was smaller,
    return a larger array with the original data in the center"""
    if arr.ndim != len(shape):
        raise Exception("Array and crop shape dimensions do not match")

    arr_shape = np.array(arr.shape)
    shape = np.array(shape)
    max_shape = np.stack([arr_shape, shape]).max(axis=0)
    output_arr = np.ones(max_shape, dtype=arr.dtype) * cval

    arr_min = ((max_shape - arr_shape) / 2).astype(np.int)
    arr_max = arr_min + arr_shape
    slicer_obj = tuple(slice(idx_min, idx_max, 1) for idx_min, idx_max in zip(arr_min, arr_max))
    output_arr[slicer_obj] = arr

    crop_min = ((max_shape - shape) / 2).astype(np.int)
    crop_max = crop_min + shape
    slicer_obj = tuple(slice(idx_min, idx_max, 1) for idx_min, idx_max in zip(crop_min, crop_max))
    return output_arr[slicer_obj].copy() # Return a copy of the view, so the rest of memory can be GC


def crop_retina(image):
    """Return a square crop of the image centered on the retina.
    This function does the following assumtions:
    - image is an np.array with dimensions [height, weight, channels] or [height, weight]
    - the background of the retinography will have a stark contrast with the rest of the image
    """
    # Check dimensionality of the array is valid
    if image.ndim > 3:
        raise Exception("image has too many dimensions. Max 3")
    elif image.ndim < 2:
        raise Exception("image has too few dimensions. Min 2")
    
    # Rescale image to ensure there will be a black border around (even if the original was already cropped)
    image = crop_to_shape(
        image,
        np.array(image.shape) + np.array([20, 20, 0])[:image.ndim],
        cval=0
    )
    
    # If image is an RGB array, convert to grayscale
    if image.ndim == 3:
        bw_image = np.mean(image, axis=-1)
    else:
        bw_image = image
    
    # Find and apply threshold, to create a binary mask
    thresh = filters.threshold_triangle(bw_image)
    binary = bw_image > thresh
        
    # Label image regions and select the largest one (the retina)
    label_image = measure.label(binary)
    eye_region = sorted(measure.regionprops(label_image), key=lambda p: -p.area)[0]
    
    # Crop around the retina
    y_start, x_start, y_end, x_end = eye_region.bbox
    y_diff = y_end - y_start
    x_diff = x_end - x_start
    if x_diff > y_diff:
        if (y_start + x_diff) <= binary.shape[0]:
            y_end_x_diff = (y_start + x_diff)
            cropped_image = image[y_start:y_end_x_diff, x_start:x_end]
        else:
            y_start_x_diff = (y_end - x_diff) if (y_end - x_diff) > 0 else 0
            cropped_image = image[y_start_x_diff:y_end, x_start:x_end]
    else:
        if (x_start + y_diff) <= binary.shape[1]:
            x_end_y_diff = (x_start + y_diff)
            cropped_image = image[y_start:y_end, x_start:x_end_y_diff]
        else:
            x_start_y_diff = (x_end - y_diff) if (x_end - y_diff) > 0 else 0
            cropped_image = image[y_start:y_end, x_start_y_diff:x_end]

    # Ensure aspect ratio will be square
    max_axis = max(cropped_image.shape)
    if cropped_image.ndim == 3:
        square_crop = (max_axis, max_axis, cropped_image.shape[-1])
    else:
        square_crop = (max_axis, max_axis)
    square_image = crop_to_shape(cropped_image, square_crop)
    return square_image

def equalize_histogram(
        img,
        mean_rgb_vals=np.array([120, 100, 80]),
        std_rgb_vals=np.array([75.40455101, 60.72748057, 46.14914927])
):
    mask = img < 10
    img_masked = np.ma.array(img, mask=mask, fill_value=0)

    if img.ndim == 3:
        equalized_img = (img - img_masked.mean(axis=(0, 1))) / img.std(axis=(0, 1)) * std_rgb_vals + mean_rgb_vals
    elif img.ndim == 2:
        equalized_img = (img - img_masked.mean()) / img.std() * std_rgb_vals[1] + mean_rgb_vals[1]
    else:
        raise Exception("img ndim is neither 2 nor 3")
    equalized_img = (equalized_img * ~mask).clip(0, 255).astype(np.uint8)
    return equalized_img


def crop_and_equalize_fd(img, crop_resize=None, equalize=True):
        retina_arr = crop_retina(np.array(img))
        if equalize:
            retina_arr = equalize_histogram(retina_arr)
        retina_img = Image.fromarray(retina_arr)
        if crop_resize:
            retina_img = retina_img.resize((crop_resize, crop_resize), Image.LANCZOS)
        return retina_img


class DummyLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return str(fname)


    @staticmethod
    def hash_image(image):
        return hash(image)


class airogs_algorithm(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        #self.learn = load_learner('resnetrsRandAug.pkl')
        self.learn1 = load_learner('convnext_base.pkl')
        #self.learn1 = load_learner('efficientnetv2_rw_s.pkl')
        #self.learn2 = load_learner('resnetrsRandAug.pkl')

        self._file_loaders = dict(input_image=DummyLoader())

        self.output_keys = ["multiple-referable-glaucoma-likelihoods", 
                            "multiple-referable-glaucoma-binary",
                            "multiple-ungradability-scores",
                            "multiple-ungradability-binary"]
    
    def load(self):
        for key, file_loader in self._file_loaders.items():
            fltr = (
                self._file_filters[key] if key in self._file_filters else None
            )
            self._cases[key] = self._load_cases(
                folder=Path("/input/images/color-fundus/"),
                file_loader=file_loader,
                file_filter=fltr,
            )

        pass
    
    def combine_dicts(self, dicts):
        out = {}
        for d in dicts:
            for k, v in d.items():
                if k not in out:
                    out[k] = []
                out[k].append(v)
        return out
    
    def process_case(self, *, idx, case):
        # Load and test the image(s) for this case
        if case.path.suffix == '.tiff':
            results = []
            with tifffile.TiffFile(case.path) as stack:
                for page in tqdm.tqdm(stack.pages):
                    input_image_array = page.asarray()
                    results.append(self.predict(input_image_array=input_image_array))
        else:
            input_image = SimpleITK.ReadImage(str(case.path))
            input_image_array = SimpleITK.GetArrayFromImage(input_image)
            results = [self.predict(input_image_array=input_image_array)]
        
        results = self.combine_dicts(results)

        # Test classification output
        if not isinstance(results, dict):
            raise ValueError("Expected a dictionary as output")

        return results

    def predict(self, *, input_image_array: np.ndarray) -> Dict:
        # From here, use the input_image to predict the output
        # We are using a not-so-smart algorithm to predict the output, you'll want to do your model inference here
        im= crop_and_equalize_fd(input_image_array,512)
        im.save('tmp.jpg')
        # Replace starting here
        pred1,_=self.learn1.tta(dl=self.learn1.dls.test_dl(['tmp.jpg']))
        #pred2,_=self.learn2.tta(dl=self.learn2.dls.test_dl(['tmp.jpg']),n=3)
        combined_pred = pred1 
        #(pred1+pred1)/2
        rg_likelihood = float(combined_pred[0][1])
        rg_binary = bool(combined_pred[0][1]>0.5)
        
        ungradability_score = 1-float(combined_pred[0][np.argmax(combined_pred[0])])
        ungradability_binary = bool(float(combined_pred[0][np.argmax(combined_pred[0])])<0.9)
        # to here with your inference algorithm

        out = {
            "multiple-referable-glaucoma-likelihoods": rg_likelihood,
            "multiple-referable-glaucoma-binary": rg_binary,
            "multiple-ungradability-scores": ungradability_score,
            "multiple-ungradability-binary": ungradability_binary
        }

        return out

    def save(self):
        for key in self.output_keys:
            with open(f"/output/{key}.json", "w") as f:
                out = []
                for case_result in self._case_results:
                    out += case_result[key]
                json.dump(out, f)


if __name__ == "__main__":
    airogs_algorithm().process()
