# Vessel_segmentation

This project was made in collaboration with [mariamagro](https://github.com/mariamagro), for the Computer vision subject.

This project performs image segmentation using either a thresholding or clustering method. It uses several image processing techniques provided by the `scikit-image` and `scikit-learn` libraries, along with metrics to evaluate the segmentation quality.

## File Structure

- **`vessel_segmentation_code.py`**: The main script that loads images, applies segmentation techniques, and evaluates the results.

## Segmentation Methods

The script supports two methods for image segmentation:

1. **Thresholding**:
   - Utilizes global and local thresholding methods like `Otsu` and adaptive thresholding.
   
2. **Clustering**:
   - Uses the `KMeans` clustering algorithm from `scikit-learn` to segment images.

The choice of the method can be selected by modifying the `method` parameter in the script.
