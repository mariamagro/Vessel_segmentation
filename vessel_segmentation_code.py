# -*- coding: utf-8 -*-
"""
Last updated on 20 October 2024

@author: Marina Gómez Rey & María Ángeles Magro Garrote
"""
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage.filters import gaussian, threshold_otsu, threshold_local
from skimage.exposure import equalize_adapthist
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from skimage.morphology import remove_small_objects, binary_dilation
from scipy.ndimage import median_filter, binary_erosion
from pathlib import Path


# --- PARAMETERS ---
method = 'threshold' #'clustering' or 'threshold'
# ------------------


# Function to load image and mask paths
def get_im_paths(im_dir, mk_dir):
    data_dir = Path('.')
    im_dir = data_dir / im_dir
    mk_dir = data_dir / mk_dir
    im_paths = sorted([f for f in im_dir.glob('*.png') if f.is_file()])
    mk_paths = sorted([f for f in mk_dir.glob('*.png') if f.is_file()])

    print(f"Number of train images: {len(im_paths)}")
    print(f"Number of train masks: {len(mk_paths)}")

    return im_paths, mk_paths

def plot_N_images(im_list, mk_list):
    """
    Plot a list of of images and their masks below.
    The lists should be the same size.
    The lists should be sort to see the images properly

    Parameters
    ----------
    im_list : (list) list of of images
    mk_list : (list) list of of corresponding masks

    """
    N = len(im_list)
    fig, ax = plt.subplots(2, N, figsize=(20, 10))
    for i in range(N):
        ax[0][i].imshow(im_list[i])
        ax[0][i].set_axis_off()
        ax[1][i].imshow(mk_list[i])
        ax[1][i].set_axis_off()
    fig.tight_layout()
    plt.show()

# CLAHE contrast enhancement function
def clahe(im, retina_mask):
    im = (im - im.min()) / (im.max() - im.min())
    contrasted_image = np.zeros_like(im)
    contrasted_image[retina_mask] = equalize_adapthist(im[retina_mask], clip_limit=0.005, kernel_size=105)
    return contrasted_image


# Function to generate retina mask using Otsu thresholding
def getting_retina_mask(img):
    blurred_image = gaussian(img, sigma=2)
    thresh_mask = threshold_otsu(blurred_image)
    retina_mask = blurred_image > thresh_mask
    return retina_mask


# k-means segmentation function
def kmeans_segmentation(im, retina_mask):
    # We want to cluster only the part of the image inside of the retina mask
    im_flat = im[retina_mask].reshape(-1, 1)

    # We apply a k-menas clustering with 4 clusters (our best result) to the image
    kmeans = KMeans(n_clusters=4, random_state=42).fit(im_flat)

    # Initialize empty mask to store the clusters
    clustered = np.zeros_like(retina_mask, dtype=int)
    clustered[retina_mask] = kmeans.labels_

    # Return the clustered mask and the cluster centers
    return clustered, kmeans.cluster_centers_


# Threshold-based segmentation
def th_based_segmentation(im, indx):
    # Getting retina mask to extract the background from the retina
    retina_mask = getting_retina_mask(im)

    # Apply binary erosion
    structure = np.ones((10, 10), dtype=int)
    retina_mask = binary_erosion(retina_mask, structure=structure)

    # Apply median filter normalization on the grayscale image by subtracting 25 to the pixels of the image
    # so that the vessels can be seen better and some parts such as the central circle of the eye affect less
    background = median_filter(im, size=25)
    im = im - background

    # Contrast enhancement using CLAHE
    im = clahe(im, retina_mask)

    # We apply the local thresholding with the parameters tuned, both the block size and the offset
    block_size = 101
    local_thresh = np.zeros_like(im)
    local_thresh[retina_mask] = threshold_local(im[retina_mask], block_size, method='median', offset=0)

    # Generate predicted mask based on thresholding, the threshold was also tuned to obtain the best result
    predicted_mask = np.zeros_like(im, dtype=bool)
    predicted_mask[retina_mask] = im[retina_mask] < 0.9 * local_thresh[retina_mask]
    predicted_mask = predicted_mask & retina_mask

    return predicted_mask, retina_mask


# Main function that manages the two types of approaches, the predetermined method
# is the threshold as it provides better results
def segmentation_evaluation(train_im, train_mk):
    # To store the scores, initially all 0s
    scores = np.zeros(len(train_im))

    # We will make the technique to every image
    for indx, img in enumerate(train_im):
        mk = img_as_float(train_mk[indx])  # Ground truth mask

        if method == 'clustering':
            # Apply the preprocessing, putting it as float and grayscale
            img = img_as_float(img)
            img = rgb2gray(img)
            mk = img_as_float(train_mk[indx])

            # Get retina mask
            retina_mask = getting_retina_mask(img)

            # Apply CLAHE on the image within the retina mask
            clahe_image = clahe(img, retina_mask)

            # Apply k-means segmentation on the CLAHE image with the enhanced contrast to obtain better results
            pred_mask, cluster_centers = kmeans_segmentation(clahe_image, retina_mask)

            # Initialize variable to track the best iou for each image
            best_iou = 0

            # Ground truth mask
            gt_mask = mk > 0.5

            # The number of clusters that worked best for us was 4 so we define define it directly here
            k = 4

            # Compare each cluster layer with ground truth
            for i in range(k):
                current_layer = (pred_mask == i)  # Binary mask for the current cluster

                # Post-process the mask: remove small objects (with tuned min_size) and dilation
                cleaned_mask = remove_small_objects(current_layer.astype(bool), min_size=10)
                cleaned_mask = binary_dilation(cleaned_mask)

                # Calculate IoU score after post-processing
                iou_score = jaccard_score(gt_mask.flatten(), cleaned_mask.flatten())
                print(f"Image {indx}, Cluster {i}, IoU={iou_score:.2f}")

                # Track the best layer based on IoU score
                if iou_score > best_iou:
                    best_iou = iou_score

            # Store the best IoU score for this image
            scores[indx] = best_iou
            print(f"Image {indx}, Best IoU={best_iou:.2f}")

        elif method == 'threshold':
            # Extract the green channel for thresholding method as it is the one where the vessels are best seen
            img_green = img[:, :, 1]
            # The image is converted to float also
            img_green = img_as_float(img_green)

            # Perform threshold-based segmentation calling the function
            pred_mask, retina_mask = th_based_segmentation(img_green, indx)

            # Apply remove small objects with tuned min_size as post-processing
            cleaned_mask = remove_small_objects(pred_mask.astype(bool), min_size=10)

            # Evaluate segmentation with IoU
            gt_mask = mk > 0.5
            iou_score = jaccard_score(gt_mask.flatten(), cleaned_mask.flatten())
            scores[indx] = iou_score
            print(f"Image {indx}, IoU={iou_score: .2f}")

        else:
            raise ValueError("Invalid method. Choose 'threshold' or 'clustering'.")

    return np.average(scores)


# Load images and masks
train_im_paths, train_mk_paths = get_im_paths('Data/image', 'Data/mask')
train_im = [imread(path) for path in train_im_paths]
train_mk = [imread(path) for path in train_mk_paths]


average_IoU = segmentation_evaluation(train_im, train_mk)
print(f"Average IoU={average_IoU: .2f}")
