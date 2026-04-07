import cv2
import numpy as np

def denoise_image(image):
    '''Applies Gaussian blur to reduce noise in DIC images.'''
    return cv2.GaussianBlur(image, (5, 5), 0)

def normalize_lighting(image):
    '''Normalizes image lighting using Histogram Equalization.'''
    return cv2.equalizeHist(image)

def align_frames(img1, img2):
    '''
    Aligns img2 to img1 using ECC Maximization to account for camera jitter.
    If correlation fails to converge, returns original img2.
    '''
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 50
    termination_eps = 1e-3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    
    try:
        _, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        aligned_img2 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_img2
    except cv2.error:
        print("ECC alignment failed to converge; using unaligned frame.")
        return img2

def preprocess_sequence(image_sequence):
    '''
    Applies the full preprocessing pipeline:
    1. Denoise
    2. Normalize
    3. Align concurrent frames
    '''
    if not image_sequence:
        return []

    processed = []
    # Process first img
    first_img = normalize_lighting(denoise_image(image_sequence[0]))
    processed.append(first_img)

    for i in range(1, len(image_sequence)):
        curr_img = normalize_lighting(denoise_image(image_sequence[i]))
        # Align to the immediate previous processed frame
        aligned = align_frames(processed[i-1], curr_img)
        processed.append(aligned)

    return processed
