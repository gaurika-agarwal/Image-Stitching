import numpy as np
import cv2
import sys
from typing import Tuple, List, Optional

# Constants for feature matching
RATIO = 0.80  # Slightly more lenient ratio
MIN_MATCH = 8  # Reduced minimum match requirement
MIN_MATCH_QUALITY = 0.02  # Adjusted for real-world images
MAX_FEATURES = 10000  # Increased max features
GOOD_MATCH_PERCENT = 0.25  # Increased percentage of best matches

# Constants for image blending
SMOOTHING_WINDOW_SIZE = 800
BLEND_WIDTH = 50  # Width of the blending zone
USE_MULTIBAND = True
PYRAMID_LEVELS = 5
USE_CYLINDRICAL = True
FOV_DEG = 60.0

# Initialize feature detectors
try:
    # Try different feature detectors in order of preference
    try:
        sift = cv2.SIFT_create(MAX_FEATURES)
    except:
        sift = cv2.xfeatures2d.SIFT_create(MAX_FEATURES)
except Exception as e:
    print(f"Warning: SIFT not available. Error: {str(e)}")
    sift = None

try:
    orb = cv2.ORB_create(MAX_FEATURES)
except Exception as e:
    print(f"Warning: ORB not available. Error: {str(e)}")
    orb = None

if sift is None and orb is None:
    raise Exception("No feature detectors available")

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image to improve feature detection
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    return denoised

def cylindrical_warp_image(img: np.ndarray, f: Optional[float] = None) -> np.ndarray:
    """
    Apply cylindrical projection to reduce perspective distortion near edges.
    If focal length `f` is not provided, estimate it from image width and FOV.
    """
    h, w = img.shape[:2]
    if f is None:
        f = w / (2.0 * np.tan(np.deg2rad(FOV_DEG) / 2.0))
    # Coordinates centered
    cx, cy = w / 2.0, h / 2.0
    # Build mapping
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x, y)
    x_shift = (xv - cx) / f
    y_shift = (yv - cy) / f
    denom = np.sqrt(x_shift**2 + 1)
    # Cylindrical projection equations
    map_x = f * np.arctan(x_shift) + cx
    map_y = f * (y_shift / denom) + cy
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped

def detect_and_match_features(img1: np.ndarray, img2: np.ndarray) -> Tuple[List, List, List, List]:
    """
    Detect and match features between two images using available methods
    """
    # Preprocess images
    img1_processed = preprocess_image(img1)
    img2_processed = preprocess_image(img2)
    
    # Scale images if they're too large
    max_dimension = 1500
    scale = 1.0
    if max(img1_processed.shape) > max_dimension:
        scale = max_dimension / max(img1_processed.shape)
        img1_processed = cv2.resize(img1_processed, None, fx=scale, fy=scale)
        img2_processed = cv2.resize(img2_processed, None, fx=scale, fy=scale)

    # Try available detectors
    if sift is not None:
        kp1, des1 = sift.detectAndCompute(img1_processed, None)
        kp2, des2 = sift.detectAndCompute(img2_processed, None)
    elif orb is not None:
        kp1, des1 = orb.detectAndCompute(img1_processed, None)
        kp2, des2 = orb.detectAndCompute(img2_processed, None)
    else:
        raise Exception("No feature detector available")
    
    # Scale keypoints back if image was resized
    if scale != 1.0:
        for kp in kp1:
            kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
        for kp in kp2:
            kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
    
    return kp1, kp2, des1, des2

def match_features(des1: np.ndarray, des2: np.ndarray, use_flann: bool = True) -> List:
    """
    Match features using appropriate matcher based on descriptor type
    """
    if use_flann:
        try:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(np.float32(des1), np.float32(des2), k=2)
            return matches
        except Exception as e:
            print(f"FLANN matching failed, falling back to BFMatcher. Error: {str(e)}")
            use_flann = False
    
    # Use BFMatcher as fallback
    if des1.dtype == np.uint8:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # For binary descriptors like ORB
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # For float descriptors like SIFT
    matches = bf.knnMatch(des1, des2, k=2)
    return matches

def registration(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Enhanced registration function with improved feature matching and error handling
    """
    try:
        # Detect features
        kp1, kp2, des1, des2 = detect_and_match_features(img1, img2)
        
        if des1 is None or des2 is None or len(kp1) < MIN_MATCH or len(kp2) < MIN_MATCH:
            raise Exception(f"Not enough features found: {len(kp1) if kp1 else 0} and {len(kp2) if kp2 else 0}")
        
        # Match features using appropriate matcher
        use_flann = des1.dtype != np.uint8  # Use FLANN for float descriptors (SIFT)
        matches = match_features(des1, des2, use_flann)
        
        # Apply ratio test
        good_points = []
        good_matches = []
        
        for m1, m2 in matches:
            if m1.distance < RATIO * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        
        # Additional quality check
        match_ratio = len(good_points) / len(matches) if matches else 0
        print(f"Found {len(good_points)} good matches with ratio {match_ratio:.3f}")
        if len(good_points) < MIN_MATCH:
            if match_ratio < MIN_MATCH_QUALITY:
                # Try alternative matching approach
                matches = match_features(des1, des2, not use_flann)
                good_points = []
                good_matches = []
                for m1, m2 in matches:
                    if m1.distance < (RATIO + 0.1) * m2.distance:  # More lenient ratio
                        good_points.append((m1.trainIdx, m1.queryIdx))
                        good_matches.append([m1])
                match_ratio = len(good_points) / len(matches) if matches else 0
                if len(good_points) < MIN_MATCH or match_ratio < MIN_MATCH_QUALITY:
                    raise Exception(f"Not enough quality matches found ({len(good_points)} matches, {match_ratio:.2f} ratio)")
        
        # Save matching visualization with only the best matches
        good_matches = sorted(good_matches, key=lambda x: x[0].distance)
        num_best_matches = max(MIN_MATCH, int(len(good_matches) * GOOD_MATCH_PERCENT))
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches[:num_best_matches], None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('static/uploads/matching.jpg', img3)
        
        # Calculate homography using RANSAC with additional refinement
        image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
        
        # Try multiple RANSAC attempts with different parameters
        ransac_reproj_threshold = 5.0
        max_attempts = 3
        best_H = None
        best_mask = None
        best_inlier_count = 0

        for attempt in range(max_attempts):
            H, mask = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 
                                       ransac_reproj_threshold + attempt)
            if H is not None and mask is not None:
                inlier_count = np.sum(mask)
                if inlier_count > best_inlier_count:
                    best_H = H
                    best_mask = mask
                    best_inlier_count = inlier_count

        if best_H is None:
            raise Exception("Could not find homography matrix")

        # Refine homography using all inlier points
        inliers1 = image1_kp[best_mask.ravel() == 1]
        inliers2 = image2_kp[best_mask.ravel() == 1]
        if len(inliers1) >= 4:
            best_H, _ = cv2.findHomography(inliers2, inliers1, cv2.LMEDS)
        
        return best_H, int(best_inlier_count)
        
    except Exception as e:
        print(f"Error in registration: {str(e)}")
        raise

def create_mask(img1: np.ndarray, img2: np.ndarray, version: str) -> np.ndarray:
    """
    Create an improved blending mask with smoother transitions
    """
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2

    # Clamp offset to avoid transition wider than the left image
    offset = min(int(SMOOTHING_WINDOW_SIZE / 2), max(1, width_img1 // 2))
    # Center transition near the seam at the end of img1
    barrier = max(0, width_img1 - offset)
    mask = np.zeros((height_panorama, width_panorama), dtype=np.float32)
    
    # Create smooth transition using sigmoid function instead of linear
    x = np.linspace(-6, 6, 2 * offset, dtype=np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    
    # Compute safe slice bounds
    start = max(0, barrier - offset)
    end = min(width_panorama, barrier + offset)

    if version == 'left_image':
        # Left image: 1 on the left, smooth drop to 0 across [start:end]
        if end > start:
            span = end - start
            mask[:, start:end] = np.tile(sigmoid[::-1][:span], (height_panorama, 1))
        else:
            # Fallback: no transition region, make left image fully 1 up to its width
            start = 0
            end = width_img1
        mask[:, :start] = 1.0
    else:
        # Right image: 0 on the left, smooth rise to 1 across [start:end], then 1 to the right
        if end > start:
            span = end - start
            mask[:, start:end] = np.tile(sigmoid[:span], (height_panorama, 1))
        else:
            # Fallback: no transition region, make right image fully 1 after the seam
            end = width_img1
        mask[:, end:] = 1.0
        
    return cv2.merge([mask, mask, mask]).astype(np.float32)

def build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    g = [img]
    for i in range(1, levels):
        g.append(cv2.pyrDown(g[-1]))
    return g

def build_laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    g = build_gaussian_pyramid(img, levels)
    l = []
    for i in range(levels - 1):
        size = (g[i].shape[1], g[i].shape[0])
        up = cv2.pyrUp(g[i + 1], dstsize=size)
        l.append(g[i] - up)
    l.append(g[-1])
    return l

def pyramid_blend(pan1: np.ndarray, pan2: np.ndarray, mask: np.ndarray, levels: int = PYRAMID_LEVELS) -> np.ndarray:
    """
    Multi-band blending using Laplacian pyramids and a Gaussian mask pyramid.
    `mask` is the weight for `pan1` (3-channel float32 in [0,1]).
    """
    pan1 = pan1.astype(np.float32)
    pan2 = pan2.astype(np.float32)
    mask = mask.astype(np.float32)

    # Ensure dimensions are even to avoid shape mismatch on pyrDown/pyrUp
    h, w = pan1.shape[:2]
    h2, w2 = h - (h % (2 ** (levels - 1))), w - (w % (2 ** (levels - 1)))
    pan1 = pan1[:h2, :w2]
    pan2 = pan2[:h2, :w2]
    mask = mask[:h2, :w2]

    # Build pyramids
    lp1 = build_laplacian_pyramid(pan1, levels)
    lp2 = build_laplacian_pyramid(pan2, levels)
    gpM = build_gaussian_pyramid(mask, levels)

    # Blend each level
    blended_pyr = []
    for L1, L2, GM in zip(lp1, lp2, gpM):
        blended = L1 * GM + L2 * (1.0 - GM)
        blended_pyr.append(blended)

    # Reconstruct
    result = blended_pyr[-1]
    for i in range(levels - 2, -1, -1):
        size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = result + blended_pyr[i]
    return np.clip(result, 0, 255)

def blend_images(panorama1: np.ndarray, panorama2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Blend two images using multi-band blending when enabled; fallback to normalized feathering.
    """
    if USE_MULTIBAND:
        # Use left-image mask as weight for pan1
        return pyramid_blend(panorama1, panorama2, mask1)
    else:
        pan1 = panorama1.astype(np.float32)
        pan2 = panorama2.astype(np.float32)
        m1 = mask1.astype(np.float32)
        m2 = mask2.astype(np.float32)
        numerator = pan1 * m1 + pan2 * m2
        denominator = m1 + m2
        denominator[denominator < 1e-6] = 1.0
        result = numerator / denominator
        return np.clip(result, 0, 255)

 

def ensure_three_channel_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.dtype != np.uint8:
        img = np.uint8(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def stitching(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Enhanced stitching function with improved blending and error handling
    """
    try:
        # Sanitize inputs: ensure 3-channel uint8
        img1 = ensure_three_channel_uint8(img1)
        img2 = ensure_three_channel_uint8(img2)
        
        # Check if images are valid
        if img1 is None or img2 is None:
            raise Exception("One or both input images are invalid")
            
        # Try registration without cylindrical warp first
        try:
            H_raw, inliers_raw = registration(img1, img2)
        except Exception:
            H_raw, inliers_raw = None, 0

        # Try registration with cylindrical warp next
        img1_cyl, img2_cyl = cylindrical_warp_image(img1), cylindrical_warp_image(img2)
        try:
            H_cyl, inliers_cyl = registration(img1_cyl, img2_cyl)
        except Exception:
            H_cyl, inliers_cyl = None, 0

        # Choose the better registration result
        use_cyl = False
        if H_raw is None and H_cyl is None:
            raise Exception("Could not find homography in either raw or cylindrical space")
        elif H_cyl is not None and (inliers_cyl > inliers_raw * 1.15 or H_raw is None):
            H = H_cyl
            img1_used, img2_used = img1_cyl, img2_cyl
            use_cyl = True
            print(f"Using cylindrical registration with inliers {inliers_cyl}")
        else:
            H = H_raw
            img1_used, img2_used = img1, img2
            print(f"Using raw registration with inliers {inliers_raw}")
        
        # Calculate output dimensions
        height_img1 = img1_used.shape[0]
        width_img1 = img1_used.shape[1]
        width_img2 = img2_used.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        # Create panorama and apply masks
        panorama1 = np.zeros((height_panorama, width_panorama, 3), dtype=np.float32)
        mask1 = create_mask(img1_used, img2_used, version='left_image')
        panorama1[0:img1_used.shape[0], 0:img1_used.shape[1]] = img1_used
        # Do not pre-multiply masks before blending when using multi-band (weights applied per-level)
        # Keep images as-is; blending function will handle masks

        mask2 = create_mask(img1_used, img2_used, version='right_image')
        panorama2 = cv2.warpPerspective(img2_used, H, (width_panorama, height_panorama))

        # Blend the images
        result = blend_images(panorama1, panorama2, mask1, mask2)

        # Crop the result to remove black borders
        gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # Increased threshold sensitivity
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (the panorama)
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            # Add small padding to avoid cutting off edges
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(result.shape[1] - x, w + 2*pad)
            h = min(result.shape[0] - y, h + 2*pad)
            result = result[y:y+h, x:x+w]

        return result.astype(np.uint8)
        
    except Exception as e:
        print(f"Error in stitching: {str(e)}")
        raise

# Main execution
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python panorama.py <img1> <img2> [<img3> ...]')
        sys.exit(1)
    imgs = []
    for p in sys.argv[1:]:
        im = cv2.imread(p)
        if im is None:
            print(f'Error reading image: {p}')
            sys.exit(1)
        imgs.append(im)
    try:
        result = imgs[0]
        for i in range(1, len(imgs)):
            result = stitching(result, imgs[i])
        cv2.imwrite('panorama.jpg', result)
        print('Successfully created panorama.jpg')
    except Exception as e:
        print(f'Error creating panorama: {str(e)}')
        sys.exit(1)