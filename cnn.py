import os
import cv2
import sys
import argparse
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import hog


def create_output_directory(output_dir):
    created = False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        created = True
    if created:
        print(f"Directory created: {output_dir}")
    else:
        print(f"Directory exists: {output_dir}")


def box_image(image):
    # Accept either grayscale or BGR images
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, 50, 150)
    contours_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    # Ensure boxed_image is BGR for drawing colored rectangles
    if len(image.shape) == 2:
        boxed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        boxed_image = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return boxed_image


def extract_colors(image, k=5, sample_size=10000):
    # Ensure color image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))

    # Downsample pixels for KMeans if image is large
    n_pixels = pixels.shape[0]
    if n_pixels > sample_size:
        idx = np.random.choice(n_pixels, sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(sample)
    colors = kmeans.cluster_centers_.astype(int)
    return colors


def adjust_contrast(image, contrast_factor):
    image_float = image.astype(np.float32)
    mean = np.mean(image_float)
    image_contrast = (image_float - mean) * contrast_factor + mean
    image_contrast = np.clip(image_contrast, 0, 255).astype(np.uint8)
    return image_contrast


def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image_sharpened = cv2.filter2D(image, -1, kernel)

    print("Sharpening image...")
    print(f"Sharpened image shape: {image_sharpened.shape}")

    return image_sharpened


def convert_to_grayscale(image):
    # If already grayscale, return as-is
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_image(image):
    if len(image.shape) == 3:  # Color image
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:  # Grayscale image
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def resize_image(image, size):
    print("Resizing image...")
    return cv2.resize(image, size)


def extract_features(image):
    gray_image = convert_to_grayscale(image)
    features, _ = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features


def pixelate_image(image, pixel_size=16):
    height, width = image.shape[:2]
    new_w = max(1, width // pixel_size)
    new_h = max(1, height // pixel_size)
    temp = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def binarize_image(image):
    # Accept color images by converting to grayscale first
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image


def process_image(image_path, output_dir, interactive=False):
    print(f"Processing image: {image_path}")

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image at {image_path}.")
        return

    print(f"Loaded image shape: {image.shape}")

    # Create a new directory for this image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_dir, image_name)
    create_output_directory(image_output_dir)

    # Resizing image
    resized_image = resize_image(image, (640, 640))
    cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_resized.jpg"), resized_image)

    # Sharpening image
    sharpened_image = sharpen_image(resized_image)
    cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_sharpened.jpg"), sharpened_image)

    # Adjusting contrast
    contrasted_image = adjust_contrast(sharpened_image, contrast_factor=1.5)
    cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_contrasted.jpg"), contrasted_image)

    # Denoising image
    print("Denoising image...")
    denoised_image = denoise_image(contrasted_image)
    print("Denoised image shape:", denoised_image.shape)
    cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_denoised.jpg"), denoised_image)

    # Feature Extraction
    print("Extracting features from denoised image...")
    features = extract_features(denoised_image)

    # Save features to a .npy file for later use
    np.save(os.path.join(image_output_dir, f"{image_name}_features.npy"), features)
    print("Features extracted and saved.")

    # **Automatic Boxing**
    print("Boxing image based on contours...")
    boxed_image = box_image(denoised_image)  # Automatically boxed image
    cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_boxed.jpg"), boxed_image)

    # Additional User Input Loop for Custom Boxing and Color Extraction (optional)
    if interactive:
        while True:
            print("Displaying original image for user input...")

            # Let the user select the region of interest (ROI) on the **denoised image** (original look)
            roi = cv2.selectROI("Select region of interest", denoised_image)  # Show denoised (original) image for selection
            x, y, w, h = map(int, roi)  # Convert to integers

            if w == 0 or h == 0:  # If no region selected, exit loop
                print("No region selected, ending input loop.")
                break

            # Crop the selected region from the denoised image
            roi_image = denoised_image[y:y + h, x:x + w]

            # Box the selected region on the already boxed image
            boxed_user_image = boxed_image.copy()
            cv2.rectangle(boxed_user_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box for user input

            # Save the boxed image with user-selected ROI
            cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_boxed_user_roi.jpg"), boxed_user_image)

            # Show the boxed image with user-defined regions
            cv2.imshow("User Boxed Image", boxed_user_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # **Color Extraction on the selected region**
            colors = extract_colors(roi_image, k=5)
            print(f"Extracted colors from user-selected ROI: {colors}")

            # Ask if the user wants to continue
            user_input = input("Do you want to select another region? (y/n): ").lower()
            if user_input != 'y':
                break
    else:
        print("Interactive mode disabled; skipping manual ROI selection.")

    print("Finished processing image.")


    # Boxing image
    print("Boxing image...")
    boxed_image = box_image(denoised_image)
    print("Boxed image shape:", boxed_image.shape)
    cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_boxed.jpg"), boxed_image)

    # Converting to grayscale
    print("Converting to grayscale...")
    grayscale_image = convert_to_grayscale(boxed_image)
    print("Grayscale image shape:", grayscale_image.shape)
    cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_grayscale.jpg"), grayscale_image)

    # Pixelating image
    print("Pixelating image...")
    pixelated_image = pixelate_image(grayscale_image)
    print("Pixelated image shape:", pixelated_image.shape)
    cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_pixelated.jpg"), pixelated_image)

    # Converting to binary
    try:
        print("Converting to binary...")
        binary_image = binarize_image(pixelated_image)

        if binary_image is not None:
            print("Binary image shape:", binary_image.shape)
            cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_binary.jpg"), binary_image)
        else:
            print("Error: No binary image to save.")
    except Exception as e:
        print(f"Error during binary conversion: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process images in a folder.")
    parser.add_argument("--input_folder", default=r"D:\cnn\input_images\flowers", help="Folder with input images")
    parser.add_argument("--output_folder", default=r"D:\cnn\output_images", help="Folder to write outputs")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive ROI selection (requires GUI)")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    interactive = args.interactive

    if not os.path.isdir(input_folder):
        print(f"Error: input_folder does not exist: {input_folder}")
        return

    create_output_directory(output_folder)

    for filename in os.listdir(input_folder):
        print(f"Checking file: {filename}")
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"Found image file: {filename}")
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, output_folder, interactive=interactive)
        else:
            print(f"Skipping non-image file: {filename}")


if __name__ == '__main__':
    main()
