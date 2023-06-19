import os
import glob
import cv2
import numpy as np
from PIL import Image
from typing import Literal
from tqdm import tqdm
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser(description='Resize images in a directory')
parser.add_argument('--input', type=str, help='path to input directory')
parser.add_argument('--output', type=str, help='path to output directory')
parser.add_argument('--size', type=int,nargs='+', help='size of output images')
parser.add_argument('--backend', type=str, default='cv2', help='backend to use for resizing')
parser.add_argument('--extension', type=str, default='jpg', help='extension of images to resize')

def resize(img: np.ndarray, size: tuple[int, int], backend: Literal["cv2", "pil"] = "cv2") -> np.ndarray | Image.Image:
    """
        Resize an image to the specified size using either OpenCV (cv2) or the Python Imaging Library (PIL).

        Parameters:
        -----------
        img : np.ndarray
            The input image as a NumPy array.
        size : tuple
            The desired output size as a (width, height) tuple.
        backend : str, optional (default='cv2')
            The image processing backend to use. Can be either 'cv2' (OpenCV) or 'pil' (Python Imaging Library).

        Returns:
        --------
        np.ndarray
            The resized image as a NumPy array.

        Raises:
        -------
        ValueError:
            If the provided backend is not 'cv2' or 'pil'.

        """

    assert isinstance(size, tuple) and len(size) == 2, "size must be a tuple of length 2"
    assert isinstance(size[0], int) and isinstance(size[1], int), "size dimensions must be integers"

    # Choose the appropriate backend based on the input argument
    if backend.lower() == "cv2":
        # if the image will be shrunk
        if size[0] < img.shape[0] or size[1] < img.shape[1]:
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_LINEAR
        img_resized = cv2.resize(img, size, interpolation=interpolation)
    elif backend.lower() == "pil":
        # Convert the input array to a PIL image object
        img_pil = Image.fromarray(img)
        # Resize the PIL image object
        img_resized_pil = img_pil.resize(size, resample=Image.LANCZOS if size < img.size else Image.BILINEAR)
        # Convert the resized PIL image object back to a NumPy array
        img_resized = np.array(img_resized_pil)
    else:
        raise ValueError(f"Invalid backend '{backend}', must be either 'cv2' or 'pil'")
    return img_resized


def resize_images_in_directory(input_dir: str, output_dir: str, size: tuple[int, int], backend: str = "cv2",
                               extension: Literal["jpg", "jpeg", "png"] = "jpg") -> None:
    """
    Resize all images in a directory to the specified size using either OpenCV (cv2) or the Python Imaging Library (PIL),
    and save the resized images to a new directory.

    Parameters:
    -----------
    input_dir : str
        The path to the input directory containing the images to be resized.
    output_dir : str
        The path to the output directory where the resized images will be saved.
    size : Tuple[int, int]
        The desired output size as a (width, height) tuple.
    backend : str, optional (default='cv2')
        The image processing backend to use. Can be either 'cv2' (OpenCV) or 'pil' (Python Imaging Library).

    Returns:
    --------
    None

    Raises:
    -------
    ValueError:
        If the provided backend is not 'cv2' or 'pil'.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all image file paths in the input directory
    img_paths = glob.glob(os.path.join(input_dir, f"*.{extension}"))
    print(
        f'Resizing all .{extension} images in {input_dir} from '
        f'({cv2.imread(img_paths[0]).shape[0]}, {cv2.imread(img_paths[0]).shape[1]}) to {size} ')
    # Loop through each image, resize it, and save the resized image to the output directory
    for img_path in tqdm(img_paths):
        # Load the image as a NumPy array
        img = cv2.imread(img_path)

        # Resize the image
        img_resized = resize(img, size, backend=backend)

        # Save the resized image to the output directory
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img_resized)


def main():
    """
    Example usage:
    python resize.py --input /path/to/input/directory --output /path/to/output/directory --size 256 256 --backend cv2 --extension jpg
    """
    # get in and out directories as argparser arguments
    args = parser.parse_args()
    in_dir = args.input
    out_dir = args.output
    size = tuple(args.size)
    backend = args.backend
    
    # in_dir = '/home/bmw/Desktop/Projects/gans/GANs/dataset/Hands/Hands/'
    # out_dir = '/home/bmw/Desktop/Projects/resized/'
    resize_images_in_directory(in_dir, out_dir, size, backend=backend)
    print('Done!')

if __name__ == '__main__':
    main()
