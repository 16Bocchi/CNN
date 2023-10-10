import os
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np
import csv

register_heif_opener()


def convert_heic_to_jpeg(input_dir, output_dir):
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List all HEIC files in the input directory
        heic_files = [f for f in os.listdir(
            input_dir) if f.lower().endswith('.heic')]

        for heic_file in heic_files:
            # Construct the full paths for input and output files
            input_file = os.path.join(input_dir, heic_file)
            output_file = os.path.join(
                output_dir, os.path.splitext(heic_file)[0] + '.jpg')

            # Open the HEIC image using Pillow and save it as JPEG
            img = Image.open(input_file)
            img.save(output_file, 'JPEG')

            print(f"Converted: {heic_file} to {output_file}")

        print("Batch conversion completed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def resize_crop_and_convert_to_greyscale(input_dir, output_dir, target_size, crop_box):
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List all JPEG files in the input directory
        jpeg_files = [f for f in os.listdir(
            input_dir) if f.lower().endswith('.jpg')]

        for jpeg_file in jpeg_files:
            # Construct the full paths for input and output files
            input_file = os.path.join(input_dir, jpeg_file)
            output_file = os.path.join(output_dir, jpeg_file)

            # Open the JPEG image using Pillow
            img = Image.open(input_file)

            # Resize the image to the target size
            img.thumbnail(target_size)

            # Crop the image using the specified box (left, upper, right, lower)
            img = img.crop(crop_box)

            # Convert the image to greyscale
            greyscale_image = img.convert("L")

            # Save the resulting image
            greyscale_image.save(output_file)

            print(f"Processed: {jpeg_file}")

        print("Batch processing completed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def convert_to_csv(input_dir, output_dir):
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List all JPEG files in the input directory
        jpeg_files = [f for f in os.listdir(
            input_dir) if f.lower().endswith('.jpg')]

        for jpeg_file in jpeg_files:
            # Construct the full paths for input and output files
            input_file = os.path.join(input_dir, jpeg_file)
            output_file = os.path.join(
                output_dir, os.path.splitext(jpeg_file)[0] + '.csv')

            img = Image.open(input_file)
            value = np.asarray(img.getdata(), dtype=int).reshape(
                (img.size[1], img.size[0]))

            value = value.flatten()
            with open(output_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(value)
            print(f"Converted: {jpeg_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    input_dir = "./inputs"  # Replace with the input directory containing HEIC files
    temp_dir = "./temp"    # Temporary directory for converted JPEG files
    # Replace with the output directory to save processed images
    output_dir = "./processed"
    output_csv = "./csv"
    # Replace with your desired target size (width, height)
    target_size = (400, 400)
    # Specify the crop box as (left, upper, right, lower)
    crop_box = (50, 150, 250, 250)

    # convert_heic_to_jpeg(input_dir, temp_dir)
    resize_crop_and_convert_to_greyscale(
        temp_dir, output_dir, target_size, crop_box)
    convert_to_csv(output_dir, output_csv)
