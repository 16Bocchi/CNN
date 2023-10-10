from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def resize_and_convert_to_greyscale(input_file, output_file, target_size):
    try:
        # Open the HEIC image
        image = Image.open(input_file)

        # Resize the image to the target size
        image.thumbnail(target_size)

        # Convert the image to greyscale
        greyscale_image = image.convert("L")

        # Save the resulting image
        greyscale_image.save(output_file)

        print("Image resized and converted to greyscale successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    input_file = "input.heic"  # Replace with your input HEIC file
    output_file = "output.jpg"  # Replace with the desired output file path
    target_size = (400, 400)   # Replace with your desired target size (width, height)

    resize_and_convert_to_greyscale(input_file, output_file, target_size)
