#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Use the non-interactive 'Agg' backend
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import logging
from pathlib import Path

PROGRESS = 15
logging.addLevelName(PROGRESS, "PROGRESS")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def extract_all_ir_heatmaps(input_file: str, output_dir: str) -> list[tuple[str, str, int]]:
    """Extract all IR thermal images from local file and generate heatmaps.

    Args:
        input_file (str): The input data file
        output_dir (str): The output directory for heatmaps

    Returns:
        list[tuple[str, str, int]]: List of tuples with (timestamp, serial, number of pixels)
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
        return []
    except Exception as e:
        logger.error(f"Unable to read file: {e}")
        return []

    rows = []
    reader = csv.reader(content.splitlines(), delimiter=';')
    for row in reader:
        if len(row) >= 4:
            rows.append(row)
    n_rows = len(rows)
    ir_images = []
    image_count = 0

    for i, row in enumerate(rows):
        
        logger.log(PROGRESS, f"Processing line {i+1}/{n_rows} ({(i+1)/n_rows*100:.2f}%)")
        
        ts = row[0].strip()
        serial = row[1].strip() if len(row) > 1 else "unknown"
        rec_type = row[2].strip()

        # Look for record type 17 (IR thermal data)
        if rec_type == "17" and len(row) > 3:
            try:
                # Extract all numeric values after the first 3 fields
                temp_values = []
                for value in row[3:]:
                    try:
                        temp_values.append(float(value) / 100.0)
                    except:
                        break

                # If we have at least 100 values, likely valid thermal data
                if len(temp_values) >= 100:
                    image_count += 1
                    success = generate_ir_heatmap(
                        temp_values,
                        ts,
                        serial,
                        output_dir,
                        image_count
                    )
                    if success:
                        ir_images.append((ts, serial, len(temp_values)))
            except Exception as e:
                logger.error(f"Unable to process line: {e}")
                continue

    return ir_images


def generate_ir_heatmap(temp_values: list[float], timestamp: str, serial: str, output_dir: str, img_number: int) -> bool:
    """Generate and save an IR heatmap image from temperature values.

    Args:
        temp_values (list[float]): List of temperature values
        timestamp (str): Timestamp string
        serial (str): Serial number
        output_dir (str): Output directory
        img_number (int): Image number for filename

    Returns:
        bool: Whether the image was saved successfully
    """
    # Array has dimensions 24x32 = 768 pixels
    image_height = 24
    image_width = 32

    if len(temp_values) < image_height * image_width:
        # Pad if necessary
        padded_values = temp_values + \
            [0.0] * (image_height * image_width - len(temp_values))
        temp_values = padded_values[:image_height * image_width]
    elif len(temp_values) > image_height * image_width:
        # Extract only the first 768 values
        temp_values = temp_values[:image_height * image_width]

    data_array = np.array(temp_values)

    try:
        image_data = data_array.reshape((image_height, image_width))
    except ValueError:
        # If not 768 values, try to find suitable dimensions
        total_pixels = len(temp_values)
        for h in range(1, int(np.sqrt(total_pixels)) + 1):
            if total_pixels % h == 0:
                image_height = h
                image_width = total_pixels // h
                if image_width >= image_height and image_width <= 100:
                    break

        try:
            image_data = data_array.reshape((image_height, image_width))
        except:
            logger.error(
                f"Unable to resize array with {len(temp_values)} items")
            return False

    # Rotate the image 90 degrees counter-clockwise
    image_data_rotated = np.rot90(image_data, 1)

    # Generate heatmap
    # fig = plt.figure(figsize=(image_height / 2, image_width / 2))
    fig, ax = plt.subplots(1, 1, figsize=(image_height / 2, image_width / 2))
    img = ax.imshow(image_data_rotated, cmap='inferno',
                    interpolation='nearest')
    ax.set_title(f'IR Image - {serial} - {timestamp}')
    fig.colorbar(img, label='Temperature (°C)')
    ax.axis('off')

    # Formatta il nome del file
    # File format: <IMG_NUMBER>_<SERIAL>_IR_DD-MM-YYYY_HH-MM-SS.jpg
    try:
        date, time = timestamp.split(" ")
        dd, mm, yy = date.split("-")
        formatted_ts = f"{dd}-{mm}-{yy}_{time.replace(':', '-')}"
    except:
        formatted_ts = timestamp.replace(" ", "_").replace(":", "-")

    filename = f"{img_number:04d}_{serial}_IR_{formatted_ts}.jpg"
    path = Path(output_dir) / filename

    try:
        fig.savefig(path, bbox_inches='tight', dpi=100)
        # plt.close()
        return True
    except Exception as e:
        logger.error(f"Unable to save {filename}: {e}")
        # plt.close()
        return False


def main():
    description = """\
This script extracts all IR thermal images (record type 17)
from the local file and generates the corresponding heatmaps.
"""
    example = """\
example:
    python3 extract_ir_heatmaps_local.py data.txt ir_heatmaps
"""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=example,
    )
    parser.add_argument("input_file", help="Input data file (e.g., data.txt)")
    parser.add_argument("output_dir", nargs='?',
                        default="ir_heatmaps", help="Output directory for heatmaps")
    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    Path(output_dir).mkdir(exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}\n")

    logger.info('Extracting all IR thermal images from local file...\n')

    ir_images = extract_all_ir_heatmaps(input_file, output_dir)

    logger.info(f"\n{'='*60}")
    logger.info(f"Final results")
    logger.info(f"{'='*60}")
    logger.info(f"Number of extracted images {len(ir_images)}")
    logger.info(f"Output directory: {output_dir}")

    if ir_images:
        logger.info(
            f"First image: {ir_images[0][0]} (Serial: {ir_images[0][1]})")
        logger.info(
            f"Last image: {ir_images[-1][0]} (Serial: {ir_images[-1][1]})")
    else:
        logger.error("No IR images found in the file.")

    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
