#!/usr/bin/env python3
import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def hex_to_bytes(hex_string: str) -> bytes | None:
    """Converts a hexadecimal string to bytes

    Args:
        hex_string (str): Hexadecimal string

    Returns:
        bytes | None: Byte representation or None if conversion fails
    """
    try:
        return bytes.fromhex(hex_string)
    except ValueError as e:
        logger.error(f"Error in hexadecimal conversion: {e}")
        return None


def is_jpeg_end(data: bytes) -> bool:
    """Veririfies if data ends with JPEG end marker

    Args:
        data (bytes): Byte data to check

    Returns:
        bool: True if ends with JPEG end marker, False otherwise
    """
    if len(data) < 2:
        return False
    return data[-2:] == b'\xFF\xD9'


def extract_images_from_csv(csv_file: str, output_dir: str = 'extracted_images'):
    """Extracts all JPG images from a CSV file

    Args:
        csv_file (str): Path to the CSV file
        output_dir (str, optional): Output directory. Defaults to 'extracted_images'.
    """

    # Create output direcoty if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    images_saved = 0
    rows_processed = 0

    # List to maintain row order
    all_rows = []

    logger.info(f"Processing file: {csv_file}")
    logger.info(f"Output directory: {output_path.absolute()}\n")

    try:
        # Read all rows first
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) >= 4:
                    all_rows.append(row)

        logger.info(f"Total rows read: {len(all_rows)}\n")

        # Find all images by grouping consecutive rows with type 18
        i = 0
        while i < len(all_rows):
            row = all_rows[i]

            timestamp = row[0].strip() if len(row) > 0 else ""
            record_id = row[1].strip() if len(row) > 1 else ""
            record_type = row[2].strip() if len(row) > 2 else ""

            # Find the start of an image (type 18, segment 00 or with marker FFD8FF)
            if record_type == '18' and len(row) > 4:
                segment_num = row[3].strip()
                hex_data = row[4].strip()

                # Check if segment number is '00' or hex data starts with JPEG start marker
                if segment_num == '00' or hex_data.upper().startswith('FFD8FF'):
                    # Start new image
                    image_segments = []
                    image_timestamp = timestamp
                    image_record_id = record_id

                    # Group all consecutive segments with the same record_id
                    j = i
                    while j < len(all_rows):
                        curr_row = all_rows[j]

                        if len(curr_row) < 4:
                            break

                        curr_record_id = curr_row[1].strip()
                        curr_record_type = curr_row[2].strip()

                        # COntinue only if type 18 and same record_id
                        if curr_record_type == '18' and curr_record_id == image_record_id:
                            curr_segment_num = curr_row[3].strip()
                            curr_hex_data = curr_row[4].strip() if len(
                                curr_row) > 4 else ""

                            if len(curr_hex_data) > 50:
                                curr_image_data = hex_to_bytes(curr_hex_data)
                                if curr_image_data:
                                    image_segments.append(
                                        (curr_segment_num, curr_image_data))
                                    logger.info(
                                        f"  Segment {curr_segment_num} collected ({len(curr_image_data)} bytes)")

                            j += 1
                        else:
                            break

                    # Reconctruct the complete image
                    if image_segments:
                        # Order by segment number
                        image_segments.sort(key=lambda x: x[0])

                        # COncatenate all segments
                        complete_image = b''.join(
                            [seg_data for _, seg_data in image_segments])

                        # Verify that it starts with JPEG marker
                        if complete_image[0:2] == b'\xFF\xD8':
                            # COnvert timestamp from dd-mm-yyyy hh:mm:ss to mm-dd-yyyy_hh-mm-ss
                            try:
                                date_part, time_part = image_timestamp.split(
                                    ' ')
                                day, month, year = date_part.split('-')
                                formatted_timestamp = f"{month}-{day}-{year}_{time_part.replace(':', '-')}"
                            except:
                                # If parsing fails, use original timestamp
                                formatted_timestamp = image_timestamp.replace(
                                    ' ', '_').replace(':', '-')

                            filename = f"img_{formatted_timestamp}_{image_record_id}.jpg"
                            filepath = output_path / filename

                            # Save image
                            with open(filepath, 'wb') as img_file:
                                img_file.write(complete_image)

                            images_saved += 1

                            # Image info
                            has_end_marker = is_jpeg_end(complete_image)
                            marker_status = "✓" if has_end_marker else "⚠"

                            logger.info(f"\n✓ Saved: {filename}")
                            logger.info(
                                f"  - Dimensions: {len(complete_image):,} bytes")
                            logger.info(f"  - Segments: {len(image_segments)}")
                            logger.info(
                                f"  - Marker end: {marker_status} {'OK' if has_end_marker else 'Missing'}\n")
                        else:
                            logger.info(
                                f"\n✗ Image {image_timestamp} does not have valid JPEG marker\n")

                    # Skip all the rows we have processed
                    i = j
                    continue

            i += 1

        # Statistiche finali
        # Final stats
        logger.info(f"{'='*60}")
        logger.info(f"Processing complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Rows processes: {len(all_rows)}")
        logger.info(f"Images saved: {images_saved}")
        logger.info(f"Output directory: {output_path.absolute()}")
        logger.info(f"{'='*60}")

    except FileNotFoundError:
        logger.error(f"File '{csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error while processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser('JPG_extractor',
                                     description='Extracts JPG images from a CSV file.',
                                     epilog='example: python extract_images.py data.csv output_dir')
    parser.add_argument('csv_file', help='Input CSV file containing image data')
    parser.add_argument('output_dir', nargs='?', default='extracted_images',
                        help='Output directory for extracted images (default: extracted_images)')
    args = parser.parse_args()
    csv_file = args.csv_file
    output_dir = args.output_dir

    extract_images_from_csv(csv_file, output_dir)


if __name__ == '__main__':
    main()
