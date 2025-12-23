import argparse
import logging
import csv
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def parse_line(row: list[str]) -> list:
    timestamp_fmt = '%d-%m-%Y %H:%M:%S'
    timestamp = datetime.datetime.strptime(row[0].strip(), timestamp_fmt)
    o3 = int(row[4])
    pm1 = int(row[5])/1000
    pm2_5 = int(row[6])/1000
    pm10 = int(row[7])/1000
    co2 = int(row[8])
    air_t = int(row[9])/100
    air_rh = int(row[10])/100
    leaf_t = int(row[11])/100
    v_batt = int(row[12])
    return [timestamp, o3, pm1, pm2_5, pm10, co2, air_t, air_rh, leaf_t, v_batt]
    

def process_data(input_filepath: str, output_filepath: str) -> bool:
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {input_filepath}")
        return []
    except Exception as e:
        logger.error(f"Unable to read file: {e}")
        return []

    rows = []
    reader = csv.reader(content.splitlines(), delimiter=';')
    for row in reader:
        if len(row) >= 4:
            rows.append(row)

    parsed_rows = []
    for row in rows:
        record_type = row[2].strip()
        if record_type == '19' and len(row) > 12:
            parsed_rows.append(parse_line(row))
            
    csv_header = [
        "timestamp",
        "o3_D.N.",
        "pm1_ug_m3",
        "pm2_5_ug_m3",
        "pm10_ug_m3",
        "co2_ppm",
        "air_temp_c",
        "air_rh_percent",
        "leaf_temp_c",
        "v_batt_mv",
    ]
    try:
        with open(output_filepath, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(parsed_rows)
    except Exception as e:
        logger.error(f"Unable to write CSV file: {e}")
        return False
    logger.info(f"Successfully wrote CSV data to {output_filepath}")
    return True


def main():
    description = """\
This script extracts all sensor data (record type 19)
from a local file and generates a CSV with all converted data.
"""
    example = """\
example:
    python3 extract_data.py data.txt data.csv
"""
    parser = argparse.ArgumentParser(
        'extract_sensor_data',
        description=description,
        epilog=example,
    )
    parser.add_argument("input_file", help="Input data file (e.g., data.txt)")
    parser.add_argument("output_file", nargs='?',
                        default="data.csv", help="Output filename for CSV data")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    process_data(input_file, output_file)


if __name__ == "__main__":
    main()
