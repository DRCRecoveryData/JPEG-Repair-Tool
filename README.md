# JPEG Repair Tool

![2024-04-24_004544](https://github.com/DRCRecoveryData/JPEG-Repair-Tool/assets/85211068/673f449b-be21-4075-8441-eef37a9f9dc0)

This is a simple GUI tool for repairing JPEG images by shifting MCU and applying auto color enhancements.

## Overview

This tool is built using Python and PyQt5. It provides a user-friendly interface to repair JPEG images corrupted due to encryption or other issues.

## Features

- Select a reference JPEG image and an encrypted folder containing corrupted JPEG images.
- Repair corrupted JPEG images by shifting MCU and applying necessary adjustments.
- Automatically enhance colors of repaired images.
- User-friendly GUI with buttons for easy interaction.

## Requirements

- Python 3.x
- PyQt5
- NumPy
- Pillow (PIL)

## Usage

1. Install the required dependencies.
2. Run the script `jpeg_repair_tool.py`.
3. Use the buttons to select the reference JPEG image and the encrypted folder.
4. Click on "Repair JPEGs" to start the repair process.
5. Optionally, click on "Auto Color" to enhance the colors of repaired images.
6. Monitor the output in the text area for progress and errors.

## How it works

The tool first identifies the MCU shift required for each corrupted JPEG image. Then, it applies the necessary shift and saves the repaired images. Optionally, it applies auto color enhancement to improve the visual quality of the images.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This tool is inspired by the need to recover corrupted JPEG images efficiently.
- Special thanks to the developers of PyQt5, NumPy, and Pillow for their amazing libraries.

