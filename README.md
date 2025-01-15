# JPEG Repair Tool

![2024-04-24_004544](https://github.com/DRCRecoveryData/JPEG-Repair-Tool/assets/85211068/673f449b-be21-4075-8441-eef37a9f9dc0)

This is a simple GUI tool for repairing JPEG images by shifting MCU and applying auto color enhancements.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Contact](#contact)

## Overview

This tool is built using Python and PyQt6. It provides a user-friendly interface to repair JPEG images corrupted due to encryption or other issues.

## Features

- Select a reference JPEG image and an encrypted folder containing corrupted JPEG images.
- Repair corrupted JPEG images by shifting MCU and applying necessary adjustments.
- Automatically enhance colors of repaired images.
- User-friendly GUI with buttons for easy interaction.

## Requirements

- Python 3.x
- PyQt6
- NumPy
- Pillow (PIL)

```pip install pyqt6 numpy pillow```

## Installation

To install the JPEG Repair Tool:

1. Download the latest release from the [releases page](https://github.com/DRCRecoveryData/JPEG-Repair-Tool/releases).
2. Extract the contents to a directory.
3. Ensure you have Python installed. You can download it from [python.org](https://www.python.org/).
4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Install the required dependencies.
2. Run the script `jpeg_repair_tool.py`.
3. Use the buttons to select the reference JPEG image and the encrypted folder.
4. Click on "Repair JPEGs" to start the repair process.
5. Optionally, click on "Auto Color" to enhance the colors of repaired images.
6. Monitor the output in the text area for progress and errors.

## How it works

The tool first identifies the MCU shift required for each corrupted JPEG image. Then, it applies the necessary shift and saves the repaired images. Optionally, it applies auto color enhancement to the images.

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

For issues or suggestions, please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License (GPL). See the [LICENSE](LICENSE) file for details.

## References

- [Stop/DJVU Ransomware Description on Bleeping Computer](https://www.bleepingcomputer.com/news/security/djvu-ransomware-updated-to-v91-uses-new-encryption-mode/)
- [Python Programming Language](https://www.python.org/)
- [PyQt6 Library](https://pypi.org/project/PyQt6/)

## Contact

For support or questions, please contact us at [hanaloginstruments@gmail.com](mailto:hanaloginstruments@gmail.com).
