# 🖼️ JPEG Repair Tool

**A User-Friendly GUI Application to Fix Corrupted and Encrypted JPEG Images**

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-538CC9?style=for-the-badge&logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/intro)
[![License](https://img.shields.io/github/license/DRCRecoveryData/JPEG-Repair-Tool?style=for-the-badge&color=green)](https://github.com/DRCRecoveryData/JPEG-Repair-Tool/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/DRCRecoveryData/JPEG-Repair-Tool?label=Latest%20Release&style=for-the-badge&include_prereleases)](https://github.com/DRCRecoveryData/JPEG-Repair-Tool/releases)

---

## 📸 Overview

This application provides a simple, graphical user interface (GUI) built with **Python** and **PyQt6** to effectively repair JPEG images that have been corrupted, often as a result of **ransomware encryption** (like Stop/DJVU) or other file-level damage.

The tool works by intelligently determining and applying the correct **Minimum Coded Unit (MCU) shift** using a reference image, and then offers an optional **auto-color enhancement** to restore image quality.

| Before Repair | After Repair |
| :---: | :---: |
| ![Corrupted JPEG Image](https://github.com/user-attachments/assets/82e97354-ef79-4ef9-81e1-2fc5b8b39a56) | ![Repaired JPEG Image](https://github.com/user-attachments/assets/83c6fe1b-befb-4e07-9e01-c1791315b6cd) |

---

## ✨ Features

* **MCU Shift Repair:** Automatically calculates and applies the necessary MCU shift to repair the header structure of corrupted JPEGs.
* **Batch Processing:** Repair multiple corrupted images within a specified folder using a single reference image.
* **Auto Color Enhancement:** Optional post-repair feature to automatically improve the color, contrast, and brightness of the restored images.
* **Intuitive GUI:** Easy-to-use interface powered by PyQt6.

---

## 🚀 Get Started

### Prerequisites

You need **Python 3.x** installed on your system.

### Installation

1.  **Clone the repository** or [download the latest release](https://github.com/DRCRecoveryData/JPEG-Repair-Tool/releases):
    ```sh
    git clone https://github.com/DRCRecoveryData/JPEG-Repair-Tool.git
    cd JPEG-Repair-Tool
    ```

2.  **Install the required dependencies** (PyQt6, NumPy, Pillow):
    ```sh
    pip install -r requirements.txt
    ```
    *Alternatively, you can install them manually:*
    ```sh
    pip install pyqt6 numpy pillow
    ```

### Usage

1.  **Run the application:**
    ```sh
    python jpeg_repair_tool.py
    ```
2.  **Select Files:**
    * Click the button to select one **known-good reference JPEG** (an uncorrupted image from the same camera/source).
    * Click the button to select the **folder containing the encrypted/corrupted JPEGs**.
3.  **Repair:** Click **"Repair JPEGs"** to start the batch repair process.
4.  **Enhance (Optional):** Click **"Auto Color"** to apply automatic color correction to the newly repaired images.

---

## 🛠️ How It Works

The core of the repair process involves **MCU (Minimum Coded Unit) alignment**. Many forms of file corruption, particularly those from simple encryption methods, often corrupt the file header, causing a shift in the data blocks. The tool leverages a known-good reference file to calculate the correct header offset (MCU shift) and applies this correction to the corrupted files, effectively restoring the image data structure and making the image viewable again.

---

## 🤝 Contributing

We welcome contributions, bug reports, and suggestions!

1.  **Fork** the repository.
2.  **Create** a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  **Commit** your changes (`git commit -m 'Add amazing feature'`).
4.  **Push** to the branch (`git push origin feature/AmazingFeature`).
5.  **Open a Pull Request**.

For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the **GNU General Public License (GPL)**. See the [LICENSE](LICENSE) file for more details.

---

## ✉️ Contact

For professional support, questions, or inquiries, feel free to reach out:

📧 **Email:** [hanaloginstruments@gmail.com](mailto:hanaloginstruments@gmail.com)
🔗 **GitHub:** [DRCRecoveryData](https://github.com/DRCRecoveryData)
