import sys
import os
import re
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTextEdit, QLabel, QLineEdit
from PyQt6.QtCore import Qt
from PIL import Image, ImageOps, ImageEnhance

class JpegToolGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("JPEG Repair Tool")
        self.setGeometry(100, 100, 400, 400)

        self.setStyleSheet('''
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }

            QPushButton:hover {
                background-color: #005F99;
            }

            QTextEdit {
                background-color: #D9E6F5;
                color: black;
                border-radius: 5px;
                padding: 8px;
            }
        ''')

        self.layout = QVBoxLayout()

        # Create buttons
        self.btnSelectReference = QPushButton("Select Reference JPEG")
        self.btnSelectEncryptedFolder = QPushButton("Select Encrypted Folder")
        self.btnRepairJpeg = QPushButton("Repair JPEGs")
        self.btnShiftMcu = QPushButton("Shift MCU")
        self.btnAutoColor = QPushButton("Auto Color")

        # Create text output area
        self.outputText = QTextEdit()
        self.outputText.setReadOnly(True)

        # Create input fields
        self.reference_jpeg_input = QLineEdit()
        self.encrypted_folder_input = QLineEdit()

        # Add widgets to layout
        self.layout.addWidget(QLabel("Reference JPEG:"))
        self.layout.addWidget(self.reference_jpeg_input)
        self.layout.addWidget(self.btnSelectReference)
        self.layout.addWidget(QLabel("Encrypted Folder:"))
        self.layout.addWidget(self.encrypted_folder_input)
        self.layout.addWidget(self.btnSelectEncryptedFolder)
        self.layout.addWidget(self.btnRepairJpeg)
        self.layout.addWidget(self.btnShiftMcu)
        self.layout.addWidget(self.btnAutoColor)
        self.layout.addWidget(self.outputText)

        self.setLayout(self.layout)

        # Connect button actions
        self.btnSelectReference.clicked.connect(self.selectReferenceJPEG)
        self.btnSelectEncryptedFolder.clicked.connect(self.selectEncryptedFolder)
        self.btnRepairJpeg.clicked.connect(self.repairJPEGs)
        self.btnShiftMcu.clicked.connect(self.shift_mcu)
        self.btnAutoColor.clicked.connect(self.autoColorImages)

    def selectReferenceJPEG(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Reference JPEG", "", "JPEG Files (*.jpg);;All Files (*)")

        if fileName:
            self.reference_jpeg_input.setText(fileName)

    def selectEncryptedFolder(self):
        folderName = QFileDialog.getExistingDirectory(self, "Select Encrypted Folder")

        if folderName:
            self.encrypted_folder_input.setText(folderName)

    def repairJPEGs(self):
        reference_jpeg = self.reference_jpeg_input.text().strip()
        encrypted_folder = self.encrypted_folder_input.text().strip()

        if not os.path.exists(reference_jpeg) or not os.path.exists(encrypted_folder):
            self.outputText.append("Please select the reference JPEG and encrypted folder.")
            return

        repaired_folder = os.path.join(encrypted_folder, "Repaired")
        if not os.path.exists(repaired_folder):
            os.makedirs(repaired_folder)

        pattern = re.compile(r".*\.JPG\..{4}$", re.I)

        for encrypted_file in os.listdir(encrypted_folder):
            if pattern.match(encrypted_file):  # Matches files like *.JPG.****
                output_file = os.path.join(repaired_folder, os.path.splitext(encrypted_file)[0].rsplit('.', 1)[0] + '.JPG')
                self.process_jpeg(reference_jpeg, os.path.join(encrypted_folder, encrypted_file), output_file)
                self.outputText.append(f"Processed {encrypted_file} to {output_file}")

        self.outputText.append("Repair complete.")

    def process_jpeg(self, reference_path, encrypted_path, output_path):
        # Load the encrypted JPEG file
        with open(encrypted_path, 'rb') as encrypted_file:
            encrypted_data = encrypted_file.read()

        # Cut the encrypted data from offset 0 to 153605 (inclusive)
        cut_encrypted_data = encrypted_data[:153605]

        # Load the reference JPEG file
        with open(reference_path, 'rb') as reference_file:
            reference_data = reference_file.read()

        # Find the last FFDA marker in the reference data
        ffda_offset = self.find_ffda_offset(reference_data)

        # Cut the reference data from offset 0 to FFDA + 12 bytes
        cut_reference_data = reference_data[:ffda_offset + 12]

        # Merge the cut reference data with the remaining encrypted data after 153605
        repaired_data = cut_reference_data + encrypted_data[153605:]

        # Remove EXIF metadata
        repaired_data = self.remove_exif(repaired_data)

        # Remove 334 bytes from the end of the repaired data
        repaired_data = repaired_data[:-334]

        # Save the repaired data to the output path
        with open(output_path, 'wb') as output_file:
            output_file.write(repaired_data)

    def find_ffda_offset(self, data):
        # Find the last occurrence of FFDA marker in the JPEG data
        ffda_marker = b'\xFF\xDA'
        ffda_offset = data.rfind(ffda_marker)
        if ffda_offset == -1:
            raise ValueError("FFDA marker not found in the reference JPEG file.")
        return ffda_offset

    def remove_exif(self, data):
        # JPEG markers that indicate the start of a segment
        marker_start = b'\xFF'
        app1_marker = b'\xFF\xE1'  # APP1 marker (where EXIF data is stored)

        i = 0
        while i < len(data) - 1:
            # Check if the current byte is a marker start
            if data[i] == 0xFF:
                marker = data[i:i+2]  # Get the marker (2 bytes)

                # If the marker is APP1 (EXIF data), remove the entire segment
                if marker == app1_marker:
                    segment_length = int.from_bytes(data[i+2:i+4], byteorder='big') + 2
                    data = data[:i] + data[i+segment_length:]  # Remove the segment
                    continue  # Continue parsing from the current position

                # Skip other markers (except SOI and EOI)
                if marker not in (b'\xFF\xD8', b'\xFF\xD9'):  # Skip SOI and EOI
                    segment_length = int.from_bytes(data[i+2:i+4], byteorder='big') + 2
                    i += segment_length  # Move to the next segment
                    continue

            i += 1  # Move to the next byte

        return data

    def autoColorImages(self):
        repaired_folder = os.path.join(self.encrypted_folder_input.text().strip(), "Repaired")
        if not os.path.exists(repaired_folder):
            self.outputText.append("Repaired folder not found.")
            return

        jpg_files = [f for f in os.listdir(repaired_folder) if f.lower().endswith(".jpg")]

        if not jpg_files:
            self.outputText.append("No JPG files found in the Repaired folder.")
            return

        for jpg_file in jpg_files:
            image_path = os.path.join(repaired_folder, jpg_file)
            try:
                im = Image.open(image_path)
                im = ImageOps.autocontrast(im, cutoff=1)
                enhancer = ImageEnhance.Sharpness(im).enhance(3)
                im = ImageOps.posterize(im, bits=8)
                enhancer = ImageEnhance.Color(im).enhance(3)
                im.save(image_path)
                self.outputText.append(f"Auto color applied to {jpg_file} and saved.")
            except Exception as e:
                self.outputText.append(f"Error processing image {jpg_file}: {str(e)}")

        self.outputText.append("Auto Color process complete.")

    def shift_mcu(self):
        reference_jpeg = self.reference_jpeg_input.text().strip()
        encrypted_folder = self.encrypted_folder_input.text().strip()

        if not os.path.exists(reference_jpeg) or not os.path.exists(encrypted_folder):
            self.outputText.append("Please select the reference JPEG and encrypted folder.")
            return

        repaired_folder = os.path.join(encrypted_folder, "Repaired")
        if not os.path.exists(repaired_folder):
            self.outputText.append("Repaired folder not found. Run Repair JPEGs first.")
            return

        jpg_files = [f for f in os.listdir(repaired_folder) if f.lower().endswith(".jpg")]

        if not jpg_files:
            self.outputText.append("No JPG files found in the Repaired folder.")
            return

        for jpg_file in jpg_files:
            image_path = os.path.join(repaired_folder, jpg_file)
            try:
                img = Image.open(image_path)
                data = np.array(img, dtype=np.uint8)
            except Exception as e:
                self.outputText.append(f"Error opening image {jpg_file}: {str(e)}")
                continue

            num_good_mcu = self.auto_detect_shift(data)
            if num_good_mcu == 0:
                self.outputText.append(f"No MCU shift needed for {jpg_file}.")
                continue

            width_to_shift = num_good_mcu * 8
            shifted_data = np.hstack((data[:, width_to_shift:], data[:, :width_to_shift]))

            height, width, _ = shifted_data.shape
            while height >= 8 and np.array_equal(shifted_data[height - 8:height, :, :], np.ones((8, width, 3), dtype=np.uint8) * 128):
                height -= 8

            height -= 8
            width -= 16

            cropped_data = shifted_data[:height, :width, :]

            output_img = Image.fromarray(cropped_data)
            output_img.save(image_path, "JPEG")

            self.outputText.append(f"Shifted and saved image {jpg_file} to: {image_path}")

        self.outputText.append("MCU Shift process complete.")

    def auto_detect_shift(self, data):
        block_size = 8
        height, width, _ = data.shape

        while height >= block_size:
            if np.array_equal(data[height - block_size:height, :, :], np.ones((block_size, width, 3), dtype=np.uint8) * 128):
                height -= block_size
            else:
                break
        cropped_data = data[:height, :, :]

        for j in range(0, width, block_size):
            mcu = cropped_data[height - block_size:height, j:j + block_size, :]
            gray_mcu = np.ones((block_size, block_size, 3), dtype=np.uint8) * 128

            diff = np.abs(mcu.astype(int) - gray_mcu.astype(int))
            avg_diff = np.mean(diff)

            if avg_diff < 10:
                return j // block_size

        return width // block_size

def main():
    app = QApplication(sys.argv)
    window = JpegToolGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
