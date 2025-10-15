import sys
import os
import re
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QTextEdit, QLabel, QLineEdit, QProgressBar, QMessageBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class JpegToolGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("JPEG Repair Tool")
        self.setGeometry(100, 100, 500, 600)

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

        self.reference_jpeg_input = QLineEdit()
        self.encrypted_folder_input = QLineEdit()

        self.btnSelectReference = QPushButton("Select Reference JPEG")
        self.btnSelectEncryptedFolder = QPushButton("Select Encrypted Folder")
        self.btnRepairJpeg = QPushButton("Repair JPEGs")
        self.btnShiftMcu = QPushButton("Shift MCU")
        self.btnAutoColor = QPushButton("Auto Color")

        self.outputText = QTextEdit()
        self.outputText.setReadOnly(True)

        self.progressBar = QProgressBar()
        self.imagePreview = QLabel()
        self.imagePreview.setAlignment(Qt.AlignmentFlag.AlignCenter)

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
        self.layout.addWidget(QLabel("Process Output:"))
        self.layout.addWidget(self.outputText)
        self.layout.addWidget(QLabel("Progress:"))
        self.layout.addWidget(self.progressBar)
        self.layout.addWidget(QLabel("Preview:"))
        self.layout.addWidget(self.imagePreview)

        self.setLayout(self.layout)

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
            QMessageBox.critical(self, "Error", "Please select the reference JPEG and encrypted folder.")
            return

        repaired_folder = os.path.join(encrypted_folder, "Repaired")
        os.makedirs(repaired_folder, exist_ok=True)

        pattern = re.compile(r".*\.JPG\..{4}$", re.I)
        encrypted_files = [f for f in os.listdir(encrypted_folder) if pattern.match(f)]

        self.progressBar.setMaximum(len(encrypted_files))
        self.progressBar.setValue(0)

        for i, encrypted_file in enumerate(encrypted_files):
            output_file = os.path.join(repaired_folder, os.path.splitext(encrypted_file)[0].rsplit('.', 1)[0] + '.JPG')
            try:
                self.process_jpeg(reference_jpeg, os.path.join(encrypted_folder, encrypted_file), output_file)
                self.outputText.append(f"Processed {encrypted_file} to {output_file}")
            except Exception as e:
                self.outputText.append(f"Error processing {encrypted_file}: {str(e)}")
            self.progressBar.setValue(i + 1)

        self.outputText.append("Repair complete.")

    def process_jpeg(self, reference_path, encrypted_path, output_path):
        with open(encrypted_path, 'rb') as encrypted_file:
            encrypted_data = encrypted_file.read()

        cut_encrypted_data = encrypted_data[:153605]

        with open(reference_path, 'rb') as reference_file:
            reference_data = reference_file.read()

        ffda_offset = self.find_ffda_offset(reference_data)
        cut_reference_data = reference_data[:ffda_offset + 12]

        repaired_data = cut_reference_data + encrypted_data[153605:]
        repaired_data = self.remove_exif(repaired_data)
        repaired_data = repaired_data[:-334]

        with open(output_path, 'wb') as output_file:
            output_file.write(repaired_data)

    def find_ffda_offset(self, data):
        ffda_marker = b'\xFF\xDA'
        ffda_offset = data.rfind(ffda_marker)
        if ffda_offset == -1:
            raise ValueError("FFDA marker not found in reference JPEG.")
        return ffda_offset

    def remove_exif(self, data):
        i = 0
        while i < len(data) - 1:
            if data[i] == 0xFF:
                marker = data[i:i+2]
                if marker == b'\xFF\xE1':
                    length = int.from_bytes(data[i+2:i+4], 'big') + 2
                    data = data[:i] + data[i+length:]
                    continue
                elif marker not in (b'\xFF\xD8', b'\xFF\xD9'):
                    length = int.from_bytes(data[i+2:i+4], 'big') + 2
                    i += length
                    continue
            i += 1
        return data

    def autoColorImages(self):
        repaired_folder = os.path.join(self.encrypted_folder_input.text().strip(), "Repaired")
        if not os.path.exists(repaired_folder):
            QMessageBox.warning(self, "Warning", "Repaired folder not found.")
            return

        jpg_files = [f for f in os.listdir(repaired_folder) if f.lower().endswith(".jpg")]

        self.progressBar.setMaximum(len(jpg_files))
        self.progressBar.setValue(0)

        for i, jpg_file in enumerate(jpg_files):
            image_path = os.path.join(repaired_folder, jpg_file)
            try:
                im = Image.open(image_path)
                im = ImageOps.autocontrast(im, cutoff=1)
                im = ImageEnhance.Sharpness(im).enhance(3)
                im = ImageOps.posterize(im, bits=8)
                im = ImageEnhance.Color(im).enhance(3)
                im.save(image_path)
                self.outputText.append(f"Auto color applied to {jpg_file}")
                if i == 0:
                    pixmap = QPixmap(image_path)
                    self.imagePreview.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
            except Exception as e:
                self.outputText.append(f"Error processing {jpg_file}: {str(e)}")
            self.progressBar.setValue(i + 1)

        self.outputText.append("Auto Color process complete.")

    def shift_mcu(self):
        reference_jpeg = self.reference_jpeg_input.text().strip()
        encrypted_folder = self.encrypted_folder_input.text().strip()
        repaired_folder = os.path.join(encrypted_folder, "Repaired")

        if not os.path.exists(reference_jpeg) or not os.path.exists(repaired_folder):
            QMessageBox.warning(self, "Warning", "Check paths or run Repair JPEGs first.")
            return

        jpg_files = [f for f in os.listdir(repaired_folder) if f.lower().endswith(".jpg")]
        self.progressBar.setMaximum(len(jpg_files))
        self.progressBar.setValue(0)

        for i, jpg_file in enumerate(jpg_files):
            image_path = os.path.join(repaired_folder, jpg_file)
            try:
                img = Image.open(image_path)
                data = np.array(img, dtype=np.uint8)
                num_good_mcu = self.auto_detect_shift(data)
                if num_good_mcu == 0:
                    self.outputText.append(f"No shift needed for {jpg_file}")
                    continue
                width_to_shift = num_good_mcu * 8
                shifted = np.hstack((data[:, width_to_shift:], data[:, :width_to_shift]))

                height, width, _ = shifted.shape
                while height >= 8 and np.array_equal(shifted[height-8:height], np.ones((8, width, 3), dtype=np.uint8) * 128):
                    height -= 8
                height -= 8
                width -= 16
                cropped = shifted[:height, :width]
                Image.fromarray(cropped).save(image_path)
                self.outputText.append(f"Shifted image {jpg_file}")
            except Exception as e:
                self.outputText.append(f"Error shifting {jpg_file}: {str(e)}")
            self.progressBar.setValue(i + 1)

        self.outputText.append("MCU Shift process complete.")

    def auto_detect_shift(self, data):
        block = 8
        height, width, _ = data.shape
        while height >= block and np.array_equal(data[height-block:height], np.ones((block, width, 3), dtype=np.uint8) * 128):
            height -= block
        cropped = data[:height]
        for j in range(0, width, block):
            mcu = cropped[height-block:height, j:j+block]
            if np.mean(np.abs(mcu - 128)) < 10:
                return j // block
        return width // block


def main():
    app = QApplication(sys.argv)
    window = JpegToolGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
