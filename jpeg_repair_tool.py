import sys
import struct
import os
import subprocess
import shutil
import numpy as np
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QGridLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QLineEdit, QHBoxLayout, QSlider, QFrame, QGroupBox, QTabWidget,
    QSpinBox, QTextEdit, QProgressBar, QScrollArea, QMenu, QDialog, 
    QListWidget, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QPalette, QMouseEvent, QFont, QImage, QAction
from PyQt6.QtCore import Qt, QRectF, QRect, QPointF, QBuffer, QIODevice

# --- Pillow Import ---
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:
    class PillowStub:
        @staticmethod
        def open(path): raise ImportError("Pillow not installed. Please run 'pip install Pillow'")
    Image = PillowStub
    TAGS = {}


# ======================================================================
# --- JPEG HEADER & SCANLINE LOGIC ---
# ======================================================================

def get_jpeg_mcu_data(filepath):
    """Calculates MCU dimensions and total MCU count by reading the SOF0/SOF2 header."""
    if not os.path.exists(filepath): return None

    try:
        with open(filepath, 'rb') as f:
            sof_data = None
            while True:
                marker = f.read(2)
                if not marker: return None
                
                if marker in (b'\xff\xc0', b'\xff\xc2'):
                    length_bytes = f.read(2)
                    if len(length_bytes) < 2: return None
                    
                    segment_length = struct.unpack('>H', length_bytes)[0]
                    sof_data = f.read(segment_length - 2)
                    
                    if len(sof_data) != (segment_length - 2): return None
                    break
                
                if marker[0] == 0xff and marker[1] not in (0x00, 0xd8, 0xd9, 0xda):
                    segment_length = struct.unpack('>H', f.read(2))[0]
                    f.seek(segment_length - 2, 1)
                elif marker[0] == 0xff and marker[1] in (0xd8, 0xd9): continue
                else: f.seek(-1, 1)

            height_y, width_x = struct.unpack('>HH', sof_data[1:5])
            num_components = sof_data[5]

            if num_components != 3: return None
            
            y_sampling_factor_byte = sof_data[7]
            y_hor = (y_sampling_factor_byte >> 4) & 0x0F
            y_ver = y_sampling_factor_byte & 0x0F
            
            mcu_x = y_hor * 8
            mcu_y = y_ver * 8
            
            n_mcu_x = (width_x + mcu_x - 1) // mcu_x
            n_mcu_y = (height_y + mcu_y - 1) // mcu_y
            
            return {
                "width": width_x, "height": height_y, "mcu_x": mcu_x, 
                "mcu_y": mcu_y, "n_mcu_x": n_mcu_x, "n_mcu_y": n_mcu_y,
            }

    except Exception:
        return None

def is_mcu_scanline_gray(pixels, gray_tolerance=10, color_std_dev_threshold=5):
    """Checks if a 2D array of YCbCr pixels is gray using NumPy."""
    if pixels.size == 0:
        return False

    Cb = pixels[:, 1]
    Cr = pixels[:, 2]

    cb_deviation = np.abs(Cb.astype(int) - 128)
    cr_deviation = np.abs(Cr.astype(int) - 128)
    
    max_cb_dev = np.max(cb_deviation)
    max_cr_dev = np.max(cr_deviation)
    is_color_neutral = (max_cb_dev <= gray_tolerance) and (max_cr_dev <= gray_tolerance)
    
    std_cb = np.std(Cb)
    std_cr = np.std(Cr)
    is_color_uniform = (std_cb <= color_std_dev_threshold) and (std_cr <= color_std_dev_threshold)
    
    return is_color_neutral and is_color_uniform

def count_gray_mcu_scanlines(filepath, mcu_data, gray_tolerance=10, color_std_dev_threshold=5, skip_top_scanlines=1):
    """Counts contiguous gray MCU scanlines from the bottom of the image."""
    if not mcu_data:
        return 0, 0, []

    try:
        img = Image.open(filepath).convert('YCbCr')
        img_array = np.array(img)
        height = img_array.shape[0]
        mcu_y = mcu_data['mcu_y']
        total_mcu_scanlines = mcu_data['n_mcu_y']
        
        gray_scanline_count = 0
        gray_scanline_indices = []

        for i in range(total_mcu_scanlines - 1, skip_top_scanlines - 1, -1):
            start_row = i * mcu_y
            end_row = min((i + 1) * mcu_y, height)

            scanline_data = img_array[start_row:end_row, :, :]
            pixels = scanline_data.reshape(-1, 3)

            if is_mcu_scanline_gray(pixels, gray_tolerance, color_std_dev_threshold):
                gray_scanline_count += 1
                gray_scanline_indices.insert(0, i)
            else:
                break

        return gray_scanline_count, total_mcu_scanlines, gray_scanline_indices

    except Exception as e:
        print(f"Error in count_gray_mcu_scanlines: {e}")
        return 0, 0, []
        
def analyze_last_scanline_mcus(filepath, mcu_data, gray_tolerance=10, color_std_dev_threshold=5):
    """Analyzes the individual MCUs in the last vertical scanline of the image."""
    if not mcu_data:
        return 0

    try:
        img = Image.open(filepath).convert('YCbCr')
        img_array = np.array(img)

        height = mcu_data['height']
        width = mcu_data['width']
        mcu_x = mcu_data['mcu_x']
        mcu_y = mcu_data['mcu_y']
        n_mcu_x = mcu_data['n_mcu_x']
        
        last_scanline_index = mcu_data['n_mcu_y'] - 1
        start_row = last_scanline_index * mcu_y
        end_row = height
        
        if start_row < 0 or start_row >= height:
            return 0

        gray_mcu_count = 0

        for i in range(n_mcu_x):
            start_col = i * mcu_x
            end_col = min((i + 1) * mcu_x, width)

            mcu_block = img_array[start_row:end_row, start_col:end_col, :]
            pixels = mcu_block.reshape(-1, 3)

            if is_mcu_scanline_gray(pixels, gray_tolerance, color_std_dev_threshold):
                gray_mcu_count += 1
                
        return gray_mcu_count
        
    except Exception:
        return 0

def _clarity_lookup_table():
    """Generates the PhotoDemon 'Clarity/Midtone Contrast' lookup table."""
    contrastLookup = np.zeros(256, dtype=np.uint8)
    factor = 0.4 
    
    for x in range(256):
        x_float = float(x)
        diff = x_float - 127.0
        
        if x < 127:
            push = (x_float / 127.0) * (diff / 2.0) * factor
        else:
            push = ((255.0 - x_float) / 127.0) * (diff / 2.0) * factor
            
        gray = x_float + push
        gray = np.clip(gray, 0, 255)
        contrastLookup[x] = int(round(gray))
        
    return contrastLookup

def photodemon_autocorrect_image(img: Image.Image) -> Image.Image:
    """Applies PhotoDemon's primary auto-correction steps: WB and Midtone Contrast."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    np_img = np.array(img, dtype=np.uint8)
    
    r, g, b = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
    low_clip = 0.05
    high_clip = 100.0 - 0.05
    corrected_channels = []
    
    for channel in [r, g, b]:
        min_val = np.percentile(channel, low_clip)
        max_val = np.percentile(channel, high_clip)
        
        if max_val <= min_val:
            corrected_channel = channel
        else:
            shifted_channel = channel.astype(np.float32) - min_val
            stretched_channel = shifted_channel * (255.0 / (max_val - min_val))
            corrected_channel = np.clip(stretched_channel, 0, 255).astype(np.uint8)
            
        corrected_channels.append(corrected_channel)
        
    np_img = np.stack(corrected_channels, axis=2)
    
    clarity_lookup = _clarity_lookup_table()
    np_img[:, :, 0] = clarity_lookup[np_img[:, :, 0]]
    np_img[:, :, 1] = clarity_lookup[np_img[:, :, 1]]
    np_img[:, :, 2] = clarity_lookup[np_img[:, :, 2]]
    
    return Image.fromarray(np_img, 'RGB')

# ======================================================================
# --- CROP/HEADER MODIFICATION LOGIC ---
# ======================================================================

def find_sof_height_position(filepath):
    """Scans the JPEG file to find the byte position of the Height field."""
    if not os.path.exists(filepath): return None
    
    try:
        with open(filepath, 'rb') as f:
            while True:
                marker_pos = f.tell()
                marker = f.read(2)
                if not marker: return None
                
                if marker in (b'\xff\xc0', b'\xff\xc2'):
                    return marker_pos + 5 
                
                if marker[0] == 0xff and marker[1] not in (0x00, 0xd8, 0xd9, 0xda):
                    segment_length = struct.unpack('>H', f.read(2))[0]
                    f.seek(segment_length - 2, 1)
                elif marker[0] == 0xff and marker[1] in (0xd8, 0xd9): continue
                else: f.seek(-1, 1)
    except Exception:
        return None

def crop_jpeg_by_header(source_filepath, output_filepath, scanlines_to_remove):
    """Copies the source file and modifies the Height field in the SOF segment."""
    mcu_data = get_jpeg_mcu_data(source_filepath)

    if not mcu_data:
        print("Error: Failed to read original image dimensions and MCU size.", file=sys.stderr)
        return False

    original_height = mcu_data['height']
    mcu_y = mcu_data['mcu_y']
    
    pixels_to_remove = scanlines_to_remove * mcu_y
    calculated_new_height = original_height - pixels_to_remove
    
    if calculated_new_height <= 0 or calculated_new_height >= original_height:
        print(f"Error: Invalid crop. New height ({calculated_new_height}) is not smaller than original or is zero/negative.", file=sys.stderr)
        return False

    try:
        shutil.copy2(source_filepath, output_filepath) 
    except Exception as e:
        print(f"Error copying file: {e}", file=sys.stderr)
        return False

    height_field_pos = find_sof_height_position(output_filepath)
    
    if height_field_pos is None:
        print("Error: Could not locate SOF Height field in the copied file.", file=sys.stderr)
        return False

    try:
        with open(output_filepath, 'r+b') as f:
            f.seek(height_field_pos)
            new_height_bytes = struct.pack('>H', calculated_new_height) 
            f.write(new_height_bytes)
            
        return True

    except Exception as e:
        print(f"An error occurred during header modification of the new file: {e}", file=sys.stderr)
        return False

# ======================================================================
# --- UTILITY MCU FUNCTIONS ---
# ======================================================================

def get_mcu_avg_ycbr_values(filepath, mcu_data, r, c):
    """Calculates the average Y, Cb, and Cr values for a specific MCU block."""
    if not filepath or not mcu_data: return None

    data = mcu_data
    x_start = c * data['mcu_x']
    y_start = r * data['mcu_y']
    
    crop_box = (x_start, y_start, x_start + data['mcu_x'], y_start + data['mcu_y'])
    
    try:
        img = Image.open(filepath).convert('YCbCr')
        mcu_img = img.crop(crop_box)
        mcu_array = np.array(mcu_img)
        avg_values = mcu_array.mean(axis=(0, 1))
        
        return avg_values.tolist()
            
    except Exception:
        return None

# ======================================================================
# --- PyQt GUI Components ---
# ======================================================================

class McuGridItem(QGraphicsItem):
    def __init__(self, mcu_data, image_pixmap, main_window):
        super().__init__()
        self.main_window = main_window
        self.pixmap = image_pixmap
        self.mcu_x = mcu_data['mcu_x']
        self.mcu_y = mcu_data['mcu_y']
        self.img_width = mcu_data['width']
        self.img_height = mcu_data['height']
        
        self.n_mcu_x = mcu_data['n_mcu_x']
        self.n_mcu_y = mcu_data['n_mcu_y']

        self.selected_mcu_coords = (0, 0)
        self.hovered_mcu_coords = None 
        self.grid_visible = False
        
        self._bounding_rect = QRectF(0, 0, self.img_width, self.img_height)
        
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptHoverEvents(True)

    def boundingRect(self): return self._bounding_rect

    def paint(self, painter, option, widget=None):
        painter.drawPixmap(self.boundingRect().toRect(), self.pixmap)
        
        grid_pen_color = QColor(100, 100, 100, 180) 
        if self.grid_visible:
            painter.setPen(QPen(grid_pen_color, 0))
            for c in range(1, self.n_mcu_x):
                painter.drawLine(c * self.mcu_x, 0, c * self.mcu_x, self.img_height)
            for r in range(1, self.n_mcu_y):
                painter.drawLine(0, r * self.mcu_y, self.img_width, r * self.mcu_y)
            
        if self.hovered_mcu_coords:
            hr, hc = self.hovered_mcu_coords
            hx = hc * self.mcu_x
            hy = hr * self.mcu_y
            h_rect = QRectF(hx, hy, self.mcu_x, self.mcu_y)
            painter.fillRect(h_rect, QColor(88, 166, 255, 60)) 
        
        if self.selected_mcu_coords:
            r, c = self.selected_mcu_coords
            
            x = c * self.mcu_x
            y = r * self.mcu_y
            w = self.mcu_x
            h = self.mcu_y
            
            highlight_rect = QRectF(x, y, w, h)
            
            painter.fillRect(highlight_rect, QColor(255, 123, 114, 70)) 
            painter.setPen(QPen(QColor(255, 123, 114), 3)) 
            painter.drawRect(highlight_rect)
    
    def _calculate_mcu_coords(self, pos):
        c = int(pos.x() // self.mcu_x)
        r = int(pos.y() // self.mcu_y)
        
        c = max(0, min(c, self.n_mcu_x - 1))
        r = max(0, min(r, self.n_mcu_y - 1)) 
        return r, c

    def hoverMoveEvent(self, event):
        pos = event.pos()
        r, c = self._calculate_mcu_coords(pos)
        new_hover_coords = (r, c)
        if self.hovered_mcu_coords != new_hover_coords:
            self.hovered_mcu_coords = new_hover_coords
            self.update() 
            self.main_window.display_hover_info(r, c)
            
    def hoverLeaveEvent(self, event):
        self.hovered_mcu_coords = None
        self.update()
        self.main_window.clear_hover_info()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.pos()
        r, c = self._calculate_mcu_coords(pos)
        self.selected_mcu_coords = (r, c)
        self.update() 
        self.main_window.display_mcu_info(r, c)
        self.setFocus(Qt.FocusReason.MouseFocusReason)

    def keyPressEvent(self, event):
        key = event.key()
        r, c = self.selected_mcu_coords
        new_r, new_c = r, c
        if key == Qt.Key.Key_Left:
            new_c = max(0, c - 1)
        elif key == Qt.Key.Key_Right:
            new_c = min(self.n_mcu_x - 1, c + 1)
        elif key == Qt.Key.Key_Up:
            new_r = max(0, r - 1)
        elif key == Qt.Key.Key_Down:
            new_r = min(self.n_mcu_y - 1, r + 1)
        else:
            super().keyPressEvent(event)
            return

        if (new_r, new_c) != (r, c):
            self.selected_mcu_coords = (new_r, new_c)
            self.hovered_mcu_coords = None 
            self.update()
            self.main_window.display_mcu_info(new_r, new_c)

class McuGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(20, 20, 20)) 

    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

class BlockPreviewWidget(QWidget):
    def __init__(self, cols=16, rows=8, is_static_preview=False, parent=None):
        super().__init__(parent)
        
        logical_dpi = QApplication.primaryScreen().logicalDotsPerInch()
        dpi_scale = logical_dpi / 96.0
        self.CELL_SIZE = int(8 * dpi_scale)
        
        self.rows = rows
        self.cols = cols
        self.is_static_preview = is_static_preview 
        self.is_selected = False 
        self.is_hovered = False 
        self.mcu_pixmap = QPixmap() 
        self.setFixedSize(self.cols * self.CELL_SIZE + 2, self.rows * self.CELL_SIZE + 2) 

    def update_pixmap(self, pixmap):
        self.mcu_pixmap = pixmap.scaled(
            self.cols * self.CELL_SIZE, 
            self.rows * self.CELL_SIZE, 
            Qt.AspectRatioMode.KeepAspectRatioByExpanding, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.update()
        
    def clear_pixmap(self):
        self.mcu_pixmap = QPixmap()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(22, 27, 34)) 

        if not self.mcu_pixmap.isNull():
            painter.drawPixmap(
                QRect(1, 1, self.cols * self.CELL_SIZE, self.rows * self.CELL_SIZE), 
                self.mcu_pixmap
            )
        else:
            for r in range(self.rows):
                for c in range(self.cols):
                    x = c * self.CELL_SIZE + 1
                    y = r * self.CELL_SIZE + 1
                    rect = QRect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                    if (r + c) % 2 == 0:
                        painter.fillRect(rect, QColor(33, 38, 45)) 
                    else:
                        painter.fillRect(rect, QColor(13, 17, 23))

        grid_line_color = QColor(48, 54, 61) 
        painter.setPen(QPen(grid_line_color, 1)) 
        for r in range(self.rows + 1):
            painter.drawLine(1, r * self.CELL_SIZE + 1, self.cols * self.CELL_SIZE + 1, r * self.CELL_SIZE + 1)
        for c in range(self.cols + 1):
            painter.drawLine(c * self.CELL_SIZE + 1, 1, c * self.CELL_SIZE + 1, self.rows * self.CELL_SIZE + 1)
        
        default_pen = QPen(QColor(48, 54, 61), 1)
        
        if self.is_static_preview and self.is_hovered:
            painter.setPen(QPen(QColor(88, 166, 255), 2)) 
            painter.drawRect(0, 0, self.width()-1, self.height()-1)
        elif not self.is_static_preview and self.is_selected:
            painter.setPen(QPen(QColor(255, 123, 114), 2)) 
            painter.drawRect(0, 0, self.width()-1, self.height()-1)
        else:
            painter.setPen(default_pen)
            painter.drawRect(0, 0, self.width()-1, self.height()-1)

    def set_active_state(self, is_selected: bool):
        if self.is_selected != is_selected:
            self.is_selected = is_selected
            self.update() 

    def set_hover_state(self, is_hovered: bool):
        if self.is_hovered != is_hovered:
            self.is_hovered = is_hovered
            self.update()

# ======================================================================
# --- Hex Viewer Components ---
# ======================================================================

def scan_jpeg_markers(filepath):
    markers = []
    if not filepath or not os.path.exists(filepath):
        return markers
        
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
            
        i = 0
        while i < len(data) - 1:
            if data[i] == 0xFF:
                marker_type = data[i+1]
                if marker_type not in (0x00, 0xFF):
                    if marker_type >= 0xD0 and marker_type <= 0xD7:
                        i += 2
                        continue
                        
                    marker_name = "Unknown"
                    if marker_type == 0xD8: marker_name = "SOI (Start of Image)"
                    elif marker_type == 0xD9: marker_name = "EOI (End of Image)"
                    elif marker_type == 0xDA: marker_name = "SOS (Start of Scan)"
                    elif marker_type == 0xDB: marker_name = "DQT (Quantization Table)"
                    elif marker_type == 0xC4: marker_name = "DHT (Huffman Table)"
                    elif marker_type in (0xC0, 0xC1, 0xC2, 0xC3): 
                        marker_name = f"SOF{marker_type - 0xC0} (Frame Header)"
                    elif marker_type >= 0xE0 and marker_type <= 0xEF:
                        marker_name = f"APP{marker_type - 0xE0} Metadata"
                    elif marker_type == 0xFE: marker_name = "COM (Comment)"
                    
                    markers.append({
                        'offset': i,
                        'name': marker_name,
                        'bytes': f"FF {marker_type:02X}"
                    })
                    
                    if marker_type not in (0xD8, 0xD9, 0x01) and not (marker_type >= 0xD0 and marker_type <= 0xD7):
                        if i + 3 < len(data):
                            length = int.from_bytes(data[i+2:i+4], 'big')
                            i += length + 2
                            continue
            i += 1
    except Exception as e:
        print(f"Error scanning JPEG markers: {e}", file=sys.stderr)
    return markers

class HexViewerDialog(QDialog):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.offset = 0
        self.chunk_size = 2048
        self.total_size = os.path.getsize(filepath) if filepath and os.path.exists(filepath) else 0
        self.scanned_markers = scan_jpeg_markers(filepath)
        
        self.init_ui()
        self.load_hex_chunk()

    def init_ui(self):
        self.setWindowTitle(f"Hex Viewer - {os.path.basename(self.filepath)}")
        self.resize(850, 580)
        self.setMinimumSize(800, 450)
        self.setStyleSheet("""
            QDialog { background-color: #0d1117; }
            QLabel { color: #c9d1d9; font-family: 'Segoe UI'; font-size: 10pt; }
            QLineEdit { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 4px; font-size: 9.5pt; }
            QLineEdit:focus { border: 1px solid #58a6ff; }
            QPushButton { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 6px 12px; font-weight: bold; font-size: 9.5pt; }
            QPushButton:hover { background-color: #30363d; }
            QPushButton:disabled { background-color: #0d1117; color: #8b949e; border-color: #30363d; }
            QListWidget { background-color: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; font-family: 'Segoe UI'; font-size: 9.5pt; padding: 4px; }
            QListWidget::item { padding: 6px; border-radius: 4px; }
            QListWidget::item:hover { background-color: #21262d; }
            QListWidget::item:selected { background-color: #21262d; color: #58a6ff; font-weight: bold; }
        """)
        
        layout = QVBoxLayout(self)
        
        top_bar = QHBoxLayout()
        self.info_label = QLabel("Offset: 0x00000000 / 0x00000000 (0.00 MB)")
        top_bar.addWidget(self.info_label)
        top_bar.addStretch(1)
        
        top_bar.addWidget(QLabel("Go to Offset:"))
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("e.g. 0xDA, 256")
        self.jump_input.setFixedWidth(120)
        self.jump_input.returnPressed.connect(self.jump_to_offset)
        top_bar.addWidget(self.jump_input)
        
        self.jump_button = QPushButton("Go")
        self.jump_button.clicked.connect(self.jump_to_offset)
        top_bar.addWidget(self.jump_button)
        layout.addLayout(top_bar)
        
        self.diagnostic_label = QLabel()
        self.run_diagnostics()
        layout.addWidget(self.diagnostic_label)
        
        split_layout = QHBoxLayout()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(QLabel("<b>JPEG Marker Directory:</b>"))
        
        self.marker_list = QListWidget()
        self.marker_list.setFixedWidth(240)
        self.populate_marker_directory()
        self.marker_list.itemClicked.connect(self.marker_directory_clicked)
        sidebar_layout.addWidget(self.marker_list)
        split_layout.addLayout(sidebar_layout)
        
        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.text_box.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
                border-radius: 6px; font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt; padding: 8px;
            }
        """)
        split_layout.addWidget(self.text_box, 1)
        layout.addLayout(split_layout)
        
        bottom_bar = QHBoxLayout()
        self.prev_button = QPushButton("← Prev 2KB")
        self.prev_button.clicked.connect(self.prev_page)
        bottom_bar.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next 2KB →")
        self.next_button.clicked.connect(self.next_page)
        bottom_bar.addWidget(self.next_button)
        bottom_bar.addStretch(1)
        
        legend = QLabel(
            "Legend: "
            "<span style='color:#58a6ff;'><b>SOI</b></span> | "
            "<span style='color:#ff7b72;'><b>EOI</b></span> | "
            "<span style='color:#7ee787;'><b>SOS</b></span> | "
            "<span style='color:#d2a8ff;'><b>DHT/DQT</b></span> | "
            "<span style='color:#ffb454;'><b>SOF</b></span>"
        )
        bottom_bar.addWidget(legend)
        bottom_bar.addStretch(1)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom_bar.addWidget(close_button)
        layout.addLayout(bottom_bar)

    def run_diagnostics(self):
        errors = []
        warnings = []
        
        if not self.scanned_markers:
            errors.append("Invalid JPEG: 0 structural markers identified.")
        else:
            if not any(m['bytes'] == "FF D8" for m in self.scanned_markers):
                errors.append("Missing SOI (Start of Image) marker at start of file.")
            elif self.scanned_markers[0]['bytes'] != "FF D8":
                warnings.append("SOI is not the very first marker in the file.")
                
            if not any(m['bytes'] == "FF DB" for m in self.scanned_markers):
                errors.append("Missing DQT (Quantization Table) structure.")
            if not any(m['bytes'] == "FF C4" for m in self.scanned_markers):
                warnings.append("Missing DHT (Huffman Table) structure.")
                
            sof_markers = [m for m in self.scanned_markers if m['bytes'] in ("FF C0", "FF C1", "FF C2", "FF C3")]
            if not sof_markers:
                errors.append("Missing SOF (Start of Frame) header.")
                
            if not any(m['bytes'] == "FF DA" for m in self.scanned_markers):
                errors.append("Missing SOS (Start of Scan) block.")
                
            if not any(m['bytes'] == "FF D9" for m in self.scanned_markers):
                errors.append("Missing EOI (End of Image) terminator.")
                
        if errors:
            status_text = f"<span style='color:#ff7b72;'><b>✖ Diagnostics:</b> {errors[0]}</span>"
        elif warnings:
            status_text = f"<span style='color:#ffb454;'><b>⚠ Diagnostics:</b> {warnings[0]}</span>"
        else:
            status_text = "<span style='color:#7ee787;'><b>✔ Diagnostics:</b> Healthy JPEG structure verified.</span>"
            
        self.diagnostic_label.setText(status_text)

    def populate_marker_directory(self):
        self.marker_list.clear()
        for marker in self.scanned_markers:
            item_text = f"{marker['bytes']}  {marker['name']} (0x{marker['offset']:04X})"
            self.marker_list.addItem(item_text)

    def marker_directory_clicked(self, item):
        row = self.marker_list.row(item)
        if 0 <= row < len(self.scanned_markers):
            target_offset = self.scanned_markers[row]['offset']
            self.offset = target_offset
            self.load_hex_chunk()

    def load_hex_chunk(self):
        if not self.filepath or not os.path.exists(self.filepath):
            self.text_box.setHtml("<span style='color:#ff7b72;'>Error: File not found or invalid path.</span>")
            return
            
        self.offset = max(0, min(self.offset, self.total_size - 1))
        self.offset = (self.offset // 16) * 16
        
        try:
            with open(self.filepath, 'rb') as f:
                f.seek(self.offset)
                data = f.read(self.chunk_size)
                
            html_dump = self.generate_hex_dump_html(data, self.offset)
            self.text_box.setHtml(html_dump)
            
            total_size_mb = self.total_size / (1024 * 1024)
            self.info_label.setText(
                f"Offset: <b>0x{self.offset:08X}</b> / 0x{self.total_size:08X} ({total_size_mb:.2f} MB)"
            )
            
            self.prev_button.setEnabled(self.offset > 0)
            self.next_button.setEnabled(self.offset + self.chunk_size < self.total_size)
            
        except Exception as e:
            self.text_box.setHtml(f"<span style='color:#ff7b72;'>Error loading hex data: {e}</span>")

    def prev_page(self):
        self.offset -= self.chunk_size
        self.load_hex_chunk()

    def next_page(self):
        self.offset += self.chunk_size
        self.load_hex_chunk()

    def jump_to_offset(self):
        text = self.jump_input.text().strip()
        if not text:
            return
            
        try:
            if text.lower().startswith("0x"):
                target = int(text, 16)
            else:
                try:
                    target = int(text, 16)
                except ValueError:
                    target = int(text, 10)
                    
            if 0 <= target < self.total_size:
                self.offset = target
                self.load_hex_chunk()
                self.jump_input.clear()
            else:
                QMessageBox.warning(self, "Invalid Offset", f"Offset must be between 0 and {self.total_size - 1}.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Format", "Please enter a valid decimal number or hex offset starting with '0x'.")

    def generate_hex_dump_html(self, data, start_offset=0):
        html_lines = []
        html_lines.append("<span style='color: #8b949e;'>Offset    00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F  Decoded Text</span>")
        html_lines.append("<span style='color: #30363d;'>------------------------------------------------------------------</span>")
        
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            offset_str = f"{start_offset + i:08X}"
            
            hex_parts = []
            ascii_parts = []
            
            j = 0
            while j < 16:
                if j < len(chunk):
                    b = chunk[j]
                    is_marker = False
                    marker_color = None
                    
                    if b == 0xFF and j + 1 < len(chunk):
                        next_b = chunk[j+1]
                        if next_b not in (0x00, 0xFF):
                            is_marker = True
                            if next_b == 0xD8: marker_color = "#58a6ff"
                            elif next_b == 0xD9: marker_color = "#ff7b72"
                            elif next_b == 0xDA: marker_color = "#7ee787"
                            elif next_b in (0xDB, 0xC4): marker_color = "#d2a8ff"
                            elif next_b in (0xC0, 0xC2): marker_color = "#ffb454"
                            else: marker_color = "#ffa198"
                    
                    if is_marker:
                        b1 = chunk[j]
                        b2 = chunk[j+1]
                        hex_parts.append(f"<span style='color: {marker_color}; font-weight: bold;'>{b1:02X}</span>")
                        hex_parts.append(f"<span style='color: {marker_color}; font-weight: bold;'>{b2:02X}</span>")
                        
                        for b_val in (b1, b2):
                            char = chr(b_val) if 32 <= b_val < 127 else "."
                            ascii_parts.append(f"<span style='color: {marker_color}; font-weight: bold;'>{char}</span>")
                        
                        j += 2
                        if j == 8: hex_parts.append("")
                        continue
                    else:
                        char = chr(b) if 32 <= b < 127 else "."
                        hex_parts.append(f"{b:02X}")
                        ascii_parts.append(char)
                else:
                    hex_parts.append("  ")
                    ascii_parts.append(" ")
                
                j += 1
                if j == 8: hex_parts.append("")
                    
            hex_str1 = " ".join(hex_parts[:8])
            hex_str2 = " ".join(hex_parts[8:])
            hex_full = f"{hex_str1}  {hex_str2}"
            
            ascii_clean = []
            for char in ascii_parts:
                if char == "<": ascii_clean.append("&lt;")
                elif char == ">": ascii_clean.append("&gt;")
                elif char == "&": ascii_clean.append("&amp;")
                else: ascii_clean.append(str(char))
            ascii_full = "".join(ascii_clean)
            
            html_lines.append(
                f"<span style='color: #8b949e;'>{offset_str}</span>  {hex_full}  "
                f"<span style='color: #8b949e;'>|</span><span style='color: #c9d1d9;'>{ascii_full}</span>"
                f"<span style='color: #8b949e;'>|</span>"
            )
            
        return "<pre style='margin: 0; font-family: \"Consolas\", \"Courier New\", monospace;'>" + "<br>".join(html_lines) + "</pre>"

# ======================================================================
# --- EXIF Metadata Inspector Components ---
# ======================================================================

class ExifInspectorDialog(QDialog):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.exif_data = []
        
        self.init_ui()
        self.load_exif_data()

    def init_ui(self):
        self.setWindowTitle(f"EXIF Metadata Inspector - {os.path.basename(self.filepath)}")
        self.resize(650, 480)
        self.setMinimumSize(550, 350)
        self.setStyleSheet("""
            QDialog { background-color: #0d1117; }
            QLabel { color: #c9d1d9; font-family: 'Segoe UI'; font-size: 10pt; }
            QLineEdit { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 6px; font-size: 9.5pt; }
            QLineEdit:focus { border: 1px solid #58a6ff; }
            QPushButton { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 6px 12px; font-weight: bold; font-size: 9.5pt; }
            QPushButton:hover { background-color: #30363d; }
            QTableWidget { background-color: #161b22; color: #c9d1d9; gridline-color: #30363d; border: 1px solid #30363d; border-radius: 6px; font-family: 'Segoe UI'; font-size: 9.5pt; }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section { background-color: #21262d; color: #58a6ff; padding: 6px; font-weight: bold; border: 1px solid #30363d; }
        """)
        
        layout = QVBoxLayout(self)
        
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search Metadata:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter tag names or values...")
        self.search_input.textChanged.connect(self.filter_exif_table)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Tag Name", "Hex ID", "Decoded Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("QTableWidget { alternate-background-color: #0d1117; }")
        layout.addWidget(self.table)
        
        bottom_bar = QHBoxLayout()
        self.status_label = QLabel("Loading metadata...")
        bottom_bar.addWidget(self.status_label)
        bottom_bar.addStretch(1)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom_bar.addWidget(close_button)
        layout.addLayout(bottom_bar)

    def load_exif_data(self):
        if not self.filepath or not os.path.exists(self.filepath):
            self.status_label.setText("Error: File not found.")
            return
            
        try:
            img = Image.open(self.filepath)
            exif = img._getexif()
            
            if not exif:
                self.table.setRowCount(0)
                self.status_label.setText("No EXIF metadata found in this JPEG APP1 header.")
                return
                
            self.exif_data = []
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, f"Unknown (Tag {tag_id})")
                hex_id = f"0x{tag_id:04X}"
                
                if isinstance(value, bytes):
                    try:
                        clean_val = value.decode('utf-8', errors='ignore').strip().replace('\x00', '')
                    except Exception:
                        clean_val = f"Binary Data ({len(value)} bytes)"
                else:
                    clean_val = str(value)
                    
                self.exif_data.append((tag_name, hex_id, clean_val))
                
            self.exif_data.sort(key=lambda x: x[0])
            self.populate_table(self.exif_data)
            self.status_label.setText(f"Found {len(self.exif_data)} metadata records.")
            self.table.setColumnWidth(0, 180)
            self.table.setColumnWidth(1, 80)
            
        except Exception as e:
            self.status_label.setText(f"Error reading metadata: {e}")

    def populate_table(self, data_list):
        self.table.setRowCount(len(data_list))
        for row_idx, (name, hex_id, value) in enumerate(data_list):
            item_name = QTableWidgetItem(name)
            item_hex = QTableWidgetItem(hex_id)
            item_val = QTableWidgetItem(value)
            item_hex.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.table.setItem(row_idx, 0, item_name)
            self.table.setItem(row_idx, 1, item_hex)
            self.table.setItem(row_idx, 2, item_val)

    def filter_exif_table(self, query):
        query = query.strip().lower()
        if not query:
            self.populate_table(self.exif_data)
            self.status_label.setText(f"Found {len(self.exif_data)} metadata records.")
            return
            
        filtered = [
            (name, hex_id, value) for name, hex_id, value in self.exif_data
            if query in name.lower() or query in value.lower() or query in hex_id.lower()
        ]
        self.populate_table(filtered)
        self.status_label.setText(f"Matches: {len(filtered)} of {len(self.exif_data)}")

# ======================================================================
# --- Main Window ---
# ======================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logical_dpi = QApplication.primaryScreen().logicalDotsPerInch()
        self.dpi_scale = logical_dpi / 96.0
        
        self.setWindowTitle("JPEG MCU Repair Tool")
        self.setGeometry(100, 100, 1200, 800) 
        self.current_mcu_data = None
        self.current_filepath = None
        self.original_filepath = None
        self.post_crop_gray_mcu_count = 0 
        self.vertical_gray_scanlines_to_remove = 0 
        
        self.scene = QGraphicsScene()
        self.view = McuGraphicsView(self.scene)
        self.grid_item = None
        
        self.view_mode = "YCbCr"
        
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_view_context_menu)
        
        self.action_ycbcr = QAction("YCbCr", self)
        self.action_ycbcr.setShortcut("Ctrl+0")
        self.action_ycbcr.setCheckable(True)
        self.action_ycbcr.setChecked(True)
        self.action_ycbcr.triggered.connect(lambda: self.set_view_mode("YCbCr"))
        self.addAction(self.action_ycbcr)
        
        self.action_y = QAction("Y", self)
        self.action_y.setShortcut("Ctrl+1")
        self.action_y.setCheckable(True)
        self.action_y.triggered.connect(lambda: self.set_view_mode("Y"))
        self.addAction(self.action_y)
        
        self.action_cb = QAction("Cb", self)
        self.action_cb.setShortcut("Ctrl+2")
        self.action_cb.setCheckable(True)
        self.action_cb.triggered.connect(lambda: self.set_view_mode("Cb"))
        self.addAction(self.action_cb)
        
        self.action_cr = QAction("Cr", self)
        self.action_cr.setShortcut("Ctrl+3")
        self.action_cr.setCheckable(True)
        self.action_cr.triggered.connect(lambda: self.set_view_mode("Cr"))
        self.addAction(self.action_cr)
        
        self.action_view_hex = QAction("View Hex", self)
        self.action_view_hex.setShortcut("Ctrl+H")
        self.action_view_hex.triggered.connect(self.show_hex_viewer)
        self.addAction(self.action_view_hex)
        
        self.action_view_exif = QAction("View EXIF Info", self)
        self.action_view_exif.setShortcut("Ctrl+E")
        self.action_view_exif.triggered.connect(self.show_exif_inspector)
        self.addAction(self.action_view_exif)
        
        central_widget = QWidget()
        outer_layout = QHBoxLayout(central_widget)

        left_content_layout = QVBoxLayout()
        left_content_layout.addWidget(self.view)
        outer_layout.addLayout(left_content_layout, 1) 

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setObjectName("RightPanelScroll")
        self.scroll_area.setMinimumWidth(int(340 * self.dpi_scale))

        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(10, 10, 10, 10)
        right_panel_widget.setObjectName("RightPanel") 
        
        # 1. Project Management
        project_management_group = QGroupBox("File Management")
        pm_layout = QVBoxLayout()
        pm_layout.setContentsMargins(10, 20, 10, 10)

        self.open_button = QPushButton("Open")
        self.open_button.setObjectName("PrimaryButton")
        self.open_button.clicked.connect(self.open_file)
        pm_layout.addWidget(self.open_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_to_original)
        self.reset_button.setEnabled(False)
        pm_layout.addWidget(self.reset_button)

        self.toggle_grid_button = QPushButton("Show grid")
        self.toggle_grid_button.clicked.connect(self.toggle_grid_visibility)
        self.toggle_grid_button.setEnabled(False) 
        pm_layout.addWidget(self.toggle_grid_button)
        
        self.view_hex_button = QPushButton("View Hex")
        self.view_hex_button.clicked.connect(self.show_hex_viewer)
        self.view_hex_button.setEnabled(False)
        pm_layout.addWidget(self.view_hex_button)
        
        self.view_exif_button = QPushButton("View EXIF")
        self.view_exif_button.clicked.connect(self.show_exif_inspector)
        self.view_exif_button.setEnabled(False)
        pm_layout.addWidget(self.view_exif_button)
        
        pm_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine)) 
        
        pm_layout.addWidget(QLabel("Current File:"))
        self.current_file_label_mini = QLabel("No file loaded")
        pm_layout.addWidget(self.current_file_label_mini)
        pm_layout.addWidget(QLabel("Resolution:"))
        self.dim_label_mini = QLabel("--- x ---")
        pm_layout.addWidget(self.dim_label_mini)
        pm_layout.addWidget(QLabel("MCU Size:"))
        self.mcu_label_mini = QLabel("--- x ---")
        pm_layout.addWidget(self.mcu_label_mini)
        
        project_management_group.setLayout(pm_layout)
        right_panel_layout.addWidget(project_management_group)
        
        # 2. MCU Previews
        mcu_previews_group = QGroupBox("MCU Previews")
        previews_layout = QGridLayout()
        previews_layout.setContentsMargins(10, 20, 10, 10)
        previews_layout.addWidget(QLabel("Active Block"), 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        previews_layout.addWidget(QLabel("Target Block"), 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.pixel_block_preview = BlockPreviewWidget(cols=16, rows=8, is_static_preview=True)
        self.selected_block_preview = BlockPreviewWidget(cols=16, rows=8, is_static_preview=False) 
        
        previews_layout.addWidget(self.pixel_block_preview, 1, 0)
        previews_layout.addWidget(self.selected_block_preview, 1, 1)

        mcu_previews_group.setLayout(previews_layout)
        right_panel_layout.addWidget(mcu_previews_group)
        
        # 3. Tab Widget
        repair_tab = QWidget()
        repair_layout = QVBoxLayout(repair_tab)
        repair_layout.setContentsMargins(0, 0, 0, 0) 
        
        mcu_repair_group = QGroupBox("MCU Edit")
        mcu_repair_layout = QGridLayout()
        mcu_repair_layout.setContentsMargins(10, 20, 10, 10)
        
        mcu_repair_layout.addWidget(QLabel("MCU Block:"), 0, 0)
        
        self.mcu_block_num_input = QSpinBox() 
        self.mcu_block_num_input.setRange(1, 1000) 
        self.mcu_block_num_input.setValue(1) 
        self.mcu_block_num_input.setFixedWidth(int(70 * self.dpi_scale)) 
        mcu_repair_layout.addWidget(self.mcu_block_num_input, 0, 1)
        
        self.insert_button = QPushButton("Insert MCU")
        self.insert_button.clicked.connect(lambda: self.run_repair("insert"))
        self.insert_button.setEnabled(False)
        mcu_repair_layout.addWidget(self.insert_button, 0, 2)
        self.delete_button = QPushButton("Delete MCU")
        self.delete_button.clicked.connect(lambda: self.run_repair("delete"))
        self.delete_button.setEnabled(False)
        mcu_repair_layout.addWidget(self.delete_button, 0, 3)

        mcu_repair_layout.addWidget(QLabel("Gray Scanlines Found:"), 1, 0)
        self.gray_scanline_count_label_mini = QLabel("--- / --- (0%)")
        mcu_repair_layout.addWidget(self.gray_scanline_count_label_mini, 1, 1, 1, 3)

        self.execute_header_crop_button = QPushButton("Remove MCU Gray Scanlines")
        self.execute_header_crop_button.clicked.connect(self.remove_gray_scanlines)
        self.execute_header_crop_button.setEnabled(False)
        mcu_repair_layout.addWidget(self.execute_header_crop_button, 2, 0, 1, 4) 
        
        self.auto_align_button = QPushButton("Auto Alignment")
        self.auto_align_button.clicked.connect(self.run_auto_alignment)
        self.auto_align_button.setEnabled(False) 
        mcu_repair_layout.addWidget(self.auto_align_button, 3, 0, 1, 4) 
        
        mcu_repair_group.setLayout(mcu_repair_layout)
        repair_layout.addWidget(mcu_repair_group)
        repair_layout.addStretch(1)

        color_tab = QWidget()
        color_layout = QVBoxLayout(color_tab)
        color_layout.setContentsMargins(0, 0, 0, 0) 
        
        manual_color_group = QGroupBox("Manual Color")
        manual_color_layout = QVBoxLayout()
        manual_color_layout.setContentsMargins(10, 20, 10, 10)
        MIN_VAL, MAX_VAL, DEFAULT_VAL = -2047, 2047, 0

        def create_slider(label_text):
            h_layout = QHBoxLayout()
            label = QLabel(f"{label_text}:")
            label.setFixedWidth(int(30 * self.dpi_scale)) 
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(MIN_VAL, MAX_VAL)
            slider.setValue(DEFAULT_VAL)
            slider.setSingleStep(1)
            slider.setPageStep(64)
            value_label = QLabel(str(DEFAULT_VAL).rjust(5))
            value_label.setFixedWidth(int(65 * self.dpi_scale)) 
            slider.setObjectName(f"{label_text.lower()}_slider") 
            slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v).rjust(5)))
            h_layout.addWidget(label)
            h_layout.addWidget(slider)
            h_layout.addWidget(value_label)
            return slider, h_layout

        self.y_slider, y_layout = create_slider("Y")
        manual_color_layout.addLayout(y_layout)
        self.cb_slider, cb_layout = create_slider("Cb")
        manual_color_layout.addLayout(cb_layout)
        self.cr_slider, cr_layout = create_slider("Cr")
        manual_color_layout.addLayout(cr_layout)
        
        self.cdelta_button = QPushButton("Apply")
        self.cdelta_button.setObjectName("SecondaryButton")
        self.cdelta_button.clicked.connect(self.run_cdelta_repair)
        self.cdelta_button.setEnabled(False)
        manual_color_layout.addWidget(self.cdelta_button)
        
        manual_color_group.setLayout(manual_color_layout)
        color_layout.addWidget(manual_color_group)

        auto_color_group = QGroupBox("Auto color")
        auto_color_layout = QVBoxLayout()
        auto_color_layout.setContentsMargins(10, 20, 10, 10)
        
        self.auto_color_button = QPushButton("Apply")
        self.auto_color_button.setObjectName("AccentButton") 
        self.auto_color_button.clicked.connect(self.run_auto_color_correction)
        self.auto_color_button.setEnabled(False)
        auto_color_layout.addWidget(self.auto_color_button) 
        
        auto_color_group.setLayout(auto_color_layout)
        color_layout.addWidget(auto_color_group)
        color_layout.addStretch(1) 
        
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
        batch_layout.setContentsMargins(10, 10, 10, 10)
        
        batch_group = QGroupBox("Batch")
        batch_grid = QGridLayout()
        batch_grid.setColumnStretch(0, 0)
        batch_grid.setColumnStretch(1, 1)
        batch_grid.setColumnStretch(2, 0)
        
        batch_grid.addWidget(QLabel("Reference JPEG:"), 0, 0)
        self.reference_jpeg_input = QLineEdit()
        self.reference_jpeg_input.setPlaceholderText("Select a known good JPEG file...")
        batch_grid.addWidget(self.reference_jpeg_input, 0, 1)
        self.select_ref_button = QPushButton("Browse")
        self.select_ref_button.clicked.connect(self.selectReferenceJPEG)
        batch_grid.addWidget(self.select_ref_button, 0, 2)
        
        batch_grid.addWidget(QLabel("Encrypted Folder:"), 1, 0)
        self.encrypted_folder_input = QLineEdit()
        self.encrypted_folder_input.setPlaceholderText("Select folder containing encrypted files...")
        batch_grid.addWidget(self.encrypted_folder_input, 1, 1)
        self.select_folder_button = QPushButton("Browse")
        self.select_folder_button.clicked.connect(self.selectEncryptedFolder)
        batch_grid.addWidget(self.select_folder_button, 1, 2)
        
        self.auto_batch_process_button = QPushButton("Start")
        self.auto_batch_process_button.setObjectName("PrimaryButton")
        self.auto_batch_process_button.clicked.connect(self.repairJPEGs)
        batch_grid.addWidget(self.auto_batch_process_button, 2, 0, 1, 3)

        batch_group.setLayout(batch_grid)
        batch_layout.addWidget(batch_group)
        
        self.progress_bar = QProgressBar(self)
        batch_layout.addWidget(self.progress_bar)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setObjectName("OutputText")
        self.output_text.setFont(QFont("Consolas", 10))
        batch_layout.addWidget(QLabel("Log:"))
        batch_layout.addWidget(self.output_text, 1)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(repair_tab, "Repair")
        self.tab_widget.addTab(color_tab, "Color")
        self.tab_widget.addTab(batch_tab, "Batch")
        
        right_panel_layout.addWidget(self.tab_widget)
        
        # 4. MCU Info
        selection_info_group = QGroupBox("MCU Info")
        selection_layout = QGridLayout()
        selection_layout.setContentsMargins(10, 20, 10, 10)
        
        selection_layout.addWidget(QLabel("MCU Address:"), 0, 0)
        self.mcu_coords_label = QLabel("--")
        selection_layout.addWidget(self.mcu_coords_label, 0, 1)
        
        selection_layout.addWidget(QLabel("MCU Index:"), 1, 0)
        self.mcu_index_label = QLabel("--")
        selection_layout.addWidget(self.mcu_index_label, 1, 1)
        
        selection_layout.addWidget(QLabel("Pixel Position:"), 2, 0)
        self.pixel_range_label = QLabel("X: ---, Y: ---")
        selection_layout.addWidget(self.pixel_range_label, 2, 1, 1, 3)

        selection_layout.addWidget(QLabel("YCbCr:"), 3, 0) 
        self.avg_ycbr_label = QLabel("Y: ---, Cb: ---, Cr: ---") 
        selection_layout.addWidget(self.avg_ycbr_label, 3, 1, 1, 3) 
        
        selection_info_group.setLayout(selection_layout)
        right_panel_layout.addWidget(selection_info_group)
        
        right_panel_layout.addStretch(1) 
        self.scroll_area.setWidget(right_panel_widget)
        outer_layout.addWidget(self.scroll_area)
        
        self.setCentralWidget(central_widget)
        self.apply_dark_theme()

    def apply_dark_theme(self):
        BG_DARK = "#0d1117"
        BG_MID = "#161b22"
        BG_LIGHT = "#21262d"
        TEXT_COLOR = "#c9d1d9"
        TEXT_MUTED = "#8b949e"
        ACCENT_BLUE = "#58a6ff"
        ACCENT_BLUE_GLOW = "#79c0ff"
        ACCENT_RED = "#ff7b72"
        ACCENT_RED_GLOW = "#ffa198"
        ACCENT_GREEN = "#3fb950"
        ACCENT_GREEN_GLOW = "#56d364"
        BORDER_GRAY = "#30363d"
        BORDER_LIGHT = "#484f58"

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG_DARK))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Base, QColor(BG_MID))
        palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Button, QColor(BG_LIGHT))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_BLUE))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(Qt.GlobalColor.white))
        QApplication.setPalette(palette)

        style_sheet = f"""
        * {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif; font-size: 10pt; }}
        QMainWindow {{ background-color: {BG_DARK}; }}
        QWidget#RightPanel {{ background-color: {BG_DARK}; border-left: 1px solid {BORDER_GRAY}; }}
        QGroupBox {{ font-weight: bold; font-size: 10pt; color: {TEXT_COLOR}; background-color: {BG_MID}; border: 1px solid {BORDER_GRAY}; border-radius: 8px; margin-top: 18px; padding: 10px; }}
        QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 2px 10px; left: 10px; font-size: 9pt; text-transform: uppercase; letter-spacing: 1px; color: {ACCENT_BLUE}; background-color: {BG_DARK}; border: 1px solid {BORDER_GRAY}; border-radius: 4px; }}
        QLabel {{ color: {TEXT_COLOR}; font-size: 10pt; padding: 2px 0; }}
        QTextEdit#OutputText {{ background-color: {BG_DARK}; color: #7ee787; border: 1px solid {BORDER_GRAY}; border-radius: 6px; padding: 8px; font-size: 10pt; }}
        QLineEdit, QSpinBox {{ background-color: {BG_LIGHT}; color: {TEXT_COLOR}; border: 1px solid {BORDER_GRAY}; border-radius: 6px; padding: 6px; font-size: 10pt; selection-background-color: {ACCENT_BLUE}; }}
        QLineEdit:focus, QSpinBox:focus {{ border: 1px solid {ACCENT_BLUE}; }}
        QSpinBox::up-button, QSpinBox::down-button {{ width: 20px; border: none; background-color: {BG_LIGHT}; }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background-color: {BORDER_GRAY}; }}
        QPushButton {{ background-color: {BG_LIGHT}; color: {TEXT_COLOR}; border: 1px solid {BORDER_GRAY}; border-radius: 6px; padding: 8px 12px; font-weight: bold; font-size: 10pt; }}
        QPushButton:hover {{ background-color: {BORDER_GRAY}; border-color: {BORDER_LIGHT}; }}
        QPushButton:pressed {{ background-color: {BORDER_LIGHT}; }}
        QPushButton:disabled {{ background-color: {BG_DARK}; border-color: {BORDER_GRAY}; color: {TEXT_MUTED}; }}
        QPushButton#PrimaryButton {{ background-color: {ACCENT_BLUE}; color: {BG_DARK}; border: none; font-weight: bold; font-size: 10pt; padding: 10px; }}
        QPushButton#PrimaryButton:hover {{ background-color: {ACCENT_BLUE_GLOW}; }}
        QPushButton#PrimaryButton:disabled {{ background-color: {BG_LIGHT}; color: {TEXT_MUTED}; }}
        QPushButton#SecondaryButton {{ background-color: {ACCENT_RED}; color: {BG_DARK}; border: none; font-weight: bold; font-size: 10pt; }}
        QPushButton#SecondaryButton:hover {{ background-color: {ACCENT_RED_GLOW}; }}
        QPushButton#SecondaryButton:disabled {{ background-color: {BG_LIGHT}; color: {TEXT_MUTED}; }}
        QPushButton#AccentButton {{ background-color: {ACCENT_GREEN}; color: {BG_DARK}; border: none; font-weight: bold; font-size: 10pt; }}
        QPushButton#AccentButton:hover {{ background-color: {ACCENT_GREEN_GLOW}; }}
        QPushButton#AccentButton:disabled {{ background-color: {BG_LIGHT}; color: {TEXT_MUTED}; }}
        QSlider::groove:horizontal {{ border: 1px solid {BORDER_GRAY}; height: 6px; background: {BG_DARK}; border-radius: 3px; }}
        QSlider::handle:horizontal {{ background: {ACCENT_BLUE}; border: 1px solid {BORDER_GRAY}; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }}
        QSlider::handle:horizontal:hover {{ background: {ACCENT_BLUE_GLOW}; }}
        QTabWidget::pane {{ border: 1px solid {BORDER_GRAY}; background-color: {BG_MID}; border-radius: 8px; top: -1px; }}
        QTabBar::tab {{ background: {BG_DARK}; color: {TEXT_MUTED}; padding: 8px 16px; border: 1px solid {BORDER_GRAY}; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; text-transform: uppercase; font-weight: bold; font-size: 9pt; letter-spacing: 0.5px; }}
        QTabBar::tab:selected {{ background: {BG_MID}; color: {ACCENT_BLUE}; border-bottom: 2px solid {ACCENT_BLUE}; }}
        QTabBar::tab:hover {{ background: {BG_LIGHT}; color: {TEXT_COLOR}; }}
        QProgressBar {{ border: 1px solid {BORDER_GRAY}; border-radius: 6px; text-align: center; color: {TEXT_COLOR}; background-color: {BG_DARK}; font-weight: bold; font-size: 9pt; height: 18px; }}
        QProgressBar::chunk {{ background-color: {ACCENT_BLUE}; border-radius: 5px; }}
        QGraphicsView {{ border: 1px solid {BORDER_GRAY}; background-color: {BG_DARK}; border-radius: 8px; }}
        QScrollArea#RightPanelScroll {{ border: none; background-color: {BG_DARK}; }}
        QScrollBar:vertical {{ border: none; background: {BG_DARK}; width: 8px; margin: 0px; }}
        QScrollBar::handle:vertical {{ background: {BORDER_GRAY}; min-height: 20px; border-radius: 4px; }}
        QScrollBar::handle:vertical:hover {{ background: {ACCENT_BLUE}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ border: none; background: none; height: 0px; }}
        QMessageBox {{ background-color: {BG_DARK}; border: 1px solid {BORDER_GRAY}; }}
        QMessageBox QLabel {{ color: {TEXT_COLOR}; font-size: 10pt; }}
        QMessageBox QPushButton {{ background-color: {BG_LIGHT}; color: {TEXT_COLOR}; border: 1px solid {BORDER_GRAY}; border-radius: 6px; padding: 6px 16px; font-weight: bold; font-size: 9.5pt; min-width: 75px; }}
        QMessageBox QPushButton:hover {{ background-color: {BORDER_GRAY}; border-color: {BORDER_LIGHT}; }}
        QMessageBox QPushButton:pressed {{ background-color: {BORDER_LIGHT}; }}
        QMenu {{ background-color: {BG_MID}; color: {TEXT_COLOR}; border: 1px solid {BORDER_GRAY}; border-radius: 6px; padding: 4px 0px; }}
        QMenu::item {{ padding: 6px 24px 6px 20px; background-color: transparent; }}
        QMenu::item:selected {{ background-color: {BG_LIGHT}; color: {ACCENT_BLUE}; }}
        QMenu::separator {{ height: 1px; background-color: {BORDER_GRAY}; margin: 4px 0px; }}
        """
        self.setStyleSheet(style_sheet)

    def get_mcu_block_pixmap(self, r, c):
        if not self.current_filepath or not self.current_mcu_data: return QPixmap()

        data = self.current_mcu_data
        x_start = c * data['mcu_x']
        y_start = r * data['mcu_y']
        crop_box = (x_start, y_start, x_start + data['mcu_x'], y_start + data['mcu_y'])
        
        try:
            img = Image.open(self.current_filepath)
            mcu_img = img.crop(crop_box)
            
            if self.view_mode != "YCbCr":
                ycbcr = mcu_img.convert('YCbCr')
                channels = ycbcr.split()
                target_channel = None
                mode = self.view_mode
                if mode == "Y": target_channel = channels[0]
                elif mode == "Cb": target_channel = channels[1]
                elif mode == "Cr": target_channel = channels[2]
                        
                if target_channel:
                    mcu_img = target_channel
            
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.ReadWrite)
            mcu_img.save(buffer, "PNG") 
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.data())
            return pixmap
            
        except ImportError:
            QMessageBox.critical(self, "Error", "Pillow library not found. Cannot display MCU pixels.")
        except Exception:
            pass
        return QPixmap()

    def display_hover_info(self, r, c):
        mcu_pixmap = self.get_mcu_block_pixmap(r, c)
        self.pixel_block_preview.update_pixmap(mcu_pixmap)
        self.pixel_block_preview.set_hover_state(True)

    def clear_hover_info(self):
        self.pixel_block_preview.clear_pixmap()
        self.pixel_block_preview.set_hover_state(False)

    def display_mcu_info(self, r, c):
        if not self.current_mcu_data or not self.grid_item: return
        data = self.current_mcu_data
        
        mcu_pixmap = self.get_mcu_block_pixmap(r, c)
        self.selected_block_preview.update_pixmap(mcu_pixmap)
        
        n_mcu_x = self.grid_item.n_mcu_x
        mcu_index = r * n_mcu_x + c + 1
        
        x_start = c * data['mcu_x']
        y_start = r * data['mcu_y']
        x_end = min((c + 1) * data['mcu_x'] - 1, data['width'] - 1)
        y_end = min((r + 1) * data['mcu_y'] - 1, data['height'] - 1)
        
        self.selected_mcu_c = c
        self.selected_mcu_r = r
        
        self.mcu_coords_label.setText(f"({c}, {r})")
        self.mcu_index_label.setText(str(mcu_index))
        self.pixel_range_label.setText(f"X: {x_start}-{x_end}, Y: {y_start}-{y_end}")
        
        avg_ycbr = get_mcu_avg_ycbr_values(self.current_filepath, data, r, c)
        if avg_ycbr:
            y, cb, cr = avg_ycbr
            self.avg_ycbr_label.setText(f"Y: {y:.2f}, Cb: {cb:.2f}, Cr: {cr:.2f}")
        else:
            self.avg_ycbr_label.setText("Y: ---, Cb: ---, Cr: --- (Error)")

        self.selected_block_preview.set_active_state(True) 

    def clear_mcu_info(self):
        self.mcu_coords_label.setText("--")
        self.mcu_index_label.setText("--")
        self.pixel_range_label.setText("X: ---, Y: ---")
        self.avg_ycbr_label.setText("Y: ---, Cb: ---, Cr: ---") 
        self.selected_mcu_c = 0
        self.selected_mcu_r = 0
        self.selected_block_preview.clear_pixmap()
        self.selected_block_preview.set_active_state(False)
        self.clear_hover_info()

    def load_and_display_image(self, filepath):
        image_pixmap = QPixmap(filepath)
        if image_pixmap.isNull():
            if filepath == self.original_filepath:
                QMessageBox.critical(self, "Error", "Failed to load image file. Check file path/permissions.")
                self.reset_gui()
            else:
                QMessageBox.critical(self, "Error", f"Failed to load repaired image: {os.path.basename(filepath)}. Keeping current view.")
            return

        mcu_data = get_jpeg_mcu_data(filepath)
        
        if mcu_data:
            mcu_data['width'] = image_pixmap.width()
            mcu_data['height'] = image_pixmap.height()
            self.current_mcu_data = mcu_data
            self.current_filepath = filepath
            
            self.dim_label_mini.setText(f"{mcu_data['width']} x {mcu_data['height']}")
            self.mcu_label_mini.setText(f"{mcu_data['mcu_x']} x {mcu_data['mcu_y']}")
            self.current_file_label_mini.setText(os.path.basename(filepath))
            
            try:
                vertical_gray_count, total_scanlines, _ = count_gray_mcu_scanlines(filepath, mcu_data)
                self.vertical_gray_scanlines_to_remove = vertical_gray_count
                
                if total_scanlines > 0:
                    percentage = (vertical_gray_count / total_scanlines) * 100
                    self.gray_scanline_count_label_mini.setText(f"{vertical_gray_count} / {total_scanlines} ({percentage:.1f}%)")
                    self.execute_header_crop_button.setEnabled(vertical_gray_count > 0)
                else:
                    self.gray_scanline_count_label_mini.setText("0 / 0 (0%)")
                    self.execute_header_crop_button.setEnabled(False)
            except Exception as e:
                self.gray_scanline_count_label_mini.setText("Error / --- (0%)")
                self.execute_header_crop_button.setEnabled(False)
                print(f"Error during gray scanline analysis: {e}", file=sys.stderr)
            
            try:
                horizontal_gray_mcu_count = analyze_last_scanline_mcus(filepath, mcu_data)
                self.post_crop_gray_mcu_count = horizontal_gray_mcu_count
                
                is_initial_file = (self.original_filepath == filepath)
                if not is_initial_file and horizontal_gray_mcu_count > 0:
                    self.auto_align_button.setEnabled(True)
                    insert_blocks = horizontal_gray_mcu_count + 1 
                    self.auto_align_button.setText(f"Auto Alignment (Insert {insert_blocks} Blocks at Header)")
                else:
                    self.auto_align_button.setEnabled(False)
                    self.auto_align_button.setText("Auto Alignment")
            except Exception as e:
                self.post_crop_gray_mcu_count = 0
                self.auto_align_button.setEnabled(False)
                self.auto_align_button.setText("Auto Alignment")
                print(f"Error during auto alignment analysis: {e}", file=sys.stderr)

            display_pixmap = self.get_channel_pixmap(filepath, self.view_mode)

            self.scene.clear()
            self.grid_item = McuGridItem(mcu_data, display_pixmap, self)
            self.scene.addItem(self.grid_item)
            self.scene.setSceneRect(self.grid_item.boundingRect())
            
            self.view.fitInView(self.grid_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.grid_item.setFocus(Qt.FocusReason.NoFocusReason)
            
            self.insert_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            self.reset_button.setEnabled(True) 
            self.toggle_grid_button.setEnabled(True) 
            self.view_hex_button.setEnabled(True)
            self.view_exif_button.setEnabled(True)
            self.cdelta_button.setEnabled(True) 
            self.auto_color_button.setEnabled(True) 
            self.toggle_grid_button.setText("Show MCU Grid")
            
            self.clear_mcu_info() 
            self.grid_item.mousePressEvent(self._create_fake_event()) 
        else:
            self.reset_gui()

    def set_view_mode(self, mode):
        self.view_mode = mode
        self.action_ycbcr.setChecked(mode == "YCbCr")
        self.action_y.setChecked(mode == "Y")
        self.action_cb.setChecked(mode == "Cb")
        self.action_cr.setChecked(mode == "Cr")
        self.update_image_view_mode()
        
        c = getattr(self, 'selected_mcu_c', 0)
        r = getattr(self, 'selected_mcu_r', 0)
        self.display_mcu_info(r, c)

    def update_image_view_mode(self):
        if not self.current_filepath or not self.grid_item: return
        display_pixmap = self.get_channel_pixmap(self.current_filepath, self.view_mode)
        self.grid_item.pixmap = display_pixmap
        self.grid_item.update()

    def get_channel_pixmap(self, filepath, mode):
        if not filepath or not os.path.exists(filepath): return QPixmap()
        if mode == "YCbCr": return QPixmap(filepath)
            
        try:
            img = Image.open(filepath)
            ycbcr = img.convert('YCbCr')
            channels = ycbcr.split()
            
            target_channel = None
            if mode == "Y": target_channel = channels[0]
            elif mode == "Cb": target_channel = channels[1]
            elif mode == "Cr": target_channel = channels[2]
                    
            if target_channel:
                img_data = target_channel.tobytes()
                qimg = QImage(
                    img_data, target_channel.size[0], target_channel.size[1], 
                    target_channel.size[0], QImage.Format.Format_Grayscale8
                )
                return QPixmap.fromImage(qimg)
        except Exception as e:
            print(f"Error extracting channel pixmap: {e}", file=sys.stderr)
            
        return QPixmap(filepath)

    def show_view_context_menu(self, pos):
        if not self.current_filepath: return
        menu = QMenu(self.view)
        menu.addAction(self.action_ycbcr)
        menu.addSeparator()
        menu.addAction(self.action_y)
        menu.addAction(self.action_cb)
        menu.addAction(self.action_cr)
        menu.addSeparator()
        
        action_menu_hex = QAction("View Hex", self)
        action_menu_hex.triggered.connect(self.show_hex_viewer)
        menu.addAction(action_menu_hex)
        
        action_menu_exif = QAction("View EXIF Info", self)
        action_menu_exif.triggered.connect(self.show_exif_inspector)
        menu.addAction(action_menu_exif)
        menu.exec(self.view.mapToGlobal(pos))

    def show_hex_viewer(self):
        if not self.current_filepath or not os.path.exists(self.current_filepath):
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return
        dialog = HexViewerDialog(self.current_filepath, self)
        dialog.exec()

    def show_exif_inspector(self):
        if not self.current_filepath or not os.path.exists(self.current_filepath):
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return
        dialog = ExifInspectorDialog(self.current_filepath, self)
        dialog.exec()

    def reset_gui(self):
        self.current_file_label_mini.setText("No file loaded")
        self.dim_label_mini.setText("--- x ---")
        self.mcu_label_mini.setText("--- x ---")
        self.gray_scanline_count_label_mini.setText("--- / --- (0%)") 
        self.auto_align_button.setText("Auto Alignment")
        self.post_crop_gray_mcu_count = 0 
        self.vertical_gray_scanlines_to_remove = 0 
        
        self.scene.clear()
        self.current_mcu_data = None
        self.current_filepath = None
        self.original_filepath = None
        self.grid_item = None
        
        self.clear_mcu_info()
        self.avg_ycbr_label.setText("Y: ---, Cb: ---, Cr: ---")
        
        self.insert_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.toggle_grid_button.setEnabled(False) 
        self.view_hex_button.setEnabled(False)
        self.view_exif_button.setEnabled(False)
        self.cdelta_button.setEnabled(False) 
        self.execute_header_crop_button.setEnabled(False) 
        self.auto_align_button.setEnabled(False) 
        self.auto_color_button.setEnabled(False) 
        
        self.mcu_block_num_input.setValue(1) 
        self.y_slider.setValue(0)
        self.cb_slider.setValue(0)
        self.cr_slider.setValue(0)
        
        self.progress_bar.setValue(0)
        self.output_text.clear()
        self.selected_block_preview.set_active_state(False)
        self.clear_hover_info()

    def toggle_grid_visibility(self):
        if self.grid_item:
            self.grid_item.grid_visible = not self.grid_item.grid_visible
            self.grid_item.update()
            self.toggle_grid_button.setText("Hide MCU Grid" if self.grid_item.grid_visible else "Show MCU Grid")
        
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open JPEG Image", os.path.expanduser("~"), "JPEG Files (*.jpg *.jpeg)"
        )
        if not filepath: return
        self.original_filepath = filepath
        self.current_filepath = filepath
        self.load_and_display_image(filepath)

    def reset_to_original(self):
        if not self.original_filepath:
            QMessageBox.warning(self, "Warning", "No original file path stored.")
            return

        if self.current_filepath == self.original_filepath:
             QMessageBox.information(self, "Info", "The original file is already being displayed.")
             return
             
        self.current_filepath = self.original_filepath
        self.load_and_display_image(self.original_filepath)
        QMessageBox.information(self, "Reset", f"Successfully reloaded: {os.path.basename(self.original_filepath)}")
        
        self.mcu_block_num_input.setValue(1) 
        self.y_slider.setValue(0)
        self.cb_slider.setValue(0)
        self.cr_slider.setValue(0)

    def _create_fake_event(self):
        class FakeMouseEvent:
            def button(self): return Qt.MouseButton.LeftButton
            def pos(self): return QPointF(0, 0)
        return FakeMouseEvent()
    
    def get_repair_filepaths(self):
        base_path = self.original_filepath 
        if not base_path:
            return None, None
            
        input_dir = os.path.dirname(base_path) 
        repaired_dir = os.path.join(input_dir, "Repaired")
        
        try:
            os.makedirs(repaired_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create Repaired directory: {e}")
            return None, None
            
        original_filename = os.path.basename(base_path)
        output_file = os.path.normpath(os.path.abspath(os.path.join(repaired_dir, original_filename)))
        input_file = os.path.normpath(os.path.abspath(self.current_filepath))
        
        return input_file, output_file[cite: 1]

    def execute_jpegrepair(self, command, operation):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exe_path = os.path.normpath(os.path.join(script_dir, "jpegrepair.exe"))

        if not os.path.exists(exe_path):
            return False, f"Executable not found: {exe_path}. Ensure it is in the same folder as this script."
        
        command.insert(0, exe_path)
        
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            if process.returncode == 0:
                return True, ""
            else:
                error_output = process.stderr if process.stderr else "No specific error output."
                return False, f"Execution failed for {operation} with return code {process.returncode}.\nError:\n{error_output}"

        except Exception as e:
            return False, f"An unexpected error occurred during execution: {e}"

    def remove_gray_scanlines(self):
        if not self.current_filepath or not self.current_mcu_data:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        scanlines_to_remove = self.vertical_gray_scanlines_to_remove
        if scanlines_to_remove <= 0:
            QMessageBox.information(self, "Info", "Zero gray MCU scanlines found. No cropping performed.")
            return
            
        input_file, output_file = self.get_repair_filepaths() 
        if not input_file: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        success = crop_jpeg_by_header(input_file, output_file, scanlines_to_remove)
        QApplication.restoreOverrideCursor()

        if success:
            QMessageBox.information(
                self, "Success", 
                f"Successfully cropped {scanlines_to_remove} gray MCU scanlines by modifying the JPEG header.<br>Output saved to: <b>{os.path.basename(output_file)}</b> (Overwritten)<br><br>Reloading view."
            )
            self.load_and_display_image(output_file) 
            if self.auto_align_button.isEnabled():
                QMessageBox.warning(self, "Post-Crop Check", f"Post-crop misalignment detected. <b>Auto Alignment is now enabled.</b>")
            else:
                QMessageBox.information(self, "Post-Crop Check", "The crop fixed the issue. Auto Alignment not necessary.")
        else:
            QMessageBox.critical(
                self, "Crop Failed", 
                "Header modification failed. Check the console for more specific error details or file write permissions."
            )
            self.auto_align_button.setEnabled(False)
            
    def run_auto_alignment(self):
        if not self.current_filepath or not self.current_mcu_data:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        gray_count_remaining = self.post_crop_gray_mcu_count
        mcu_block_num = gray_count_remaining + 1
        
        if mcu_block_num <= 1:
            QMessageBox.information(self, "Alignment Info", "Post-crop gray MCU count is 0. Auto Alignment not necessary.")
            self.auto_align_button.setEnabled(False)
            return

        operation = f"Align_{mcu_block_num}B" 
        c = 0
        r = 0 
        
        input_file, output_file = self.get_repair_filepaths() 
        if not input_file: return
        
        command = [input_file, output_file, "dest", str(c), str(r), "insert", str(mcu_block_num)]
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        success, error_msg = self.execute_jpegrepair(command, f"Auto Alignment ({operation})")
        QApplication.restoreOverrideCursor()
        
        if success:
            QMessageBox.information(
                self, "Auto Alignment Success", 
                f"Auto Alignment completed by inserting {mcu_block_num} blocks at MCU ({c}, {r}).<br>Output saved to: <b>{os.path.basename(output_file)}</b> (Overwritten)<br><br>Reloading view."
            )
            self.load_and_display_image(output_file)
            self.auto_align_button.setEnabled(False) 
        else:
            QMessageBox.critical(self, "Auto Alignment Failed", error_msg)
            self.auto_align_button.setEnabled(True) 
            
    def run_auto_color_correction(self):
        if not self.current_filepath:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        input_file, output_file = self.get_repair_filepaths() 
        if not input_file: return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            original_img = Image.open(input_file)
            corrected_img = photodemon_autocorrect_image(original_img)
            corrected_img.save(output_file, quality=92, optimize=True) 
            success = True
        except Exception as e:
            success = False
            error_msg = f"Failed to apply PhotoDemon Auto-Correction (WB/Clarity).\nError details: {e}"
            
        QApplication.restoreOverrideCursor()

        if success:
            QMessageBox.information(
                self, "Auto-Correction Success", 
                f"PhotoDemon Auto Color/Lighting Correction applied.<br>Output saved to: <b>{os.path.basename(output_file)}</b> (Overwritten)<br><br>Reloading view."
            )
            self.load_and_display_image(output_file)
            self.y_slider.setValue(0)
            self.cb_slider.setValue(0)
            self.cr_slider.setValue(0)
        else:
            QMessageBox.critical(self, "Auto-Correction Failed", error_msg)

    def run_cdelta_repair(self):
        if not self.current_filepath:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return
            
        deltas = {
            0: self.y_slider.value(),
            1: self.cb_slider.value(),
            2: self.cr_slider.value()
        }
        
        if all(v == 0 for v in deltas.values()):
            QMessageBox.information(self, "Info", "All color corrections are set to 0. No operation executed.")
            return

        _, final_output_file = self.get_repair_filepaths() 
        if not final_output_file: return
        
        active_components = [i for i, v in deltas.items() if v != 0]

        base_dir = os.path.dirname(final_output_file)
        ext = os.path.splitext(final_output_file)[1]
        temp_output_file = os.path.join(base_dir, f"temp_cdelta_{os.getpid()}{ext}")

        current_input_file = self.current_filepath
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        all_successful = True
        error_message = ""

        try:
            for comp_index in active_components:
                value = deltas[comp_index]
                is_last_component = (comp_index == active_components[-1])
                output_path = final_output_file if is_last_component else temp_output_file
                operation = f"cdelta {comp_index} {value}"
                
                command = [
                    current_input_file, output_path, "dest", 
                    str(0), str(0), "cdelta", str(comp_index), str(value)
                ]
                
                success, error_msg = self.execute_jpegrepair(command, operation)
                if not success:
                    all_successful = False
                    error_message = error_msg
                    break 

                current_input_file = output_path

        finally:
            QApplication.restoreOverrideCursor()
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)
        
        if all_successful:
            QMessageBox.information(
                self, "Success", 
                f"Color Correction (cdelta) completed. Output saved to: {os.path.basename(final_output_file)} (Overwritten)\n\nReloading view."
            )
            self.load_and_display_image(final_output_file)
        else:
            QMessageBox.critical(
                self, "Repair Failed", 
                f"One or more color corrections failed.\n\nError:\n{error_message}"
            )

    def run_repair(self, operation):
        if not self.current_filepath:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        mcu_block_num = self.mcu_block_num_input.value()
        c = getattr(self, 'selected_mcu_c', 0)
        r = getattr(self, 'selected_mcu_r', 0)
        
        input_file, output_file = self.get_repair_filepaths()
        if not input_file: return
        
        command = [input_file, output_file, "dest", str(c), str(r), operation, str(mcu_block_num)]
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        success, error_msg = self.execute_jpegrepair(command, operation)
        QApplication.restoreOverrideCursor()
        
        if success:
            QMessageBox.information(
                self, "Success", 
                f"MCU {operation} completed. Output saved to: {os.path.basename(output_file)} (Overwritten)\n\nReloading view."
            )
            self.load_and_display_image(output_file)
        else:
            QMessageBox.critical(self, "Repair Failed", error_msg)

    def selectReferenceJPEG(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Reference JPEG", "", "JPEG Files (*.jpg *.jpeg);;All Files (*)")
        if fileName:
            self.reference_jpeg_input.setText(fileName)

    def selectEncryptedFolder(self):
        folderName = QFileDialog.getExistingDirectory(self, "Select Encrypted Folder")
        if folderName:
            self.encrypted_folder_input.setText(folderName)

    def find_ffda_offset(self, data):
        ffda_marker = b'\xFF\xDA'
        ffda_offset = data.rfind(ffda_marker)
        if ffda_offset == -1:
            raise ValueError("FFDA marker (Start of Scan) not found in reference JPEG.")
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
                elif marker == b'\xFF\xDA':
                    break
                elif marker not in (b'\xFF\xD8', b'\xFF\xD9'):
                    if len(data) < i + 4: break
                    try:
                        length = int.from_bytes(data[i+2:i+4], 'big') + 2
                        i += length
                    except ValueError:
                        break
                    continue
            i += 1
        return data

    def _initial_file_manipulation(self, reference_path, encrypted_path, output_path):
        with open(encrypted_path, 'rb') as encrypted_file:
            encrypted_data = encrypted_file.read()

        with open(reference_path, 'rb') as reference_file:
            reference_data = reference_file.read()

        ffda_offset = self.find_ffda_offset(reference_data)
        cut_reference_data = reference_data[:ffda_offset + 12]
        
        repaired_data = cut_reference_data + encrypted_data[153605:]
        repaired_data = self.remove_exif(repaired_data)
        repaired_data = repaired_data[:-334]

        with open(output_path, 'wb') as output_file:
            output_file.write(repaired_data)

    def _batch_step_header_crop(self, input_path, output_path):
        mcu_data = get_jpeg_mcu_data(input_path)
        if not mcu_data: return False, f"Error: Cannot read MCU data from {os.path.basename(input_path)}."

        vertical_gray_count, _, _ = count_gray_mcu_scanlines(input_path, mcu_data)
        if vertical_gray_count <= 0:
            shutil.copy2(input_path, output_path)
            return True, output_path
            
        success = crop_jpeg_by_header(input_path, output_path, vertical_gray_count)
        if success:
            return True, output_path
        else:
            return False, f"Failed to crop {vertical_gray_count} scanlines."

    def _batch_step_auto_align(self, input_path, output_path):
        mcu_data = get_jpeg_mcu_data(input_path)
        if not mcu_data: 
            shutil.copy2(input_path, output_path)
            return True, output_path
        
        horizontal_gray_mcu_count = analyze_last_scanline_mcus(input_path, mcu_data)
        mcu_block_num = horizontal_gray_mcu_count + 1
        
        if mcu_block_num <= 1:
            shutil.copy2(input_path, output_path)
            return True, output_path

        command = [input_path, output_path, "dest", str(0), str(0), "insert", str(mcu_block_num)]
        success, error_msg = self.execute_jpegrepair(command, f"Auto Align ({mcu_block_num} blocks)")
        
        if success:
            return True, output_path
        else:
            return False, error_msg

    def _batch_step_auto_color(self, input_path, output_path):
        try:
            original_img = Image.open(input_path)
            corrected_img = photodemon_autocorrect_image(original_img)
            corrected_img.save(output_path, quality=92, optimize=True) 
            return True, output_path
        except Exception as e:
            return False, f"Failed to apply PhotoDemon Auto-Correction (WB/Clarity): {e}"

    def process_jpeg_batch(self, input_file, reference_path, output_file):
        base_dir = os.path.dirname(output_file)
        temp_pre_process_file = os.path.join(base_dir, f"temp_pre_process_{os.getpid()}_{os.path.basename(input_file)}")
        
        try:
            self._initial_file_manipulation(reference_path, input_file, temp_pre_process_file)
        except Exception as e:
            return False, f"Initial file manipulation failed: {e}"
        
        current_input_file = temp_pre_process_file
        temp_files_to_cleanup = [temp_pre_process_file]
        
        try:
            temp_crop_file = os.path.join(base_dir, f"temp_batch_{os.getpid()}_crop.jpg")
            success, result = self._batch_step_header_crop(current_input_file, temp_crop_file)
            if not success: return False, f"Header Crop failed: {result}"
            current_input_file = result
            temp_files_to_cleanup.append(temp_crop_file)
            
            temp_align_file = os.path.join(base_dir, f"temp_batch_{os.getpid()}_align.jpg")
            success, result = self._batch_step_auto_align(current_input_file, temp_align_file)
            if not success: return False, f"Auto Alignment failed: {result}"
            current_input_file = result
            temp_files_to_cleanup.append(temp_align_file)
            
            success, result = self._batch_step_auto_color(current_input_file, output_file)
            if not success: return False, f"Auto Color failed: {result}"
            
            return True, output_file
            
        finally:
            for f in temp_files_to_cleanup:
                if os.path.exists(f): 
                    try:
                        os.remove(f)
                    except Exception:
                        pass

    def repairJPEGs(self):
        reference_jpeg = self.reference_jpeg_input.text().strip()
        encrypted_folder = self.encrypted_folder_input.text().strip()

        if not os.path.exists(reference_jpeg) or not os.path.isdir(encrypted_folder):
            QMessageBox.critical(self, "Error", "Please ensure the reference JPEG and encrypted folder paths are valid.")
            return

        repaired_folder = os.path.join(encrypted_folder, "Repaired")
        os.makedirs(repaired_folder, exist_ok=True)
        
        pattern = re.compile(r".*\.JPG\..{4}$", re.I)
        encrypted_files = [f for f in os.listdir(encrypted_folder) if pattern.match(f)]
        
        if not encrypted_files:
            QMessageBox.information(self, "Info", "No files matching the pattern '*.JPG.xxxx' found in the folder.")
            return

        self.progress_bar.setMaximum(len(encrypted_files))
        self.progress_bar.setValue(0)
        self.output_text.clear()
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        successful_count = 0
        total_count = len(encrypted_files)

        try:
            for i, encrypted_file in enumerate(encrypted_files):
                input_file_path = os.path.join(encrypted_folder, encrypted_file)
                output_name = os.path.splitext(encrypted_file)[0].rsplit('.', 1)[0] + '.JPG'
                output_file_path = os.path.join(repaired_folder, output_name)
                
                self.output_text.append(f"Starting: {encrypted_file} -> {output_name}")
                success, result_path_or_error = self.process_jpeg_batch(input_file_path, reference_jpeg, output_file_path)
                
                if success:
                    self.output_text.append(f"  SUCCESS: All 3 steps completed.")
                    successful_count += 1
                    self.original_filepath = output_file_path
                    self.load_and_display_image(output_file_path) 
                else:
                    self.output_text.append(f"  ERROR: {result_path_or_error}")
                    
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()
            
        self.output_text.append(f"\nBatch Repair complete. {successful_count} of {total_count} files successfully processed.")
        QMessageBox.information(
            self, "Batch Complete", 
            f"Batch Repair finished.<br>Processed: <b>{total_count}</b><br>Successful: <b>{successful_count}</b>"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.grid_item:
            self.view.fitInView(self.grid_item, Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == '__main__':
    if hasattr(sys, 'frozen') and sys.platform == 'win32':
        qt_plugin_path = os.path.join(os.path.dirname(sys.executable), 'PyQt6', 'Qt6', 'plugins')
        if os.path.isdir(qt_plugin_path):
             os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
             
    app = QApplication(sys.argv)
    
    app_font = app.font()
    app_font.setFamily("Segoe UI")
    app_font.setPointSize(10)
    app.setFont(app_font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
import sys
import struct
import os
import subprocess
import shutil
import numpy as np
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QGridLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QLineEdit, QHBoxLayout, QSlider, QFrame, QGroupBox, QTabWidget,
    QSpinBox, QTextEdit, QProgressBar, QScrollArea, QMenu, QDialog, 
    QListWidget, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QPalette, QMouseEvent, QFont, QImage, QAction
from PyQt6.QtCore import Qt, QRectF, QRect, QPointF, QBuffer, QIODevice

# --- Pillow Import ---
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:
    class PillowStub:
        @staticmethod
        def open(path): raise ImportError("Pillow not installed. Please run 'pip install Pillow'")
    Image = PillowStub
    TAGS = {}


# ======================================================================
# --- JPEG HEADER & SCANLINE LOGIC ---
# ======================================================================

def get_jpeg_mcu_data(filepath):
    """Calculates MCU dimensions and total MCU count by reading the SOF0/SOF2 header."""
    if not os.path.exists(filepath): return None

    try:
        with open(filepath, 'rb') as f:
            sof_data = None
            while True:
                marker = f.read(2)
                if not marker: return None
                
                if marker in (b'\xff\xc0', b'\xff\xc2'):
                    length_bytes = f.read(2)
                    if len(length_bytes) < 2: return None
                    
                    segment_length = struct.unpack('>H', length_bytes)[0]
                    sof_data = f.read(segment_length - 2)
                    
                    if len(sof_data) != (segment_length - 2): return None
                    break
                
                if marker[0] == 0xff and marker[1] not in (0x00, 0xd8, 0xd9, 0xda):
                    segment_length = struct.unpack('>H', f.read(2))[0]
                    f.seek(segment_length - 2, 1)
                elif marker[0] == 0xff and marker[1] in (0xd8, 0xd9): continue
                else: f.seek(-1, 1)

            height_y, width_x = struct.unpack('>HH', sof_data[1:5])
            num_components = sof_data[5]

            if num_components != 3: return None
            
            y_sampling_factor_byte = sof_data[7]
            y_hor = (y_sampling_factor_byte >> 4) & 0x0F
            y_ver = y_sampling_factor_byte & 0x0F
            
            mcu_x = y_hor * 8
            mcu_y = y_ver * 8
            
            n_mcu_x = (width_x + mcu_x - 1) // mcu_x
            n_mcu_y = (height_y + mcu_y - 1) // mcu_y
            
            return {
                "width": width_x, "height": height_y, "mcu_x": mcu_x, 
                "mcu_y": mcu_y, "n_mcu_x": n_mcu_x, "n_mcu_y": n_mcu_y,
            }

    except Exception:
        return None

def is_mcu_scanline_gray(pixels, gray_tolerance=10, color_std_dev_threshold=5):
    """Checks if a 2D array of YCbCr pixels is gray using NumPy."""
    if pixels.size == 0:
        return False

    Cb = pixels[:, 1]
    Cr = pixels[:, 2]

    cb_deviation = np.abs(Cb.astype(int) - 128)
    cr_deviation = np.abs(Cr.astype(int) - 128)
    
    max_cb_dev = np.max(cb_deviation)
    max_cr_dev = np.max(cr_deviation)
    is_color_neutral = (max_cb_dev <= gray_tolerance) and (max_cr_dev <= gray_tolerance)
    
    std_cb = np.std(Cb)
    std_cr = np.std(Cr)
    is_color_uniform = (std_cb <= color_std_dev_threshold) and (std_cr <= color_std_dev_threshold)
    
    return is_color_neutral and is_color_uniform

def count_gray_mcu_scanlines(filepath, mcu_data, gray_tolerance=10, color_std_dev_threshold=5, skip_top_scanlines=1):
    """Counts contiguous gray MCU scanlines from the bottom of the image."""
    if not mcu_data:
        return 0, 0, []

    try:
        img = Image.open(filepath).convert('YCbCr')
        img_array = np.array(img)
        height = img_array.shape[0]
        mcu_y = mcu_data['mcu_y']
        total_mcu_scanlines = mcu_data['n_mcu_y']
        
        gray_scanline_count = 0
        gray_scanline_indices = []

        for i in range(total_mcu_scanlines - 1, skip_top_scanlines - 1, -1):
            start_row = i * mcu_y
            end_row = min((i + 1) * mcu_y, height)

            scanline_data = img_array[start_row:end_row, :, :]
            pixels = scanline_data.reshape(-1, 3)

            if is_mcu_scanline_gray(pixels, gray_tolerance, color_std_dev_threshold):
                gray_scanline_count += 1
                gray_scanline_indices.insert(0, i)
            else:
                break

        return gray_scanline_count, total_mcu_scanlines, gray_scanline_indices

    except Exception as e:
        print(f"Error in count_gray_mcu_scanlines: {e}")
        return 0, 0, []
        
def analyze_last_scanline_mcus(filepath, mcu_data, gray_tolerance=10, color_std_dev_threshold=5):
    """Analyzes the individual MCUs in the last vertical scanline of the image."""
    if not mcu_data:
        return 0

    try:
        img = Image.open(filepath).convert('YCbCr')
        img_array = np.array(img)

        height = mcu_data['height']
        width = mcu_data['width']
        mcu_x = mcu_data['mcu_x']
        mcu_y = mcu_data['mcu_y']
        n_mcu_x = mcu_data['n_mcu_x']
        
        last_scanline_index = mcu_data['n_mcu_y'] - 1
        start_row = last_scanline_index * mcu_y
        end_row = height
        
        if start_row < 0 or start_row >= height:
            return 0

        gray_mcu_count = 0

        for i in range(n_mcu_x):
            start_col = i * mcu_x
            end_col = min((i + 1) * mcu_x, width)

            mcu_block = img_array[start_row:end_row, start_col:end_col, :]
            pixels = mcu_block.reshape(-1, 3)

            if is_mcu_scanline_gray(pixels, gray_tolerance, color_std_dev_threshold):
                gray_mcu_count += 1
                
        return gray_mcu_count
        
    except Exception:
        return 0

def _clarity_lookup_table():
    """Generates the PhotoDemon 'Clarity/Midtone Contrast' lookup table."""
    contrastLookup = np.zeros(256, dtype=np.uint8)
    factor = 0.4 
    
    for x in range(256):
        x_float = float(x)
        diff = x_float - 127.0
        
        if x < 127:
            push = (x_float / 127.0) * (diff / 2.0) * factor
        else:
            push = ((255.0 - x_float) / 127.0) * (diff / 2.0) * factor
            
        gray = x_float + push
        gray = np.clip(gray, 0, 255)
        contrastLookup[x] = int(round(gray))
        
    return contrastLookup

def photodemon_autocorrect_image(img: Image.Image) -> Image.Image:
    """Applies PhotoDemon's primary auto-correction steps: WB and Midtone Contrast."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    np_img = np.array(img, dtype=np.uint8)
    
    r, g, b = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
    low_clip = 0.05
    high_clip = 100.0 - 0.05
    corrected_channels = []
    
    for channel in [r, g, b]:
        min_val = np.percentile(channel, low_clip)
        max_val = np.percentile(channel, high_clip)
        
        if max_val <= min_val:
            corrected_channel = channel
        else:
            shifted_channel = channel.astype(np.float32) - min_val
            stretched_channel = shifted_channel * (255.0 / (max_val - min_val))
            corrected_channel = np.clip(stretched_channel, 0, 255).astype(np.uint8)
            
        corrected_channels.append(corrected_channel)
        
    np_img = np.stack(corrected_channels, axis=2)
    
    clarity_lookup = _clarity_lookup_table()
    np_img[:, :, 0] = clarity_lookup[np_img[:, :, 0]]
    np_img[:, :, 1] = clarity_lookup[np_img[:, :, 1]]
    np_img[:, :, 2] = clarity_lookup[np_img[:, :, 2]]
    
    return Image.fromarray(np_img, 'RGB')

# ======================================================================
# --- CROP/HEADER MODIFICATION LOGIC ---
# ======================================================================

def find_sof_height_position(filepath):
    """Scans the JPEG file to find the byte position of the Height field."""
    if not os.path.exists(filepath): return None
    
    try:
        with open(filepath, 'rb') as f:
            while True:
                marker_pos = f.tell()
                marker = f.read(2)
                if not marker: return None
                
                if marker in (b'\xff\xc0', b'\xff\xc2'):
                    return marker_pos + 5 
                
                if marker[0] == 0xff and marker[1] not in (0x00, 0xd8, 0xd9, 0xda):
                    segment_length = struct.unpack('>H', f.read(2))[0]
                    f.seek(segment_length - 2, 1)
                elif marker[0] == 0xff and marker[1] in (0xd8, 0xd9): continue
                else: f.seek(-1, 1)
    except Exception:
        return None

def crop_jpeg_by_header(source_filepath, output_filepath, scanlines_to_remove):
    """Copies the source file and modifies the Height field in the SOF segment."""
    mcu_data = get_jpeg_mcu_data(source_filepath)

    if not mcu_data:
        print("Error: Failed to read original image dimensions and MCU size.", file=sys.stderr)
        return False

    original_height = mcu_data['height']
    mcu_y = mcu_data['mcu_y']
    
    pixels_to_remove = scanlines_to_remove * mcu_y
    calculated_new_height = original_height - pixels_to_remove
    
    if calculated_new_height <= 0 or calculated_new_height >= original_height:
        print(f"Error: Invalid crop. New height ({calculated_new_height}) is not smaller than original or is zero/negative.", file=sys.stderr)
        return False

    try:
        shutil.copy2(source_filepath, output_filepath) 
    except Exception as e:
        print(f"Error copying file: {e}", file=sys.stderr)
        return False

    height_field_pos = find_sof_height_position(output_filepath)
    
    if height_field_pos is None:
        print("Error: Could not locate SOF Height field in the copied file.", file=sys.stderr)
        return False

    try:
        with open(output_filepath, 'r+b') as f:
            f.seek(height_field_pos)
            new_height_bytes = struct.pack('>H', calculated_new_height) 
            f.write(new_height_bytes)
            
        return True

    except Exception as e:
        print(f"An error occurred during header modification of the new file: {e}", file=sys.stderr)
        return False

# ======================================================================
# --- UTILITY MCU FUNCTIONS ---
# ======================================================================

def get_mcu_avg_ycbr_values(filepath, mcu_data, r, c):
    """Calculates the average Y, Cb, and Cr values for a specific MCU block."""
    if not filepath or not mcu_data: return None

    data = mcu_data
    x_start = c * data['mcu_x']
    y_start = r * data['mcu_y']
    
    crop_box = (x_start, y_start, x_start + data['mcu_x'], y_start + data['mcu_y'])
    
    try:
        img = Image.open(filepath).convert('YCbCr')
        mcu_img = img.crop(crop_box)
        mcu_array = np.array(mcu_img)
        avg_values = mcu_array.mean(axis=(0, 1))
        
        return avg_values.tolist()
            
    except Exception:
        return None

# ======================================================================
# --- PyQt GUI Components & Main Window ---
# ======================================================================

class McuGridItem(QGraphicsItem):
    def __init__(self, mcu_data, image_pixmap, main_window):
        super().__init__()
        self.main_window = main_window
        self.pixmap = image_pixmap
        self.mcu_x = mcu_data['mcu_x']
        self.mcu_y = mcu_data['mcu_y']
        self.img_width = mcu_data['width']
        self.img_height = mcu_data['height']
        
        self.n_mcu_x = mcu_data['n_mcu_x']
        self.n_mcu_y = mcu_data['n_mcu_y']

        self.selected_mcu_coords = (0, 0)
        self.hovered_mcu_coords = None 
        self.grid_visible = False
        
        self._bounding_rect = QRectF(0, 0, self.img_width, self.img_height)
        
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptHoverEvents(True)

    def boundingRect(self): return self._bounding_rect

    def paint(self, painter, option, widget=None):
        painter.drawPixmap(self.boundingRect().toRect(), self.pixmap)
        
        grid_pen_color = QColor(100, 100, 100, 180) 
        if self.grid_visible:
            painter.setPen(QPen(grid_pen_color, 0))
            for c in range(1, self.n_mcu_x):
                painter.drawLine(c * self.mcu_x, 0, c * self.mcu_x, self.img_height)
            for r in range(1, self.n_mcu_y):
                painter.drawLine(0, r * self.mcu_y, self.img_width, r * self.mcu_y)
            
        if self.hovered_mcu_coords:
            hr, hc = self.hovered_mcu_coords
            hx = hc * self.mcu_x
            hy = hr * self.mcu_y
            h_rect = QRectF(hx, hy, self.mcu_x, self.mcu_y)
            painter.fillRect(h_rect, QColor(88, 166, 255, 60)) 
        
        if self.selected_mcu_coords:
            r, c = self.selected_mcu_coords
            x = c * self.mcu_x
            y = r * self.mcu_y
            w = self.mcu_x
            h = self.mcu_y
            highlight_rect = QRectF(x, y, w, h)
            
            painter.fillRect(highlight_rect, QColor(255, 123, 114, 70)) 
            painter.setPen(QPen(QColor(255, 123, 114), 3)) 
            painter.drawRect(highlight_rect)
    
    def _calculate_mcu_coords(self, pos):
        c = int(pos.x() // self.mcu_x)
        r = int(pos.y() // self.mcu_y)
        c = max(0, min(c, self.n_mcu_x - 1))
        r = max(0, min(r, self.n_mcu_y - 1)) 
        return r, c

    def hoverMoveEvent(self, event):
        pos = event.pos()
        r, c = self._calculate_mcu_coords(pos)
        new_hover_coords = (r, c)
        if self.hovered_mcu_coords != new_hover_coords:
            self.hovered_mcu_coords = new_hover_coords
            self.update() 
            self.main_window.display_hover_info(r, c)
            
    def hoverLeaveEvent(self, event):
        self.hovered_mcu_coords = None
        self.update()
        self.main_window.clear_hover_info()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.pos()
        r, c = self._calculate_mcu_coords(pos)
        self.selected_mcu_coords = (r, c)
        self.update() 
        self.main_window.display_mcu_info(r, c)
        self.setFocus(Qt.FocusReason.MouseFocusReason)

    def keyPressEvent(self, event):
        key = event.key()
        r, c = self.selected_mcu_coords
        new_r, new_c = r, c
        if key == Qt.Key.Key_Left:
            new_c = max(0, c - 1)
        elif key == Qt.Key.Key_Right:
            new_c = min(self.n_mcu_x - 1, c + 1)
        elif key == Qt.Key.Key_Up:
            new_r = max(0, r - 1)
        elif key == Qt.Key.Key_Down:
            new_r = min(self.n_mcu_y - 1, r + 1)
        else:
            super().keyPressEvent(event)
            return

        if (new_r, new_c) != (r, c):
            self.selected_mcu_coords = (new_r, new_c)
            self.hovered_mcu_coords = None 
            self.update()
            self.main_window.display_mcu_info(new_r, new_c)

class McuGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(20, 20, 20)) 

    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

class BlockPreviewWidget(QWidget):
    def __init__(self, cols=16, rows=8, is_static_preview=False, parent=None):
        super().__init__(parent)
        logical_dpi = QApplication.primaryScreen().logicalDotsPerInch()
        dpi_scale = logical_dpi / 96.0
        self.CELL_SIZE = int(8 * dpi_scale)
        self.rows = rows
        self.cols = cols
        self.is_static_preview = is_static_preview 
        self.is_selected = False 
        self.is_hovered = False 
        self.mcu_pixmap = QPixmap() 
        self.setFixedSize(self.cols * self.CELL_SIZE + 2, self.rows * self.CELL_SIZE + 2) 

    def update_pixmap(self, pixmap):
        self.mcu_pixmap = pixmap.scaled(
            self.cols * self.CELL_SIZE, 
            self.rows * self.CELL_SIZE, 
            Qt.AspectRatioMode.KeepAspectRatioByExpanding, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.update()
        
    def clear_pixmap(self):
        self.mcu_pixmap = QPixmap()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(22, 27, 34)) 

        if not self.mcu_pixmap.isNull():
            painter.drawPixmap(
                QRect(1, 1, self.cols * self.CELL_SIZE, self.rows * self.CELL_SIZE), 
                self.mcu_pixmap
            )
        else:
            for r in range(self.rows):
                for c in range(self.cols):
                    x = c * self.CELL_SIZE + 1
                    y = r * self.CELL_SIZE + 1
                    rect = QRect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                    if (r + c) % 2 == 0:
                        painter.fillRect(rect, QColor(33, 38, 45)) 
                    else:
                        painter.fillRect(rect, QColor(13, 17, 23))

        grid_line_color = QColor(48, 54, 61) 
        painter.setPen(QPen(grid_line_color, 1)) 
        for r in range(self.rows + 1):
            painter.drawLine(1, r * self.CELL_SIZE + 1, self.cols * self.CELL_SIZE + 1, r * self.CELL_SIZE + 1)
        for c in range(self.cols + 1):
            painter.drawLine(c * self.CELL_SIZE + 1, 1, c * self.CELL_SIZE + 1, self.rows * self.CELL_SIZE + 1)
        
        default_pen = QPen(QColor(48, 54, 61), 1)
        if self.is_static_preview and self.is_hovered:
            painter.setPen(QPen(QColor(88, 166, 255), 2)) 
            painter.drawRect(0, 0, self.width()-1, self.height()-1)
        elif not self.is_static_preview and self.is_selected:
            painter.setPen(QPen(QColor(255, 123, 114), 2)) 
            painter.drawRect(0, 0, self.width()-1, self.height()-1)
        else:
            painter.setPen(default_pen)
            painter.drawRect(0, 0, self.width()-1, self.height()-1)

    def set_active_state(self, is_selected: bool):
        if self.is_selected != is_selected:
            self.is_selected = is_selected
            self.update() 

    def set_hover_state(self, is_hovered: bool):
        if self.is_hovered != is_hovered:
            self.is_hovered = is_hovered
            self.update()

# ======================================================================
# --- Hex & EXIF Dialogs ---
# ======================================================================

def scan_jpeg_markers(filepath):
    markers = []
    if not filepath or not os.path.exists(filepath):
        return markers
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        i = 0
        while i < len(data) - 1:
            if data[i] == 0xFF:
                marker_type = data[i+1]
                if marker_type not in (0x00, 0xFF):
                    if 0xD0 <= marker_type <= 0xD7:
                        i += 2
                        continue
                    marker_name = "Unknown"
                    if marker_type == 0xD8: marker_name = "SOI (Start of Image)"
                    elif marker_type == 0xD9: marker_name = "EOI (End of Image)"
                    elif marker_type == 0xDA: marker_name = "SOS (Start of Scan)"
                    elif marker_type == 0xDB: marker_name = "DQT (Quantization Table)"
                    elif marker_type == 0xC4: marker_name = "DHT (Huffman Table)"
                    elif marker_type in (0xC0, 0xC1, 0xC2, 0xC3): 
                        marker_name = f"SOF{marker_type - 0xC0} (Frame Header)"
                    elif 0xE0 <= marker_type <= 0xEF:
                        marker_name = f"APP{marker_type - 0xE0} Metadata"
                    elif marker_type == 0xFE: marker_name = "COM (Comment)"
                    
                    markers.append({
                        'offset': i,
                        'name': marker_name,
                        'bytes': f"FF {marker_type:02X}"
                    })
                    if marker_type not in (0xD8, 0xD9, 0x01) and not (0xD0 <= marker_type <= 0xD7):
                        if i + 3 < len(data):
                            length = int.from_bytes(data[i+2:i+4], 'big')
                            i += length + 2
                            continue
            i += 1
    except Exception as e:
        print(f"Error scanning JPEG markers: {e}", file=sys.stderr)
    return markers

class HexViewerDialog(QDialog):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.offset = 0
        self.chunk_size = 2048
        self.total_size = os.path.getsize(filepath) if filepath and os.path.exists(filepath) else 0
        self.scanned_markers = scan_jpeg_markers(filepath)
        self.init_ui()
        self.load_hex_chunk()

    def init_ui(self):
        self.setWindowTitle(f"Hex Viewer - {os.path.basename(self.filepath)}")
        self.resize(850, 580)
        self.setMinimumSize(800, 450)
        self.setStyleSheet("""
            QDialog { background-color: #0d1117; }
            QLabel { color: #c9d1d9; font-family: 'Segoe UI'; font-size: 10pt; }
            QLineEdit { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 4px; font-size: 9.5pt; }
            QLineEdit:focus { border: 1px solid #58a6ff; }
            QPushButton { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 6px 12px; font-weight: bold; font-size: 9.5pt; }
            QPushButton:hover { background-color: #30363d; }
            QPushButton:disabled { background-color: #0d1117; color: #8b949e; border-color: #30363d; }
            QListWidget { background-color: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; font-family: 'Segoe UI'; font-size: 9.5pt; padding: 4px; }
            QListWidget::item { padding: 6px; border-radius: 4px; }
            QListWidget::item:hover { background-color: #21262d; }
            QListWidget::item:selected { background-color: #21262d; color: #58a6ff; font-weight: bold; }
        """)
        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        self.info_label = QLabel("Offset: 0x00000000 / 0x00000000 (0.00 MB)")
        top_bar.addWidget(self.info_label)
        top_bar.addStretch(1)
        top_bar.addWidget(QLabel("Go to Offset:"))
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("e.g. 0xDA, 256")
        self.jump_input.setFixedWidth(120)
        self.jump_input.returnPressed.connect(self.jump_to_offset)
        top_bar.addWidget(self.jump_input)
        self.jump_button = QPushButton("Go")
        self.jump_button.clicked.connect(self.jump_to_offset)
        top_bar.addWidget(self.jump_button)
        layout.addLayout(top_bar)
        
        self.diagnostic_label = QLabel()
        self.run_diagnostics()
        layout.addWidget(self.diagnostic_label)
        
        split_layout = QHBoxLayout()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(QLabel("<b>JPEG Marker Directory:</b>"))
        self.marker_list = QListWidget()
        self.marker_list.setFixedWidth(240)
        self.populate_marker_directory()
        self.marker_list.itemClicked.connect(self.marker_directory_clicked)
        sidebar_layout.addWidget(self.marker_list)
        split_layout.addLayout(sidebar_layout)
        
        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.text_box.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
                border-radius: 6px; font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt; padding: 8px;
            }
        """)
        split_layout.addWidget(self.text_box, 1)
        layout.addLayout(split_layout)
        
        bottom_bar = QHBoxLayout()
        self.prev_button = QPushButton("← Prev 2KB")
        self.prev_button.clicked.connect(self.prev_page)
        bottom_bar.addWidget(self.prev_button)
        self.next_button = QPushButton("Next 2KB →")
        self.next_button.clicked.connect(self.next_page)
        bottom_bar.addWidget(self.next_button)
        bottom_bar.addStretch(1)
        
        legend = QLabel(
            "Legend: "
            "<span style='color:#58a6ff;'><b>SOI</b></span> | "
            "<span style='color:#ff7b72;'><b>EOI</b></span> | "
            "<span style='color:#7ee787;'><b>SOS</b></span> | "
            "<span style='color:#d2a8ff;'><b>DHT/DQT</b></span> | "
            "<span style='color:#ffb454;'><b>SOF</b></span>"
        )
        bottom_bar.addWidget(legend)
        bottom_bar.addStretch(1)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom_bar.addWidget(close_button)
        layout.addLayout(bottom_bar)

    def run_diagnostics(self):
        errors, warnings = [], []
        if not self.scanned_markers:
            errors.append("Invalid JPEG: 0 structural markers identified.")
        else:
            if not any(m['bytes'] == "FF D8" for m in self.scanned_markers):
                errors.append("Missing SOI (Start of Image) marker at start of file.")
            elif self.scanned_markers[0]['bytes'] != "FF D8":
                warnings.append("SOI is not the very first marker in the file.")
            if not any(m['bytes'] == "FF DB" for m in self.scanned_markers):
                errors.append("Missing DQT (Quantization Table) structure.")
            if not any(m['bytes'] == "FF C4" for m in self.scanned_markers):
                warnings.append("Missing DHT (Huffman Table) structure.")
            sof_markers = [m for m in self.scanned_markers if m['bytes'] in ("FF C0", "FF C1", "FF C2", "FF C3")]
            if not sof_markers:
                errors.append("Missing SOF (Start of Frame) header.")
            if not any(m['bytes'] == "FF DA" for m in self.scanned_markers):
                errors.append("Missing SOS (Start of Scan) block.")
            if not any(m['bytes'] == "FF D9" for m in self.scanned_markers):
                errors.append("Missing EOI (End of Image) terminator.")
                
        if errors:
            status_text = f"<span style='color:#ff7b72;'><b>✖ Diagnostics:</b> {errors[0]}</span>"
        elif warnings:
            status_text = f"<span style='color:#ffb454;'><b>⚠ Diagnostics:</b> {warnings[0]}</span>"
        else:
            status_text = "<span style='color:#7ee787;'><b>✔ Diagnostics:</b> Healthy JPEG structure verified.</span>"
        self.diagnostic_label.setText(status_text)

    def populate_marker_directory(self):
        self.marker_list.clear()
        for marker in self.scanned_markers:
            item_text = f"{marker['bytes']}  {marker['name']} (0x{marker['offset']:04X})"
            self.marker_list.addItem(item_text)

    def marker_directory_clicked(self, item):
        row = self.marker_list.row(item)
        if 0 <= row < len(self.scanned_markers):
            self.offset = self.scanned_markers[row]['offset']
            self.load_hex_chunk()

    def load_hex_chunk(self):
        if not self.filepath or not os.path.exists(self.filepath):
            self.text_box.setHtml("<span style='color:#ff7b72;'>Error: File not found or invalid path.</span>")
            return
        self.offset = max(0, min(self.offset, self.total_size - 1))
        self.offset = (self.offset // 16) * 16
        try:
            with open(self.filepath, 'rb') as f:
                f.seek(self.offset)
                data = f.read(self.chunk_size)
            html_dump = self.generate_hex_dump_html(data, self.offset)
            self.text_box.setHtml(html_dump)
            total_size_mb = self.total_size / (1024 * 1024)
            self.info_label.setText(f"Offset: <b>0x{self.offset:08X}</b> / 0x{self.total_size:08X} ({total_size_mb:.2f} MB)")
            self.prev_button.setEnabled(self.offset > 0)
            self.next_button.setEnabled(self.offset + self.chunk_size < self.total_size)
        except Exception as e:
            self.text_box.setHtml(f"<span style='color:#ff7b72;'>Error loading hex data: {e}</span>")

    def prev_page(self):
        self.offset -= self.chunk_size
        self.load_hex_chunk()

    def next_page(self):
        self.offset += self.chunk_size
        self.load_hex_chunk()

    def jump_to_offset(self):
        text = self.jump_input.text().strip()
        if not text: return
        try:
            target = int(text, 16) if text.lower().startswith("0x") else int(text, 10)
            if 0 <= target < self.total_size:
                self.offset = target
                self.load_hex_chunk()
                self.jump_input.clear()
            else:
                QMessageBox.warning(self, "Invalid Offset", f"Offset must be between 0 and {self.total_size - 1}.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Format", "Please enter a valid decimal number or hex offset starting with '0x'.")

    def generate_hex_dump_html(self, data, start_offset=0):
        html_lines = [
            "<span style='color: #8b949e;'>Offset    00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F  Decoded Text</span>",
            "<span style='color: #30363d;'>------------------------------------------------------------------</span>"
        ]
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            offset_str = f"{start_offset + i:08X}"
            hex_parts, ascii_parts = [], []
            j = 0
            while j < 16:
                if j < len(chunk):
                    b = chunk[j]
                    is_marker, marker_color = False, None
                    if b == 0xFF and j + 1 < len(chunk):
                        next_b = chunk[j+1]
                        if next_b not in (0x00, 0xFF):
                            is_marker = True
                            if next_b == 0xD8: marker_color = "#58a6ff"
                            elif next_b == 0xD9: marker_color = "#ff7b72"
                            elif next_b == 0xDA: marker_color = "#7ee787"
                            elif next_b in (0xDB, 0xC4): marker_color = "#d2a8ff"
                            elif next_b in (0xC0, 0xC2): marker_color = "#ffb454"
                            else: marker_color = "#ffa198"
                    if is_marker:
                        b1, b2 = chunk[j], chunk[j+1]
                        hex_parts.append(f"<span style='color: {marker_color}; font-weight: bold;'>{b1:02X}</span>")
                        hex_parts.append(f"<span style='color: {marker_color}; font-weight: bold;'>{b2:02X}</span>")
                        for b_val in (b1, b2):
                            char = chr(b_val) if 32 <= b_val < 127 else "."
                            ascii_parts.append(f"<span style='color: {marker_color}; font-weight: bold;'>{char}</span>")
                        j += 2
                        if j == 8: hex_parts.append("")
                        continue
                    else:
                        char = chr(b) if 32 <= b < 127 else "."
                        hex_parts.append(f"{b:02X}")
                        ascii_parts.append(char)
                else:
                    hex_parts.append("  ")
                    ascii_parts.append(" ")
                j += 1
                if j == 8: hex_parts.append("")
            hex_str1 = " ".join(hex_parts[:8])
            hex_str2 = " ".join(hex_parts[8:])
            hex_full = f"{hex_str1}  {hex_str2}"
            ascii_clean = "".join([("&lt;" if c == "<" else "&gt;" if c == ">" else "&amp;" if c == "&" else str(c)) for c in ascii_parts])
            html_lines.append(f"<span style='color: #8b949e;'>{offset_str}</span>  {hex_full}  <span style='color: #8b949e;'>|</span><span style='color: #c9d1d9;'>{ascii_clean}</span><span style='color: #8b949e;'>|</span>")
        return "<pre style='margin: 0; font-family: \"Consolas\", \"Courier New\", monospace;'>" + "<br>".join(html_lines) + "</pre>"

class ExifInspectorDialog(QDialog):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.exif_data = []
        self.init_ui()
        self.load_exif_data()

    def init_ui(self):
        self.setWindowTitle(f"EXIF Metadata Inspector - {os.path.basename(self.filepath)}")
        self.resize(650, 480)
        self.setMinimumSize(550, 350)
        self.setStyleSheet("""
            QDialog { background-color: #0d1117; }
            QLabel { color: #c9d1d9; font-family: 'Segoe UI'; font-size: 10pt; }
            QLineEdit { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 6px; font-size: 9.5pt; }
            QLineEdit:focus { border: 1px solid #58a6ff; }
            QPushButton { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 6px 12px; font-weight: bold; font-size: 9.5pt; }
            QPushButton:hover { background-color: #30363d; }
            QTableWidget { background-color: #161b22; color: #c9d1d9; gridline-color: #30363d; border: 1px solid #30363d; border-radius: 6px; font-family: 'Segoe UI'; font-size: 9.5pt; }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section { background-color: #21262d; color: #58a6ff; padding: 6px; font-weight: bold; border: 1px solid #30363d; }
        """)
        layout = QVBoxLayout(self)
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search Metadata:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter tag names or values...")
        self.search_input.textChanged.connect(self.filter_exif_table)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Tag Name", "Hex ID", "Decoded Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("QTableWidget { alternate-background-color: #0d1117; }")
        layout.addWidget(self.table)
        
        bottom_bar = QHBoxLayout()
        self.status_label = QLabel("Loading metadata...")
        bottom_bar.addWidget(self.status_label)
        bottom_bar.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom_bar.addWidget(close_button)
        layout.addLayout(bottom_bar)

    def load_exif_data(self):
        if not self.filepath or not os.path.exists(self.filepath):
            self.status_label.setText("Error: File not found.")
            return
        try:
            img = Image.open(self.filepath)
            exif = img._getexif()
            if not exif:
                self.table.setRowCount(0)
                self.status_label.setText("No EXIF metadata found in this JPEG APP1 header.")
                return
            self.exif_data = []
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, f"Unknown (Tag {tag_id})")
                hex_id = f"0x{tag_id:04X}"
                if isinstance(value, bytes):
                    try:
                        clean_val = value.decode('utf-8', errors='ignore').strip().replace('\x00', '')
                    except Exception:
                        clean_val = f"Binary Data ({len(value)} bytes)"
                else:
                    clean_val = str(value)
                self.exif_data.append((tag_name, hex_id, clean_val))
            self.exif_data.sort(key=lambda x: x[0])
            self.populate_table(self.exif_data)
            self.status_label.setText(f"Found {len(self.exif_data)} metadata records.")
            self.table.setColumnWidth(0, 180)
            self.table.setColumnWidth(1, 80)
        except Exception as e:
            self.status_label.setText(f"Error reading metadata: {e}")

    def populate_table(self, data_list):
        self.table.setRowCount(len(data_list))
        for row_idx, (name, hex_id, value) in enumerate(data_list):
            item_name = QTableWidgetItem(name)
            item_hex = QTableWidgetItem(hex_id)
            item_val = QTableWidgetItem(value)
            item_hex.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 0, item_name)
            self.table.setItem(row_idx, 1, item_hex)
            self.table.setItem(row_idx, 2, item_val)

    def filter_exif_table(self, query):
        query = query.strip().lower()
        if not query:
            self.populate_table(self.exif_data)
            self.status_label.setText(f"Found {len(self.exif_data)} metadata records.")
            return
        filtered = [(n, h, v) for n, h, v in self.exif_data if query in n.lower() or query in v.lower() or query in h.lower()]
        self.populate_table(filtered)
        self.status_label.setText(f"Matches: {len(filtered)} of {len(self.exif_data)}")

# ======================================================================
# --- MainWindow Implementation ---
# ======================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logical_dpi = QApplication.primaryScreen().logicalDotsPerInch()
        self.dpi_scale = logical_dpi / 96.0
        
        self.setWindowTitle("JPEG MCU Repair Tool")
        self.setGeometry(100, 100, 1200, 800) 
        self.current_mcu_data = None
        self.current_filepath = None
        self.original_filepath = None
        self.post_crop_gray_mcu_count = 0 
        self.vertical_gray_scanlines_to_remove = 0 
        
        self.scene = QGraphicsScene()
        self.view = McuGraphicsView(self.scene)
        self.grid_item = None
        self.view_mode = "YCbCr"
        
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_view_context_menu)
        
        self.action_ycbcr = QAction("YCbCr", self)
        self.action_ycbcr.setShortcut("Ctrl+0")
        self.action_ycbcr.setCheckable(True)
        self.action_ycbcr.setChecked(True)
        self.action_ycbcr.triggered.connect(lambda: self.set_view_mode("YCbCr"))
        self.addAction(self.action_ycbcr)
        
        self.action_y = QAction("Y", self)
        self.action_y.setShortcut("Ctrl+1")
        self.action_y.setCheckable(True)
        self.action_y.triggered.connect(lambda: self.set_view_mode("Y"))
        self.addAction(self.action_y)
        
        self.action_cb = QAction("Cb", self)
        self.action_cb.setShortcut("Ctrl+2")
        self.action_cb.setCheckable(True)
        self.action_cb.triggered.connect(lambda: self.set_view_mode("Cb"))
        self.addAction(self.action_cb)
        
        self.action_cr = QAction("Cr", self)
        self.action_cr.setShortcut("Ctrl+3")
        self.action_cr.setCheckable(True)
        self.action_cr.triggered.connect(lambda: self.set_view_mode("Cr"))
        self.addAction(self.action_cr)
        
        self.action_view_hex = QAction("View Hex", self)
        self.action_view_hex.setShortcut("Ctrl+H")
        self.action_view_hex.triggered.connect(self.show_hex_viewer)
        self.addAction(self.action_view_hex)
        
        self.action_view_exif = QAction("View EXIF Info", self)
        self.action_view_exif.setShortcut("Ctrl+E")
        self.action_view_exif.triggered.connect(self.show_exif_inspector)
        self.addAction(self.action_view_exif)
        
        central_widget = QWidget()
        outer_layout = QHBoxLayout(central_widget)

        left_content_layout = QVBoxLayout()
        left_content_layout.addWidget(self.view)
        outer_layout.addLayout(left_content_layout, 1) 

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setObjectName("RightPanelScroll")
        self.scroll_area.setMinimumWidth(int(340 * self.dpi_scale))

        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(10, 10, 10, 10)
        right_panel_widget.setObjectName("RightPanel") 
        
        # 1. Project Management
        project_management_group = QGroupBox("File Management")
        pm_layout = QVBoxLayout()
        pm_layout.setContentsMargins(10, 20, 10, 10)

        self.open_button = QPushButton("Open")
        self.open_button.setObjectName("PrimaryButton")
        self.open_button.clicked.connect(self.open_file)
        pm_layout.addWidget(self.open_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_to_original)
        self.reset_button.setEnabled(False)
        pm_layout.addWidget(self.reset_button)

        self.toggle_grid_button = QPushButton("Show grid")
        self.toggle_grid_button.clicked.connect(self.toggle_grid_visibility)
        self.toggle_grid_button.setEnabled(False) 
        pm_layout.addWidget(self.toggle_grid_button)
        
        self.view_hex_button = QPushButton("View Hex")
        self.view_hex_button.clicked.connect(self.show_hex_viewer)
        self.view_hex_button.setEnabled(False)
        pm_layout.addWidget(self.view_hex_button)
        
        self.view_exif_button = QPushButton("View EXIF")
        self.view_exif_button.clicked.connect(self.show_exif_inspector)
        self.view_exif_button.setEnabled(False)
        pm_layout.addWidget(self.view_exif_button)
        
        pm_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine)) 
        pm_layout.addWidget(QLabel("Current File:"))
        self.current_file_label_mini = QLabel("No file loaded")
        pm_layout.addWidget(self.current_file_label_mini)
        pm_layout.addWidget(QLabel("Resolution:"))
        self.dim_label_mini = QLabel("--- x ---")
        pm_layout.addWidget(self.dim_label_mini)
        pm_layout.addWidget(QLabel("MCU Size:"))
        self.mcu_label_mini = QLabel("--- x ---")
        pm_layout.addWidget(self.mcu_label_mini)
        project_management_group.setLayout(pm_layout)
        right_panel_layout.addWidget(project_management_group)
        
        # 2. MCU Previews
        mcu_previews_group = QGroupBox("MCU Previews")
        previews_layout = QGridLayout()
        previews_layout.setContentsMargins(10, 20, 10, 10)
        previews_layout.addWidget(QLabel("Active Block"), 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        previews_layout.addWidget(QLabel("Target Block"), 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.pixel_block_preview = BlockPreviewWidget(cols=16, rows=8, is_static_preview=True)
        self.selected_block_preview = BlockPreviewWidget(cols=16, rows=8, is_static_preview=False) 
        previews_layout.addWidget(self.pixel_block_preview, 1, 0)
        previews_layout.addWidget(self.selected_block_preview, 1, 1)
        mcu_previews_group.setLayout(previews_layout)
        right_panel_layout.addWidget(mcu_previews_group)
        
        # 3. Tabs
        repair_tab = QWidget()
        repair_layout = QVBoxLayout(repair_tab)
        repair_layout.setContentsMargins(0, 0, 0, 0) 
        
        mcu_repair_group = QGroupBox("MCU Edit")
        mcu_repair_layout = QGridLayout()
        mcu_repair_layout.setContentsMargins(10, 20, 10, 10)
        mcu_repair_layout.addWidget(QLabel("MCU Block:"), 0, 0)
        
        self.mcu_block_num_input = QSpinBox() 
        self.mcu_block_num_input.setRange(1, 1000) 
        self.mcu_block_num_input.setValue(1) 
        self.mcu_block_num_input.setFixedWidth(int(70 * self.dpi_scale)) 
        mcu_repair_layout.addWidget(self.mcu_block_num_input, 0, 1)
        
        self.insert_button = QPushButton("Insert MCU")
        self.insert_button.clicked.connect(lambda: self.run_repair("insert"))
        self.insert_button.setEnabled(False)
        mcu_repair_layout.addWidget(self.insert_button, 0, 2)
        
        self.delete_button = QPushButton("Delete MCU")
        self.delete_button.clicked.connect(lambda: self.run_repair("delete"))
        self.delete_button.setEnabled(False)
        mcu_repair_layout.addWidget(self.delete_button, 0, 3)

        mcu_repair_layout.addWidget(QLabel("Gray Scanlines Found:"), 1, 0)
        self.gray_scanline_count_label_mini = QLabel("--- / --- (0%)")
        mcu_repair_layout.addWidget(self.gray_scanline_count_label_mini, 1, 1, 1, 3)

        self.execute_header_crop_button = QPushButton("Remove MCU Gray Scanlines")
        self.execute_header_crop_button.clicked.connect(self.remove_gray_scanlines)
        self.execute_header_crop_button.setEnabled(False)
        mcu_repair_layout.addWidget(self.execute_header_crop_button, 2, 0, 1, 4) 
        
        self.auto_align_button = QPushButton("Auto Alignment")
        self.auto_align_button.clicked.connect(self.run_auto_alignment)
        self.auto_align_button.setEnabled(False) 
        mcu_repair_layout.addWidget(self.auto_align_button, 3, 0, 1, 4) 
        
        mcu_repair_group.setLayout(mcu_repair_layout)
        repair_layout.addWidget(mcu_repair_group)
        repair_layout.addStretch(1)

        color_tab = QWidget()
        color_layout = QVBoxLayout(color_tab)
        color_layout.setContentsMargins(0, 0, 0, 0) 
        
        manual_color_group = QGroupBox("Manual Color")
        manual_color_layout = QVBoxLayout()
        manual_color_layout.setContentsMargins(10, 20, 10, 10)
        MIN_VAL, MAX_VAL, DEFAULT_VAL = -2047, 2047, 0

        def create_slider(label_text):
            h_layout = QHBoxLayout()
            label = QLabel(f"{label_text}:")
            label.setFixedWidth(int(30 * self.dpi_scale)) 
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(MIN_VAL, MAX_VAL)
            slider.setValue(DEFAULT_VAL)
            slider.setSingleStep(1)
            slider.setPageStep(64)
            value_label = QLabel(str(DEFAULT_VAL).rjust(5))
            value_label.setFixedWidth(int(65 * self.dpi_scale)) 
            slider.setObjectName(f"{label_text.lower()}_slider") 
            slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v).rjust(5)))
            h_layout.addWidget(label)
            h_layout.addWidget(slider)
            h_layout.addWidget(value_label)
            return slider, h_layout

        self.y_slider, y_layout = create_slider("Y")
        manual_color_layout.addLayout(y_layout)
        self.cb_slider, cb_layout = create_slider("Cb")
        manual_color_layout.addLayout(cb_layout)
        self.cr_slider, cr_layout = create_slider("Cr")
        manual_color_layout.addLayout(cr_layout)
        
        self.cdelta_button = QPushButton("Apply")
        self.cdelta_button.setObjectName("SecondaryButton")
        self.cdelta_button.clicked.connect(self.run_cdelta_repair)
        self.cdelta_button.setEnabled(False)
        manual_color_layout.addWidget(self.cdelta_button)
        manual_color_group.setLayout(manual_color_layout)
        color_layout.addWidget(manual_color_group)

        auto_color_group = QGroupBox("Auto color")
        auto_color_layout = QVBoxLayout()
        auto_color_layout.setContentsMargins(10, 20, 10, 10)
        self.auto_color_button = QPushButton("Apply")
        self.auto_color_button.setObjectName("AccentButton") 
        self.auto_color_button.clicked.connect(self.run_auto_color_correction)
        self.auto_color_button.setEnabled(False)
        auto_color_layout.addWidget(self.auto_color_button) 
        auto_color_group.setLayout(auto_color_layout)
        color_layout.addWidget(auto_color_group)
        color_layout.addStretch(1) 
        
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
        batch_layout.setContentsMargins(10, 10, 10, 10)
        batch_group = QGroupBox("Batch")
        batch_grid = QGridLayout()
        batch_grid.setColumnStretch(0, 0)
        batch_grid.setColumnStretch(1, 1)
        batch_grid.setColumnStretch(2, 0)
        
        batch_grid.addWidget(QLabel("Reference JPEG:"), 0, 0)
        self.reference_jpeg_input = QLineEdit()
        self.reference_jpeg_input.setPlaceholderText("Select a known good JPEG file...")
        batch_grid.addWidget(self.reference_jpeg_input, 0, 1)
        self.select_ref_button = QPushButton("Browse")
        self.select_ref_button.clicked.connect(self.selectReferenceJPEG)
        batch_grid.addWidget(self.select_ref_button, 0, 2)
        
        batch_grid.addWidget(QLabel("Encrypted Folder:"), 1, 0)
        self.encrypted_folder_input = QLineEdit()
        self.encrypted_folder_input.setPlaceholderText("Select folder containing encrypted files...")
        batch_grid.addWidget(self.encrypted_folder_input, 1, 1)
        self.select_folder_button = QPushButton("Browse")
        self.select_folder_button.clicked.connect(self.selectEncryptedFolder)
        batch_grid.addWidget(self.select_folder_button, 1, 2)
        
        self.auto_batch_process_button = QPushButton("Start")
        self.auto_batch_process_button.setObjectName("PrimaryButton")
        self.auto_batch_process_button.clicked.connect(self.repairJPEGs)
        batch_grid.addWidget(self.auto_batch_process_button, 2, 0, 1, 3)

        batch_group.setLayout(batch_grid)
        batch_layout.addWidget(batch_group)
        
        self.progress_bar = QProgressBar(self)
        batch_layout.addWidget(self.progress_bar)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setObjectName("OutputText")
        self.output_text.setFont(QFont("Consolas", 10))
        batch_layout.addWidget(QLabel("Log:"))
        batch_layout.addWidget(self.output_text, 1)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(repair_tab, "Repair")
        self.tab_widget.addTab(color_tab, "Color")
        self.tab_widget.addTab(batch_tab, "Batch")
        right_panel_layout.addWidget(self.tab_widget)
        
        # 4. MCU Info
        selection_info_group = QGroupBox("MCU Info")
        selection_layout = QGridLayout()
        selection_layout.setContentsMargins(10, 20, 10, 10)
        selection_layout.addWidget(QLabel("MCU Address:"), 0, 0)
        self.mcu_coords_label = QLabel("--")
        selection_layout.addWidget(self.mcu_coords_label, 0, 1)
        
        selection_layout.addWidget(QLabel("MCU Index:"), 1, 0)
        self.mcu_index_label = QLabel("--")
        selection_layout.addWidget(self.mcu_index_label, 1, 1)
        
        selection_layout.addWidget(QLabel("Pixel Position:"), 2, 0)
        self.pixel_range_label = QLabel("X: ---, Y: ---")
        selection_layout.addWidget(self.pixel_range_label, 2, 1, 1, 3)

        selection_layout.addWidget(QLabel("YCbCr:"), 3, 0) 
        self.avg_ycbr_label = QLabel("Y: ---, Cb: ---, Cr: ---") 
        selection_layout.addWidget(self.avg_ycbr_label, 3, 1, 1, 3) 
        
        selection_info_group.setLayout(selection_layout)
        right_panel_layout.addWidget(selection_info_group)
        right_panel_layout.addStretch(1) 
        
        self.scroll_area.setWidget(right_panel_widget)
        outer_layout.addWidget(self.scroll_area)
        self.setCentralWidget(central_widget)
        self.apply_dark_theme()

    def apply_dark_theme(self):
        BG_DARK, BG_MID, BG_LIGHT = "#0d1117", "#161b22", "#21262d"
        TEXT_COLOR, TEXT_MUTED = "#c9d1d9", "#8b949e"
        ACCENT_BLUE, ACCENT_BLUE_GLOW = "#58a6ff", "#79c0ff"
        ACCENT_RED, ACCENT_RED_GLOW = "#ff7b72", "#ffa198"
        ACCENT_GREEN, ACCENT_GREEN_GLOW = "#3fb950", "#56d364"
        BORDER_GRAY, BORDER_LIGHT = "#30363d", "#484f58"

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG_DARK))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Base, QColor(BG_MID))
        palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Button, QColor(BG_LIGHT))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_BLUE))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(Qt.GlobalColor.white))
        QApplication.setPalette(palette)

        style_sheet = f"""
        * {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif; font-size: 10pt; }}
        QMainWindow {{ background-color: {BG_DARK}; }}
        QWidget#RightPanel {{ background-color: {BG_DARK}; border-left: 1px solid {BORDER_GRAY}; }}
        QGroupBox {{ font-weight: bold; font-size: 10pt; color: {TEXT_COLOR}; background-color: {BG_MID}; border: 1px solid {BORDER_GRAY}; border-radius: 8px; margin-top: 18px; padding: 10px; }}
        QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 2px 10px; left: 10px; font-size: 9pt; text-transform: uppercase; letter-spacing: 1px; color: {ACCENT_BLUE}; background-color: {BG_DARK}; border: 1px solid {BORDER_GRAY}; border-radius: 4px; }}
        QLabel {{ color: {TEXT_COLOR}; font-size: 10pt; padding: 2px 0; }}
        QTextEdit#OutputText {{ background-color: {BG_DARK}; color: #7ee787; border: 1px solid {BORDER_GRAY}; border-radius: 6px; padding: 8px; font-size: 10pt; }}
        QLineEdit, QSpinBox {{ background-color: {BG_LIGHT}; color: {TEXT_COLOR}; border: 1px solid {BORDER_GRAY}; border-radius: 6px; padding: 6px; font-size: 10pt; selection-background-color: {ACCENT_BLUE}; }}
        QLineEdit:focus, QSpinBox:focus {{ border: 1px solid {ACCENT_BLUE}; }}
        QPushButton {{ background-color: {BG_LIGHT}; color: {TEXT_COLOR}; border: 1px solid {BORDER_GRAY}; border-radius: 6px; padding: 8px 12px; font-weight: bold; font-size: 10pt; }}
        QPushButton:hover {{ background-color: {BORDER_GRAY}; border-color: {BORDER_LIGHT}; }}
        QPushButton:pressed {{ background-color: {BORDER_LIGHT}; }}
        QPushButton:disabled {{ background-color: {BG_DARK}; border-color: {BORDER_GRAY}; color: {TEXT_MUTED}; }}
        QPushButton#PrimaryButton {{ background-color: {ACCENT_BLUE}; color: {BG_DARK}; border: none; font-weight: bold; font-size: 10pt; padding: 10px; }}
        QPushButton#PrimaryButton:hover {{ background-color: {ACCENT_BLUE_GLOW}; }}
        QPushButton#SecondaryButton {{ background-color: {ACCENT_RED}; color: {BG_DARK}; border: none; font-weight: bold; font-size: 10pt; }}
        QPushButton#SecondaryButton:hover {{ background-color: {ACCENT_RED_GLOW}; }}
        QPushButton#AccentButton {{ background-color: {ACCENT_GREEN}; color: {BG_DARK}; border: none; font-weight: bold; font-size: 10pt; }}
        QPushButton#AccentButton:hover {{ background-color: {ACCENT_GREEN_GLOW}; }}
        QSlider::groove:horizontal {{ border: 1px solid {BORDER_GRAY}; height: 6px; background: {BG_DARK}; border-radius: 3px; }}
        QSlider::handle:horizontal {{ background: {ACCENT_BLUE}; border: 1px solid {BORDER_GRAY}; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }}
        QTabWidget::pane {{ border: 1px solid {BORDER_GRAY}; background-color: {BG_MID}; border-radius: 8px; top: -1px; }}
        QTabBar::tab {{ background: {BG_DARK}; color: {TEXT_MUTED}; padding: 8px 16px; border: 1px solid {BORDER_GRAY}; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; text-transform: uppercase; font-weight: bold; font-size: 9pt; }}
        QTabBar::tab:selected {{ background: {BG_MID}; color: {ACCENT_BLUE}; border-bottom: 2px solid {ACCENT_BLUE}; }}
        QTabBar::tab:hover {{ background: {BG_LIGHT}; color: {TEXT_COLOR}; }}
        QProgressBar {{ border: 1px solid {BORDER_GRAY}; border-radius: 6px; text-align: center; color: {TEXT_COLOR}; background-color: {BG_DARK}; font-weight: bold; font-size: 9pt; height: 18px; }}
        QProgressBar::chunk {{ background-color: {ACCENT_BLUE}; border-radius: 5px; }}
        QGraphicsView {{ border: 1px solid {BORDER_GRAY}; background-color: {BG_DARK}; border-radius: 8px; }}
        QScrollArea#RightPanelScroll {{ border: none; background-color: {BG_DARK}; }}
        QScrollBar:vertical {{ border: none; background: {BG_DARK}; width: 8px; }}
        QScrollBar::handle:vertical {{ background: {BORDER_GRAY}; min-height: 20px; border-radius: 4px; }}
        QScrollBar::handle:vertical:hover {{ background: {ACCENT_BLUE}; }}
        QMenu {{ background-color: {BG_MID}; color: {TEXT_COLOR}; border: 1px solid {BORDER_GRAY}; border-radius: 6px; padding: 4px 0px; }}
        QMenu::item {{ padding: 6px 24px 6px 20px; background-color: transparent; }}
        QMenu::item:selected {{ background-color: {BG_LIGHT}; color: {ACCENT_BLUE}; }}
        QMenu::separator {{ height: 1px; background-color: {BORDER_GRAY}; margin: 4px 0px; }}
        """
        self.setStyleSheet(style_sheet)

    def get_mcu_block_pixmap(self, r, c):
        if not self.current_filepath or not self.current_mcu_data: return QPixmap()
        data = self.current_mcu_data
        x_start = c * data['mcu_x']
        y_start = r * data['mcu_y']
        crop_box = (x_start, y_start, x_start + data['mcu_x'], y_start + data['mcu_y'])
        try:
            img = Image.open(self.current_filepath)
            mcu_img = img.crop(crop_box)
            if self.view_mode != "YCbCr":
                ycbcr = mcu_img.convert('YCbCr')
                channels = ycbcr.split()
                target_channel = None
                mode = self.view_mode
                if mode == "Y": target_channel = channels[0]
                elif mode == "Cb": target_channel = channels[1]
                elif mode == "Cr": target_channel = channels[2]
                if target_channel: mcu_img = target_channel
            
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.ReadWrite)
            mcu_img.save(buffer, "PNG") 
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.data())
            return pixmap
        except Exception:
            pass
        return QPixmap()

    def display_hover_info(self, r, c):
        mcu_pixmap = self.get_mcu_block_pixmap(r, c)
        self.pixel_block_preview.update_pixmap(mcu_pixmap)
        self.pixel_block_preview.set_hover_state(True)

    def clear_hover_info(self):
        self.pixel_block_preview.clear_pixmap()
        self.pixel_block_preview.set_hover_state(False)

    def display_mcu_info(self, r, c):
        if not self.current_mcu_data or not self.grid_item: return
        data = self.current_mcu_data
        mcu_pixmap = self.get_mcu_block_pixmap(r, c)
        self.selected_block_preview.update_pixmap(mcu_pixmap)
        
        n_mcu_x = self.grid_item.n_mcu_x
        mcu_index = r * n_mcu_x + c + 1
        x_start = c * data['mcu_x']
        y_start = r * data['mcu_y']
        x_end = min((c + 1) * data['mcu_x'] - 1, data['width'] - 1)
        y_end = min((r + 1) * data['mcu_y'] - 1, data['height'] - 1)
        
        self.selected_mcu_c = c
        self.selected_mcu_r = r
        
        self.mcu_coords_label.setText(f"({c}, {r})")
        self.mcu_index_label.setText(str(mcu_index))
        self.pixel_range_label.setText(f"X: {x_start}-{x_end}, Y: {y_start}-{y_end}")
        
        avg_ycbr = get_mcu_avg_ycbr_values(self.current_filepath, data, r, c)
        if avg_ycbr:
            y, cb, cr = avg_ycbr
            self.avg_ycbr_label.setText(f"Y: {y:.2f}, Cb: {cb:.2f}, Cr: {cr:.2f}")
        else:
            self.avg_ycbr_label.setText("Y: ---, Cb: ---, Cr: --- (Error)")

        self.selected_block_preview.set_active_state(True) 

    def clear_mcu_info(self):
        self.mcu_coords_label.setText("--")
        self.mcu_index_label.setText("--")
        self.pixel_range_label.setText("X: ---, Y: ---")
        self.avg_ycbr_label.setText("Y: ---, Cb: ---, Cr: ---") 
        self.selected_mcu_c = 0
        self.selected_mcu_r = 0
        self.selected_block_preview.clear_pixmap()
        self.selected_block_preview.set_active_state(False)
        self.clear_hover_info()

    def load_and_display_image(self, filepath):
        image_pixmap = QPixmap(filepath)
        if image_pixmap.isNull():
            if filepath == self.original_filepath:
                QMessageBox.critical(self, "Error", "Failed to load image file.")
                self.reset_gui()
            else:
                QMessageBox.critical(self, "Error", f"Failed to load repaired image: {os.path.basename(filepath)}.")
            return

        mcu_data = get_jpeg_mcu_data(filepath)
        if mcu_data:
            mcu_data['width'] = image_pixmap.width()
            mcu_data['height'] = image_pixmap.height()
            self.current_mcu_data = mcu_data
            self.current_filepath = filepath
            
            self.dim_label_mini.setText(f"{mcu_data['width']} x {mcu_data['height']}")
            self.mcu_label_mini.setText(f"{mcu_data['mcu_x']} x {mcu_data['mcu_y']}")
            self.current_file_label_mini.setText(os.path.basename(filepath))
            
            try:
                vertical_gray_count, total_scanlines, _ = count_gray_mcu_scanlines(filepath, mcu_data)
                self.vertical_gray_scanlines_to_remove = vertical_gray_count
                if total_scanlines > 0:
                    percentage = (vertical_gray_count / total_scanlines) * 100
                    self.gray_scanline_count_label_mini.setText(f"{vertical_gray_count} / {total_scanlines} ({percentage:.1f}%)")
                    self.execute_header_crop_button.setEnabled(vertical_gray_count > 0)
                else:
                    self.gray_scanline_count_label_mini.setText("0 / 0 (0%)")
                    self.execute_header_crop_button.setEnabled(False)
            except Exception as e:
                self.gray_scanline_count_label_mini.setText("Error / --- (0%)")
                self.execute_header_crop_button.setEnabled(False)
            
            try:
                horizontal_gray_mcu_count = analyze_last_scanline_mcus(filepath, mcu_data)
                self.post_crop_gray_mcu_count = horizontal_gray_mcu_count
                is_initial_file = (self.original_filepath == filepath)
                if not is_initial_file and horizontal_gray_mcu_count > 0:
                    self.auto_align_button.setEnabled(True)
                    insert_blocks = horizontal_gray_mcu_count + 1 
                    self.auto_align_button.setText(f"Auto Alignment (Insert {insert_blocks} Blocks at Header)")
                else:
                    self.auto_align_button.setEnabled(False)
                    self.auto_align_button.setText("Auto Alignment")
            except Exception as e:
                self.post_crop_gray_mcu_count = 0
                self.auto_align_button.setEnabled(False)

            display_pixmap = self.get_channel_pixmap(filepath, self.view_mode)
            self.scene.clear()
            self.grid_item = McuGridItem(mcu_data, display_pixmap, self)
            self.scene.addItem(self.grid_item)
            self.scene.setSceneRect(self.grid_item.boundingRect())
            self.view.fitInView(self.grid_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.grid_item.setFocus(Qt.FocusReason.NoFocusReason)
            
            self.insert_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            self.reset_button.setEnabled(True) 
            self.toggle_grid_button.setEnabled(True) 
            self.view_hex_button.setEnabled(True)
            self.view_exif_button.setEnabled(True)
            self.cdelta_button.setEnabled(True) 
            self.auto_color_button.setEnabled(True) 
            self.toggle_grid_button.setText("Show MCU Grid")
            
            self.clear_mcu_info() 
            self.grid_item.mousePressEvent(self._create_fake_event()) 
        else:
            self.reset_gui()

    def set_view_mode(self, mode):
        self.view_mode = mode
        self.action_ycbcr.setChecked(mode == "YCbCr")
        self.action_y.setChecked(mode == "Y")
        self.action_cb.setChecked(mode == "Cb")
        self.action_cr.setChecked(mode == "Cr")
        self.update_image_view_mode()
        c = getattr(self, 'selected_mcu_c', 0)
        r = getattr(self, 'selected_mcu_r', 0)
        self.display_mcu_info(r, c)

    def update_image_view_mode(self):
        if not self.current_filepath or not self.grid_item: return
        display_pixmap = self.get_channel_pixmap(self.current_filepath, self.view_mode)
        self.grid_item.pixmap = display_pixmap
        self.grid_item.update()

    def get_channel_pixmap(self, filepath, mode):
        if not filepath or not os.path.exists(filepath): return QPixmap()
        if mode == "YCbCr": return QPixmap(filepath)
        try:
            img = Image.open(filepath)
            ycbcr = img.convert('YCbCr')
            channels = ycbcr.split()
            target_channel = None
            if mode == "Y": target_channel = channels[0]
            elif mode == "Cb": target_channel = channels[1]
            elif mode == "Cr": target_channel = channels[2]
            if target_channel:
                img_data = target_channel.tobytes()
                qimg = QImage(img_data, target_channel.size[0], target_channel.size[1], target_channel.size[0], QImage.Format.Format_Grayscale8)
                return QPixmap.fromImage(qimg)
        except Exception:
            pass
        return QPixmap(filepath)

    def show_view_context_menu(self, pos):
        if not self.current_filepath: return
        menu = QMenu(self.view)
        menu.addAction(self.action_ycbcr)
        menu.addSeparator()
        menu.addAction(self.action_y)
        menu.addAction(self.action_cb)
        menu.addAction(self.action_cr)
        menu.addSeparator()
        action_menu_hex = QAction("View Hex", self)
        action_menu_hex.triggered.connect(self.show_hex_viewer)
        menu.addAction(action_menu_hex)
        action_menu_exif = QAction("View EXIF Info", self)
        action_menu_exif.triggered.connect(self.show_exif_inspector)
        menu.addAction(action_menu_exif)
        menu.exec(self.view.mapToGlobal(pos))

    def show_hex_viewer(self):
        if not self.current_filepath or not os.path.exists(self.current_filepath):
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return
        HexViewerDialog(self.current_filepath, self).exec()

    def show_exif_inspector(self):
        if not self.current_filepath or not os.path.exists(self.current_filepath):
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return
        ExifInspectorDialog(self.current_filepath, self).exec()

    def reset_gui(self):
        self.current_file_label_mini.setText("No file loaded")
        self.dim_label_mini.setText("--- x ---")
        self.mcu_label_mini.setText("--- x ---")
        self.gray_scanline_count_label_mini.setText("--- / --- (0%)") 
        self.auto_align_button.setText("Auto Alignment")
        self.post_crop_gray_mcu_count = 0 
        self.vertical_gray_scanlines_to_remove = 0 
        self.scene.clear()
        self.current_mcu_data = None
        self.current_filepath = None
        self.original_filepath = None
        self.grid_item = None
        self.clear_mcu_info()
        self.avg_ycbr_label.setText("Y: ---, Cb: ---, Cr: ---")
        self.insert_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.toggle_grid_button.setEnabled(False) 
        self.view_hex_button.setEnabled(False)
        self.view_exif_button.setEnabled(False)
        self.cdelta_button.setEnabled(False) 
        self.execute_header_crop_button.setEnabled(False) 
        self.auto_align_button.setEnabled(False) 
        self.auto_color_button.setEnabled(False) 
        self.mcu_block_num_input.setValue(1) 
        self.y_slider.setValue(0)
        self.cb_slider.setValue(0)
        self.cr_slider.setValue(0)
        self.progress_bar.setValue(0)
        self.output_text.clear()
        self.selected_block_preview.set_active_state(False)
        self.clear_hover_info()

    def toggle_grid_visibility(self):
        if self.grid_item:
            self.grid_item.grid_visible = not self.grid_item.grid_visible
            self.grid_item.update()
            self.toggle_grid_button.setText("Hide MCU Grid" if self.grid_item.grid_visible else "Show MCU Grid")
        
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open JPEG Image", os.path.expanduser("~"), "JPEG Files (*.jpg *.jpeg)")
        if not filepath: return
        self.original_filepath = filepath
        self.current_filepath = filepath
        self.load_and_display_image(filepath)

    def reset_to_original(self):
        if not self.original_filepath: return
        if self.current_filepath == self.original_filepath: return
        self.current_filepath = self.original_filepath
        self.load_and_display_image(self.original_filepath)
        self.mcu_block_num_input.setValue(1) 
        self.y_slider.setValue(0)
        self.cb_slider.setValue(0)
        self.cr_slider.setValue(0)

    def _create_fake_event(self):
        class FakeMouseEvent:
            def button(self): return Qt.MouseButton.LeftButton
            def pos(self): return QPointF(0, 0)
        return FakeMouseEvent()
    
    def get_repair_filepaths(self):
        base_path = self.original_filepath 
        if not base_path: return None, None
        input_dir = os.path.dirname(base_path) 
        repaired_dir = os.path.join(input_dir, "Repaired")
        try:
            os.makedirs(repaired_dir, exist_ok=True)
        except Exception:
            return None, None
        original_filename = os.path.basename(base_path)
        output_file = os.path.normpath(os.path.abspath(os.path.join(repaired_dir, original_filename)))
        input_file = os.path.normpath(os.path.abspath(self.current_filepath))
        return input_file, output_file

    def execute_jpegrepair(self, command, operation):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exe_path = os.path.normpath(os.path.join(script_dir, "jpegrepair.exe"))

        if not os.path.exists(exe_path):
            return False, f"Executable not found: {exe_path}. Ensure it is in the same folder as this script."
        
        command.insert(0, exe_path)
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            if process.returncode == 0:
                return True, ""
            else:
                error_output = process.stderr if process.stderr else "No specific error output."
                return False, f"Execution failed for {operation} with return code {process.returncode}.\nError:\n{error_output}"
        except Exception as e:
            return False, f"An unexpected error occurred during execution: {e}"

    def remove_gray_scanlines(self):
        if not self.current_filepath or not self.current_mcu_data: return
        scanlines_to_remove = self.vertical_gray_scanlines_to_remove
        if scanlines_to_remove <= 0: return
            
        input_file, output_file = self.get_repair_filepaths() 
        if not input_file: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        success = crop_jpeg_by_header(input_file, output_file, scanlines_to_remove)
        QApplication.restoreOverrideCursor()

        if success:
            self.load_and_display_image(output_file) 
        else:
            self.auto_align_button.setEnabled(False)
            
    def run_auto_alignment(self):
        if not self.current_filepath or not self.current_mcu_data: return
        gray_count_remaining = self.post_crop_gray_mcu_count
        mcu_block_num = gray_count_remaining + 1
        if mcu_block_num <= 1: return

        input_file, output_file = self.get_repair_filepaths() 
        if not input_file: return
        command = [input_file, output_file, "dest", "0", "0", "insert", str(mcu_block_num)]
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        success, error_msg = self.execute_jpegrepair(command, "Auto Alignment")
        QApplication.restoreOverrideCursor()
        
        if success:
            self.load_and_display_image(output_file)
            self.auto_align_button.setEnabled(False) 
        else:
            self.auto_align_button.setEnabled(True) 
            
    def run_auto_color_correction(self):
        if not self.current_filepath: return
        input_file, output_file = self.get_repair_filepaths() 
        if not input_file: return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            original_img = Image.open(input_file)
            corrected_img = photodemon_autocorrect_image(original_img)
            corrected_img.save(output_file, quality=92, optimize=True) 
            success = True
        except Exception as e:
            success = False
            
        QApplication.restoreOverrideCursor()
        if success:
            self.load_and_display_image(output_file)
            self.y_slider.setValue(0)
            self.cb_slider.setValue(0)
            self.cr_slider.setValue(0)

    def run_cdelta_repair(self):
        if not self.current_filepath: return
        deltas = {0: self.y_slider.value(), 1: self.cb_slider.value(), 2: self.cr_slider.value()}
        if all(v == 0 for v in deltas.values()): return

        _, final_output_file = self.get_repair_filepaths() 
        if not final_output_file: return
        active_components = [i for i, v in deltas.items() if v != 0]

        base_dir = os.path.dirname(final_output_file)
        ext = os.path.splitext(final_output_file)[1]
        temp_output_file = os.path.join(base_dir, f"temp_cdelta_{os.getpid()}{ext}")
        current_input_file = self.current_filepath
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        all_successful = True
        try:
            for comp_index in active_components:
                value = deltas[comp_index]
                is_last = (comp_index == active_components[-1])
                output_path = final_output_file if is_last else temp_output_file
                command = [current_input_file, output_path, "dest", "0", "0", "cdelta", str(comp_index), str(value)]
                success, _ = self.execute_jpegrepair(command, f"cdelta {comp_index}")
                if not success:
                    all_successful = False
                    break 
                current_input_file = output_path
        finally:
            QApplication.restoreOverrideCursor()
            if os.path.exists(temp_output_file): os.remove(temp_output_file)
        
        if all_successful:
            self.load_and_display_image(final_output_file)

    def run_repair(self, operation):
        if not self.current_filepath: return
        mcu_block_num = self.mcu_block_num_input.value()
        c, r = getattr(self, 'selected_mcu_c', 0), getattr(self, 'selected_mcu_r', 0)
        input_file, output_file = self.get_repair_filepaths()
        if not input_file: return
        
        command = [input_file, output_file, "dest", str(c), str(r), operation, str(mcu_block_num)]
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        success, _ = self.execute_jpegrepair(command, operation)
        QApplication.restoreOverrideCursor()
        
        if success:
            self.load_and_display_image(output_file)

    def selectReferenceJPEG(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Reference JPEG", "", "JPEG Files (*.jpg *.jpeg);;All Files (*)")
        if fileName: self.reference_jpeg_input.setText(fileName)

    def selectEncryptedFolder(self):
        folderName = QFileDialog.getExistingDirectory(self, "Select Encrypted Folder")
        if folderName: self.encrypted_folder_input.setText(folderName)

    def find_ffda_offset(self, data):
        ffda_offset = data.rfind(b'\xFF\xDA')
        if ffda_offset == -1: raise ValueError("FFDA marker not found.")
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
                elif marker == b'\xFF\xDA':
                    break
                elif marker not in (b'\xFF\xD8', b'\xFF\xD9'):
                    if len(data) < i + 4: break
                    try:
                        length = int.from_bytes(data[i+2:i+4], 'big') + 2
                        i += length
                    except ValueError:
                        break
                    continue
            i += 1
        return data

    def _initial_file_manipulation(self, reference_path, encrypted_path, output_path):
        with open(encrypted_path, 'rb') as f: encrypted_data = f.read()
        with open(reference_path, 'rb') as f: reference_data = f.read()
        ffda_offset = self.find_ffda_offset(reference_data)
        cut_reference_data = reference_data[:ffda_offset + 12]
        repaired_data = cut_reference_data + encrypted_data[153605:]
        repaired_data = self.remove_exif(repaired_data)
        repaired_data = repaired_data[:-334]
        with open(output_path, 'wb') as f: f.write(repaired_data)

    def _batch_step_header_crop(self, input_path, output_path):
        mcu_data = get_jpeg_mcu_data(input_path)
        if not mcu_data: return False, "Error reading MCU data."
        vertical_gray_count, _, _ = count_gray_mcu_scanlines(input_path, mcu_data)
        if vertical_gray_count <= 0:
            shutil.copy2(input_path, output_path)
            return True, output_path
        success = crop_jpeg_by_header(input_path, output_path, vertical_gray_count)
        return success, output_path

    def _batch_step_auto_align(self, input_path, output_path):
        mcu_data = get_jpeg_mcu_data(input_path)
        if not mcu_data: 
            shutil.copy2(input_path, output_path)
            return True, output_path
        horizontal_gray_mcu_count = analyze_last_scanline_mcus(input_path, mcu_data)
        mcu_block_num = horizontal_gray_mcu_count + 1
        if mcu_block_num <= 1:
            shutil.copy2(input_path, output_path)
            return True, output_path
        command = [input_path, output_path, "dest", "0", "0", "insert", str(mcu_block_num)]
        success, error_msg = self.execute_jpegrepair(command, "Auto Align")
        return success, error_msg

    def _batch_step_auto_color(self, input_path, output_path):
        try:
            original_img = Image.open(input_path)
            corrected_img = photodemon_autocorrect_image(original_img)
            corrected_img.save(output_path, quality=92, optimize=True) 
            return True, output_path
        except Exception as e:
            return False, str(e)

    def process_jpeg_batch(self, input_file, reference_path, output_file):
        base_dir = os.path.dirname(output_file)
        temp_pre_process_file = os.path.join(base_dir, f"temp_pre_process_{os.getpid()}_{os.path.basename(input_file)}")
        try:
            self._initial_file_manipulation(reference_path, input_file, temp_pre_process_file)
        except Exception as e:
            return False, str(e)
        
        current_input_file = temp_pre_process_file
        temp_files_to_cleanup = [temp_pre_process_file]
        try:
            temp_crop_file = os.path.join(base_dir, f"temp_batch_{os.getpid()}_crop.jpg")
            success, result = self._batch_step_header_crop(current_input_file, temp_crop_file)
            if not success: return False, result
            current_input_file = result
            temp_files_to_cleanup.append(temp_crop_file)
            
            temp_align_file = os.path.join(base_dir, f"temp_batch_{os.getpid()}_align.jpg")
            success, result = self._batch_step_auto_align(current_input_file, temp_align_file)
            if not success: return False, result
            current_input_file = result
            temp_files_to_cleanup.append(temp_align_file)
            
            success, result = self._batch_step_auto_color(current_input_file, output_file)
            if not success: return False, result
            return True, output_file
        finally:
            for f in temp_files_to_cleanup:
                if os.path.exists(f):
                    try: os.remove(f)
                    except Exception: pass

    def repairJPEGs(self):
        reference_jpeg = self.reference_jpeg_input.text().strip()
        encrypted_folder = self.encrypted_folder_input.text().strip()
        if not os.path.exists(reference_jpeg) or not os.path.isdir(encrypted_folder):
            QMessageBox.critical(self, "Error", "Please check your paths.")
            return

        repaired_folder = os.path.join(encrypted_folder, "Repaired")
        os.makedirs(repaired_folder, exist_ok=True)
        pattern = re.compile(r".*\.JPG\..{4}$", re.I)
        encrypted_files = [f for f in os.listdir(encrypted_folder) if pattern.match(f)]
        if not encrypted_files: return

        self.progress_bar.setMaximum(len(encrypted_files))
        self.progress_bar.setValue(0)
        self.output_text.clear()
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        successful_count = 0
        total_count = len(encrypted_files)
        try:
            for i, encrypted_file in enumerate(encrypted_files):
                input_file_path = os.path.join(encrypted_folder, encrypted_file)
                output_name = os.path.splitext(encrypted_file)[0].rsplit('.', 1)[0] + '.JPG'
                output_file_path = os.path.join(repaired_folder, output_name)
                
                self.output_text.append(f"Starting: {encrypted_file} -> {output_name}")
                success, _ = self.process_jpeg_batch(input_file_path, reference_jpeg, output_file_path)
                if success:
                    successful_count += 1
                    self.original_filepath = output_file_path
                    self.load_and_display_image(output_file_path) 
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.grid_item:
            self.view.fitInView(self.grid_item, Qt.AspectRatioMode.KeepAspectRatio)

if __name__ == '__main__':
    if hasattr(sys, 'frozen') and sys.platform == 'win32':
        qt_plugin_path = os.path.join(os.path.dirname(sys.executable), 'PyQt6', 'Qt6', 'plugins')
        if os.path.isdir(qt_plugin_path):
             os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
             
    app = QApplication(sys.argv)
    app_font = app.font()
    app_font.setFamily("Segoe UI")
    app_font.setPointSize(10)
    app.setFont(app_font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
