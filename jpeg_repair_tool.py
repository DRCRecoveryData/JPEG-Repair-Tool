import sys
import struct
import os
import subprocess
import shutil
import numpy as np
import re # ADDED: For file pattern matching in batch process
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QGridLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QLineEdit, QHBoxLayout, QSlider, QFrame, QGroupBox, QTabWidget,
    QSpinBox, # ADDED: QSpinBox
    QTextEdit, # ADDED: For batch process log
    QProgressBar # ADDED: For batch process progress
)
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QPalette, QMouseEvent
from PyQt6.QtCore import Qt, QRectF, QRect, QPointF, QBuffer, QIODevice

# --- Pillow Import ---
try:
    from PIL import Image
except ImportError:
    class PillowStub:
        @staticmethod
        def open(path): raise ImportError("Pillow not installed. Please run 'pip install Pillow'")
    Image = PillowStub


# ======================================================================
# --- JPEG HEADER & SCANLINE LOGIC (Utility functions remain the same) ---
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
                
                # Check for SOF0 (Baseline DCT) or SOF2 (Progressive DCT)
                if marker in (b'\xff\xc0', b'\xff\xc2'):
                    length_bytes = f.read(2)
                    if len(length_bytes) < 2: return None
                    
                    segment_length = struct.unpack('>H', length_bytes)[0]
                    sof_data = f.read(segment_length - 2)
                    
                    if len(sof_data) != (segment_length - 2): return None
                    break
                
                # Skip other markers (0xFFXX where XX is not D8, D9, DA)
                if marker[0] == 0xff and marker[1] not in (0x00, 0xd8, 0xd9, 0xda):
                    segment_length = struct.unpack('>H', f.read(2))[0]
                    f.seek(segment_length - 2, 1)
                elif marker[0] == 0xff and marker[1] in (0xd8, 0xd9): continue
                else: f.seek(-1, 1)

            height_y, width_x = struct.unpack('>HH', sof_data[1:5])
            
            # Simplified component check for common YCbCr 4:2:0/4:4:4
            num_components = sof_data[5]

            # Assume 3 components (Y, Cb, Cr)
            if num_components != 3: return None
            
            # Read Y component sampling factor byte (index 7 from SOF data)
            y_sampling_factor_byte = sof_data[7]
            y_hor = (y_sampling_factor_byte >> 4) & 0x0F
            y_ver = y_sampling_factor_byte & 0x0F
            
            # MCU size in pixels
            mcu_x = y_hor * 8
            mcu_y = y_ver * 8
            
            # Number of MCUs required to cover the width/height
            n_mcu_x = (width_x + mcu_x - 1) // mcu_x
            n_mcu_y = (height_y + mcu_y - 1) // mcu_y
            
            return {
                "width": width_x, "height": height_y, "mcu_x": mcu_x, 
                "mcu_y": mcu_y, "n_mcu_x": n_mcu_x, "n_mcu_y": n_mcu_y,
            }

    except Exception:
        return None

def is_mcu_scanline_gray(pixels, gray_tolerance=10, color_std_dev_threshold=5):
    """
    Checks if a 2D array of YCbCr pixels (whether full scanline or single MCU) is gray using NumPy.
    """
    if pixels.size == 0:
        return False

    # Check the Cb and Cr components (indices 1 and 2)
    Cb = pixels[:, 1]
    Cr = pixels[:, 2]

    # Chrominance Neutrality Check: Max deviation from 128 (neutral)
    cb_deviation = np.abs(Cb.astype(int) - 128)
    cr_deviation = np.abs(Cr.astype(int) - 128)
    
    max_cb_dev = np.max(cb_deviation)
    max_cr_dev = np.max(cr_deviation)
    is_color_neutral = (max_cb_dev <= gray_tolerance) and (max_cr_dev <= gray_tolerance)
    
    # Chrominance Uniformity Check: Low standard deviation
    std_cb = np.std(Cb)
    std_cr = np.std(Cr)
    is_color_uniform = (std_cb <= color_std_dev_threshold) and (std_cr <= color_std_dev_threshold)
    
    return is_color_neutral and is_color_uniform

# --- FIXED: Iterates backward to find contiguous gray blocks at the footer ---
def count_gray_mcu_scanlines(filepath, mcu_data, gray_tolerance=10, color_std_dev_threshold=5):
    """
    (Used for Header Crop) Iterates backward through vertical MCU scanlines 
    and counts how many contiguous gray scanlines are at the end (footer).
    """
    if not mcu_data: return 0, 0, []

    try:
        img = Image.open(filepath).convert('YCbCr')
        img_array = np.array(img)
        height = img_array.shape[0]
        mcu_y = mcu_data['mcu_y']
        total_mcu_scanlines = mcu_data['n_mcu_y'] 
        
        gray_scanline_count = 0
        gray_scanline_indices = []

        # Iterate backward from the last scanline
        for i in range(total_mcu_scanlines - 1, -1, -1):
            start_row = i * mcu_y
            end_row = min((i + 1) * mcu_y, height)

            scanline_data = img_array[start_row:end_row, :, :]
            pixels = scanline_data.reshape(-1, 3)

            if is_mcu_scanline_gray(pixels, gray_tolerance, color_std_dev_threshold):
                gray_scanline_count += 1
                gray_scanline_indices.insert(0, i) # Keep indices in ascending order
            else:
                 # Break on the first non-gray scanline encountered from the bottom
                 break 

        return gray_scanline_count, total_mcu_scanlines, gray_scanline_indices
        
    except Exception:
        # If Pillow/NumPy fails, return 0
        return 0, 0, []
        
        
def analyze_last_scanline_mcus(filepath, mcu_data, gray_tolerance=10, color_std_dev_threshold=5):
    """
    (Used for Auto Alignment) Analyzes the individual MCUs in the last vertical scanline 
    of the image to count 'Gray MCUs Found' (horizontal analysis).
    """
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

        # Iterate horizontally across MCUs in the last scanline
        for i in range(n_mcu_x):
            start_col = i * mcu_x
            end_col = min((i + 1) * mcu_x, width)

            # Extract the current MCU block
            mcu_block = img_array[start_row:end_row, start_col:end_col, :]
            pixels = mcu_block.reshape(-1, 3)

            # Check if the MCU block is gray
            if is_mcu_scanline_gray(pixels, gray_tolerance, color_std_dev_threshold):
                gray_mcu_count += 1
                
        return gray_mcu_count
        
    except Exception:
        return 0

# --- PhotoDemon Clarity Lookup Table Generator ---

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
        
        # Crop the lookup value to [0, 255] range
        gray = np.clip(gray, 0, 255)
            
        contrastLookup[x] = int(round(gray))
        
    return contrastLookup

# --- Combined PhotoDemon Auto-Correction Logic (WB + Clarity) ---
def photodemon_autocorrect_image(img: Image.Image) -> Image.Image:
    """Applies PhotoDemon's primary auto-correction steps: WB and Midtone Contrast."""
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    np_img = np.array(img, dtype=np.uint8)
    
    # 1. White Balance (Independent Channel Histogram Stretch, 0.05% threshold)
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
    
    # 2. Clarity/Midtone Contrast (Lookup Table Application)
    clarity_lookup = _clarity_lookup_table()
    
    # Apply lookup table to all channels
    np_img[:, :, 0] = clarity_lookup[np_img[:, :, 0]] # Red
    np_img[:, :, 1] = clarity_lookup[np_img[:, :, 1]] # Green
    np_img[:, :, 2] = clarity_lookup[np_img[:, :, 2]] # Blue
    
    return Image.fromarray(np_img, 'RGB')

# ======================================================================
# --- CROP/HEADER MODIFICATION LOGIC ---
# ======================================================================

def find_sof_height_position(filepath):
    """
    Scans the JPEG file to find the byte position of the Height field
    within the SOF segment (0xFFC0 or 0xFFC2).
    """
    if not os.path.exists(filepath): return None
    
    try:
        with open(filepath, 'rb') as f:
            while True:
                marker_pos = f.tell()
                marker = f.read(2)
                if not marker: return None
                
                if marker in (b'\xff\xc0', b'\xff\xc2'):
                    # The height field is 5 bytes after the SOF marker
                    # 2 bytes for marker, 2 bytes for length, 1 byte for precision
                    return marker_pos + 5 
                
                if marker[0] == 0xff and marker[1] not in (0x00, 0xd8, 0xd9, 0xda):
                    segment_length = struct.unpack('>H', f.read(2))[0]
                    f.seek(segment_length - 2, 1)
                elif marker[0] == 0xff and marker[1] in (0xd8, 0xd9): continue
                else: f.seek(-1, 1)
    except Exception:
        return None

def crop_jpeg_by_header(source_filepath, output_filepath, scanlines_to_remove):
    """
    Copies the source file and modifies the Height field in the SOF segment
    of the new file based on the number of MCU scanlines to remove.
    """
    mcu_data = get_jpeg_mcu_data(source_filepath)

    if not mcu_data:
        print("Error: Failed to read original image dimensions and MCU size.", file=sys.stderr)
        return False

    original_height = mcu_data['height']
    mcu_y = mcu_data['mcu_y']
    
    pixels_to_remove = scanlines_to_remove * mcu_y
    calculated_new_height = original_height - pixels_to_remove
    
    if calculated_new_height <= 0 or calculated_new_height >= original_height:
        print(f"Error: Invalid crop. New height ({calculated_new_height}) is not smaller than original or is zero/negative. Aborting.", file=sys.stderr)
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
    # (Unchanged)
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
            painter.fillRect(h_rect, QColor(0, 119, 255, 60)) 
        
        if self.selected_mcu_coords:
            r, c = self.selected_mcu_coords
            
            x = c * self.mcu_x
            y = r * self.mcu_y
            w = self.mcu_x
            h = self.mcu_y
            
            highlight_rect = QRectF(x, y, w, h)
            
            painter.fillRect(highlight_rect, QColor(255, 60, 60, 70)) 
            painter.setPen(QPen(QColor(255, 60, 60), 3)) 
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
    # (Unchanged)
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
    
    CELL_SIZE = 8
    
    def __init__(self, cols=16, rows=8, is_static_preview=False, parent=None):
        super().__init__(parent)
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
        
        painter.fillRect(self.rect(), QColor(45, 45, 55)) 

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
                        painter.fillRect(rect, QColor(50, 50, 60)) 
                    else:
                        painter.fillRect(rect, QColor(40, 40, 50))

        grid_line_color = QColor(60, 60, 70) 
        painter.setPen(QPen(grid_line_color, 1)) 
        for r in range(self.rows + 1):
            painter.drawLine(1, r * self.CELL_SIZE + 1, self.cols * self.CELL_SIZE + 1, r * self.CELL_SIZE + 1)
        for c in range(self.cols + 1):
            painter.drawLine(c * self.CELL_SIZE + 1, 1, c * self.CELL_SIZE + 1, self.rows * self.CELL_SIZE + 1)
        
        default_pen = QPen(QColor(80, 80, 90), 1)
        
        if self.is_static_preview and self.is_hovered:
            painter.setPen(QPen(QColor(0, 119, 255), 2)) 
            painter.drawRect(0, 0, self.width()-1, self.height()-1)
        elif not self.is_static_preview and self.is_selected:
            painter.setPen(QPen(QColor(255, 60, 60), 2)) 
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


# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JPEG MCU Repair & Alignment Tool (Modern UI)")
        self.setGeometry(100, 100, 1200, 800) 
        self.current_mcu_data = None
        self.current_filepath = None
        self.original_filepath = None
        self.post_crop_gray_mcu_count = 0 
        self.vertical_gray_scanlines_to_remove = 0 
        
        self.scene = QGraphicsScene()
        self.view = McuGraphicsView(self.scene)
        self.grid_item = None
        
        central_widget = QWidget()
        outer_layout = QHBoxLayout(central_widget)

        # --- Left/Main Content Area ---
        left_content_layout = QVBoxLayout()
        left_content_layout.addWidget(self.view)
        outer_layout.addLayout(left_content_layout, 1) 

        # --- Right Panel (Controls) ---
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(10, 10, 10, 10)
        right_panel_widget.setObjectName("RightPanel") 
        
        # 1. Project Management
        project_management_group = QGroupBox("Project Management")
        pm_layout = QVBoxLayout()
        pm_layout.setContentsMargins(10, 20, 10, 10)

        self.open_button = QPushButton("Open JPEG File...")
        self.open_button.setObjectName("PrimaryButton")
        self.open_button.clicked.connect(self.open_file)
        pm_layout.addWidget(self.open_button)

        self.reset_button = QPushButton("Reset to Original")
        self.reset_button.clicked.connect(self.reset_to_original)
        self.reset_button.setEnabled(False)
        pm_layout.addWidget(self.reset_button)

        self.toggle_grid_button = QPushButton("Show MCU Grid")
        self.toggle_grid_button.clicked.connect(self.toggle_grid_visibility)
        self.toggle_grid_button.setEnabled(False) 
        pm_layout.addWidget(self.toggle_grid_button)
        
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
        previews_layout.addWidget(QLabel("HOVERED"), 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        previews_layout.addWidget(QLabel("SELECTED"), 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.pixel_block_preview = BlockPreviewWidget(cols=16, rows=8, is_static_preview=True)
        self.selected_block_preview = BlockPreviewWidget(cols=16, rows=8, is_static_preview=False) 
        
        previews_layout.addWidget(self.pixel_block_preview, 1, 0)
        previews_layout.addWidget(self.selected_block_preview, 1, 1)

        mcu_previews_group.setLayout(previews_layout)
        right_panel_layout.addWidget(mcu_previews_group)
        
        # 3. Tab Widget for Repair/Color Correction
        
        # --- Repair/Alignment Tab ---
        repair_tab = QWidget()
        repair_layout = QVBoxLayout(repair_tab)
        repair_layout.setContentsMargins(0, 0, 0, 0) 
        
        # MCU Repair/Alignment
        mcu_repair_group = QGroupBox("MCU Repair/Alignment")
        mcu_repair_layout = QGridLayout()
        mcu_repair_layout.setContentsMargins(10, 20, 10, 10)
        
        mcu_repair_layout.addWidget(QLabel("MCU Block Number (k):"), 0, 0)
        
        self.mcu_block_num_input = QSpinBox() 
        self.mcu_block_num_input.setRange(1, 1000) 
        self.mcu_block_num_input.setValue(1) 
        self.mcu_block_num_input.setFixedWidth(70) 
        mcu_repair_layout.addWidget(self.mcu_block_num_input, 0, 1)
        
        self.insert_button = QPushButton("Insert MCU")
        self.insert_button.clicked.connect(lambda: self.run_repair("insert"))
        self.insert_button.setEnabled(False)
        mcu_repair_layout.addWidget(self.insert_button, 0, 2)
        self.delete_button = QPushButton("Delete MCU")
        self.delete_button.clicked.connect(lambda: self.run_repair("delete"))
        self.delete_button.setEnabled(False)
        mcu_repair_layout.addWidget(self.delete_button, 0, 3)

        # Header Crop Analysis 
        mcu_repair_layout.addWidget(QLabel("Gray Scanlines Found:"), 1, 0)
        self.gray_scanline_count_label_mini = QLabel("--- / --- (0%)")
        mcu_repair_layout.addWidget(self.gray_scanline_count_label_mini, 1, 1, 1, 3)

        self.execute_header_crop_button = QPushButton("Execute Header Crop (Remove All Gray Scanlines)")
        self.execute_header_crop_button.clicked.connect(self.remove_gray_scanlines)
        self.execute_header_crop_button.setEnabled(False)
        mcu_repair_layout.addWidget(self.execute_header_crop_button, 2, 0, 1, 4) 
        
        self.auto_align_button = QPushButton("Auto Alignment (Insert Blocks at Header)")
        self.auto_align_button.clicked.connect(self.run_auto_alignment)
        self.auto_align_button.setEnabled(False) 
        mcu_repair_layout.addWidget(self.auto_align_button, 3, 0, 1, 4) 
        
        mcu_repair_group.setLayout(mcu_repair_layout)
        repair_layout.addWidget(mcu_repair_group)
        repair_layout.addStretch(1)


        # --- Color Correction Tab ---
        color_tab = QWidget()
        color_layout = QVBoxLayout(color_tab)
        color_layout.setContentsMargins(0, 0, 0, 0) 
        
        # Manual Color Adjustment
        manual_color_group = QGroupBox("Manual Color Component Adjustment (cdelta)")
        manual_color_layout = QVBoxLayout()
        manual_color_layout.setContentsMargins(10, 20, 10, 10)
        MIN_VAL, MAX_VAL, DEFAULT_VAL = -2047, 2047, 0

        def create_slider(label_text):
            h_layout = QHBoxLayout()
            label = QLabel(f"{label_text}:")
            label.setFixedWidth(30) 
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(MIN_VAL, MAX_VAL)
            slider.setValue(DEFAULT_VAL)
            slider.setSingleStep(1)
            slider.setPageStep(64)
            value_label = QLabel(str(DEFAULT_VAL).rjust(5))
            value_label.setFixedWidth(50) 
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
        
        self.cdelta_button = QPushButton("Apply CDelta Adjustments")
        self.cdelta_button.setObjectName("SecondaryButton")
        self.cdelta_button.clicked.connect(self.run_cdelta_repair)
        self.cdelta_button.setEnabled(False)
        manual_color_layout.addWidget(self.cdelta_button)
        
        manual_color_group.setLayout(manual_color_layout)
        color_layout.addWidget(manual_color_group)

        # Automatic Color Correction
        auto_color_group = QGroupBox("Automatic Color Correction")
        auto_color_layout = QVBoxLayout()
        auto_color_layout.setContentsMargins(10, 20, 10, 10)
        
        self.auto_color_button = QPushButton("Apply PhotoDemon WB + Clarity")
        self.auto_color_button.setObjectName("AccentButton") 
        self.auto_color_button.clicked.connect(self.run_auto_color_correction)
        self.auto_color_button.setEnabled(False)
        auto_color_layout.addWidget(self.auto_color_button) 
        
        auto_color_group.setLayout(auto_color_layout)
        color_layout.addWidget(auto_color_group)
        color_layout.addStretch(1) 
        
        # --- NEW: Batch Processing Tab ---
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
        batch_layout.setContentsMargins(10, 10, 10, 10)
        
        batch_group = QGroupBox("Auto Batch Process (Reference Header Merge)")
        batch_grid = QGridLayout()
        
        # 1. Reference JPEG Path
        batch_grid.addWidget(QLabel("Reference JPEG Path:"), 0, 0)
        self.reference_jpeg_input = QLineEdit()
        self.reference_jpeg_input.setPlaceholderText("Select a known good JPEG file...")
        batch_grid.addWidget(self.reference_jpeg_input, 0, 1)
        self.select_ref_button = QPushButton("Browse")
        self.select_ref_button.clicked.connect(self.selectReferenceJPEG)
        batch_grid.addWidget(self.select_ref_button, 0, 2)
        
        # 2. Encrypted Folder Path
        batch_grid.addWidget(QLabel("Encrypted Folder Path:"), 1, 0)
        self.encrypted_folder_input = QLineEdit()
        self.encrypted_folder_input.setPlaceholderText("Select folder containing encrypted files...")
        batch_grid.addWidget(self.encrypted_folder_input, 1, 1)
        self.select_folder_button = QPushButton("Browse")
        self.select_folder_button.clicked.connect(self.selectEncryptedFolder)
        batch_grid.addWidget(self.select_folder_button, 1, 2)
        
        # 3. Process Button
        self.auto_batch_process_button = QPushButton("Start Auto Batch Process")
        self.auto_batch_process_button.setObjectName("PrimaryButton")
        self.auto_batch_process_button.clicked.connect(self.repairJPEGs)
        batch_grid.addWidget(self.auto_batch_process_button, 2, 0, 1, 3)

        batch_group.setLayout(batch_grid)
        batch_layout.addWidget(batch_group)
        
        # 4. Progress and Output
        self.progress_bar = QProgressBar(self)
        batch_layout.addWidget(self.progress_bar)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setObjectName("OutputText")
        batch_layout.addWidget(QLabel("Batch Log:"))
        batch_layout.addWidget(self.output_text, 1)
        
        # Tab Widget container
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(repair_tab, "Repair/Alignment")
        self.tab_widget.addTab(color_tab, "Color Correction")
        self.tab_widget.addTab(batch_tab, "Batch Processing") # ADDED BATCH TAB
        
        right_panel_layout.addWidget(self.tab_widget)
        
        # 4. Selected MCU Information
        selection_info_group = QGroupBox("Selected MCU Information")
        selection_layout = QGridLayout()
        selection_layout.setContentsMargins(10, 20, 10, 10)
        
        selection_layout.addWidget(QLabel("MCU Coords (Col, Row):"), 0, 0)
        self.mcu_coords_label = QLabel("--")
        selection_layout.addWidget(self.mcu_coords_label, 0, 1)
        
        selection_layout.addWidget(QLabel("MCU Index (1-based):"), 1, 0)
        self.mcu_index_label = QLabel("--")
        selection_layout.addWidget(self.mcu_index_label, 1, 1)
        
        selection_layout.addWidget(QLabel("Pixel Position Range:"), 2, 0)
        self.pixel_range_label = QLabel("X: ---, Y: ---")
        selection_layout.addWidget(self.pixel_range_label, 2, 1, 1, 3)

        selection_layout.addWidget(QLabel("Avg YCbCr:"), 3, 0) 
        self.avg_ycbr_label = QLabel("Y: ---, Cb: ---, Cr: ---") 
        selection_layout.addWidget(self.avg_ycbr_label, 3, 1, 1, 3) 
        
        selection_info_group.setLayout(selection_layout)
        right_panel_layout.addWidget(selection_info_group)
        
        right_panel_layout.addStretch(1) 
        outer_layout.addWidget(right_panel_widget)
        
        self.setCentralWidget(central_widget)
        
        # Apply the dark theme
        self.apply_dark_theme()
        
    # --- Dark Theme Implementation (Unchanged) ---
    def apply_dark_theme(self):
        
        BG_DARK = "#282c36"     
        BG_MID = "#21252b"      
        BG_LIGHT = "#3b404d"    
        TEXT_COLOR = "#abb2bf"  
        ACCENT_BLUE = "#0077ff" 
        ACCENT_RED = "#ff3c4a"  
        BORDER_GRAY = "#4d515a" 
        
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG_DARK))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Base, QColor(BG_LIGHT))
        palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Button, QColor(BG_LIGHT))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_BLUE))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(Qt.GlobalColor.white))
        QApplication.setPalette(palette)
        
        style_sheet = f"""
        QMainWindow {{
            background-color: {BG_DARK};
        }}

        QWidget#RightPanel {{
            background-color: {BG_MID};
        }}
        
        QGroupBox {{
            font-weight: bold;
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_GRAY};
            border-radius: 6px;
            margin-top: 10px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            left: 10px;
            color: {TEXT_COLOR};
            background-color: {BG_MID};
        }}

        QLabel {{
            color: {TEXT_COLOR};
            padding: 2px 0;
        }}
        
        QTextEdit#OutputText {{
            background-color: {BG_LIGHT};
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_GRAY};
            border-radius: 4px;
            padding: 5px;
        }}

        QLineEdit, QSpinBox {{
            background-color: {BG_LIGHT};
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_GRAY};
            border-radius: 4px;
            padding: 5px;
            selection-background-color: {ACCENT_BLUE};
        }}

        /* --- SpinBox Specific Styles --- */
        QSpinBox::up-button, QSpinBox::down-button {{
            width: 16px; 
            border: 1px solid {BORDER_GRAY};
            border-radius: 2px;
            background-color: #4a5160;
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
            background-color: #555d6e;
        }}
        QSpinBox::up-arrow {{
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDhMMTYgMTJIOEwxMiA4WiIgZmlsbD0iI2FiYjJiZiIvPgo8L3N2Zz4=);
        }}
        QSpinBox::down-arrow {{
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDE2TDE2IDEySDhMMTIgMTZaIiBmaWxsPSIjYWJiMmJmIi8+Cjwvc3ZnPg==);
        }}

        /* --- Buttons --- */
        QPushButton {{
            background-color: {BG_LIGHT};
            color: {TEXT_COLOR};
            border: none;
            border-radius: 4px;
            padding: 8px;
            margin: 4px 0;
        }}

        QPushButton:hover {{
            background-color: #4a5160;
        }}

        QPushButton:pressed {{
            background-color: #555d6e;
        }}
        
        QPushButton:disabled {{
            background-color: #33363d;
            color: #6c717a;
        }}

        /* Primary Button (Open/Submit - Accent Blue) */
        QPushButton#PrimaryButton {{
            background-color: {ACCENT_BLUE};
            color: white;
            font-weight: bold;
            padding: 10px;
        }}
        QPushButton#PrimaryButton:hover {{
            background-color: #3d9dff;
        }}
        
        /* Secondary Button (Apply CDelta - Accent Red) */
        QPushButton#SecondaryButton {{
            background-color: {ACCENT_RED};
            color: white;
            font-weight: bold;
        }}
        QPushButton#SecondaryButton:hover {{
            background-color: #ff606b;
        }}

        /* Accent Button (Auto Color - Soft Green/Yellow) */
        QPushButton#AccentButton {{
            background-color: #4e825a; 
            color: white;
            font-weight: bold;
        }}
        QPushButton#AccentButton:hover {{
            background-color: #63a26f;
        }}
        
        /* --- Sliders --- */
        QSlider::groove:horizontal {{
            border: 1px solid {BORDER_GRAY};
            height: 8px;
            background: {BG_LIGHT};
            margin: 2px 0;
            border-radius: 4px;
        }}
        
        QSlider::handle:horizontal {{
            background: {ACCENT_BLUE};
            border: none;
            width: 14px;
            margin: -3px 0;
            border-radius: 7px;
        }}
        
        /* --- Tab Widget --- */
        QTabWidget::pane {{ 
            border: 1px solid {BORDER_GRAY};
            background-color: {BG_MID};
            padding: 1px;
            border-radius: 6px;
        }}
        
        QTabBar::tab {{
            background: {BG_LIGHT};
            color: {TEXT_COLOR};
            padding: 8px 15px;
            border: 1px solid {BORDER_GRAY};
            border-bottom: none; 
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        
        QTabBar::tab:selected {{
            background: {BG_MID}; 
            color: white;
            font-weight: bold;
            border-bottom: 2px solid {ACCENT_BLUE}; 
        }}
        
        QTabBar::tab:hover {{
            background: #4a5160;
        }}
        
        /* --- Progress Bar --- */
        QProgressBar {{
            border: 1px solid {BORDER_GRAY};
            border-radius: 5px;
            text-align: center;
            color: {TEXT_COLOR};
            background-color: {BG_LIGHT};
        }}
        QProgressBar::chunk {{
            background-color: {ACCENT_BLUE};
            border-radius: 5px;
        }}

        
        /* --- Graphics View (Main Image Area) --- */
        QGraphicsView {{
            border: 1px solid {BORDER_GRAY};
            border-radius: 6px;
        }}
        """
        self.setStyleSheet(style_sheet)
        
    
    # --- MCU Pixel Extraction with Pillow (Unchanged) ---
    def get_mcu_block_pixmap(self, r, c):
        if not self.current_filepath or not self.current_mcu_data: return QPixmap()

        data = self.current_mcu_data
        x_start = c * data['mcu_x']
        y_start = r * data['mcu_y']
        
        # Calculate crop box (x_min, y_min, x_max, y_max)
        crop_box = (x_start, y_start, x_start + data['mcu_x'], y_start + data['mcu_y'])
        
        try:
            img = Image.open(self.current_filepath)
            mcu_img = img.crop(crop_box)
            
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.ReadWrite)
            # Save as PNG to QBuffer to get a Pixmap for display
            mcu_img.save(buffer, "PNG") 
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.data())
            return pixmap
            
        except ImportError:
            QMessageBox.critical(self, "Error", "Pillow library not found. Cannot display MCU pixels.")
        except Exception:
            pass
        return QPixmap()


    # --- Display/Feedback Methods (Unchanged) ---
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
            ycbr_text = f"Y: {y:.2f}, Cb: {cb:.2f}, Cr: {cr:.2f}"
            self.avg_ycbr_label.setText(ycbr_text)
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


    # --- Image Loading Logic (Unchanged) ---
    def load_and_display_image(self, filepath):
        
        image_pixmap = QPixmap(filepath)
        if image_pixmap.isNull():
            # Only reset GUI if original load fails
            if filepath == self.original_filepath:
                QMessageBox.critical(self, "Error", "Failed to load image file. Check file path/permissions.")
                self.reset_gui()
            # If a repair file fails to load, keep the original loaded state
            else:
                 QMessageBox.critical(self, "Error", f"Failed to load repaired image: {os.path.basename(filepath)}. Keeping current view.")
            return

        mcu_data = get_jpeg_mcu_data(filepath)
        
        if mcu_data:
            mcu_data['width'] = image_pixmap.width()
            mcu_data['height'] = image_pixmap.height()
            self.current_mcu_data = mcu_data
            self.current_filepath = filepath # Update current path if successful
            
            dim_text = f"{mcu_data['width']} x {mcu_data['height']}"
            mcu_text = f"{mcu_data['mcu_x']} x {mcu_data['mcu_y']}"
            file_name = os.path.basename(filepath)
            
            self.dim_label_mini.setText(dim_text)
            self.mcu_label_mini.setText(mcu_text)
            self.current_file_label_mini.setText(file_name)
            
            # 1. Vertical Gray Scanline Detection (for Header Crop)
            try:
                vertical_gray_count, total_scanlines, _ = count_gray_mcu_scanlines(filepath, mcu_data)
                
                self.vertical_gray_scanlines_to_remove = vertical_gray_count
                
                if total_scanlines > 0:
                    percentage = (vertical_gray_count / total_scanlines) * 100
                    scanline_text = f"{vertical_gray_count} / {total_scanlines} ({percentage:.1f}%)"
                    self.gray_scanline_count_label_mini.setText(scanline_text)
                    self.execute_header_crop_button.setEnabled(vertical_gray_count > 0)
                else:
                    self.gray_scanline_count_label_mini.setText("0 / 0 (0%)")
                    self.execute_header_crop_button.setEnabled(False)
            except Exception as e:
                self.gray_scanline_count_label_mini.setText("Error / --- (0%)")
                self.execute_header_crop_button.setEnabled(False)
                print(f"Error during gray scanline analysis: {e}", file=sys.stderr)
            
            # 2. Horizontal MCU Analysis (for Auto Alignment)
            try:
                horizontal_gray_mcu_count = analyze_last_scanline_mcus(filepath, mcu_data)
                self.post_crop_gray_mcu_count = horizontal_gray_mcu_count
                
                # Auto alignment is enabled only if blocks need to be inserted 
                # AND it's a file that has been processed (i.e., not the initial load)
                is_initial_file = (self.original_filepath == filepath)
                
                if not is_initial_file and horizontal_gray_mcu_count > 0:
                    self.auto_align_button.setEnabled(True)
                    # Inserted blocks = gray count remaining + 1 (the original block that was short)
                    insert_blocks = horizontal_gray_mcu_count + 1 
                    self.auto_align_button.setText(f"Auto Alignment (Insert {insert_blocks} Blocks at Header)")
                else:
                    self.auto_align_button.setEnabled(False)
                    self.auto_align_button.setText("Auto Alignment (Insert Blocks at Header)")


            except Exception as e:
                self.post_crop_gray_mcu_count = 0
                self.auto_align_button.setEnabled(False)
                self.auto_align_button.setText("Auto Alignment (Insert Blocks at Header)")
                print(f"Error during auto alignment analysis: {e}", file=sys.stderr)


            self.scene.clear()
            self.grid_item = McuGridItem(mcu_data, image_pixmap, self)
            self.scene.addItem(self.grid_item)
            self.scene.setSceneRect(self.grid_item.boundingRect())
            
            self.view.fitInView(self.grid_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.grid_item.setFocus(Qt.FocusReason.NoFocusReason)
            
            # Enable all main controls
            self.insert_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            self.reset_button.setEnabled(True) 
            self.toggle_grid_button.setEnabled(True) 
            self.cdelta_button.setEnabled(True) 
            self.auto_color_button.setEnabled(True) 
            
            self.toggle_grid_visibility()
            self.toggle_grid_visibility() 
            
            self.clear_mcu_info() 
            # Simulate click on (0, 0) to initialize selection
            self.grid_item.mousePressEvent(self._create_fake_event()) 
        else:
            self.reset_gui()
            
    # --- Reset GUI (Unchanged) ---
    def reset_gui(self):
        # Reset Mini Labels
        self.current_file_label_mini.setText("No file loaded")
        self.dim_label_mini.setText("--- x ---")
        self.mcu_label_mini.setText("--- x ---")
        self.gray_scanline_count_label_mini.setText("--- / --- (0%)") 
        self.auto_align_button.setText("Auto Alignment (Insert Blocks at Header)")
        self.post_crop_gray_mcu_count = 0 
        self.vertical_gray_scanlines_to_remove = 0 
        
        self.scene.clear()
        self.current_mcu_data = None
        self.current_filepath = None
        self.original_filepath = None
        self.grid_item = None
        
        # Clear Selection/Averge Info
        self.clear_mcu_info()
        self.avg_ycbr_label.setText("Y: ---, Cb: ---, Cr: ---")
        
        # Disable all controls
        self.insert_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.toggle_grid_button.setEnabled(False) 
        self.cdelta_button.setEnabled(False) 
        self.execute_header_crop_button.setEnabled(False) 
        self.auto_align_button.setEnabled(False) 
        self.auto_color_button.setEnabled(False) 
        
        # Reset SpinBox/Sliders
        self.mcu_block_num_input.setValue(1) 
        self.y_slider.setValue(0)
        self.cb_slider.setValue(0)
        self.cr_slider.setValue(0)
        
        # Reset Batch UI
        self.progress_bar.setValue(0)
        self.output_text.clear()
        
        self.selected_block_preview.set_active_state(False)
        self.clear_hover_info()

    # --- Utility Methods (Unchanged) ---
    def toggle_grid_visibility(self):
        if self.grid_item:
            self.grid_item.grid_visible = not self.grid_item.grid_visible
            self.grid_item.update()
            
            if self.grid_item.grid_visible:
                self.toggle_grid_button.setText("Hide MCU Grid")
            else:
                self.toggle_grid_button.setText("Show MCU Grid")
        
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
        
        # Reset cdelta sliders and MCU Block Num
        self.mcu_block_num_input.setValue(1) 
        self.y_slider.setValue(0)
        self.cb_slider.setValue(0)
        self.cr_slider.setValue(0)

    def _create_fake_event(self):
        class FakeMouseEvent:
            def button(self): return Qt.MouseButton.LeftButton
            def pos(self): return QPointF(0, 0)
        return FakeMouseEvent()
    
    # ======================================================================
    # --- SINGLE-FILE REPAIR LOGIC (Unchanged) ---
    # ======================================================================
    def get_repair_filepaths(self):
        """
        Determines input and output file paths for single-file operation.
        The output file is always saved to 'Repaired/[Original Filename]' and OVERWRITES.
        """
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
        output_file = os.path.join(repaired_dir, original_filename)
        input_file = self.current_filepath
        
        return input_file, output_file

    def execute_jpegrepair(self, command, operation):
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        exe_path = os.path.join(script_dir, "jpegrepair.exe")

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

    # --- Remove Gray Scanlines Method ---
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
                self, 
                "Success", 
                f"Successfully cropped {scanlines_to_remove} gray MCU scanlines by modifying the JPEG header.\nOutput saved to: {os.path.basename(output_file)} (Overwritten)\n\nReloading view."
            )
            self.load_and_display_image(output_file) 
            
            if self.auto_align_button.isEnabled():
                QMessageBox.warning(self, "Post-Crop Check", f"Post-crop misalignment detected. **Auto Alignment is now enabled.**")
            else:
                QMessageBox.information(self, "Post-Crop Check", "The crop fixed the issue. Auto Alignment not necessary.")
        else:
            QMessageBox.critical(
                self, 
                "Crop Failed", 
                "Header modification failed. Check the console for more specific error details or file write permissions."
            )
            self.auto_align_button.setEnabled(False)
            
    # --- Auto Alignment Method ---
    def run_auto_alignment(self):
        if not self.current_filepath or not self.current_mcu_data:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        gray_count_remaining = self.post_crop_gray_mcu_count
        mcu_block_num = gray_count_remaining + 1
        
        if mcu_block_num <= 1:
            QMessageBox.information(
                self, 
                "Alignment Info", 
                "Post-crop gray MCU count is 0. Auto Alignment not necessary."
            )
            self.auto_align_button.setEnabled(False)
            return

        operation = f"Align_{mcu_block_num}B" 
        c = 0
        r = 0 
        
        input_file, output_file = self.get_repair_filepaths() 
        if not input_file: return
        
        command = [
            input_file,
            output_file,
            "dest",
            str(c), 
            str(r),
            "insert", 
            str(mcu_block_num)
        ]
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        success, error_msg = self.execute_jpegrepair(command, f"Auto Alignment ({operation})")
        QApplication.restoreOverrideCursor()
        
        if success:
            QMessageBox.information(
                self, 
                "Auto Alignment Success", 
                f"Auto Alignment completed by inserting {mcu_block_num} blocks at MCU ({c}, {r}).\nOutput saved to: {os.path.basename(output_file)} (Overwritten)\n\nReloading view."
            )
            self.load_and_display_image(output_file)
            self.auto_align_button.setEnabled(False) 
        else:
            QMessageBox.critical(self, "Auto Alignment Failed", error_msg)
            self.auto_align_button.setEnabled(True) 
            
    # --- Auto Color Correction Method (Pillow/PhotoDemon) ---
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
                self, 
                "Auto-Correction Success", 
                f"PhotoDemon Auto Color/Lighting Correction applied.\nOutput saved to: {os.path.basename(output_file)} (Overwritten)\n\nReloading view."
            )
            self.load_and_display_image(output_file)
            
            # Reset CDelta sliders since this is a new color process
            self.y_slider.setValue(0)
            self.cb_slider.setValue(0)
            self.cr_slider.setValue(0)
            
        else:
            QMessageBox.critical(self, "Auto-Correction Failed", error_msg)

    # --- Run CDelta Repair Method ---
    def run_cdelta_repair(self):
        if not self.current_filepath:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return
            
        deltas = {
            0: self.y_slider.value(),   # Y
            1: self.cb_slider.value(),  # Cb
            2: self.cr_slider.value()   # Cr
        }
        
        if all(v == 0 for v in deltas.values()):
            QMessageBox.information(self, "Info", "All color corrections are set to 0. No operation executed.")
            return

        # 1. Get the final desired output path
        _, final_output_file = self.get_repair_filepaths() 
        if not final_output_file: return
        
        active_components = [i for i, v in deltas.items() if v != 0]

        # Setup temp file path for intermediate steps
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
                    current_input_file,
                    output_path, 
                    "dest",
                    str(0), 
                    str(0), 
                    "cdelta",
                    str(comp_index),
                    str(value)
                ]
                
                success, error_msg = self.execute_jpegrepair(command, operation)
                
                if not success:
                    all_successful = False
                    error_message = error_msg
                    break 

                current_input_file = output_path

        finally:
            QApplication.restoreOverrideCursor()
            # Clean up the temporary file if it was created and still exists
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)
        
        # 2. Final Load
        if all_successful:
            QMessageBox.information(
                self, 
                "Success", 
                f"Color Correction (cdelta) completed. Output saved to: {os.path.basename(final_output_file)} (Overwritten)\n\nReloading view."
            )
            self.load_and_display_image(final_output_file)
            
        else:
            QMessageBox.critical(
                self, 
                "Repair Failed", 
                f"One or more color corrections failed.\n\nError:\n{error_message}"
            )

    # --- Run Insert/Delete MCU Method ---
    def run_repair(self, operation):
        if not self.current_filepath:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        mcu_block_num = self.mcu_block_num_input.value()
        
        c = getattr(self, 'selected_mcu_c', 0)
        r = getattr(self, 'selected_mcu_r', 0)
        
        input_file, output_file = self.get_repair_filepaths()
        if not input_file: return
        
        command = [
            input_file,
            output_file,
            "dest",
            str(c), 
            str(r),
            operation,
            str(mcu_block_num)
        ]
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        success, error_msg = self.execute_jpegrepair(command, operation)
        QApplication.restoreOverrideCursor()
        
        if success:
            QMessageBox.information(
                self, 
                "Success", 
                f"MCU {operation} completed. Output saved to: {os.path.basename(output_file)} (Overwritten)\n\nReloading view."
            )
            self.load_and_display_image(output_file)
        else:
            QMessageBox.critical(self, "Repair Failed", error_msg)

    # ======================================================================
    # --- BATCH PROCESSING LOGIC (NEW) ---
    # ======================================================================

    # --- Button Callbacks (Provided by user) ---
    def selectReferenceJPEG(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Reference JPEG", "", "JPEG Files (*.jpg);;All Files (*)")
        if fileName:
            self.reference_jpeg_input.setText(fileName)

    def selectEncryptedFolder(self):
        folderName = QFileDialog.getExistingDirectory(self, "Select Encrypted Folder")
        if folderName:
            self.encrypted_folder_input.setText(folderName)

    # --- User's File Manipulation Helpers ---
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
                if marker == b'\xFF\xE1': # APP1 (EXIF) marker
                    length = int.from_bytes(data[i+2:i+4], 'big') + 2
                    data = data[:i] + data[i+length:]
                    continue
                elif marker not in (b'\xFF\xD8', b'\xFF\xD9'): # Not SOI or EOI
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
        """Initial merge/cleaning step using user-provided fixed offsets."""
        with open(encrypted_path, 'rb') as encrypted_file:
            encrypted_data = encrypted_file.read()

        with open(reference_path, 'rb') as reference_file:
            reference_data = reference_file.read()

        ffda_offset = self.find_ffda_offset(reference_data)
        # Use fixed offsets provided by the user's process_jpeg logic
        cut_reference_data = reference_data[:ffda_offset + 12]
        
        repaired_data = cut_reference_data + encrypted_data[153605:]
        repaired_data = self.remove_exif(repaired_data)
        repaired_data = repaired_data[:-334]

        with open(output_path, 'wb') as output_file:
            output_file.write(repaired_data)

    # --- Internal Batch Repair Helpers ---

    def _batch_step_header_crop(self, input_path, output_path):
        """1. Execute Header Crop (Remove All Gray Scanlines) - Batch Version"""
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
        """2. Auto Alignment (Insert Blocks at Header) - Batch Version"""
        mcu_data = get_jpeg_mcu_data(input_path)
        if not mcu_data: 
            shutil.copy2(input_path, output_path) # Copy to ensure file exists for next step
            return True, output_path # Alignment not possible/needed
        
        horizontal_gray_mcu_count = analyze_last_scanline_mcus(input_path, mcu_data)
        mcu_block_num = horizontal_gray_mcu_count + 1
        
        if mcu_block_num <= 1:
            shutil.copy2(input_path, output_path)
            return True, output_path

        command = [
            input_path, output_path, "dest", str(0), str(0), "insert", str(mcu_block_num)
        ]
        
        success, error_msg = self.execute_jpegrepair(command, f"Auto Align ({mcu_block_num} blocks)")
        
        if success:
            return True, output_path
        else:
            return False, error_msg

    def _batch_step_auto_color(self, input_path, output_path):
        """3. Apply PhotoDemon WB + Clarity - Batch Version"""
        try:
            original_img = Image.open(input_path)
            corrected_img = photodemon_autocorrect_image(original_img)
            corrected_img.save(output_path, quality=92, optimize=True) 
            return True, output_path
        except Exception as e:
            return False, f"Failed to apply PhotoDemon Auto-Correction (WB/Clarity): {e}"

    def process_jpeg_batch(self, input_file, reference_path, output_file):
        """Chains the full three-step repair for a single file in the batch."""
        base_dir = os.path.dirname(output_file)
        # Use unique PID temp file to manage intermediate results
        temp_pre_process_file = os.path.join(base_dir, f"temp_pre_process_{os.getpid()}_{os.path.basename(input_file)}")
        
        # 1. Initial Merge/Clean (User's original logic)
        try:
            self._initial_file_manipulation(reference_path, input_file, temp_pre_process_file)
        except Exception as e:
            return False, f"Initial file manipulation failed: {e}"
        
        current_input_file = temp_pre_process_file
        temp_files_to_cleanup = [temp_pre_process_file]
        
        try:
            # 2. Header Crop
            temp_crop_file = os.path.join(base_dir, f"temp_batch_{os.getpid()}_crop.jpg")
            success, result = self._batch_step_header_crop(current_input_file, temp_crop_file)
            if not success: return False, f"Header Crop failed: {result}"
            current_input_file = result
            temp_files_to_cleanup.append(temp_crop_file)
            
            # 3. Auto Alignment
            temp_align_file = os.path.join(base_dir, f"temp_batch_{os.getpid()}_align.jpg")
            success, result = self._batch_step_auto_align(current_input_file, temp_align_file)
            if not success: return False, f"Auto Alignment failed: {result}"
            current_input_file = result
            temp_files_to_cleanup.append(temp_align_file)
            
            # 4. Auto Color Correction (Final Step - outputs to final path)
            success, result = self._batch_step_auto_color(current_input_file, output_file)
            if not success: return False, f"Auto Color failed: {result}"
            
            return True, output_file
            
        finally:
            # Cleanup all intermediate files
            for f in temp_files_to_cleanup:
                if os.path.exists(f): 
                    try:
                        os.remove(f)
                    except Exception:
                        pass # Ignore cleanup errors


    def repairJPEGs(self):
        """Main batch loop provided by the user, modified to use chained repair logic."""
        reference_jpeg = self.reference_jpeg_input.text().strip()
        encrypted_folder = self.encrypted_folder_input.text().strip()

        if not os.path.exists(reference_jpeg) or not os.path.isdir(encrypted_folder):
            QMessageBox.critical(self, "Error", "Please ensure the reference JPEG and encrypted folder paths are valid.")
            return

        repaired_folder = os.path.join(encrypted_folder, "Repaired")
        os.makedirs(repaired_folder, exist_ok=True)
        
        # Pattern from user's request
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
                # Calculate the final output name based on user's original logic
                input_file_path = os.path.join(encrypted_folder, encrypted_file)
                output_name = os.path.splitext(encrypted_file)[0].rsplit('.', 1)[0] + '.JPG'
                output_file_path = os.path.join(repaired_folder, output_name)
                
                self.output_text.append(f"Starting: {encrypted_file} -> {output_name}")
                
                success, result_path_or_error = self.process_jpeg_batch(input_file_path, reference_jpeg, output_file_path)
                
                if success:
                    self.output_text.append(f"  SUCCESS: All 3 steps completed.")
                    successful_count += 1
                    
                    # Auto load the image when processed, as requested by the user
                    self.original_filepath = output_file_path
                    self.load_and_display_image(output_file_path) 
                else:
                    self.output_text.append(f"  ERROR: {result_path_or_error}")
                    
                self.progress_bar.setValue(i + 1)
        finally:
            QApplication.restoreOverrideCursor()
            
        self.output_text.append(f"\nBatch Repair complete. {successful_count} of {total_count} files successfully processed.")
        QMessageBox.information(
            self, 
            "Batch Complete", 
            f"Batch Repair finished.\nProcessed: {total_count}\nSuccessful: {successful_count}"
        )


# --- Run the Application ---
if __name__ == '__main__':
    if hasattr(sys, 'frozen') and sys.platform == 'win32':
        qt_plugin_path = os.path.join(os.path.dirname(sys.executable), 'PyQt6', 'Qt6', 'plugins')
        if os.path.isdir(qt_plugin_path):
             os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
             
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
