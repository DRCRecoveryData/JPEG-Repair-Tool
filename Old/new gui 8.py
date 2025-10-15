import sys
import struct
import os
import subprocess
import shutil
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QGridLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QLineEdit, QHBoxLayout, QSlider, QSizePolicy
)
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap
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
# --- JPEG HEADER & SCANLINE LOGIC ---
# ======================================================================

def get_jpeg_mcu_data(filepath: str) -> dict | None:
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

            # Check if SOF data is long enough and indicates 3 components (Y, Cb, Cr)
            if sof_data is None or len(sof_data) < 12 or sof_data[5] != 3: 
                # Basic check for 3 components (sof_data[5] is num_components)
                return None 

            height_y, width_x = struct.unpack('>HH', sof_data[1:5])
            
            # The sampling factor is in sof_data[7] for the first component (Y)
            y_sampling_factor_byte = sof_data[7]
            y_hor = (y_sampling_factor_byte >> 4) & 0x0F
            y_ver = y_sampling_factor_byte & 0x0F
            
            mcu_x = y_hor * 8
            mcu_y = y_ver * 8
            
            if mcu_x == 0 or mcu_y == 0: return None # Invalid MCU size

            n_mcu_x = (width_x + mcu_x - 1) // mcu_x
            n_mcu_y = (height_y + mcu_y - 1) // mcu_y
            
            return {
                "width": width_x, "height": height_y, "mcu_x": mcu_x, 
                "mcu_y": mcu_y, "n_mcu_x": n_mcu_x, "n_mcu_y": n_mcu_y,
            }

    except Exception:
        return None

def is_mcu_scanline_gray(pixels: np.ndarray, gray_tolerance: int = 10, color_std_dev_threshold: int = 5) -> bool:
    """
    Checks if a 2D array of YCbCr pixels (whether full scanline or single MCU) is gray using NumPy.
    """
    if pixels.size == 0:
        return False

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

def count_gray_mcu_scanlines(filepath: str, mcu_data: dict, gray_tolerance: int = 10, color_std_dev_threshold: int = 5) -> tuple[int, int, list[int]]:
    """
    (Used for Header Crop) Iterates through all vertical MCU scanlines and counts how many are gray.
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

        for i in range(total_mcu_scanlines):
            start_row = i * mcu_y
            end_row = min((i + 1) * mcu_y, height)

            scanline_data = img_array[start_row:end_row, :, :]
            pixels = scanline_data.reshape(-1, 3)

            if is_mcu_scanline_gray(pixels, gray_tolerance, color_std_dev_threshold):
                gray_scanline_count += 1
                gray_scanline_indices.append(i) 

        return gray_scanline_count, total_mcu_scanlines, gray_scanline_indices
        
    except Exception:
        return 0, 0, []
        
        
def analyze_last_scanline_mcus(filepath: str, mcu_data: dict, gray_tolerance: int = 10, color_std_dev_threshold: int = 5) -> int:
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

# --- NEW: PhotoDemon Clarity Lookup Table Generator ---

def _clarity_lookup_table() -> np.ndarray:
    """
    Generates the PhotoDemon 'Clarity/Midtone Contrast' lookup table 
    based on the fxAutoEnhance VBA function logic.
    """
    contrastLookup = np.zeros(256, dtype=np.uint8)
    factor = 0.4 # (50 / 100) * 0.8
    
    for x in range(256):
        x_float = float(x)
        diff = x_float - 127.0
        
        if x < 127:
            # The push is negative, making darks darker (contrast).
            push = (x_float / 127.0) * (diff / 2.0) * factor
        else:
            # The push is positive, making lights lighter (contrast).
            push = ((255.0 - x_float) / 127.0) * (diff / 2.0) * factor
            
        gray = x_float + push
        
        # Crop the lookup value to [0, 255] range
        if gray > 255.0:
            gray = 255.0
        elif gray < 0.0:
            gray = 0.0
            
        # Round before converting to integer/byte
        contrastLookup[x] = int(round(gray))
        
    return contrastLookup

# --- NEW: Combined PhotoDemon Auto-Correction Logic (WB + Clarity) ---
def photodemon_autocorrect_image(img: Image.Image) -> Image.Image:
    """
    Applies PhotoDemon's primary auto-correction steps: 
    1. Independent channel White Balance (0.05% clip - from AutoCorrectImage)
    2. Midtone Contrast/Clarity (from fxAutoEnhance)
    
    FIXED: Removed 'mode' parameter from Image.fromarray to fix DeprecationWarning.
    """
    
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
            # Linear Contrast Stretch
            shifted_channel = channel.astype(np.float32) - min_val
            stretched_channel = shifted_channel * (255.0 / (max_val - min_val))
            corrected_channel = np.clip(stretched_channel, 0, 255).astype(np.uint8)
            
        corrected_channels.append(corrected_channel)
        
    np_img = np.stack(corrected_channels, axis=2)
    
    # 2. Clarity/Midtone Contrast (Lookup Table Application)
    
    clarity_lookup = _clarity_lookup_table()
    
    # Apply lookup table to all channels in one go (fastest method)
    np_img[:, :, 0] = clarity_lookup[np_img[:, :, 0]] # Red
    np_img[:, :, 1] = clarity_lookup[np_img[:, :, 1]] # Green
    np_img[:, :, 2] = clarity_lookup[np_img[:, :, 2]] # Blue
    
    # FIX: Remove 'mode' parameter to eliminate DeprecationWarning
    return Image.fromarray(np_img)

# ======================================================================
# --- CROP/HEADER MODIFICATION LOGIC ---
# ======================================================================

def find_sof_height_position(filepath: str) -> int | None:
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
                    return marker_pos + 5 
                
                if marker[0] == 0xff and marker[1] not in (0x00, 0xd8, 0xd9, 0xda):
                    segment_length = struct.unpack('>H', f.read(2))[0]
                    f.seek(segment_length - 2, 1)
                elif marker[0] == 0xff and marker[1] in (0xd8, 0xd9): continue
                else: f.seek(-1, 1)
    except Exception:
        return None

def crop_jpeg_by_header(source_filepath: str, output_filepath: str, scanlines_to_remove: int) -> bool:
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

def get_mcu_avg_ycbr_values(filepath: str, mcu_data: dict, r: int, c: int) -> list[float] | None:
    """Calculates the average Y, Cb, and Cr values for a specific MCU block."""
    if not filepath or not mcu_data: return None

    data = mcu_data
    x_start = c * data['mcu_x']
    y_start = r * data['mcu_y']
    
    # Ensure crop box doesn't exceed image boundaries (Pillow handles this automatically, but good practice)
    x_end = x_start + data['mcu_x']
    y_end = y_start + data['mcu_y']
    crop_box = (x_start, y_start, x_end, y_end)
    
    try:
        img = Image.open(filepath).convert('YCbCr')
        mcu_img = img.crop(crop_box)
        mcu_array = np.array(mcu_img)
        # Calculate mean across the block (rows, then columns)
        avg_values = mcu_array.mean(axis=(0, 1)) 
        
        return avg_values.tolist()
            
    except Exception:
        return None

# ======================================================================
# --- PyQt GUI Components ---
# ======================================================================

class McuGridItem(QGraphicsItem):
    
    def __init__(self, mcu_data: dict, image_pixmap: QPixmap, main_window: 'MainWindow'):
        super().__init__()
        self.main_window = main_window
        self.pixmap = image_pixmap
        self.mcu_x = mcu_data['mcu_x']
        self.mcu_y = mcu_data['mcu_y']
        self.img_width = mcu_data['width']
        self.img_height = mcu_data['height']
        
        self.n_mcu_x = mcu_data['n_mcu_x']
        self.n_mcu_y = mcu_data['n_mcu_y']

        # Ensure selected_mcu_coords is initialized (0, 0)
        self.selected_mcu_coords = (0, 0) 
        self.hovered_mcu_coords = None 
        self.grid_visible = False
        
        self._bounding_rect = QRectF(0, 0, self.img_width, self.img_height)
        
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptHoverEvents(True)

    def boundingRect(self): return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None):
        painter.drawPixmap(self.boundingRect().toRect(), self.pixmap)
        
        if self.grid_visible:
            painter.setPen(QPen(QColor(255, 255, 255, 120), 0))
            for c in range(1, self.n_mcu_x):
                painter.drawLine(c * self.mcu_x, 0, c * self.mcu_x, self.img_height)
            for r in range(1, self.n_mcu_y):
                painter.drawLine(0, r * self.mcu_y, self.img_width, r * self.mcu_y)
            
        if self.hovered_mcu_coords:
            hr, hc = self.hovered_mcu_coords
            hx = hc * self.mcu_x
            hy = hr * self.mcu_y
            h_rect = QRectF(hx, hy, self.mcu_x, self.mcu_y)
            painter.fillRect(h_rect, QColor(0, 160, 255, 40)) 

        if self.selected_mcu_coords:
            r, c = self.selected_mcu_coords
            
            x = c * self.mcu_x
            y = r * self.mcu_y
            w = self.mcu_x
            h = self.mcu_y
            
            highlight_rect = QRectF(x, y, w, h)
            
            painter.fillRect(highlight_rect, QColor(0, 120, 215, 90))
            painter.setPen(QPen(QColor(0, 120, 215), 5))
            painter.drawRect(highlight_rect)
    
    def _calculate_mcu_coords(self, pos: QPointF) -> tuple[int, int]:
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

    def mousePressEvent(self, event):
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
    
    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)


class BlockPreviewWidget(QWidget):
    
    CELL_SIZE = 8
    
    def __init__(self, cols: int = 16, rows: int = 8, is_static_preview: bool = False, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.is_static_preview = is_static_preview 
        self.is_selected = False 
        self.is_hovered = False 
        self.mcu_pixmap = QPixmap() 
        self.setFixedSize(self.cols * self.CELL_SIZE + 2, self.rows * self.CELL_SIZE + 2) 
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed) # Ensures fixed size

    def update_pixmap(self, pixmap: QPixmap):
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
        painter.fillRect(self.rect(), QColor(240, 240, 240)) 

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
                        painter.fillRect(rect, QColor(220, 220, 220)) 
                    else:
                        painter.fillRect(rect, QColor(200, 200, 200))

        painter.setPen(QPen(QColor(180, 180, 180), 1)) 
        for r in range(self.rows + 1):
            painter.drawLine(1, r * self.CELL_SIZE + 1, self.cols * self.CELL_SIZE + 1, r * self.CELL_SIZE + 1)
        for c in range(self.cols + 1):
            painter.drawLine(c * self.CELL_SIZE + 1, 1, c * self.CELL_SIZE + 1, self.rows * self.CELL_SIZE + 1)
        
        # Hover/Selection Border Logic
        border_pen = QPen(QColor(0, 0, 0), 1) # Default
        if self.is_static_preview and self.is_hovered:
            border_pen = QPen(QColor(0, 160, 255), 2) 
        elif not self.is_static_preview and self.is_selected:
            border_pen = QPen(QColor(255, 0, 0), 2) 
        
        painter.setPen(border_pen)
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
        self.setWindowTitle("JPEG MCU Viewer (Interactive)")
        self.setGeometry(100, 100, 1000, 750) 
        self.current_mcu_data = None
        self.current_filepath = None
        self.original_filepath = None
        
        self.post_crop_gray_mcu_count = 0 
        
        self.scene = QGraphicsScene()
        self.view = McuGraphicsView(self.scene)
        self.grid_item = None
        
        # Initialize selected MCU coordinates
        self.selected_mcu_c = 0 
        self.selected_mcu_r = 0

        central_widget = QWidget()
        outer_layout = QHBoxLayout(central_widget)

        # --- Left/Main Content Area (Image Viewer + Controls) ---
        left_content_layout = QVBoxLayout()
        
        # --- Top Info Layout (MCU, DIMS, File) ---
        top_layout = QGridLayout()
        
        # Row 0: File Operations
        self.open_button = QPushButton("Open JPEG File...")
        self.open_button.clicked.connect(self.open_file)
        top_layout.addWidget(self.open_button, 0, 0, 1, 1)
        self.reset_button = QPushButton("Reset to Original")
        self.reset_button.clicked.connect(self.reset_to_original)
        self.reset_button.setEnabled(False)
        top_layout.addWidget(self.reset_button, 0, 1, 1, 1)
        self.toggle_grid_button = QPushButton("Turn Grid On")
        self.toggle_grid_button.clicked.connect(self.toggle_grid_visibility)
        self.toggle_grid_button.setEnabled(False) 
        top_layout.addWidget(self.toggle_grid_button, 0, 2, 1, 1)
        
        # Row 1-4: File/Image Info
        top_layout.addWidget(QLabel("File:"), 1, 0)
        self.file_label = QLabel("No file loaded")
        top_layout.addWidget(self.file_label, 1, 1, 1, 2)
        top_layout.addWidget(QLabel("Dimensions:"), 2, 0)
        self.dim_label = QLabel("--- x ---")
        top_layout.addWidget(self.dim_label, 2, 1)
        top_layout.addWidget(QLabel("MCU Size:"), 3, 0)
        self.mcu_label = QLabel("--- x ---")
        top_layout.addWidget(self.mcu_label, 3, 1)

        top_layout.addWidget(QLabel("Gray Scanlines (Vertical):"), 4, 0)
        self.gray_scanline_count_label = QLabel("--- / --- (0%)")
        top_layout.addWidget(self.gray_scanline_count_label, 4, 1, 1, 2)
        
        left_content_layout.addLayout(top_layout)
        left_content_layout.addSpacing(10)

        left_content_layout.addWidget(self.view)
        
        # --- Selection Info Layout ---
        selection_layout = QGridLayout()
        selection_layout.addWidget(QLabel("MCU Coords (Col, Row):"), 0, 0)
        self.mcu_coords_label = QLabel("--")
        selection_layout.addWidget(self.mcu_coords_label, 0, 1)
        selection_layout.addWidget(QLabel("Pixel Position Range:"), 0, 2)
        self.pixel_range_label = QLabel("X: ---, Y: ---")
        selection_layout.addWidget(self.pixel_range_label, 0, 3)
        selection_layout.addWidget(QLabel("MCU Index (1-based):"), 1, 0)
        self.mcu_index_label = QLabel("--")
        selection_layout.addWidget(self.mcu_index_label, 1, 1)
        selection_layout.addWidget(QLabel("Avg YCbCr:"), 2, 0) 
        self.avg_ycbr_label = QLabel("Y: ---, Cb: ---, Cr: ---") 
        selection_layout.addWidget(self.avg_ycbr_label, 2, 1, 1, 3) 
        left_content_layout.addLayout(selection_layout)
        
        # --- Color Correction Sliders ---
        color_correction_group = QVBoxLayout()
        color_correction_group.addWidget(QLabel("--- Color Component Adjustment (cdelta) ---"))
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
            slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v).rjust(5)))
            h_layout.addWidget(label)
            h_layout.addWidget(slider)
            h_layout.addWidget(value_label)
            return slider, h_layout

        self.y_slider, y_layout = create_slider("Y")
        color_correction_group.addLayout(y_layout)
        self.cb_slider, cb_layout = create_slider("Cb")
        color_correction_group.addLayout(cb_layout)
        self.cr_slider, cr_layout = create_slider("Cr")
        color_correction_group.addLayout(cr_layout)
        self.cdelta_button = QPushButton("Apply Color Correction (cdelta)")
        self.cdelta_button.clicked.connect(self.run_cdelta_repair)
        self.cdelta_button.setEnabled(False)
        color_correction_group.addWidget(self.cdelta_button)
        
        self.auto_color_button = QPushButton("Auto Color/Lighting Correction (PhotoDemon Logic)")
        self.auto_color_button.clicked.connect(self.run_auto_color_correction)
        self.auto_color_button.setEnabled(False)
        color_correction_group.addWidget(self.auto_color_button) 
        
        left_content_layout.addLayout(color_correction_group)
        
        left_content_layout.addSpacing(10)
        
        # --- MCU Block Operations ---
        repair_layout = QGridLayout()
        repair_layout.addWidget(QLabel("--- MCU Block Operations ---"), 0, 0, 1, 4)
        repair_layout.addWidget(QLabel("MCU Block Number (k):"), 1, 0)
        self.mcu_block_num_input = QLineEdit("1") 
        self.mcu_block_num_input.setFixedWidth(50)
        repair_layout.addWidget(self.mcu_block_num_input, 1, 1)
        self.insert_button = QPushButton("Insert MCU")
        self.insert_button.clicked.connect(lambda: self.run_repair("insert"))
        self.insert_button.setEnabled(False)
        repair_layout.addWidget(self.insert_button, 1, 2)
        self.delete_button = QPushButton("Delete MCU")
        self.delete_button.clicked.connect(lambda: self.run_repair("delete"))
        self.delete_button.setEnabled(False)
        repair_layout.addWidget(self.delete_button, 1, 3)

        self.remove_gray_scanlines_button = QPushButton("Remove Gray MCU Scanlines (Crop Header)")
        self.remove_gray_scanlines_button.clicked.connect(self.remove_gray_scanlines)
        self.remove_gray_scanlines_button.setEnabled(False)
        repair_layout.addWidget(self.remove_gray_scanlines_button, 2, 0, 1, 4) 
        
        self.auto_align_button = QPushButton("Auto Alignment (Insert MCU)")
        self.auto_align_button.clicked.connect(self.run_auto_alignment)
        self.auto_align_button.setEnabled(False) 
        repair_layout.addWidget(self.auto_align_button, 3, 0, 1, 4) 
        
        left_content_layout.addLayout(repair_layout)
        
        outer_layout.addLayout(left_content_layout, 1) # Give viewer space

        # --- Right Panel (Previews) ---
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop) 
        right_panel_layout.addWidget(QLabel("Pixel block preview (on hover):"))
        self.pixel_block_preview = BlockPreviewWidget(cols=16, rows=8, is_static_preview=True)
        right_panel_layout.addWidget(self.pixel_block_preview)
        right_panel_layout.addSpacing(20)

        right_panel_layout.addWidget(QLabel("Selected block:"))
        self.selected_block_preview = BlockPreviewWidget(cols=16, rows=8, is_static_preview=False) 
        right_panel_layout.addWidget(self.selected_block_preview)
        right_panel_layout.addStretch(1) 

        outer_layout.addLayout(right_panel_layout)
        
        self.setCentralWidget(central_widget)
    
    # --- MCU Pixel Extraction with Pillow ---
    def get_mcu_block_pixmap(self, r: int, c: int) -> QPixmap:
        if not self.current_filepath or not self.current_mcu_data: return QPixmap()

        data = self.current_mcu_data
        x_start = c * data['mcu_x']
        y_start = r * data['mcu_y']
        x_end = x_start + data['mcu_x']
        y_end = y_start + data['mcu_y']
        
        crop_box = (x_start, y_start, x_end, y_end)
        
        try:
            img = Image.open(self.current_filepath)
            mcu_img = img.crop(crop_box)
            
            # Convert Pillow Image to QPixmap via in-memory buffer
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


    # --- Display/Feedback Methods ---
    def display_hover_info(self, r: int, c: int):
        mcu_pixmap = self.get_mcu_block_pixmap(r, c)
        self.pixel_block_preview.update_pixmap(mcu_pixmap)
        self.pixel_block_preview.set_hover_state(True)

    def clear_hover_info(self):
        self.pixel_block_preview.clear_pixmap()
        self.pixel_block_preview.set_hover_state(False)

    def display_mcu_info(self, r: int, c: int):
        if not self.current_mcu_data or not self.grid_item: 
            self.clear_mcu_info()
            return
            
        data = self.current_mcu_data
        
        mcu_pixmap = self.get_mcu_block_pixmap(r, c)
        self.selected_block_preview.update_pixmap(mcu_pixmap)
        
        n_mcu_x = self.grid_item.n_mcu_x

        mcu_index = r * n_mcu_x + c + 1
        
        x_start = c * data['mcu_x']
        y_start = r * data['mcu_y']
        
        # Pixel end coordinates are inclusive
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


    # --- Image Loading Logic ---
    def load_and_display_image(self, filepath: str):
        
        image_pixmap = QPixmap(filepath)
        if image_pixmap.isNull():
            QMessageBox.critical(self, "Error", "Failed to load image file.")
            self.reset_gui()
            return

        mcu_data = get_jpeg_mcu_data(filepath)
        
        if mcu_data:
            # Update data with actual QPixmap dimensions (in case header reading was truncated)
            mcu_data['width'] = image_pixmap.width()
            mcu_data['height'] = image_pixmap.height()
            self.current_mcu_data = mcu_data
            
            self.dim_label.setText(f"{mcu_data['width']} x {mcu_data['height']}")
            self.mcu_label.setText(f"{mcu_data['mcu_x']} x {mcu_data['mcu_y']}")

            # 1. Run Vertical Gray Scanline Detection (for Header Crop)
            try:
                vertical_gray_count, total_scanlines, _ = count_gray_mcu_scanlines(filepath, mcu_data)
                
                if total_scanlines > 0:
                    percentage = (vertical_gray_count / total_scanlines) * 100
                    self.gray_scanline_count_label.setText(f"{vertical_gray_count} / {total_scanlines} ({percentage:.1f}%)")
                    self.remove_gray_scanlines_button.setEnabled(vertical_gray_count > 0)
                else:
                    self.gray_scanline_count_label.setText("0 / 0 (0%)")
                    self.remove_gray_scanlines_button.setEnabled(False)
            except Exception:
                self.gray_scanline_count_label.setText("Error / --- (0%)")
                self.remove_gray_scanlines_button.setEnabled(False)
            
            # 2. Run Horizontal MCU Analysis (for Auto Alignment)
            try:
                horizontal_gray_mcu_count = analyze_last_scanline_mcus(filepath, mcu_data)
                self.post_crop_gray_mcu_count = horizontal_gray_mcu_count
                
                # Auto Alignment is only enabled if: 
                # a) The current file is *not* the original file (i.e., a crop has already happened) AND
                # b) Horizontal gray MCUs were found.
                is_repaired_file = (self.original_filepath is not None and filepath != self.original_filepath)
                
                if is_repaired_file and horizontal_gray_mcu_count > 0:
                    self.auto_align_button.setEnabled(True)
                else:
                    self.auto_align_button.setEnabled(False)

            except Exception:
                self.post_crop_gray_mcu_count = 0
                self.auto_align_button.setEnabled(False)


            self.file_label.setText(os.path.basename(filepath))

            self.scene.clear()
            self.grid_item = McuGridItem(mcu_data, image_pixmap, self)
            self.scene.addItem(self.grid_item)
            self.scene.setSceneRect(self.grid_item.boundingRect())
            
            self.view.fitInView(self.grid_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.grid_item.setFocus(Qt.FocusReason.NoFocusReason)
            
            # Enable controls
            self.insert_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            self.reset_button.setEnabled(True) 
            self.toggle_grid_button.setEnabled(True) 
            self.cdelta_button.setEnabled(True) 
            self.auto_color_button.setEnabled(True) 
            
            self.toggle_grid_button.setText("Turn Grid On") 
            
            # Select and display info for MCU (0, 0)
            self.clear_mcu_info() 
            self.grid_item.mousePressEvent(self._create_fake_event()) 
        else:
            self.reset_gui()
            
    # --- Reset GUI ---
    def reset_gui(self):
        self.file_label.setText("No file loaded")
        self.dim_label.setText("--- x ---")
        self.mcu_label.setText("--- x ---")
        self.gray_scanline_count_label.setText("--- / --- (0%)") 
        self.post_crop_gray_mcu_count = 0 
        self.scene.clear()
        self.current_mcu_data = None
        self.current_filepath = None
        self.original_filepath = None
        self.grid_item = None
        self.clear_mcu_info()
        self.avg_ycbr_label.setText("Y: ---, Cb: ---, Cr: ---")
        
        # Disable controls
        self.insert_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.toggle_grid_button.setEnabled(False) 
        self.cdelta_button.setEnabled(False) 
        self.remove_gray_scanlines_button.setEnabled(False) 
        self.auto_align_button.setEnabled(False) 
        self.auto_color_button.setEnabled(False) 
        
        # Reset Sliders
        self.y_slider.setValue(0)
        self.cb_slider.setValue(0)
        self.cr_slider.setValue(0)

    # --- Utility Methods ---
    def toggle_grid_visibility(self):
        if self.grid_item:
            self.grid_item.grid_visible = not self.grid_item.grid_visible
            self.grid_item.update()
            
            if self.grid_item.grid_visible:
                self.toggle_grid_button.setText("Turn Grid Off")
            else:
                self.toggle_grid_button.setText("Turn Grid On")
        
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
        
        self.y_slider.setValue(0)
        self.cb_slider.setValue(0)
        self.cr_slider.setValue(0)

    def _create_fake_event(self):
        """Creates a fake mouse event for programmatically selecting the (0, 0) MCU."""
        class FakeMouseEvent:
            def button(self): return Qt.MouseButton.LeftButton
            def pos(self): return QPointF(1, 1) # Small offset to ensure it's inside the first MCU
        return FakeMouseEvent()
    
    # ======================================================================
    # --- File Path Logic (Simplified Naming - NO COUNTER, OVERWRITE) ---
    # ======================================================================
    def get_repair_filepaths(self, operation: str) -> tuple[str, str] | tuple[None, None]:
        """
        Determines input and output file paths.
        All non-temp operations output to the original filename in the Repaired folder, enabling overwrite.
        """
        base_path = self.original_filepath if self.original_filepath else self.current_filepath
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
        base_name_no_ext = os.path.splitext(original_filename)[0]
        ext = os.path.splitext(original_filename)[1]
        
        if operation.startswith("temp_cdelta"):
            # Temporary files for multi-step cdelta operation
            output_filename = f"{base_name_no_ext}_{operation.replace(' ', '_')}{ext}"
        else:
            # All final repairs use the original name (e.g., file.jpg) in the Repaired folder
            output_filename = original_filename

        output_file = os.path.join(repaired_dir, output_filename)
        input_file = self.current_filepath
        
        return input_file, output_file
    # ======================================================================

    def execute_jpegrepair(self, command: list[str], operation: str) -> tuple[bool, str]:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        exe_path = os.path.join(script_dir, "jpegrepair.exe")

        if not os.path.exists(exe_path):
            return False, f"Executable not found: {exe_path}"
        
        command.insert(0, exe_path) 
        
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            
            if process.returncode == 0:
                return True, ""
            else:
                error_output = process.stderr if process.stderr else "No specific error output."
                # Check for "Invalid SOS parameters" to inform the user what caused the jpegrepair failure.
                if "Invalid SOS parameters" in process.stderr or "Invalid SOS parameters" in process.stdout:
                    error_output += "\n\n(Note: The tool reported 'Invalid SOS parameters' which usually means the JPEG stream is structurally corrupted. This often prevents further operations.)"
                
                return False, f"Execution failed for {operation} with return code {process.returncode}.\nError:\n{error_output}"

        except Exception as e:
            return False, f"An unexpected error occurred during execution: {e}"

    # --- Remove Gray Scanlines Method ---
    def remove_gray_scanlines(self):
        if not self.current_filepath or not self.current_mcu_data:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        try:
            # Parse the scanlines to remove from the current label text (vertical count)
            # The label is formatted as "COUNT / TOTAL (PERCENT%)"
            label_text = self.gray_scanline_count_label.text().split('/')[0].strip()
            scanlines_to_remove = int(label_text)
        except:
            QMessageBox.critical(self, "Error", "Could not determine the number of gray scanlines to remove.")
            return

        if scanlines_to_remove <= 0:
            QMessageBox.information(self, "Info", "Zero gray MCU scanlines found. No cropping performed.")
            return
            
        input_file, output_file = self.get_repair_filepaths("header_crop") 
        if not input_file: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        success = crop_jpeg_by_header(input_file, output_file, scanlines_to_remove)

        QApplication.restoreOverrideCursor()

        if success:
            QMessageBox.information(
                self, 
                "Success", 
                f"Successfully cropped {scanlines_to_remove} gray MCU scanlines from the bottom by modifying the JPEG header.\nOutput saved to:\n{os.path.basename(output_file)}\n\nReloading view."
            )
            self.current_filepath = output_file
            self.load_and_display_image(output_file) 
            
            if self.auto_align_button.isEnabled():
                QMessageBox.warning(self, "Post-Crop Check", f"{self.post_crop_gray_mcu_count} gray MCUs remaining in the last scanline. **Auto Alignment enabled.**")
            else:
                QMessageBox.information(self, "Post-Crop Check", "No gray MCUs remaining in the last scanline after crop. Auto Alignment not necessary.")
        else:
            QMessageBox.critical(
                self, 
                "Crop Failed", 
                "Header modification failed. Check the console for more specific error details from the operation."
            )
            self.auto_align_button.setEnabled(False)
            
    # --- Auto Alignment Method ---
    def run_auto_alignment(self):
        if not self.current_filepath or not self.current_mcu_data:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        gray_count_remaining = self.post_crop_gray_mcu_count
        
        # Logic: MCU Block Number (k) = (Gray MCUs Found) + 1
        mcu_block_num = gray_count_remaining + 1
        
        if mcu_block_num <= 1:
            QMessageBox.information(
                self, 
                "Alignment Info", 
                "Post-crop gray MCU count is 0. Auto Alignment not necessary (or invalid block count)."
            )
            self.auto_align_button.setEnabled(False)
            return

        operation = "auto_align_insert" 
        
        # Insertion point must be (0, 0) for header alignment
        c = 0
        r = 0
        
        input_file, output_file = self.get_repair_filepaths(operation) 
        if not input_file: return
        
        self.mcu_block_num_input.setText(str(mcu_block_num))
        
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
        success, error_msg = self.execute_jpegrepair(command, f"Auto Alignment ({operation} {mcu_block_num} blocks)")
        QApplication.restoreOverrideCursor()
        
        if success:
            QMessageBox.information(
                self, 
                "Auto Alignment Success", 
                f"Auto Alignment completed by inserting {mcu_block_num} blocks at MCU ({c}, {r}).\nOutput saved to:\n{os.path.basename(output_file)}\n\nReloading view."
            )
            self.current_filepath = output_file
            self.load_and_display_image(output_file)
            self.auto_align_button.setEnabled(False) 
        else:
            QMessageBox.critical(self, "Auto Alignment Failed", error_msg)
            
    # --- Auto Color Correction Method (replaces AWB) ---
    def run_auto_color_correction(self):
        if not self.current_filepath:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        input_file, output_file = self.get_repair_filepaths("autowb") 
        if not input_file: return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        success = False
        error_msg = ""
        
        try:
            original_img = Image.open(input_file)
            corrected_img = photodemon_autocorrect_image(original_img)
            # Use high quality when saving as JPEG
            corrected_img.save(output_file, quality=95, optimize=True)
            success = True
            
        except Exception as e:
            error_msg = f"Failed to apply PhotoDemon Auto-Correction (WB/Clarity).\nError details: {e}"
            
        QApplication.restoreOverrideCursor()

        if success:
            QMessageBox.information(
                self, 
                "Auto-Correction Success", 
                f"PhotoDemon Auto Color/Lighting Correction applied (WB + Clarity).\nOutput saved to:\n{os.path.basename(output_file)}\n\nReloading view."
            )
            self.current_filepath = output_file
            self.load_and_display_image(output_file)
            
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

        _, final_output_file = self.get_repair_filepaths("cdelta") 
        if not final_output_file: return
        
        active_components = [i for i, v in deltas.items() if v != 0]

        current_input_file = self.current_filepath
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        all_successful = True
        failed_messages = []

        # Execute component corrections sequentially
        for comp_index in active_components:
            value = deltas[comp_index]
            is_last_component = (comp_index == active_components[-1])
            
            if is_last_component:
                temp_output_file = final_output_file 
            else:
                _, temp_output_file = self.get_repair_filepaths(f"temp_cdelta_{comp_index}")
                
            operation = f"cdelta {comp_index} {value}"
            
            command = [
                current_input_file,
                temp_output_file, 
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
                failed_messages.append(error_msg)
                break 

            current_input_file = temp_output_file # Input for the next step

            # Clean up temporary files (optional, but good practice)
            if not is_last_component and os.path.exists(temp_output_file):
                 try:
                     os.remove(temp_output_file)
                 except Exception:
                     # Ignore cleanup errors
                     pass
        
        QApplication.restoreOverrideCursor()

        if all_successful:
            QMessageBox.information(
                self, 
                "Success", 
                f"Color Correction (cdelta) completed. Output saved to:\n{os.path.basename(final_output_file)}\n\nReloading view with the repaired image."
            )
            self.current_filepath = final_output_file 
            self.load_and_display_image(final_output_file)
            
        else:
            QMessageBox.critical(
                self, 
                "Repair Failed", 
                "One or more color corrections failed.\n\nErrors:\n" + "\n".join(failed_messages)
            )

    # --- Run Insert/Delete MCU Method ---
    def run_repair(self, operation: str):
        if not self.current_filepath:
            QMessageBox.warning(self, "Warning", "Please load a JPEG file first.")
            return

        try:
            mcu_block_num = int(self.mcu_block_num_input.text())
            if mcu_block_num <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "MCU Block Number must be a positive integer.")
            return

        # Get selected MCU coordinates
        c = self.selected_mcu_c
        r = self.selected_mcu_r
        
        input_file, output_file = self.get_repair_filepaths(operation)
        if not input_file: return
        
        command = [
            input_file,
            output_file,
            "dest",
            str(c), 
            str(r),
            operation, # "insert" or "delete"
            str(mcu_block_num)
        ]
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        success, error_msg = self.execute_jpegrepair(command, operation)
        QApplication.restoreOverrideCursor()
        
        if success:
            QMessageBox.information(
                self, 
                "Success", 
                f"MCU {operation} completed. Output saved to:\n{os.path.basename(output_file)}\n\nReloading view with the repaired image."
            )
            self.current_filepath = output_file
            self.load_and_display_image(output_file)
        else:
            QMessageBox.critical(self, "Repair Failed", error_msg)


# --- Run the Application (Unchanged) ---
if __name__ == '__main__':
    if hasattr(sys, 'frozen') and sys.platform == 'win32':
        qt_plugin_path = os.path.join(os.path.dirname(sys.executable), 'PyQt6', 'Qt6', 'plugins')
        if os.path.isdir(qt_plugin_path):
             os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
             
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
