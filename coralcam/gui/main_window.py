import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QSpinBox, QDoubleSpinBox, 
                             QPushButton, QGroupBox, QSlider, QLineEdit, 
                             QFileDialog, QProgressBar, QCheckBox, QTabWidget,
                             QApplication, QMessageBox, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QBrush, QColor
import numpy as np
import cv2
import time

from ..hardware.camera import CoralCameras
from ..hardware.light import Light
from ..hardware.motor_TMC2209 import Motor

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.setMaximumSize(400, 250)
        
        # Store histogram data
        self.hist_data = [np.zeros(64), np.zeros(64), np.zeros(64)]  # R, G, B with fewer bins
        self.colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255)]  # Red, Green, Blue
        self.max_val = 1
        
    def update_histogram(self, image):
        if image is not None:
            try:
                # Downsample image for faster calculation
                small_image = cv2.resize(image, (80, 60))  # Very small for speed
                
                # Calculate histograms with fewer bins (64 instead of 256)
                for i in range(3):
                    hist = cv2.calcHist([small_image], [i], None, [64], [0, 256])
                    self.hist_data[i] = hist.flatten()
                
                # Find max value for scaling
                self.max_val = max(1, max([np.max(h) for h in self.hist_data]))
                
                # Trigger repaint
                self.update()
                
            except Exception as e:
                print(f"Error updating histogram: {e}")
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set up drawing area
        rect = self.rect()
        margin = 30
        plot_rect = QRect(margin, margin, rect.width() - 2*margin, rect.height() - 2*margin)
        
        # Draw background
        painter.fillRect(rect, QColor(240, 240, 240))
        painter.fillRect(plot_rect, QColor(255, 255, 255))
        
        # Draw border
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(plot_rect)
        
        # Draw title
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        title_rect = QRect(0, 5, rect.width(), 20)
        painter.drawText(title_rect, Qt.AlignCenter, "RGB Histogram")
        
        if self.max_val > 1:
            # Draw histograms
            bin_width = plot_rect.width() / 64.0
            
            for channel in range(3):
                color = self.colors[channel]
                color.setAlpha(150)  # Semi-transparent
                painter.setPen(QPen(color, 1))
                painter.setBrush(QBrush(color))
                
                # Draw bars
                for i, val in enumerate(self.hist_data[channel]):
                    if val > 0:
                        bar_height = int((val / self.max_val) * plot_rect.height())
                        bar_x = plot_rect.x() + int(i * bin_width)
                        bar_y = plot_rect.y() + plot_rect.height() - bar_height
                        bar_width = max(1, int(bin_width))
                        
                        # Only draw if height is significant
                        if bar_height > 1:
                            painter.drawRect(bar_x, bar_y, bar_width, bar_height)
        
        # Draw simple grid lines
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        # Vertical lines
        for i in range(1, 4):
            x = plot_rect.x() + (i * plot_rect.width()) // 4
            painter.drawLine(x, plot_rect.y(), x, plot_rect.y() + plot_rect.height())
        # Horizontal lines  
        for i in range(1, 4):
            y = plot_rect.y() + (i * plot_rect.height()) // 4
            painter.drawLine(plot_rect.x(), y, plot_rect.x() + plot_rect.width(), y)
        
        # Draw axis labels
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 8))
        
        # X-axis labels
        for i in range(5):
            x = plot_rect.x() + (i * plot_rect.width()) // 4
            val = int((i * 256) // 4)
            painter.drawText(x - 10, plot_rect.y() + plot_rect.height() + 15, f"{val}")
        
        painter.end()

class CameraWidget(QWidget):
    def __init__(self, camera_id, cameras, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.cameras = cameras
        self.roi_start = None
        self.roi_end = None
        self.roi_active = False
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Camera label
        title = QLabel(f"Camera {self.camera_id}")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Main content layout
        content_layout = QHBoxLayout()
        
        # Left side - Camera view and ROI controls
        left_layout = QVBoxLayout()
        
        # Camera view with fixed size
        self.camera_label = ClickableLabel()
        self.camera_label.setFixedSize(480, 360)  # Fixed 4:3 aspect ratio
        self.camera_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.camera_label.setScaledContents(True)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.clicked.connect(self.on_roi_click)
        left_layout.addWidget(self.camera_label)
        
        # ROI controls
        roi_group = QGroupBox("Region of Interest")
        roi_layout = QHBoxLayout(roi_group)
        
        self.roi_enable = QCheckBox("Enable ROI")
        self.roi_enable.toggled.connect(self.toggle_roi)
        roi_layout.addWidget(self.roi_enable)
        
        roi_layout.addWidget(QLabel("X:"))
        self.roi_x = QSpinBox()
        self.roi_x.setRange(0, 4608)
        roi_layout.addWidget(self.roi_x)
        
        roi_layout.addWidget(QLabel("Y:"))
        self.roi_y = QSpinBox()
        self.roi_y.setRange(0, 2592)
        roi_layout.addWidget(self.roi_y)
        
        roi_layout.addWidget(QLabel("W:"))
        self.roi_w = QSpinBox()
        self.roi_w.setRange(1, 4608)
        self.roi_w.setValue(100)
        roi_layout.addWidget(self.roi_w)
        
        roi_layout.addWidget(QLabel("H:"))
        self.roi_h = QSpinBox()
        self.roi_h.setRange(1, 2592)
        self.roi_h.setValue(100)
        roi_layout.addWidget(self.roi_h)
        
        clear_btn = QPushButton("Clear ROI")
        clear_btn.clicked.connect(self.clear_roi)
        roi_layout.addWidget(clear_btn)
        
        left_layout.addWidget(roi_group)
        content_layout.addLayout(left_layout)
        
        # Right side - Controls and histogram
        right_layout = QVBoxLayout()
        
        # Camera controls
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Exposure control
        controls_layout.addWidget(QLabel("Exposure (µs):"), 0, 0)
        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(100, 100000)
        self.exposure_spin.setValue(3000)
        self.exposure_spin.valueChanged.connect(self.on_exposure_changed)
        controls_layout.addWidget(self.exposure_spin, 0, 1)
        
        # Gain control
        controls_layout.addWidget(QLabel("Analogue Gain:"), 1, 0)
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(1.0, 16.0)
        self.gain_spin.setValue(1.5)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.valueChanged.connect(self.on_gain_changed)
        controls_layout.addWidget(self.gain_spin, 1, 1)
        
        # Focus controls
        controls_layout.addWidget(QLabel("Focus:"), 2, 0)
        focus_layout = QHBoxLayout()
        self.auto_focus_btn = QPushButton("Auto Focus")
        self.auto_focus_btn.clicked.connect(self.on_auto_focus)
        focus_layout.addWidget(self.auto_focus_btn)
        
        self.manual_focus_spin = QDoubleSpinBox()
        self.manual_focus_spin.setRange(0.0, 15.0)
        self.manual_focus_spin.setSingleStep(0.1)
        self.manual_focus_spin.valueChanged.connect(self.on_manual_focus)
        focus_layout.addWidget(self.manual_focus_spin)
        
        controls_layout.addLayout(focus_layout, 2, 1)
        
        right_layout.addWidget(controls_group)
        
        # Image Enhancement controls
        enhancement_group = QGroupBox("Image Enhancement")
        enhancement_layout = QGridLayout(enhancement_group)
        
        # Color Limits (Levels)
        enhancement_layout.addWidget(QLabel("Lower Limit:"), 0, 0)
        self.lower_limit_spin = QSpinBox()
        self.lower_limit_spin.setRange(0, 254)
        self.lower_limit_spin.setValue(0)
        self.lower_limit_spin.valueChanged.connect(self.on_enhancement_changed)
        enhancement_layout.addWidget(self.lower_limit_spin, 0, 1)
        
        enhancement_layout.addWidget(QLabel("Upper Limit:"), 1, 0)
        self.upper_limit_spin = QSpinBox()
        self.upper_limit_spin.setRange(1, 255)
        self.upper_limit_spin.setValue(255)
        self.upper_limit_spin.valueChanged.connect(self.on_enhancement_changed)
        enhancement_layout.addWidget(self.upper_limit_spin, 1, 1)
        
        # Gamma correction
        enhancement_layout.addWidget(QLabel("Gamma:"), 2, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 3.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.valueChanged.connect(self.on_enhancement_changed)
        enhancement_layout.addWidget(self.gamma_spin, 2, 1)
        
        # Contrast enhancement
        enhancement_layout.addWidget(QLabel("Contrast:"), 3, 0)
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.5, 3.0)
        self.contrast_spin.setValue(1.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.valueChanged.connect(self.on_enhancement_changed)
        enhancement_layout.addWidget(self.contrast_spin, 3, 1)
        
        # Brightness adjustment
        enhancement_layout.addWidget(QLabel("Brightness:"), 4, 0)
        self.brightness_spin = QSpinBox()
        self.brightness_spin.setRange(-100, 100)
        self.brightness_spin.setValue(0)
        self.brightness_spin.valueChanged.connect(self.on_enhancement_changed)
        enhancement_layout.addWidget(self.brightness_spin, 4, 1)
        
        # Enhancement presets
        preset_layout = QHBoxLayout()
        self.auto_levels_btn = QPushButton("Auto Levels")
        self.auto_levels_btn.clicked.connect(self.on_auto_levels)
        preset_layout.addWidget(self.auto_levels_btn)
        
        self.reset_enhancement_btn = QPushButton("Reset")
        self.reset_enhancement_btn.clicked.connect(self.on_reset_enhancement)
        preset_layout.addWidget(self.reset_enhancement_btn)
        
        enhancement_layout.addLayout(preset_layout, 5, 0, 1, 2)
        
        # Enhancement enable/disable
        self.enhancement_enable = QCheckBox("Enable Enhancement")
        self.enhancement_enable.setChecked(False)
        self.enhancement_enable.toggled.connect(self.on_enhancement_changed)
        enhancement_layout.addWidget(self.enhancement_enable, 6, 0, 1, 2)
        
        right_layout.addWidget(enhancement_group)
        
        # Histogram
        histogram_group = QGroupBox("Histogram")
        histogram_layout = QVBoxLayout(histogram_group)
        self.histogram = HistogramWidget()
        histogram_layout.addWidget(self.histogram)
        right_layout.addWidget(histogram_group)
        
        content_layout.addLayout(right_layout)
        layout.addLayout(content_layout)
        
    def toggle_roi(self, enabled):
        self.roi_active = enabled
        
    def clear_roi(self):
        self.roi_start = None
        self.roi_end = None
        self.roi_x.setValue(0)
        self.roi_y.setValue(0)
        self.roi_w.setValue(100)
        self.roi_h.setValue(100)
        
    def on_roi_click(self, pos):
        if self.roi_enable.isChecked():
            # Convert click position to image coordinates
            label_size = self.camera_label.size()
            x = int(pos.x() * 4608 / label_size.width())
            y = int(pos.y() * 2592 / label_size.height())
            
            self.roi_x.setValue(x)
            self.roi_y.setValue(y)
        
    def on_exposure_changed(self, value):
        self.cameras.set_exposure(value, cameras=self.camera_id)
        
    def on_gain_changed(self, value):
        self.cameras.set_gain(value, cameras=self.camera_id)
        
    def on_auto_focus(self):
        self.cameras.focus_auto(cameras=self.camera_id)
        
    def on_manual_focus(self, value):
        self.cameras.focus_manual(value, cameras=self.camera_id)
    
    def on_enhancement_changed(self):
        """Sync enhancement settings to camera when GUI controls change"""
        settings = {
            'enabled': self.enhancement_enable.isChecked(),
            'lower_limit': self.lower_limit_spin.value(),
            'upper_limit': self.upper_limit_spin.value(),
            'gamma': self.gamma_spin.value(),
            'contrast': self.contrast_spin.value(),
            'brightness': self.brightness_spin.value(),
            'remove_black_background': True,  # Always enabled for coral imaging
            'black_threshold': 15,
            'denoise': False,
            'sharpen': False,
        }
        self.cameras.set_enhancement_settings(self.camera_id, **settings)
    
    def on_auto_levels(self):
        """Automatically calculate optimal levels based on current frame"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            # Calculate histogram to find optimal levels
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Find 1st and 99th percentiles for auto levels
            total_pixels = gray.shape[0] * gray.shape[1]
            cumsum = np.cumsum(hist)
            
            # Find 1% and 99% points
            lower_idx = np.where(cumsum >= total_pixels * 0.01)[0]
            upper_idx = np.where(cumsum >= total_pixels * 0.99)[0]
            
            if len(lower_idx) > 0 and len(upper_idx) > 0:
                self.lower_limit_spin.setValue(int(lower_idx[0]))
                self.upper_limit_spin.setValue(int(upper_idx[0]))
                self.enhancement_enable.setChecked(True)
    
    def on_reset_enhancement(self):
        """Reset all enhancement settings to default"""
        self.lower_limit_spin.setValue(0)
        self.upper_limit_spin.setValue(255)
        self.gamma_spin.setValue(1.0)
        self.contrast_spin.setValue(1.0)
        self.brightness_spin.setValue(0)
        self.enhancement_enable.setChecked(False)
    
    def apply_image_enhancement(self, image):
        """Apply image enhancement based on current settings"""
        if not self.enhancement_enable.isChecked():
            return image
        
        enhanced = image.copy().astype(np.float32)
        
        # Apply levels adjustment (most important for black backgrounds)
        lower = self.lower_limit_spin.value()
        upper = self.upper_limit_spin.value()
        
        if upper > lower:
            # Clip and stretch levels
            enhanced = np.clip(enhanced, lower, upper)
            enhanced = (enhanced - lower) / (upper - lower) * 255.0
        
        # Apply gamma correction
        gamma = self.gamma_spin.value()
        if gamma != 1.0:
            enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255.0
        
        # Apply contrast
        contrast = self.contrast_spin.value()
        if contrast != 1.0:
            enhanced = ((enhanced - 127.5) * contrast) + 127.5
        
        # Apply brightness
        brightness = self.brightness_spin.value()
        if brightness != 0:
            enhanced = enhanced + brightness
        
        # Clip final result
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)
        
    def update_frame(self, frame):
        if frame is not None:
            try:
                # Store current frame for auto levels calculation
                self.current_frame = frame
                
                # Rotate 90 degrees
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Convert BGR to RGB for correct color display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply image enhancement
                enhanced_frame = self.apply_image_enhancement(frame_rgb)
                
                # Draw ROI if enabled (after enhancement)
                if self.roi_enable.isChecked():
                    x, y, w, h = self.roi_x.value(), self.roi_y.value(), self.roi_w.value(), self.roi_h.value()
                    # Scale ROI coordinates to current frame size
                    frame_h, frame_w = enhanced_frame.shape[:2]
                    scale_x = frame_w / 4608
                    scale_y = frame_h / 2592
                    roi_x = int(x * scale_x)
                    roi_y = int(y * scale_y)
                    roi_w_scaled = int(w * scale_x)
                    roi_h_scaled = int(h * scale_y)
                    cv2.rectangle(enhanced_frame, (roi_x, roi_y), (roi_x + roi_w_scaled, roi_y + roi_h_scaled), (0, 255, 0), 2)
                
                # Convert to QImage and display
                height, width, channel = enhanced_frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(enhanced_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # Create pixmap and scale to fit the fixed-size label
                pixmap = QPixmap.fromImage(q_image)
                # Use KeepAspectRatio to maintain image proportions within the fixed window
                scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.camera_label.setPixmap(scaled_pixmap)
                
                # Update histogram with enhanced frame
                self.histogram.update_histogram(enhanced_frame)
                
            except Exception as e:
                print(f"Error in update_frame: {e}")
                import traceback
                traceback.print_exc()

class ClickableLabel(QLabel):
    clicked = pyqtSignal(object)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos())

class CaptureThread(QThread):
    progress = pyqtSignal(int)
    finished_capture = pyqtSignal()
    
    def __init__(self, cameras, motor, output_dir, name, num_images, roi_settings, delay=0.5):
        super().__init__()
        self.cameras = cameras
        self.motor = motor
        self.output_dir = Path(output_dir) / name
        self.num_images = num_images
        self.roi_settings = roi_settings
        self.name = name
        self.delay = delay
        
    def run(self):
        if not (self.output_dir).exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        total_captures = self.num_images
        current_capture = 0
        
        rotation_step = 360.0 / self.num_images if self.num_images > 0 else 0
        
        for image in range(self.num_images):
            # Capture images
            filename = self.output_dir / f"{self.name}_{image:03}.jpg"

            time.sleep(self.delay)  # Small delay between captures
            self.cameras.capture_enhanced(filename, apply_enhancement=True)
            
            current_capture += 1
            progress_percent = int((current_capture / total_captures) * 100)
            self.progress.emit(progress_percent)
            
            self.motor.rotation(rotation_step)
            
        self.finished_capture.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CoralCam Control")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize hardware
        try:
            self.cameras = CoralCameras()
            self.light = Light()
            self.motor = Motor()
        except Exception as e:
            QMessageBox.critical(self, "Hardware Error", f"Failed to initialize hardware: {str(e)}")
            sys.exit(1)
        
        self.capture_thread = None
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("CoralCam Control System")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Left side - Camera views
        cameras_layout = QVBoxLayout()
        
        # Camera widgets
        self.camera_widgets = {}
        for cam_id in self.cameras.ALL_CAMERAS:
            camera_widget = CameraWidget(cam_id, self.cameras)
            self.camera_widgets[cam_id] = camera_widget
            cameras_layout.addWidget(camera_widget)
        
        content_layout.addLayout(cameras_layout)
        
        # Right side - System controls
        controls_layout = QVBoxLayout()
        
        # Light controls
        light_group = QGroupBox("Light Control")
        light_layout = QGridLayout(light_group)
        
        self.light_on_btn = QPushButton("Lights ON")
        self.light_on_btn.setCheckable(True)
        self.light_on_btn.clicked.connect(self.toggle_lights)
        light_layout.addWidget(self.light_on_btn, 0, 0, 1, 2)
        
        light_layout.addWidget(QLabel("Brightness:"), 1, 0)
        self.light_slider = QSlider(Qt.Horizontal)
        self.light_slider.setRange(0, 100)
        self.light_slider.setValue(50)
        self.light_slider.valueChanged.connect(self.on_light_brightness_changed)
        light_layout.addWidget(self.light_slider, 1, 1)
        
        self.light_value_label = QLabel("50%")
        light_layout.addWidget(self.light_value_label, 2, 1)
        
        controls_layout.addWidget(light_group)
        
        # Capture controls
        capture_group = QGroupBox("Capture Control")
        capture_layout = QGridLayout(capture_group)
        
        capture_layout.addWidget(QLabel("Output Directory:"), 0, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(str(Path.home() / "coral_images"))
        capture_layout.addWidget(self.output_dir_edit, 0, 1)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_output_dir)
        capture_layout.addWidget(browse_btn, 0, 2)
        
        capture_layout.addWidget(QLabel("Name:"), 1, 0)
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setText("capture")
        capture_layout.addWidget(self.output_name_edit, 1, 1)

        capture_layout.addWidget(QLabel("Number of Images:"), 2, 0)
        self.num_images_spin = QSpinBox()
        self.num_images_spin.setRange(1, 1000)
        self.num_images_spin.setValue(30)
        capture_layout.addWidget(self.num_images_spin, 2, 1)

        self.start_capture_btn = QPushButton("Start Capture Sequence")
        self.start_capture_btn.clicked.connect(self.start_capture_sequence)
        capture_layout.addWidget(self.start_capture_btn, 3, 0, 1, 3)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        capture_layout.addWidget(self.progress_bar, 4, 0, 1, 3)
        
        controls_layout.addWidget(capture_group)
        
        # System status
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("System Ready")
        status_layout.addWidget(self.status_label)
        
        controls_layout.addWidget(status_group)
        
        # Add stretch to push controls to top
        controls_layout.addStretch()
        
        content_layout.addLayout(controls_layout)
        main_layout.addLayout(content_layout)
        
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_feeds)
        self.timer.start(100)  # Update every 100ms
        
        # Start cameras
        self.cameras.start()
        
    def update_camera_feeds(self):
        for cam_id, widget in self.camera_widgets.items():
            try:
                # Get enhanced preview frame
                frame = self.cameras.capture_array_rgb(cam_id, apply_enhancement=True)
                
                if frame is not None:
                    widget.update_frame(frame)
                    
            except Exception as e:
                print(f"Error updating camera {cam_id}: {e}")
                
    def toggle_lights(self, checked):
        if checked:
            self.light.on()
            self.light_on_btn.setText("Lights OFF")
            self.status_label.setText("Lights ON")
        else:
            self.light.off()
            self.light_on_btn.setText("Lights ON")
            self.status_label.setText("Lights OFF")
            
    def on_light_brightness_changed(self, value):
        self.light.set_brightness(value)
        self.light_value_label.setText(f"{value}%")
        
    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            
    def start_capture_sequence(self):
        if self.capture_thread and self.capture_thread.isRunning():
            QMessageBox.warning(self, "Capture in Progress", "A capture sequence is already running.")
            return
            
        output_dir = Path(self.output_dir_edit.text())
            
        num_images = self.num_images_spin.value()
        
        # Get ROI settings from camera widgets
        roi_settings = {}
        for widget in self.camera_widgets.values():
            if widget.roi_enable.isChecked():
                roi_settings[widget.camera_id] = {
                    'x': widget.roi_x.value(),
                    'y': widget.roi_y.value(),
                    'w': widget.roi_w.value(),
                    'h': widget.roi_h.value()
                }
                
        # Sync enhancement settings to cameras before capture
        for widget in self.camera_widgets.values():
            enhancement_settings = {
                'enabled': widget.enhancement_enable.isChecked(),
                'lower_limit': widget.lower_limit_spin.value(),
                'upper_limit': widget.upper_limit_spin.value(),
                'gamma': widget.gamma_spin.value(),
                'contrast': widget.contrast_spin.value(),
                'brightness': widget.brightness_spin.value(),
                'remove_black_background': True,
                'black_threshold': 15,
                'denoise': False,
                'sharpen': False,
            }
            self.cameras.set_enhancement_settings(widget.camera_id, **enhancement_settings)
        
        self.capture_thread = CaptureThread(
            self.cameras, self.motor, output_dir, self.output_name_edit.text(),
            num_images, roi_settings
        )
        self.capture_thread.progress.connect(self.update_progress)
        self.capture_thread.finished_capture.connect(self.capture_finished)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_capture_btn.setEnabled(False)
        self.status_label.setText("Capture sequence running...")
        
        self.capture_thread.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def capture_finished(self):
        self.progress_bar.setVisible(False)
        self.start_capture_btn.setEnabled(True)
        self.status_label.setText("Capture sequence completed!")
        QMessageBox.information(self, "Capture Complete", "Image capture sequence has finished successfully.")
        
    def closeEvent(self, event):
        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.quit()
            self.capture_thread.wait()
            
        self.timer.stop()
        self.cameras.stop()
        self.light.off()
        super().closeEvent(event)


