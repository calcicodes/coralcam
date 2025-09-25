import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QSpinBox, QDoubleSpinBox, 
                             QPushButton, QGroupBox, QSlider, QLineEdit, 
                             QFileDialog, QProgressBar, QCheckBox, QTabWidget,
                             QApplication, QMessageBox, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time

from ..hardware.camera import CoralCameras
from ..hardware.light import Light
from ..hardware.motor_TMC2209 import Motor

class HistogramWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(4, 3), dpi=80)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title('RGB Histogram')
        self.axes.set_xlabel('Pixel Intensity')
        self.axes.set_ylabel('Frequency')
        self.figure.tight_layout()
        
    def update_histogram(self, image):
        self.axes.clear()
        if image is not None:
            # Calculate histograms for each channel
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                self.axes.plot(hist, color=color, alpha=0.7, linewidth=1)
            
            self.axes.set_title('RGB Histogram')
            self.axes.set_xlabel('Pixel Intensity')
            self.axes.set_ylabel('Frequency')
            self.axes.grid(True, alpha=0.3)
            self.axes.set_xlim([0, 255])
        self.draw()

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
        self.exposure_spin.setValue(20000)
        self.exposure_spin.valueChanged.connect(self.on_exposure_changed)
        controls_layout.addWidget(self.exposure_spin, 0, 1)
        
        # Gain control
        controls_layout.addWidget(QLabel("Analogue Gain:"), 1, 0)
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(1.0, 16.0)
        self.gain_spin.setValue(1.0)
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
        
    def update_frame(self, frame):
        if frame is not None:
            try:
                # Rotate 90 degrees
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Draw ROI if enabled
                if self.roi_enable.isChecked():
                    x, y, w, h = self.roi_x.value(), self.roi_y.value(), self.roi_w.value(), self.roi_h.value()
                    # Scale ROI coordinates to current frame size
                    frame_h, frame_w = frame.shape[:2]
                    scale_x = frame_w / 4608
                    scale_y = frame_h / 2592
                    roi_x = int(x * scale_x)
                    roi_y = int(y * scale_y)
                    roi_w_scaled = int(w * scale_x)
                    roi_h_scaled = int(h * scale_y)
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w_scaled, roi_y + roi_h_scaled), (0, 255, 0), 2)
                
                # Convert to QImage and display
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # Create pixmap and scale to fit the fixed-size label
                pixmap = QPixmap.fromImage(q_image)
                # Use KeepAspectRatio to maintain image proportions within the fixed window
                scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.camera_label.setPixmap(scaled_pixmap)
                
                # Update histogram
                self.histogram.update_histogram(frame)
                
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
            self.cameras.capture(filename)
            
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
                # Capture frame from camera
                frame = self.cameras.cameras[cam_id].capture_array("lores")  # Use low-res for preview
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
        for cam_id, widget in self.camera_widgets.items():
            if widget.roi_enable.isChecked():
                roi_settings[cam_id] = {
                    'x': widget.roi_x.value(),
                    'y': widget.roi_y.value(),
                    'w': widget.roi_w.value(),
                    'h': widget.roi_h.value()
                }
        
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
        
        
