from pathlib import Path
import time
import numpy as np
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available - image enhancement disabled")

from picamera2 import Picamera2, Preview
from libcamera import controls, Transform, ColorSpace

config = {
    'use_case': 'still',
    'transform': Transform(),
    'colour_space': ColorSpace.Sycc(),
    'buffer_count': 3,
    'queue': True,
    'main': {
        'format': 'RGB888',
        'size': (4608, 2592),
        'stride': 13824,
        'framesize': 35831808,
        'preserve_ar': True
        },
    'lores': {
        'format': 'RGB888',
        'size': (320, 240),
        'stride': 640,
        'framesize': 230400,
        'preserve_ar': True
        },
    'raw': {
        'format': 'BGGR_PISP_COMP1',
        'size': (4608, 2592),
        'stride': 4608,
        'framesize': 11943936,
        'preserve_ar': True
        },
    'controls': {
        'NoiseReductionMode': controls.draft.NoiseReductionModeEnum.HighQuality,
        'FrameDurationLimits': (100, 1000000000),
        'AfMode': controls.AfModeEnum.Auto,
        'AeEnable': False,
        'AwbEnable': False,
        'ExposureTime': 3500,
        'AnalogueGain': 1.5,
        },
    'sensor': {},
    'display': 'lores',
    'encode': None
}

class CoralCameras:
    def __init__(self):
        self.cameras = {}
        self.cameras[0] = Picamera2(0)
        try:
            self.cameras[1] = Picamera2(1)
        except IndexError:
            print("Only one camera available.")
        
        self.ALL_CAMERAS = self.cameras.keys()
        
        self.ExposureTime = {k: 20000 for k in self.ALL_CAMERAS}
        self.AnalogueGain = {k: 1.0 for k in self.ALL_CAMERAS}
        
        # Add enhancement settings for each camera
        self.enhancement_settings = {}
        for cam_id in self.ALL_CAMERAS:
            self.enhancement_settings[cam_id] = {
                'enabled': False,
                'remove_black_background': True,
                'black_threshold': 15,
                'lower_limit': 0,
                'upper_limit': 255,
                'gamma': 1.0,
                'contrast': 1.0,
                'brightness': 0,
                'denoise': False,
                'sharpen': False,
            }
        
        self.setup()
    
    def setup(self):
        for cam in self.cameras.values():
            cam.configure(config)
    
    def set_enhancement_settings(self, camera_id, **settings):
        """Update enhancement settings for a specific camera"""
        if camera_id in self.enhancement_settings:
            self.enhancement_settings[camera_id].update(settings)
    
    def get_enhancement_settings(self, camera_id):
        """Get current enhancement settings for a camera"""
        return self.enhancement_settings.get(camera_id, {}).copy()
    
    def apply_image_enhancement(self, image, camera_id):
        """Apply image enhancement based on camera settings"""
        if not OPENCV_AVAILABLE:
            return image
            
        settings = self.enhancement_settings.get(camera_id, {})
        if not settings.get('enabled', False):
            return image
        
        enhanced = image.copy().astype(np.float32)
        
        # 1. Remove black background
        if settings.get('remove_black_background', False):
            threshold = settings.get('black_threshold', 15)
            black_mask = np.all(enhanced <= threshold, axis=2)
            enhanced[black_mask] = [0, 0, 0]
        
        # 2. Apply levels adjustment
        lower = settings.get('lower_limit', 0)
        upper = settings.get('upper_limit', 255)
        if upper > lower:
            enhanced = np.clip((enhanced - lower) * (255.0 / (upper - lower)), 0, 255)
        
        # 3. Apply gamma correction
        gamma = settings.get('gamma', 1.0)
        if gamma != 1.0:
            enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255.0
        
        # 4. Apply contrast
        contrast = settings.get('contrast', 1.0)
        if contrast != 1.0:
            enhanced = ((enhanced - 127.5) * contrast) + 127.5
        
        # 5. Apply brightness
        brightness = settings.get('brightness', 0)
        if brightness != 0:
            enhanced = enhanced + brightness
        
        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # 6. Optional denoising and sharpening
        if settings.get('denoise', False):
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        if settings.get('sharpen', False):
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return enhanced
    
    def capture_array_rgb(self, camera_id, apply_enhancement=True):
        """Capture RGB array for preview with optional enhancement"""
        if camera_id not in self.cameras:
            return None
            
        try:
            # Capture from lores stream for preview
            frame = self.cameras[camera_id].capture_array("lores")
            
            if frame is not None and apply_enhancement:
                frame = self.apply_image_enhancement(frame, camera_id)
            
            return frame
                
        except Exception as e:
            print(f"Error capturing RGB array from camera {camera_id}: {e}")
            return None
    
    def auto_enhance_for_dark_background(self, camera_id):
        """Automatically set enhancement parameters optimized for dark backgrounds"""
        if not OPENCV_AVAILABLE:
            return
            
        # Capture a sample image for analysis
        sample_image = self.capture_array_rgb(camera_id, apply_enhancement=False)
        
        if sample_image is None:
            return
        
        # Analyze image histogram
        gray = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Find percentiles for auto-levels
        cumsum = np.cumsum(hist)
        total_pixels = cumsum[-1]
        
        # Find 1st and 99th percentiles
        lower_percentile = np.searchsorted(cumsum, total_pixels * 0.01)
        upper_percentile = np.searchsorted(cumsum, total_pixels * 0.99)
        
        # Set optimal settings for dark background
        optimal_settings = {
            'enabled': True,
            'remove_black_background': True,
            'black_threshold': max(10, lower_percentile),
            'lower_limit': max(5, lower_percentile - 5),
            'upper_limit': min(250, upper_percentile + 10),
            'gamma': 0.7,  # Brighten mid-tones
            'contrast': 1.2,  # Increase contrast
            'brightness': 5,  # Slight brightness boost
            'denoise': True,
            'sharpen': False,
        }
        
        self.set_enhancement_settings(camera_id, **optimal_settings)
        print(f"Auto-enhancement applied for camera {camera_id}")
        return optimal_settings
    
    def capture_array(self, camera_id, stream='lores'):
        """Capture image as numpy array"""
        if camera_id in self.cameras:
            try:
                return self.cameras[camera_id].capture_array(stream)
            except Exception as e:
                print(f"Error capturing array from camera {camera_id}: {e}")
        return None
    
    def capture_enhanced(self, output_file, cameras=None, delay=5, apply_enhancement=True):
        """Capture images with optional enhancement applied"""
        if isinstance(output_file, str):
            output_file = Path(output_file)
        
        for cam in self._get_cameras(cameras):
            try:
                filename = output_file.parent / (output_file.stem + f'_cam{cam}' + output_file.suffix)
                
                # Pause before taking pic
                shutter_s = self.ExposureTime[cam] * 1e-6
                time.sleep(shutter_s * (delay + 1))
                
                if apply_enhancement and OPENCV_AVAILABLE:
                    # Capture as array, enhance, then save
                    image_array = self.cameras[cam].capture_array('main')
                    if image_array is not None:
                        enhanced_image = self.apply_image_enhancement(image_array, cam)
                        
                        # Convert RGB to BGR for OpenCV saving
                        enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(filename), enhanced_bgr)
                        print(f"Enhanced image saved: {filename}")
                    else:
                        print(f"Failed to capture array from camera {cam}")
                else:
                    # Standard capture
                    self.cameras[cam].capture_file(str(filename), 'main')
                    print(f"Standard image saved: {filename}")
                    
            except Exception as e:
                print(f"Error capturing from camera {cam}: {e}")
    
    # Keep all your existing methods
    def start(self, cameras=None):
        for cam in self._get_cameras(cameras):
            self.cameras[cam].start()
            
    def stop(self, cameras=None):
        for cam in self._get_cameras(cameras):
            self.cameras[cam].stop()
    
    def start_preview(self, cameras=None, preview_engine=None):
        if preview_engine is None:
            preview_engine = Preview.QTGL
        for cam in self._get_cameras(cameras):
            self.cameras[cam].start_preview(preview_engine)
    
    def stop_preview(self, cameras=None):
        for cam in self._get_cameras(cameras):
            self.cameras[cam].stop_preview()
     
    def focus_auto(self, cameras=None):
        for cam in self._get_cameras(cameras):
            self.cameras[cam].set_controls({
                'AfMode': controls.AfModeEnum.Auto
                })
            self.cameras[cam].autofocus_cycle()

    def focus_manual(self, lens_position=0.0, cameras=None):
        for cam in self._get_cameras(cameras):
            self.cameras[cam].set_controls({
                'AfMode': controls.AfModeEnum.Manual,
                'LensPosition': lens_position
                })
            
    def set_exposure(self, exposure_us=15000, cameras=None):
        for cam in self._get_cameras(cameras):
            self.cameras[cam].set_controls({"ExposureTime": exposure_us})
            self.ExposureTime[cam] = exposure_us
    
    def set_gain(self, analogue_gain=1.0, cameras=None):
        for cam in self._get_cameras(cameras):
            self.cameras[cam].set_controls({"AnalogueGain": analogue_gain})
            self.AnalogueGain[cam] = analogue_gain
    
    def set_controls(self, control_input, cameras=None):
        for cam in self._get_cameras(cameras):
            self.cameras[cam].set_controls(control_input)
    
    def _get_cameras(self, cameras=None):
        if cameras is None:
            cameras = self.ALL_CAMERAS 
        if isinstance(cameras, int):
            cameras = [cameras]
        
        return [c for c in cameras if c in self.cameras]
    
    def capture(self, output_file=None, cameras=None, delay=5):
        """Legacy capture method - use capture_enhanced for enhanced images"""
        return self.capture_enhanced(output_file, cameras, delay, apply_enhancement=False)

