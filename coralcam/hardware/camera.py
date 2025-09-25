from pathlib import Path
import time
from picamera2 import Picamera2, Preview
from libcamera import controls, Transform, ColorSpace

config = {
    'use_case': 'still',
    'transform': Transform(),  #<libcamera.Transform 'identity'>,
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
        'AfMode': controls.AfModeEnum.Auto,  # allow auto focus
        'AeEnable': False,  # no auto exposure
        'AwbEnable': False,  # no auto white balance
        'ExposureTime': 20000,
        'AnalogueGain': 1.0,
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
        
        self.setup()
    
    def setup(self):
        for cam in self.cameras.values():
            cam.configure(config)
        
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
        if isinstance(output_file, str):
            output_file = Path(output_file)
        
        for cam in self._get_cameras(cameras):
            filename = output_file.parent / (output_file.stem + f'_cam{cam}' + output_file.suffix)
            
            # pause before taking pic
            shutter_s = self.ExposureTime[cam] * 1e-6
            time.sleep(shutter_s * (delay + 1))
            
            # take pic
            img = self.cameras[cam].capture_file(filename, 'main')
            
    