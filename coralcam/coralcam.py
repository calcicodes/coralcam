from pathlib import Path
from .hardware.camera import CoralCameras
from .hardware.motor_TMC2209 import Motor
from .hardware.light import Light

class CoralScanner:
    def __init__(self):
        self.cam = CoralCameras()
        self.cam.start_preview()
        self.cam.start()    
        
        self.motor = Motor()
        
        self.light = Light()
    
    def autofocus(self):
        self.cam.focus_auto()
    
    def set_exposure(self, exposure_us, camera=None):
        self.cam.set_exposure(exposure_us, camera)
        
    def light_off(self):
        self.light.off()
        
    def light_on(self, brightness_percent=100):
        self.light.set_brightness(brightness_percent)
    
    def capture_revolution(self, name, n_angles=90, folder='~/coral_scans'):
        step = 360.0 / n_angles
        
        if isinstance(folder, str):
            folder = Path(folder)
        
        folder = folder.expanduser() / name
        if not folder.is_dir():
            folder.mkdir(parents=True, exist_ok=True)
        
        for i in range(n_angles):
            filename = folder / f'{name}_{i:03}.jpg'
            self.cam.capture(filename)
            
            self.motor.rotation(step)
            