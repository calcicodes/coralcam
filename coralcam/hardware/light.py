from rpi_hardware_pwm import HardwarePWM

class Light:
    def __init__(self):
        self.led = HardwarePWM(pwm_channel=2, hz=333, chip=2)
        self.brightness = 100
        
    def set_brightness(self, value=50):
        self.brightness = value
        self.led.start(self.brightness)
    
    def on(self):
        self.led.start(self.brightness)
    
    def off(self):
        self.led.stop()
