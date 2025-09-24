from tmc_driver.tmc_2209 import *
from .config import pins

class Motor:
    def __init__(self, gear_ratio=5):

        self.motor = Tmc2209(
            TmcEnableControlPin(pins['EN']),
            TmcMotionControlStepDir(pins['STEP'], pins['DIR']),
            TmcComUart("/dev/ttyAMA0"),
            loglevel=Loglevel.DEBUG
        )
        
        self.gear_ratio = gear_ratio
        
        self.setup()
    
    def setup(self):

        self.motor.movement_abs_rel = MovementAbsRel.RELATIVE

        self.motor.set_direction_reg(False)
        self.motor.set_current(300)
        self.motor.set_interpolation(True)
        self.motor.set_spreadcycle(False)
        self.motor.set_microstepping_resolution(8)
        self.motor.set_internal_rsense(False)

        self.motor.acceleration_fullstep = 1000
        self.motor.max_speed_fullstep = 100

        self.motor.set_motor_enabled(True)

    def rotation(self, degrees):
        revs = round(self.gear_ratio * degrees / 360.0, 2)
        self.motor.run_to_position_revolutions(revs)
