# pin setup for Adafruit TMC2209 Stepper Driver

# VCC-----3V3  (1) (2)  5V    
#       GPIO2  (3) (4)  5V    
#       GPIO3  (5) (6)  GND-----GND
#       GPIO4  (7) (8)  GPIO14--UART via 1K resistor
#         GND  (9) (10) GPIO15--UART
#      GPIO17 (11) (12) GPIO18--LED control (hardware PWM)
#      GPIO27 (13) (14) GND   
#      GPIO22 (15) (16) GPIO23
#         3V3 (17) (18) GPIO24
#      GPIO10 (19) (20) GND   
#       GPIO9 (21) (22) GPIO25
#      GPIO11 (23) (24) GPIO8 
#         GND (25) (26) GPIO7
#       GPIO0 (27) (28) GPIO1 
#       GPIO5 (29) (30) GND   
#       GPIO6 (31) (32) GPIO12
#       GPIO13 (33) (34) GND   
#       GPIO19 (35) (36) GPIO16-STEP
#       GPIO26 (37) (38) GPIO20-DIR
#          GND (39) (40) GPIO21-EN (enable motor)

import board

pins = {
    'EN': 21,
    'DIR': 20,
    'STEP': 16,
    'LED': 18,
}