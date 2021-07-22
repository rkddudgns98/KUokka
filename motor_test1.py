import RPi.GPIO as GPIO
from time import sleep

# Pins for Motor Driver Inputs 
Motor1A = 20
Motor1B = 21
Motor1E = 26

GPIO.setmode(GPIO.BCM)				# GPIO Numbering
GPIO.setup(Motor1A, GPIO.OUT)  		# All pins as Outputs
GPIO.setup(Motor1B, GPIO.OUT)
GPIO.setup(Motor1E, GPIO.OUT)

print("start")

# Going forwards
GPIO.output(Motor1A, 1)
GPIO.output(Motor1B, 0)
GPIO.output(Motor1E, 1)

sleep(5)

# Going backwards
GPIO.output(Motor1A, 0)
GPIO.output(Motor1B, 1)
GPIO.output(Motor1E, 1)
 
sleep(5)

# Stop
GPIO.output(Motor1A, 0)
GPIO.output(Motor1B, 0)
GPIO.output(Motor1E, 0)

GPIO.cleanup()

print("end")