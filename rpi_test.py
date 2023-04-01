# Write your code here :-)
import time
import board
import RPi.GPIO as GPIO
from audiocore import WaveFile
from audiopwmio import PWMAudioOut as AudioOut

# Pin setup
play_squirtle = True ## This controls when squirtle starts speaking: have to create a function that writes this
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(37, GPIO.OUT) # Set pin 37 to the speaker output
audio = AudioOut(board.GP37)

# play sound
wave = WaveFile("squirtle.wav")

While play_squirtle == True:
    audio.play(wave)
