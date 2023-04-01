import pygame
import time
import RPi.GPIO as GPIO

# Pin setup
play_squirtle = True ## This controls when squirtle starts speaking: have to create a function that writes this
GPIO.setmode(GPIO.BOARD)
GPIO.setup(37, GPIO.OUT) # Set pin 37 to the speaker output

pygame.mixer.init()
pygame.mixer.music.load("squirtle.wav")
pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue