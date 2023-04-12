#!/usr/bin/python3

# Install necessary dependences before starting,
#
# $ sudo apt update
# $ sudo apt install build-essential
# $ sudo apt install libatlas-base-dev
# $ sudo apt install python3-pip
# $ pip3 install tflite-runtime
# $ pip3 install opencv-python==4.4.0.46
# $ pip3 install pillow
# $ pip3 install numpy
#
# $ python3 real_time_with_labels.py --model mobilenet_v2.tflite --label coco_labels.txt

import argparse

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from picamera2 import MappedArray, Picamera2, Preview

import I2C_LCD_driver
import time
import RPi.GPIO as GPIO
import os
import random

normalSize = (640, 480)
lowresSize = (320, 240)

rectangles = []
phoneCount = [0]
LCDSequenceStart = [False]


def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def DrawRectangles(request):
    with MappedArray(request, "main") as m:
        for rect in rectangles:
            # print(rect)
            rect_start = (int(rect[0] * 2) - 5, int(rect[1] * 2) - 5)
            rect_end = (int(rect[2] * 2) + 5, int(rect[3] * 2) + 5)
            cv2.rectangle(m.array, rect_start, rect_end, (0, 255, 0, 0))
            if len(rect) == 5:
                text = rect[4]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(m.array, text, (int(rect[0] * 2) + 10, int(rect[1] * 2) + 10),
                            font, 1, (255, 255, 255), 2, cv2.LINE_AA)


def InferenceTensorFlow(image, model, output, label=None):
    global rectangles

    if label:
        labels = ReadLabelFile(label)
    else:
        labels = None

    interpreter = tflite.Interpreter(model_path=model, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = False
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    initial_h, initial_w, channels = rgb.shape

    picture = cv2.resize(rgb, (width, height))

    input_data = np.expand_dims(picture, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])

    rectangles = []
    for i in range(int(num_boxes)):
        top, left, bottom, right = detected_boxes[0][i]
        classId = int(detected_classes[0][i])
        score = detected_scores[0][i]
        if classId == 76:
            # Only draw detection if it is a phone
            if score > 0.5:
                phoneCount[0] += 1
                xmin = left * initial_w
                ymin = bottom * initial_h
                xmax = right * initial_w
                ymax = top * initial_h
                box = [xmin, ymin, xmax, ymax]
                rectangles.append(box)
                if labels:
                    # print(labels[classId], 'score = ', score)
                    rectangles[-1].append(labels[classId])
                else:
                    # print('score = ', score)
                    pass
async def countdown():
    '''
    Async function that executes a 60 second countdown, printing the remaining time on the screen
    '''
    timeCount = 60
    while timeCount >= 0:
        if LCDSequenceStart[0] == False:
            break
        if timeCount < 10:
            mylcd.lcdclear()
        mylcd.lcd_display_string(f"Timeout in {timeCount} s",1,0)
        timeCount -= 1
        
        time.sleep(1)
        
async def countIsDone():
    '''
    awaits the countdown, and if countdown reaches end, execute punishment sequence
    '''
    await countdown()
    mylcd.lcd_clear()
    mylcd.lcd_display_string("COUNT OVER", 1, 0)
    
def main():
    
    # initialize LCD
    mylcd = I2C_LCD_driver.lcd()
    defaultMsg = "Hi, I am"
    mylcd.lcd_display_string(defaultMsg, 1, 0)
    defaultMsg = "Squirtlebot!"
    mylcd.lcd_display_string(defaultMsg, 2, 0)
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', help='Path of the detection model.', required=True)
    # parser.add_argument('--label', help='Path of the labels file.')
    # parser.add_argument('--output', help='File path of the output image.')
    # args = parser.parse_args()

    # if (args.output):
    #     output_file = args.output
    # else:
    #     output_file = 'out.jpg'

    # if (args.label):
    #     label_file = args.label
    # else:
    #     label_file = None
    model = 'mobilenet_v2.tflite'
    label_file = 'coco_labels.txt'
    output_file = 'out.jpg'

    picam2 = Picamera2()
    picam2.start_preview(Preview.QTGL)
    config = picam2.create_preview_configuration(main={"size": normalSize},
                                                 lores={"size": lowresSize, "format": "YUV420"})
    picam2.configure(config)

    stride = picam2.stream_configuration("lores")["stride"]
    picam2.post_callback = DrawRectangles

    picam2.start()
    
    # Sequence variable initialization
    nophoneCount = 0
    timeCount = 60
    counterForCounter = 0
    # LCDSequenceStart = False
    

    while True:
        buffer = picam2.capture_buffer("lores")
        grey = buffer[:stride * lowresSize[1]].reshape((lowresSize[1], stride))
        prevCount = phoneCount[0]
        _ = InferenceTensorFlow(grey, model, output_file, label_file)
        
        # Check if phone counter is greater than 10 and sequence hasnt been started. If it is, start LCD sequence
        print(phoneCount[0])
        
        if phoneCount[0] >= 10 and LCDSequenceStart[0] == False:
            LCDSequenceStart[0] = True # run countdown
            phoneCount[0] = 0
            # Run Squirtle sound to indicate LCD countdown start
            os.chdir('/home/mie438/SquirtleBot/squirtleSounds')
            introSounds = ['squirtle_timerstart1.wav', 'squirtle_timerstart2.wav',
                           'squirtle_timerstart3.wav', 'squirtle_timerstart4.wav',
                           'squirtle_timerstart5.wav']
            choiceIdx = np.random.choice(3,1)
            if choiceIdx == 0:
                os.system('python3 squirtlebotSound.py squirtle_timerstart1.wav')
            elif choiceIdx == 1:
                os.system('python3 squirtlebotSound.py squirtle_timerstart2.wav')
            elif choiceIdx == 2:
                os.system('python3 squirtlebotSound.py squirtle_timerstart4.wav')
                
            os.chdir('/home/mie438/SquirtleBot')
            
        if LCDSequenceStart[0] and counterForCounter >= 5:
            counterForCounter = 0
            if timeCount >= 0:
                mylcd.lcd_clear()
                mylcd.lcd_display_string(f"Timeout in {timeCount}s",1,0)
                timeCount -= 1
            else:
                # punishment
                mylcd.lcd_clear()
                mylcd.lcd_display_string("TIME'S UP: USING", 1, 0)
                mylcd.lcd_display_string("WATER GUN!", 2, 0)
                # Run Squirtle sound to indicate pump activation
                os.chdir('/home/mie438/SquirtleBot/squirtleSounds')
                os.system('python3 squirtlebotSound.py squirtle_timerend.wav')
                
                # activating pump
                # initialize pump
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(17,GPIO.OUT)
                GPIO.output(17, GPIO.LOW)
                os.system('python3 squirtlebotSound.py squirtlewatergun.wav') #pump runs during this
                GPIO.output(17, GPIO.HIGH)
                GPIO.cleanup()
                os.chdir('/home/mie438/SquirtleBot')
                
                # Reset counters and LCD to continue looping
                time.sleep(3)
                LCDSequenceStart[0] = False
                nophoneCount = 0
                phoneCount[0] = 0
                timeCount = 60
                mylcd = I2C_LCD_driver.lcd()
                mylcd.lcd_display_string("Hi, I am", 1, 0)
                mylcd.lcd_display_string("Squirtlebot!", 2, 0)
                
        if phoneCount[0] == prevCount:
            # Get to this section of code when nothing is being detected from model
            nophoneCount += 1
        else:
            nophoneCount = 0

        # Reset phone count if detection does not exceed 10 (timer has not begun)
        if LCDSequenceStart[0] == False and nophoneCount >= 10:
            phoneCount[0] = 0
            nophoneCount = 0
    
        # Sequence has already started, so keep running until LCD interrupt or no phone being detected
        if LCDSequenceStart[0] == True and nophoneCount > 30:
            LCDSequenceStart[0] = False
            nophoneCount = 0
            phoneCount[0] = 0
            timeCount = 60
            mylcd.lcd_clear()
            defaultMsg = "Hi, I am"
            mylcd.lcd_display_string(defaultMsg, 1, 0)
            defaultMsg = "Squirtlebot!"
            mylcd.lcd_display_string(defaultMsg, 2, 0)
            # Run Squirtle sound to indicate that the LCD timer has stopped
            os.chdir('/home/mie438/SquirtleBot/squirtleSounds')
            os.system('python3 squirtlebotSound.py squirtle.wav')
            os.chdir('/home/mie438/SquirtleBot')
        time.sleep(0.01)
        counterForCounter += 1

if __name__ == '__main__':
    main()
