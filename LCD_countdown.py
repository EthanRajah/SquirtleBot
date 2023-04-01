import I2C_LCD_driver
import time
mylcd = I2C_LCD_driver.lcd()
mylcd.lcd_display_string("Test sequence!", 1,0)

lookingAtPhone = True # to be implemented/fed by camera control
activatePunishment = False
breakTime = 60 # break time specified
timeCount = breakTime

def resetLCD(lcd):
    '''
    Resets the LCD screen after the punishment occurs
    '''
    time.sleep(3) # wait some time for the punishment to occur
    lcd.lcd_display_string("clearing", 1, 0)
    time.sleep(3)

if __name__ == "__main__":
    defaultMsg = "Hi im Squirtle"
    mylcd.lcd_display_string(defaultMsg,1,0)
    while True:
        if lookingAtPhone == True:
            if timeCount <= 0:
                activatePunishment == True # to be implemented what this does
                mylcd.lcd_display_string("COUNT UP: USING WATER GUN", 1, 0)
            else:
                mylcd.lcd_display_string(f"Timing out in {timeCount} s",1,0)
                timeCount -= 1
        else:
            timeCount = breakTime # reset the timer
            mylcd.lcd_display_string(defaultMsg,1,0)
        time.sleep(1)