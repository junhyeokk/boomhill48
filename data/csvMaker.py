import ctypes
import time

import pandas as pd

import win32gui
from mss import mss
from PIL import Image

IMG_PATH = "../img/"
FILENAME = "../test.csv"

INTERVAL = 0.1

ENTER = 0x0D
HANJA = 0x19
SPACE = 0x20

LEFT    = 0x25
UP      = 0x26
RIGHT   = 0x27
DOWN    = 0x28
LSHIFT  = 0xA0
LCTRL   = 0xA2

keymap = [UP, LEFT, RIGHT, LCTRL, LSHIFT] #, DOWN]

def getkey(vkKeyCode):
    return ctypes.windll.user32.GetAsyncKeyState(vkKeyCode) > 1

if __name__ == "__main__":

    hwnd = win32gui.FindWindow(None, "KartRider Client")
    if hwnd == 0:
        quit("Please run KartRider")
    rect = win32gui.GetWindowRect(hwnd)
    win_pos = {"top" : rect[1], "left" : rect[0], "width" : 1920, "height" : 1080}

    print("Press ENTER to record...")
    while True:
        if getkey(ENTER):
            break
        time.sleep(INTERVAL)

    print("START!!")
    print("Press SPACEBAR to stop...")

    imgnames = []
    inputs = []
    cnt = 0

    while True:

        start_time = time.time()

        if getkey(SPACE):
            break

        sct = mss()
        sct_img = sct.grab(win_pos)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        imgfile = str(cnt) + ".jpg"
        imgnames.append(imgfile)
        img.save(IMG_PATH + imgfile)

        keystr = ""
        for idx in range(len(keymap)):
            if getkey(keymap[idx]):
                keystr = keystr + '1'
            else:
                keystr = keystr + '0'
        inputs.append(keystr)

        print(keystr)

        t = time.time() - start_time
        if t < 0.1:
            time.sleep(INTERVAL - t)
        
        cnt += 1
    
    df = pd.DataFrame({"imgname" : imgnames, "input" : inputs})
    df.to_csv(FILENAME, header = False, index = False)