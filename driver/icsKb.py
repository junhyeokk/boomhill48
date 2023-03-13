import directkeys as kb

keymap = [kb.UP, kb.LEFT, kb.RIGHT, kb.LCTRL, kb.LSHIFT, kb.DOWN]

def str2keys(rst_str: str):
    for idx in range(len(keymap)):
        if rst_str[idx] == '1':
            kb.PressKey(keymap[idx])
        else:
            kb.ReleaseKey(keymap[idx])

import time

if __name__ == "__main__":
    time.sleep(3)
    str2keys("100000")
    time.sleep(2)
    str2keys("000000")