import pyautogui
import time

# Display the mouse position
pyautogui.displayMousePosition()

time.sleep(5)
# If popup -> close
px = pyautogui.pixel(1593, 436)
print(px)