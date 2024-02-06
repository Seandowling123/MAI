import pyautogui
import time
import random

# Most recent article downloaded
#beginning = 500
with open("Current_Articles_downloaded.txt") as f:
    beginning = int(f.read())
print("Downloading articles beginning from:", beginning)

time.sleep(4)

# Chrome options
pyautogui.moveTo(1890, 80, duration=random.uniform(.3, .7))
pyautogui.click()

# Open tab
pyautogui.moveTo(1633, 203, duration=random.uniform(.2, .5))
pyautogui.click()

# Search bar
pyautogui.moveTo(685, 78, duration=random.uniform(.3, .7))
pyautogui.click()
pyautogui.typewrite("tcd lexisnexis", interval=random.uniform(.01, .1))
pyautogui.press('enter')

# Privacy notice
pyautogui.moveTo(1103, 1012, duration=random.uniform(.3, .7))
pyautogui.click()

# Tcd law website
time.sleep(1)
pyautogui.moveTo(468, 600, duration=random.uniform(.7, 1))
pyautogui.click()
time.sleep(1.5)
pyautogui.scroll(-4000)

# Lexis Nexis News and Business
pyautogui.moveTo(819, 637, duration=random.uniform(.2, .5))
pyautogui.click()

# If login page
time.sleep(1.5)
px = pyautogui.pixel(100, 500)
if px != (0, 24, 46):
    # Tcd email
    pyautogui.moveTo(602, 487, duration=random.uniform(.3, .5))
    pyautogui.click()

    # Autofill
    pyautogui.moveTo(741, 552, duration=random.uniform(.5, .7))
    pyautogui.click()

    # Login
    pyautogui.moveTo(574, 569, duration=random.uniform(.1, .3))
    pyautogui.click()
    time.sleep(15)

# News
pyautogui.moveTo(362, 255, duration=random.uniform(.3, .7))
pyautogui.click()

# Advanced search
time.sleep(6)
pyautogui.moveTo(1511, 595, duration=random.uniform(.3, .7))
pyautogui.click()

# Search query
time.sleep(4)
pyautogui.moveTo(922, 551, duration=random.uniform(.3, .7))
pyautogui.click()
pyautogui.typewrite("company(Ryanair) and >=1/1/2003 <=31/12/2023", interval=random.uniform(.003, .005))
pyautogui.scroll(-4000)

# Language English
pyautogui.moveTo(303, 594, duration=random.uniform(.3, .7))
pyautogui.click()

# Search
pyautogui.moveTo(331, 741, duration=random.uniform(.3, .7))
pyautogui.click()
time.sleep(11)

# If popup -> close
px = pyautogui.pixel(1593, 436)
if px == (43, 59, 96):
    pyautogui.moveTo(1855, 444, duration=random.uniform(.5, .1))
    pyautogui.click()

# Sort 
pyautogui.moveTo(1795, 431, duration=random.uniform(.7, 1))
pyautogui.click()

# Newewst to oldest
pyautogui.moveTo(1778, 531, duration=random.uniform(.3, .7))
pyautogui.click()
time.sleep(6)

# Actions
pyautogui.moveTo(470, 302, duration=random.uniform(.3, .7))
pyautogui.click()

# Moderate similarity
pyautogui.moveTo(519, 663, duration=random.uniform(.3, .7))
pyautogui.click()
time.sleep(6)

# Wait to select company
time.sleep(25)

# Download
pyautogui.moveTo(775, 435, duration=random.uniform(.3, .7))
pyautogui.click()
time.sleep(1)

# Options
pyautogui.moveTo(693, 362, duration=random.uniform(.2, .5))
pyautogui.click()

# Remove formatting
pyautogui.moveTo(432, 567, duration=random.uniform(.1, .3))
pyautogui.click()

pyautogui.moveTo(431, 603, duration=random.uniform(.1, .3))
pyautogui.click()

pyautogui.moveTo(432, 638, duration=random.uniform(.1, .3))
pyautogui.click()

pyautogui.moveTo(431, 678, duration=random.uniform(.1, .3))
pyautogui.click()

# Basic options
pyautogui.moveTo(502, 371, duration=random.uniform(.1, .3))
pyautogui.click()

# Article numbers
pyautogui.moveTo(609, 598, duration=random.uniform(.3, .7))
pyautogui.click()
article_range = [beginning+1, beginning+500]
pyautogui.typewrite(f"{article_range[0]}-{article_range[1]}", interval=random.uniform(.01, .1))

# Download
pyautogui.moveTo(1238, 872, duration=random.uniform(.3, .7))
pyautogui.click()

# Check download wheel
time.sleep(20)
px = pyautogui.pixel(275, 611)
while px == (55, 55, 57):
    px = pyautogui.pixel(275, 611)
    i = 0

print("Download wheel disappeared awaiting download bubble.")

# Check for download bubble
time.sleep(2)
px = pyautogui.pixel(900, 25)
while px == (31, 32, 32):
    px = pyautogui.pixel(900, 25)
    i = 0
print("Download bubble detected.")

# File name # London Stock Exchange Regulatory News Service
time.sleep(1)
pyautogui.typewrite(f"ThomsonFinancial({article_range[0]}-{article_range[1]})")
print("Articles Downloaded.")

# Update the number of articles downloaded
f = open("Current_Articles_downloaded.txt", "w")
f.write(str(article_range[1]))
f.close()