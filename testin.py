import pyautogui
import time

time.sleep(4)

# Chrome options
pyautogui.moveTo(1890, 80, duration=1)
pyautogui.click()

# Open tab
pyautogui.moveTo(1633, 203, duration=1)
pyautogui.click()

# Search bar
pyautogui.moveTo(685, 78, duration=1)
pyautogui.click()
pyautogui.typewrite("tcd lexisnexis", interval=0.05)
pyautogui.press('enter')

# Tcd law website
pyautogui.moveTo(541, 623, duration=1)
pyautogui.click()
time.sleep(1)
pyautogui.scroll(-4000)

# Lexis Nexis News and Business
pyautogui.moveTo(819, 637, duration=1.12)
pyautogui.click()

# If no login page
time.sleep(15)
px = pyautogui.pixel(100, 500)
if px != (0, 24, 46):
    # Tcd email
    time.sleep(3)
    pyautogui.moveTo(602, 487, duration=0.3534)
    pyautogui.click()

    # Autofill
    pyautogui.moveTo(741, 552, duration=0.5347835478)
    pyautogui.click()

    # Login
    pyautogui.moveTo(574, 569, duration=0.578635478)
    pyautogui.click()

# News
time.sleep(15)
pyautogui.moveTo(362, 255, duration=0.7892347823)
pyautogui.click()

# Advanced search
time.sleep(5)
pyautogui.moveTo(1511, 595, duration=0.7892347823)
pyautogui.click()

# Search query
time.sleep(2)
pyautogui.moveTo(922, 551, duration=0.7892347823)
pyautogui.click()
pyautogui.typewrite("company(Ryanair) and >=1/1/2003 <=31/12/2023 and (publication(Wall Street Journal Abstracts) OR publication(Financial Times Online) OR publication(Associated Press Financial Wire) OR publication(AirGuide Business & AirGuideBusiness.com) OR publication(Proactive Investors (UK)) OR publication(BNS News Service in English by Baltic News Service (BNS) English) OR publication(Newstex Blogs) OR publication(Live Briefs PRO Global Markets) OR publication(MT Newswires Live Briefs) OR publication(Business World (Digest)) OR publication(MarketLine NewsWire) OR publication(London Stock Exchange Regulatory News Service) OR publication(Sunday Business Post) OR publication(International Business Times News) OR publication(The Investors Chronicle) OR publication(Financial Times (London, England)) OR publication(AirFinance Journal) OR publication(Flight International) OR publication(dpa-AFX International ProFeed) OR publication(dpa international (Englischer Dienst)) OR publication(RTT News (United States)) OR publication(Citywire) OR publication(City A.M.) OR publication(ANSA English Corporate Service) OR publication(American Banking and Market News) OR publication(Transcript Daily) OR publication(Watchlist News) OR publication(DailyPolitical) OR publication(Alliance News UK) OR publication(Thomson Financial News Super Focus))", interval=0.001)
pyautogui.scroll(-4000)
pyautogui.press('enter')

# Language English
time.sleep(5)
pyautogui.moveTo(303, 594, duration=0.7892347823)
pyautogui.click()

# Search
time.sleep(5)
pyautogui.moveTo(359, 684, duration=0.7892347823)
pyautogui.click()

# Display the current mouse position
print(pyautogui.position())

