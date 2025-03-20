from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json
import keyboard
# Set the path to the Chromedriver
DRIVER_PATH = 'G:\Downloads\chromedriver-win64(1).zip\chromedriver-win64\chromedriver.exe'
options = Options()
options.headless = True  # Enable headless mode
options.add_argument("--window-size=1920,1200")
# Initialize the Chrome driver
driver = webdriver.Chrome()

#go to search results
# Navigate to the URL
driver.get('https://www.canlii.org/#search/type=decision&ccType=tribunals&startDate=2010-01-01&endDate=2025-12-31&sort=decisionDateAsc&text=ottawa%20K%20-k9%20-k7%20-k8%20-k6&id=LTB-T&searchId=2025-03-19T17%3A58%3A53%3A392%2F34de77dc924f4354bd0f81ba44703290')

#smaller
# driver.get('https://www.canlii.org/en/on/onltb/#search/type=decision&ccId=onltb&startDate=2022-02-01&endDate=2025-12-31&sort=decisionDateDesc&text=landlord%20ottawa&id=EAL-&searchId=2025-03-14T19%3A09%3A51%3A011%2F286c111e180845fd943ff2cba82a1c34&origType=decision&origCcId=onltb')

time.sleep(2.5)
cookies = driver.find_element(By.ID, 'understandCookieConsent')
cookies.click()

print('Stopping script for captcha.\nPress "ctrl+shift+x" to continue script')
while True:        
    if keyboard.is_pressed("ctrl+shift+x"):
        print('Script will continue in 2 seconds :)')
        time.sleep(3) 
        break


# while True:
for i in range(10):
    try:
        link = driver.find_element(By.LINK_TEXT, "Load more...")
        link.click()
        time.sleep(3)
    except:
        break


div = driver.find_element(By.ID, 'searchResults')
print(div)
list = div.find_element(By.CLASS_NAME, 'list-unstyled')
print(list)
results = div.find_elements(By.CLASS_NAME, 'result')
print(len(results))

textmap = {}
with open('newdata.json', encoding='utf-8') as f:
    textmap = json.load(f)
f.close()
for e in results:
    print(len(results))
# for i in range(10):
    # e = results[i]
    link = e.find_element(By.TAG_NAME, 'a')
    # print(link.text())
    print(link.get_attribute('href'))
    #case id
    caseid = link.get_attribute('href').split('/')[8]
    keywords = e.find_element(By.CLASS_NAME, 'keyword')
    t = keywords.text.split('—')
    # print(caseid)
    driver.get(link.get_attribute('href'))
    time.sleep(5)
    try:
        h1 = driver.find_element(By.ID, 'documentContent')
        # print(h1.text)
        textmap[caseid] = [h1.text, t]
    except Exception as e:
        print("trying class name: " + caseid)
        # print(e)
        try:
            h1 = driver.find_elements(By.CLASS_NAME, 'documentContent')[-1]
            # print(h1.text)
            textmap[caseid] = [h1.text, t]
        except Exception as f:
            print('final error: ' + caseid)
            print(f)       


    driver.back()
    time.sleep(5)

# print('https://www.canlii.org/en/on/onltb/doc/2023/2023onltb19375/2023onltb19375.html?resultId=f00833f2794647cb84083c3406ffc036&searchId=2025-03-14T19:09:51:011/286c111e180845fd943ff2cba82a1c34&searchUrlHash=AAAAAQAPbGFuZGxvcmQgb3R0YXdhAAAAAAE'.split())


print(len(textmap))
# print(textmap.keys()[-1])
# js = json.loads(textmap)
f = open("newdata.json", "w", encoding='utf-8')
json.dump(textmap, f, ensure_ascii=False)
f.close()
# f.write(js)

# Navigate to the URL
# driver.get('https://www.canlii.org/en/on/onltb/doc/2014/2014canlii76727/2014canlii76727.html')
# h1 = driver.find_element(By.ID, 'documentContent')
# print(h1.text)
# It's a good practice to close the browser when done
driver.quit()


