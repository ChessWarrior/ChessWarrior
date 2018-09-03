'''
多线程爬虫
'''
import time
from queue import Queue
import threading

from selenium import webdriver
from bs4 import BeautifulSoup
import requests

browser = webdriver.Chrome()

urls = Queue()
val = 0
num = 0

def login(base_url):
    browser.set_page_load_timeout(60)
    try:
        browser.get(base_url)
    except Exception as e:
        print(e)
    
    browser.find_element_by_css_selector('[class="signin button text"]').click()
    
    browser.find_element_by_css_selector("[name='username']").send_keys("izhongyuchen@gmail.com")
    browser.find_element_by_css_selector("[name='password']").send_keys("zyc759631647")
    browser.find_element_by_css_selector('[class="submit button"]').click()
    time.sleep(10)
    while browser.current_url != "https://lichess.org/games/search":
        time.sleep(5)
    

    return True

def change(i, sort=True, winner=True, oppo=True):
    
    url = "https://lichess.org/games/search?"
    
    url += "ratingMin=" + str(i*100) + "&"
    url += "ratingMax=" + str(i*100) + "&"

    if oppo:
        url += "hasAi=1&"
    else:
        url += "hasAi=0&"

    if winner:
        url += "winnerColor=1&"
    else:
        url += "winnerColor=0&"

    if sort:
        url += "sort.order=desc&"
    else:
        url += "sort.order=asc&"


    url += "analysed=1"

    try:
        browser.get(url)
        browser.find_element_by_xpath('//*[@id="lichess"]/div/form/table/tbody/tr[21]/td/button').click()
        return True
    except Exception as e:
        print(e)
        return False


class Producer(threading.Thread):
    def __init__(self,  cond, urls):
        super(Producer, self).__init__()
        self.urls = urls
        self.cond = cond
        self._running = True
    
    def run(self):
        url_set = set()
        global val
        while val <= 3:
            result_raw = browser.page_source
            result_soup = BeautifulSoup(result_raw, 'html.parser')
            pgn_list = result_soup.find_all('div', class_='game_row paginated_element')
            
            cnt = 0
            if self.cond.acquire():
                for pgn in pgn_list:
                    mid = pgn.a.attrs['href'].split('/')[1]
                    url = "https://lichess.org/game/export/" + mid + '?literate=1'
                    if url not in url_set:
                        self.urls.put(url)
                        cnt += 1
                    url_set.add(url)
                
                self.cond.notify() #唤醒comsumer
                self.cond.release()
                print("producer release")
                print("Add %d into the queue" % (cnt))
                browser.execute_script('window.scrollTo(0,document.body.scrollHeight);')
            time.sleep(5)
            if cnt == 0:
                val += 1
            else:
                val = 0
            print("sleep over")
        print("producer out")

    def terminate(self):
        self._running = False
    
class Comsumer(threading.Thread):
    def __init__(self, cond, urls):
        super(Comsumer, self).__init__()
        self.urls = urls
        self.cond = cond
        self._running = True
    
    def run(self):
        global num
        global val
        while val <= 3:
            if self.cond.acquire():
                if self.urls.empty():
                    self.cond.wait()
                else:
                    rs = []
                    
                    while not self.urls.empty():
                        url = self.urls.get()
                        try:
                            r = requests.get(url, timeout=5)
                            rs.append(r)
                        except Exception as e:
                            print(e)
                       
                    
                    print("comsumer release")
                    self.cond.release()
                    #保存啊
                    for r in rs:
                        content = r.content.decode()
                        content = content.strip('\r\n').split('\r\n')
                        with open('data/'+str(num)+'.pgn','w', encoding='utf-8') as f:
                            f.writelines(content)
                        num += 1
                    print("write %d pgn" % num)
        print("comsumer out")

    def terminate(self):
        print("comsumer out")
        global val
        val = 0
        self._running = False



if __name__=='__main__':
    base_url = "https://lichess.org/games/search?ratingMin=2500&ratingMax=2900&sort.field=d&sort.order=desc&analysed=1"
    login(base_url)
    
    for i in range(29,0,-1):
        for j in [False, True]:
            for k in [False, True]:
                for z in [False, True]:
                    print("new term begin! i:%d, sort:%d, winner:%d oppo:%d" % (i, j, k, z))
                    while not change(i,j,k,z):
                        try:
                            browser.refresh()
                        except Exception as e:
                            print(e)
                        continue
                    print("start crawling")
                    cond = threading.Condition()
                    procuder = Producer(cond, urls)
                    comsumer = Comsumer(cond, urls)

                    procuder.start()
                    comsumer.start()

                    procuder.join()
                    comsumer.terminate()
                    
                    del procuder
                    del comsumer #防止产生僵尸进程
                    print("terminate thread.")
    
    

