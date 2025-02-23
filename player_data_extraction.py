import csv
import time
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

options = Options()
options.headless = True
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

player_names = [
    
]
try:
    with open("skaters.csv", mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        skater_names = [row[0].strip() for row in reader if row]
        player_names.extend(skater_names)
        print(f"✅ Loaded {len(skater_names)} skaters from skaters.csv")
except Exception as e:
    print(f"❌ Error reading skaters.csv: {e}")

csv_filename = "hockey_players_stats.csv"
headers = ["Player", "Season", "Team", "League", "GP", "G", "A", "PTS", "PIM", "+/-"]

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(headers)

    for player_name in player_names:
        print(f"Scraping data for {player_name}...")

        formatted_name = player_name.replace(" ", "+")

        print(formatted_name)
        url = f"https://www.hockeydb.com/ihdb/stats/find_player.php?full_name={formatted_name}"

        driver.set_page_load_timeout(30)
        try:
            driver.get(url)
        except:
            print(f"Timeout, skipping")
            continue

        time.sleep(2)

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        table = soup.find("table")
        
        if table:
            for row in table.find_all("tr"):
                cols = row.find_all("td")

                if len(cols) >= 9 and "NHL Totals" not in row.text:
                    season = cols[0].text.strip()
                    team = cols[1].text.strip()
                    league = cols[2].text.strip()
                    gp = cols[3].text.strip()
                    g = cols[4].text.strip()
                    a = cols[5].text.strip()
                    pts = cols[6].text.strip()
                    pim = cols[7].text.strip()
                    plus_minus = cols[8].text.strip()

                    writer.writerow([player_name, season, team, league, gp, g, a, pts, pim, plus_minus])

print(f"\n Data saved to {csv_filename}.")
driver.quit()
