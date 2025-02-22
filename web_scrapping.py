import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Configure Selenium
options = Options()
options.headless = True  # Runs in the background
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Load the HockeyDB page
url = "https://www.hockeydb.com/ihdb/stats/pdisplay.php?pid=160293"
driver.get(url)

# Extract HTML and parse with BeautifulSoup
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
driver.quit()  # Close browser after scraping

# Locate the main stats table
table = soup.find("table")  # Adjust if necessary

# Extract all rows (excluding header and footer)
data = []
for row in table.find_all("tr"):
    cols = row.find_all("td")
    
    # Ensure the row has enough data and ignore summary rows
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

        data.append([season, team, league, gp, g, a, pts, pim, plus_minus])

# Save to CSV file
csv_filename = "player_stats.csv"
headers = ["Season", "Team", "League", "GP", "G", "A", "PTS", "PIM", "+/-"]

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write headers
    writer.writerows(data)  # Write data rows

print(f"Data saved to {csv_filename} ✅")