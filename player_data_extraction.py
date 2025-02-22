import csv
import time
import urllib.parse  # For encoding player names in URLs
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

# List of player names
player_names = [
    
]  # Replace with your actual list of players
try:
    with open("skaters.csv", mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        skater_names = [row[0].strip() for row in reader if row]  # Read names, remove empty lines
        player_names.extend(skater_names)  # Append to the existing list
        print(f"✅ Loaded {len(skater_names)} skaters from skaters.csv")
except Exception as e:
    print(f"❌ Error reading skaters.csv: {e}")

# Open CSV file and write headers
csv_filename = "hockey_players_stats.csv"
headers = ["Player", "Season", "Team", "League", "GP", "G", "A", "PTS", "PIM", "+/-"]

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write headers

    # Loop through each player
    for player_name in player_names:
        print(f"Scraping data for {player_name}...")

        # Format player name for URL
        formatted_name = player_name.replace(" ", "+")

        print(formatted_name)
        url = f"https://www.hockeydb.com/ihdb/stats/find_player.php?full_name={formatted_name}"

        # Load the player's page
        driver.set_page_load_timeout(30)  # Increase timeout to 30 seconds
        try:
            driver.get(url)
        except:
            print(f"⚠️ Timeout loading {url}. Skipping...")
            continue  # Move to the next player if the page fails

        time.sleep(2)  # Prevent rate limiting

        # Extract HTML and parse with BeautifulSoup
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Locate the stats table
        table = soup.find("table")
        
        if table:
            # Extract all rows (excluding header and footer)
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

                    writer.writerow([player_name, season, team, league, gp, g, a, pts, pim, plus_minus])

print(f"\n✅ Data saved to {csv_filename} with stats from {len(player_names)} players!")
driver.quit()
