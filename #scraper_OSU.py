from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

# Set up Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run without opening a browser
driver = webdriver.Chrome(options=chrome_options)

# Load page
url = "https://www.espn.com/college-football/playbyplay/_/gameId/401628455"
driver.get(url)
time.sleep(5)  # Wait for JavaScript to load

# Get page source
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

# Target the play-by-play accordion section
play_section = soup.find('div', class_='AccordionPanel')
if not play_section:
    print("Accordion section not found. Check HTML structure or class name.")
    play_section = soup.find('div', class_=lambda c: c and 'Accordion' in c)
    if not play_section:
        print("No alternative play-by-play section found. Verify URL and HTML.")
        exit()

# Extract play list
plays_list = play_section.find('ul', class_='Playlist')
if not plays_list:
    print("Playlist not found. Check ul class or nesting.")
    exit()

data = []
for item in plays_list.find_all('li', class_='PlaylistItem'):
    description = item.find('span', class_='PlaylistItem_Description')
    if description:
        text = description.text.strip()
        parts = text.split(' - ')
        if len(parts) >= 2:
            situation = parts[0]
            play = parts[1]
            
            down_distance = situation.split(' at ')[0] if ' at ' in situation else 'N/A'
            yard_line = situation.split(' at ')[1].split('(#')[0] if ' at ' in situation else 'N/A'
            play_type = 'run' if 'run' in play.lower() else 'pass' if 'pass' in play.lower() else 'other'

            data.append({
                'down_distance': down_distance,
                'yard_line': yard_line,
                'play_type': play_type,
                'description': play
            })

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv('ohio_state_game_pbp.csv', index=False)
print("Data saved to ohio_state_game_pbp.csv")