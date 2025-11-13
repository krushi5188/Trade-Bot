from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
import time

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    return driver

def scrape_economic_calendar(data_dir, days_to_fetch=3):
    driver = setup_driver()
    base_url = "https://www.investing.com/economic-calendar/"
    all_events = []
    print("Starting economic calendar scrape with headless browser...")
    try:
        driver.get(base_url)
        try:
            cookie_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
            cookie_button.click()
            print("Accepted cookie consent.")
            time.sleep(2)
        except Exception as e:
            print(f"Could not find or click cookie button: {e}")
        for _ in range(min(days_to_fetch - 1, 2)):
             date_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "datePickerBtnForward")))
             date_button.click()
             time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', {'id': 'economicCalendarData'})
        if not table:
            print("Could not find events table even with headless browser.")
            with open("debug_selenium_page.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            return
        rows = table.find_all('tr', class_='js-event-item')
        print(f"Found {len(rows)} total events on the page.")
        for row in rows:
            event_date_str = row.get('data-event-datetime')
            if not event_date_str:
                continue
            importance_tag = row.find('td', class_='sentiment')
            if importance_tag and importance_tag.get('data-img_key') == 'bull3':
                event_dt = datetime.strptime(event_date_str, '%Y-%m-%d %H:%M:%S')
                event_name_tag = row.find('td', class_='event')
                event_name = event_name_tag.text.strip() if event_name_tag else "N/A"
                all_events.append({'timestamp_utc': event_dt, 'event_name': event_name, 'importance': 'High'})
                print(f"  -> High-Impact Event Found: {event_dt} - {event_name}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        with open("debug_selenium_error.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
    finally:
        driver.quit()
    if all_events:
        df = pd.DataFrame(all_events)
        df.set_index('timestamp_utc', inplace=True)
        df.sort_index(inplace=True)
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, "economic_calendar.parquet")
        df.to_parquet(filepath)
        print(f"\\nSuccessfully scraped {len(df)} high-impact events and saved to {filepath}")
    else:
        print("\\nNo high-impact events were found to save.")
