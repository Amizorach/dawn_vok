import requests
from bs4 import BeautifulSoup
import os
import time

# --- Configuration ---
IMS_DATA_PORTAL_URL = "https://ims.gov.il/en/stations"
OSCAR_BASE_URL = "https://oscar.wmo.int/surface/index.html#/search/station/stationReportDetails/"
OUTPUT_DIR = "oscar_html_reports"
REQUEST_TIMEOUT = 30 # seconds
DELAY_BETWEEN_REQUESTS = 2 # seconds to be polite to servers

# --- Helper Functions ---
def fetch_html(url):
    """Fetches HTML content from a URL."""
    print(f"Fetching HTML from: {url}")
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        print(f"Successfully fetched.")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def save_content(filename, content):
    """Saves content to a file in the output directory."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved content to: {filepath}")
    except IOError as e:
        print(f"Error saving file {filepath}: {e}")

# --- Main Script Logic ---
print("--- Starting IMS Station & OSCAR Fetch Script ---")

# 1. Fetch the IMS Data Portal page
ims_html = fetch_html(IMS_DATA_PORTAL_URL)
if not ims_html:
    print("Failed to fetch IMS data portal page. Exiting.")
    exit()

# 2. Parse the IMS page to find stations and OSCAR IDs
print("\nParsing IMS page to find stations...")
soup = BeautifulSoup(ims_html, 'html.parser')
stations = {} # Dictionary to store {oscar_id: station_name}

# --- !!! CRITICAL PART: Finding the station data !!! ---
# This part depends HEAVILY on the IMS website's HTML structure.
# We need to find the <select> element or similar containing station options.
# Assumption: The <select> element has id='stations' and each <option>
# has a value attribute containing the OSCAR ID and the text is the station name.
# *** This might need adjustment based on actual inspection of the page source ***
station_select_element = soup.find('select', id='station_link') # Adjust selector if needed

if station_select_element:
    options = station_select_element.find_all('option')
    print(f"Found {len(options)} potential station options in dropdown.")
    for option in options:
        oscar_id = option.get('value')
        station_name = option.text.strip()
        # Basic validation: Check if value seems like an OSCAR ID (contains hyphens, etc.)
        # and if name is not empty. Adjust validation as needed.
        if oscar_id and '-' in oscar_id and station_name and station_name != "Type/select station":
             # Clean up OSCAR ID if it includes extra text (adjust logic as needed)
            # Example: if value="0-376-0-645 (Primary)" -> extract "0-376-0-645"
            if ' ' in oscar_id:
                oscar_id = oscar_id.split(' ')[0] # Simple split, might need refinement
            stations[oscar_id] = station_name
        # else:
            # print(f"Skipping option: Value='{oscar_id}', Name='{station_name}'")

else:
    print("Could not find the expected station select element on the IMS page.")
    print("You may need to inspect the page HTML source ('View Source' in browser)")
    print("and adjust the 'station_select_element' selector in the script.")
    print("Attempting to extract from the known Bet Dagan example link as a fallback...")
    # Fallback: Try finding the known Bet Dagan OSCAR ID link text
    link_element = soup.find('a', text='0-376-0-645') # Adjust if needed
    if link_element:
         print("Found Bet Dagan link, but cannot automatically get others this way.")
         # stations['0-376-0-645'] = 'Bet Dagan' # Manually add if needed for testing

if not stations:
     print("\nCould not automatically extract any station OSCAR IDs.")
     print("Please check the IMS page structure and update the script selector,")
     print("or consider manually creating a list of OSCAR IDs if needed.")
     print("Exiting.")
     exit()

print(f"\nExtracted {len(stations)} stations with potential OSCAR IDs.")
print("Stations found:", list(stations.values())) # Display names

# 3. Loop through stations and attempt to fetch OSCAR HTML
print(f"\n--- Attempting to fetch OSCAR HTML reports (saving to '{OUTPUT_DIR}' directory) ---")
for oscar_id, station_name in stations.items():
    print(f"\nProcessing Station: {station_name} (OSCAR ID: {oscar_id})")
    oscar_url = f"{OSCAR_BASE_URL}{oscar_id}"
    oscar_html = fetch_html(oscar_url)

    if oscar_html:
        # Sanitize station name for filename
        safe_station_name = "".join(c for c in station_name if c.isalnum() or c in (' ', '_')).rstrip()
        filename = f"OSCAR_{oscar_id}_{safe_station_name}.html"
        save_content(filename, oscar_html)
    else:
        print(f"Skipping save for {station_name} due to fetch error.")

    # Be polite to the server
    print(f"Waiting for {DELAY_BETWEEN_REQUESTS} seconds...")
    time.sleep(DELAY_BETWEEN_REQUESTS)

print("\n--- Script Finished ---")
print(f"Check the '{OUTPUT_DIR}' directory for downloaded HTML files.")
print("Remember: The HTML files might not contain the full report details due to Javascript rendering on the OSCAR site.")