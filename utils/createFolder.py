from playwright.sync_api import sync_playwright
import time
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

found = []
with sync_playwright() as p:
    wait_seconds = 20
    time_out_seconds = min(wait_seconds//4, 5)
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.set_default_navigation_timeout((wait_seconds + time_out_seconds) * 1000)
    page.set_default_timeout((wait_seconds + time_out_seconds) * 1000)
    page.goto("https://leetgpu.com/challenges", wait_until="domcontentloaded")

    time.sleep(wait_seconds)

    anchors = page.query_selector_all('a[href^="/challenges/"]')

    for a in anchors:
        href = a.get_attribute("href")
        if href:
            name = href.split("/")[-1]
            if name and name not in found:
                found.append(name)

    browser.close()

for idx, item in enumerate(found):
    folder_path = os.path.join(parent_dir, f"{idx+1:02d}-{item}")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

digital_folders = [
    name for name in os.listdir(parent_dir)
    if os.path.isdir(os.path.join(parent_dir, name))
    and name[0].isdigit() and name[1].isdigit()
    and name[2] == '-'
]

for item in digital_folders:
    folder_path = os.path.join(parent_dir, item, "Triton")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_path = os.path.join(folder_path, "native.py")
    if not os.path.exists(file_path):
        open(file_path, 'w')
