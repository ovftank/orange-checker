import base64
import json
import re
import urllib.parse

import requests
import urllib3

from modules.label import label

urllib3.disable_warnings()
count = 0


def image_url_to_base64(url):
    global count
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image_bytes = response.content
        data = base64.b64encode(image_bytes).decode("utf-8")
        return data
    except Exception:
        return None


def remove_duplicate(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
    unique_lines = {line.strip() for line in lines if line.strip()}
    with open(filename, "w", encoding="utf-8") as file:
        file.write("\n".join(unique_lines) + "\n")


def extract_data(data_json):
    all_urls = []

    if "rows" not in data_json:
        return

    for row in data_json["rows"]:
        for item in row:
            if "data" in item and item["data"]:
                all_urls.append(item["data"])

    for url in all_urls:
        try:
            file_name = f"{url.split('images/')[1]}.jpeg"
            base64_str = image_url_to_base64(url)
            if base64_str:
                label(base64_str, file_name)
        except Exception as e:
            print(e)
            continue


def get_captcha_url():
    session = requests.session()
    session.verify = False
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    )

    LOGIN_URL = "https://login.orange.fr"
    API_ACCESS_URL = "https://login.orange.fr/api/access"
    IZ_ORANGE_REGEX = (
        r'<script[^>]*?src\s*=\s*["\']([^"\']*iz\.orange\.fr[^"\']*?)["\']'
    )
    CAPTCHA_VALUE = [0, 0, 0, 0, 0, 0]

    login_html = session.get(LOGIN_URL).text
    iz_url = re.findall(IZ_ORANGE_REGEX, login_html, re.IGNORECASE)[0]
    session.get(iz_url)

    response = session.post(
        API_ACCESS_URL,
        data=json.dumps({}),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-requested-with": "idme",
        },
    )
    query = urllib.parse.urlparse(response.json()["data"]["location"]).query
    csid = urllib.parse.parse_qs(query).get("csid", [None])[0]
    if not csid:
        return
    url = "https://captcha.orange.fr/api/verify"
    data = session.post(
        url=url,
        data={
            "csid": csid,
            "captchaService": "idme",
            "value": CAPTCHA_VALUE,
        },
    )
    if data.status_code != 200:
        return
    data_json = data.json()
    extract_data(data_json)


try:
    while True:
        try:
            get_captcha_url()
        except Exception:
            pass
except KeyboardInterrupt:
    print("done!")
