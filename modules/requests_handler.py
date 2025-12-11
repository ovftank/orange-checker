import requests
import urllib3
import re
import json
import urllib.parse
from bs4 import BeautifulSoup
from modules import captcha_handler
import tempfile
import asyncio
lock = asyncio.Lock()
# Tắt cảnh báo của urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

async def check_mail(mail="barrieredom@orange.fr",captcha_handle=None):

# Tạo session để giữ cookie
    session = requests.Session()

    # Cấu hình headers để mô phỏng trình duyệt
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    })

    # URL target
    url = "https://login.orange.fr/"

    print(f"Đang gửi request đến: {url}")

    try:
        response = session.get(url, verify=False)
        # Nếu status code là 200, tìm URL iz.orange.fr
        if response.status_code == 200:
            # Pattern để tìm URLs chứa iz.orange.fr
            pattern = r'https?://iz\.orange\.fr/[^\s<>"\'{}|\\^`\[\]]+'

            match = re.search(pattern, response.text, re.IGNORECASE)

            if match:
                url = match.group(1) if match.lastindex else match.group(0)
                session.get(url, verify=False, timeout=30)

                # Thêm header x-requested-with vào session
                session.headers.update({'x-requested-with': 'idme'})

                # POST đến API access
                api_url = "https://login.orange.fr/api/access"
                response = session.post(api_url, json={}, verify=False, timeout=30)
                print(f"POST to {api_url} - Status: {response.status_code}")

                # Parse response JSON
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        if response_data.get("nextStep") == "redirect" and "data" in response_data:
                            location = response_data["data"]["location"]
                            print(f"URL: {location}")

                            # Lấy csid từ URL

                            parsed_url = urllib.parse.urlparse(location)
                            query_params = urllib.parse.parse_qs(parsed_url.query)
                            csid = query_params.get('csid', [None])[0]

                            if csid:
                                print(f"CSID: {csid}")

                            # Truy cập đến location URL
                            response = session.get(location, verify=False, timeout=30, allow_redirects=True)
                            print(f"GET to {location} - Status: {response.status_code}")

                            # Parse HTML với BeautifulSoup
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.text, 'html.parser')

                                # Tìm script tag với id="__NEXT_DATA__"
                                next_data_script = soup.find('script', id='__NEXT_DATA__', type='application/json')

                                if next_data_script:
                                    # Lấy content từ script tag
                                    json_content = next_data_script.string

                                    # Parse JSON
                                    try:
                                        if json_content is not None:
                                            data = json.loads(json_content)
                                            print("\n=== JSON Data từ __NEXT_DATA__ ===")

                                            # Lấy các thông tin quan trọng
                                            page_props = data.get('props', {}).get('pageProps', {})

                                            # Lấy csid từ query parameters
                                            query = data.get('query', {})
                                            csid = query.get('csid')
                                            if csid:
                                                print(f"CSID: {csid}")

                                            # Lấy captcha data
                                            indications = page_props.get('indications', []) #class_names
                                            if indications:
                                                print(f"Captcha instructions: {indications}")

                                            # Lấy image URLs
                                            rows = page_props.get('rows', [])
                                            image_urls = []
                                            for row in rows:
                                                for item in row:
                                                    image_urls.append({
                                                        'value': item.get('value'),
                                                        'url': item.get('data')
                                                    })

                                            print(f"\nTìm thấy {len(image_urls)} images:")
                                            async def download_one(img):
                                                response = await asyncio.to_thread(requests.get, img['url'])
                                                img_data = response.content
                                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                                    tmp.write(img_data)
                                                    return tmp.name
                                            tasks = [download_one(img) for img in image_urls]
                                            img_paths = await asyncio.gather(*tasks)

                                            captcha_handle = captcha_handle
                                            predicts={}
                                            async def solve_one(idx, path):
    # load ảnh
                                                with open(path, "rb") as f:
                                                    img_bytes = f.read()

                                                # chạy predict trong thread -> non-blocking
                                                predict = await asyncio.to_thread(captcha_handle.predict, img_bytes) # type: ignore

                                                # vì nhiều task ghi vào dict -> cần lock
                                                async with lock:
                                                    predicts[predict] = idx + 1

                                            solve_captcha = [solve_one(idx, path) for idx, path in enumerate(img_paths)]
                                            await asyncio.gather(*solve_captcha)

                                            ans =[predicts.get(i) for i  in indications ]
                                            payload = {
                                                "csid": csid,
                                                "captchaService": "idme",
                                                "value": ans
                                            }
                                            print(ans)
                                            bypass =  session.post("https://captcha.orange.fr/api/verify",json=payload)
                                            redirect_url = bypass.json()["redirect"]

                                            res= session.get(redirect_url, allow_redirects=True)
                                            soup_res =  BeautifulSoup(res.text,'html.parser')
                                            next_url =soup_res.find('script',type="text/javascript")
                                            if next_url is not None:
                                                next_src = next_url.get('src')
                                                print(next_src)
                                                if next_src is not None:
                                                    check = session.get(str(next_src))
                                                    print(check.status_code)
                                            session.post("https://login.orange.fr/api/access", json={"csid":csid,"appName":"idme"}, verify=False, timeout=30)
                                            login =  session.post("https://login.orange.fr/api/login",json={"login":mail,"loginOrigin": "input"})
                                            if "mobileConnect" in login.text:
                                                print("ok")
                                                return True
                                            else:
                                                return False
                                    except json.JSONDecodeError as e:
                                        print(f"Lỗi khi parse JSON: {e}")
                                else:
                                    print("Không tìm thấy script tag với id='__NEXT_DATA__'")
                    except json.JSONDecodeError:
                        print("Response không phải là JSON hợp lệ")

    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi kết nối: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")




