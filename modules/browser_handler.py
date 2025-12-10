import asyncio
from .captcha_handler import CaptchaHandler
from playwright.async_api import async_playwright, Page,Browser
from playwright_stealth import Stealth
from .config import Config
config = Config()
class BrowserHandler:
    def __init__(self, captcha_handler: CaptchaHandler, numthread=2,mails=[]) -> None:
        self.playwright = None
        self.pages: list[Page] = []
        self.numthread:int = numthread
        self.browser : Browser|None = None
        self.stealth :Stealth |None = None
        self.captcha_handler = captcha_handler
        self.mails = mails
    async def setup(self):
        if self.browser is not None and self.stealth is not None:
            ctx =  await self.browser.new_context()
            await self.stealth.apply_stealth_async(ctx)
            await ctx.add_init_script(script=str(config.cookie_click_script))
            ctx.set_default_timeout(timeout=30000)
            page =await ctx.new_page()
            self.pages.append(page)

    async def __aenter__(self):
        self.stealth =Stealth()
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            channel="chromium", headless=False
        )
        task = []

        for _ in range(self.numthread):
            task.append(self.setup())
        await asyncio.gather(*task)
        return self

    async def __aexit__(self, _, aa, _bb):
        if self.playwright is not None:
            self.playwright = None


    async def process_single_mail(self, page: Page, mail: str):
        """Xử lý một email - kiểm tra có captcha hay không"""
        try:
            print(f"Processing: {mail.strip()}")
            await page.goto("https://login.orange.fr")

            # Kiểm tra có phải trang captcha không
            try:
                await page.wait_for_url(
                    "**://captcha.orange.fr/**", wait_until="domcontentloaded",
                )

                # Có captcha - giải captcha
                print(f"Phát hiện captcha cho {mail.strip()}")
                image_index = {}
                captcha_frame = await page.locator(".captcha_btn__1Pngd").all()

                for captcha in captcha_frame:
                    a = await captcha.screenshot()
                    image_name = self.captcha_handler.predict(a)
                    index = await captcha.get_attribute("aria-label")
                    image_index[image_name] = index

                for _ in range(6):
                    requirement = await page.locator("li.ob1-timeline-step.active .ob1-timeline-title").text_content()
                    locate_click = image_index.get(requirement.strip().lower()) # type: ignore
                    if locate_click:
                        await page.locator(f'[aria-label="{locate_click}"]').click()

                await page.get_by_role("button", name="Continuer").click()
                print(f"Đã giải captcha xong cho {mail.strip()}")

            except Exception:
                # Không có captcha - đi thẳng đến trang login
                print(f"Không có captcha cho {mail.strip()} - đi thẳng đến login")
                # Try to go directly to login or wait for login form
                try:
                    await page.wait_for_selector("#login", timeout=5000)
                except:
                    # Nếu không thấy #login, có thể cần click hoặc navigate
                    await page.wait_for_load_state("networkidle")

            # Fill mail và submit (dù có captcha hay không)
            await page.locator("#login").fill(mail.strip())

            async with page.expect_response(
                lambda response: "orange.fr" in response.url
                and (
                    "idme" in response.url.lower() or "api" in response.url.lower()
                ),
                timeout=15000,
            ) as response_info:
                await page.locator("#btnSubmit").click()

            api_response = await response_info.value
            api_text = await api_response.text()

            if "mobileConnect" in api_text:
                print(f"✓ {mail.strip()} - SUCCESS")
                return True
            else:
                print(f"✗ {mail.strip()} - FAILED")
                return False

        except Exception as e:
            print(f"ERROR {mail.strip()} - {str(e)}")
            return False

    async def process_mail_queue(self, page: Page, mail_queue: list, page_id: int):
        """Worker xử lý hàng đợi email liên tục"""
        processed_count = 0
        success_count = 0

        print(f"Worker {page_id}: Bắt đầu xử lý...")

        while mail_queue:
            # Lấy mail đầu tiên từ hàng đợi
            mail = mail_queue.pop(0)
            print(f"Worker {page_id}: Lấy mail {mail.strip()} (còn lại: {len(mail_queue)})")

            # Xử lý mail này
            result = await self.process_single_mail(page, mail)
            processed_count += 1
            if result:
                success_count += 1

        print(f"Worker {page_id}: Hoàn thành! Xử lý {processed_count} mail, thành công {success_count}")
        return {
            "processed": processed_count,
            "success": success_count,
            "failed": processed_count - success_count
        }

    async def check_all(self):
        """Xử lý tất cả mail - mỗi page lấy mail liên tục cho đến khi hết"""
        if not self.pages or not self.mails:
            print("Không có trang hoặc email nào để xử lý")
            return

        total_mails = len(self.mails)
        total_pages = len(self.pages)
        print(f"Bắt đầu xử lý {total_mails} emails với {total_pages} trình duyệt...")
        print("Mỗi trình duyệt sẽ xử lý email liên tục, tự động detect có captcha hay không\n")

        # Tạo hàng đợi mail chia sẻ
        mail_queue = self.mails.copy()

        # Mỗi page chạy một worker process
        tasks = []
        for i, page in enumerate(self.pages):
            page_id = i + 1
            task = self.process_mail_queue(page, mail_queue, page_id)
            tasks.append(task)

        # Chạy tất cả workers song song
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Thống kê kết quả
        total_processed = 0
        total_success = 0
        total_failed = 0

        print(f"\n=== KẾT QUẢ ===")
        for i, result in enumerate(results):
            if isinstance(result, dict):
                print(f"Trình duyệt {i+1}: xử lý {result['processed']} mail, thành công {result['success']}, thất bại {result['failed']}")
                total_processed += result['processed']
                total_success += result['success']
                total_failed += result['failed']
            else:
                print(f"Trình duyệt {i+1}: Lỗi - {str(result)}")

        print(f"\nTổng cộng:")
        print(f"Đã xử lý: {total_processed} emails")
        print(f"Thành công: {total_success}")
        print(f"Thất bại: {total_failed}")

        return results















