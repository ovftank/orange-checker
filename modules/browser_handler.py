import asyncio
from playwright.async_api import async_playwright, Page,Browser
from playwright_stealth import Stealth
from .config import Config
config = Config()
from .captcha_handler import CaptchaHandler
class BrowserHandler:
    def __init__(self, captcha_handler: CaptchaHandler, numthread=1) -> None:
        self.playwright = None
        self.pages: list[Page] = []
        self.numthread:int = numthread
        self.browser : Browser|None = None
        self.stealth :Stealth |None = None
        self.captcha_handler = captcha_handler
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

    async def check(self):
        if len(self.pages) >0:
            tasks=[]
            for page in self.pages:
                await page.goto("https://login.orange.fr")
                try:
                    await page.wait_for_url(
                "**://captcha.orange.fr/**", wait_until="domcontentloaded",
                )
                    captcha_frame = await page.locator(".captcha_btn__1Pngd").all()
                    for captcha  in captcha_frame:
                        a =await captcha.screenshot()
                        self.captcha_handler.predict(a)

                    print(captcha_frame)


                except Exception as e:
                    print(e)














