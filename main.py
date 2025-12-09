import asyncio
from modules import BrowserHandler
from modules.captcha_handler import CaptchaHandler

async def main():
    # Create CaptchaHandler instance once
    captcha_handler = CaptchaHandler(model_path="model.pth")

    # Pass the handler to BrowserHandler
    async with BrowserHandler(captcha_handler=captcha_handler) as b:
        await b.check()
        await asyncio.to_thread(input)


asyncio.run(main())
