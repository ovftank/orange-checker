import asyncio
from modules import BrowserHandler
from modules.captcha_handler import CaptchaHandler

async def main():
    captcha_handler = CaptchaHandler(model_path="model.pth")
    num_threads = 2
    mails = []
    with open("html.txt","r") as f:
        mails= f.readlines()
    async with BrowserHandler(captcha_handler=captcha_handler, numthread=num_threads,mails=mails) as b:
        await b.check_all()
        await asyncio.to_thread(input)

if __name__ == "__main__":
    asyncio.run(main())
