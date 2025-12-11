import asyncio
from modules.requests_handler import check_mail
from modules.captcha_handler import CaptchaHandler

captcha_handler = CaptchaHandler(model_path="model.pth")
num_threads = 2  # chạy tối đa 2 mail cùng lúc

async def worker(mail, captcha_handler, sem):
    async with sem:                    # giới hạn số luồng
        await check_mail(mail, captcha_handle=captcha_handler)

async def main():
    sem = asyncio.Semaphore(num_threads)

    # đọc mail
    with open("html.txt", "r") as f:
        mails = [m.strip() for m in f.readlines()]

    # tạo toàn bộ task
    tasks = [worker(mail, captcha_handler, sem) for mail in mails]

    # chạy song song theo limit
    await asyncio.gather(*tasks)

# chạy chương trình
asyncio.run(main())
