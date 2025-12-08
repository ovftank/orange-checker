import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any

from config import Config

def read_emails(file_path: str) -> List[str]:
    """Đọc danh sách email từ file"""
    try:
        with open(file_path, "r") as f:
            emails = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Read {len(emails)} emails from {file_path}")
        return emails
    except Exception as e:
        print(f"Error reading emails file: {e}")
        return []

async def run_with_context(context_id: int, emails: List[str], automation_class: Any) -> None:
    """Chạy automation với một context"""
    while emails:
        email = emails.pop(0)
        print(f"[CTX {context_id}] - Checking: {email}")
        
        try:
            automation = automation_class(email=email, headless=True)
            result = await automation.run()
            if result and result.get("success"):
                print(f"[CTX {context_id}] - Success: {email}")
                print(f"  - Title: {result.get('title')}")
                print(f"  - URL: {result.get('url')}")
                print(f"  - Captcha: {result.get('is_captcha')}")
                print(f"  - Mobile Connect: {result.get('is_mobile_connect')}")
            else:
                print(f"[CTX {context_id}] - Failed: {email}")
                
        except Exception as e:
            print(f"[CTX {context_id}] ERROR: {e}")
    
    print(f"[CTX {context_id}] Completed all emails.")

import asyncio

async def worker(worker_id: int, email_queue: asyncio.Queue, automation_class: Any):
    """Một worker chạy liên tục cho đến khi hết email trong queue."""
    while True:
        try:
            email = email_queue.get_nowait()
        except asyncio.QueueEmpty:
            break  # hết email → worker dừng

        try:
            await run_with_context(worker_id, [email], automation_class)
        except Exception as e:
            print(f"Worker {worker_id} error on {email}: {e}")

        email_queue.task_done()


async def main(emails: list[str], automation_class, browser_count=3):
    """Luôn giữ 3 context chạy song song, worker nào xong thì chạy email tiếp theo."""
    email_queue = asyncio.Queue()

    # Đưa tất cả email vào queue
    for email in emails:
        email_queue.put_nowait(email)

    # Tạo 3 worker chạy song song
    workers = [
        asyncio.create_task(worker(i, email_queue, automation_class))
        for i in range(browser_count)
    ]

    # Đợi xử lý xong toàn bộ email
    await email_queue.join()

    # Đợi tất cả worker kết thúc
    for w in workers:
        await w



