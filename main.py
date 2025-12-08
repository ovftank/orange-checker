import asyncio
from utils.helpers import read_emails, main
from automation.orange_automation import OrangeAutomation

def main_entry():
    """Điểm entry của chương trình"""
    # Đọc danh sách email
    emails = read_emails("html.txt")
    if not emails:
        print("No emails found or error reading file")
        return
    
    print(f"Starting automation with {len(emails)} emails...")
    
    # Chạy automation
    asyncio.run(main(emails, OrangeAutomation, browser_count=3))

if __name__ == "__main__":
    main_entry()
