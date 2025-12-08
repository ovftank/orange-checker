import asyncio
from typing import Optional, Dict, Any

from playwright.async_api import (
    Browser, 
    BrowserContext, 
    Page, 
    Playwright,
    async_playwright,
    TimeoutError as PlaywrightTimeoutError
)
from playwright_stealth import Stealth
from python_ghost_cursor.playwright_async import install_mouse_helper
from config import Config

class BrowserSetup:
    """Quản lý việc thiết lập và cấu hình trình duyệt"""
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright: Optional[Playwright] = None
    
    async def setup_playwright(self):
        """Khởi tạo Playwright"""
        self.playwright_cm = async_playwright()         # context manager
        self.playwright = await self.playwright_cm.start() 
    
    async def create_context(self):
        """Tạo context mới"""
        if not self.playwright:
            raise RuntimeError("Playwright not initialized")
        
        browser = await self.playwright.chromium.launch(
            channel="chromium", 
            headless=self.headless,
            proxy =  {
                "server": 'na.lunaproxy.com:32233',
                "username": 'user-anan11211_J3mhI-region-vn',
                "password": 'Anhan3122'
            }
            
        
        )
        
        self.context = await browser.new_context()
        
        # Cài đặt thời gian chờ mặc định
        self.context.set_default_timeout(Config.DEFAULT_TIMEOUT)
        self.context.set_default_navigation_timeout(Config.DEFAULT_NAVIGATION_TIMEOUT)
        
        return browser
    
    async def setup_page(self):
        """Tạo và cấu hình page"""
        if not self.context:
            raise RuntimeError("Context not initialized")
        
        self.page = await self.context.new_page()
        
        # Cài đặt thời gian chờ mặc định cho page
        self.page.set_default_timeout(Config.DEFAULT_TIMEOUT)
        self.page.set_default_navigation_timeout(Config.DEFAULT_NAVIGATION_TIMEOUT)
        
        # Cài đặt mouse helper
        await install_mouse_helper(self.page)
        
        # Thêm script cookie handling
        await self._setup_cookie_handler()
    
    async def _setup_cookie_handler(self):
        """Cài đặt xử lý cookie banner"""
        cookie_click_script = """
        (() => {
            const clickCookieBanner = () => {
                const btn = document.querySelector('.didomi-continue-without-agreeing');
                if (btn && btn.offsetParent !== null) {
                    btn.click();
                    return true;
                }
                return false;
            };

            const startObserver = () => {
                if (!document.body) {
                    return false;
                }

                if (clickCookieBanner()) return true;

                const observer = new MutationObserver(() => {
                    if (clickCookieBanner()) {
                        observer.disconnect();
                    }
                });

                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });

                let attempts = 0;
                const interval = setInterval(() => {
                    attempts++;
                    if (clickCookieBanner() || attempts >= 240) {
                        clearInterval(interval);
                        observer.disconnect();
                    }
                }, 500);

                return true;
            };

            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', startObserver);
            } else {
                let retries = 0;
                const checkInterval = setInterval(() => {
                    if (startObserver() || retries >= 10) {
                        clearInterval(checkInterval);
                    }
                    retries++;
                }, 100);
            }
        })();
        """
        await self.context.add_init_script(script=cookie_click_script)
    
    async def inject_stealth(self):
        """Inject stealth để tránh phát hiện bot"""
        stealth = Stealth()
        await stealth.apply_stealth_async(self.context)
    
    async def cleanup(self):
        """Dọn dẹp tài nguyên"""
        try:
            if self.page:
                await self.page.close()
        except:
            pass
        
        try:
            if self.context:
                await self.context.close()
        except:
            pass
        
        try:
            if self.playwright:
                await self.playwright.stop()
        except:
            pass
