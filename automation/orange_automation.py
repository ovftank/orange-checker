from typing import Dict, Optional, Any
from automation.browser_setup import BrowserSetup
from automation.captcha_handler import CaptchaHandler
from automation.form_handler import FormHandler
from models.model_loader import ModelLoader
from models.image_processor import ImageProcessor
from config import Config

class OrangeAutomation:
    """Lớp chính để tự động hóa quy trình đăng nhập Orange"""
    
    def __init__(self, email: str = "", headless: bool = True):
        self.email = email
        self.headless = headless
        
        # Khởi tạo các thành phần
        self.browser_setup = BrowserSetup(headless=False)
        self.model_loader = ModelLoader()
        self.image_processor = ImageProcessor(self.model_loader)
        self.captcha_handler = CaptchaHandler(self.browser_setup, self.image_processor)
        self.form_handler = FormHandler(self.browser_setup)
    
    async def initialize(self):
        await self.browser_setup.setup_playwright()
       
        await self.browser_setup.create_context()
        
        await self.browser_setup.setup_page()
        
        await self.browser_setup.inject_stealth()
        
        self.model_loader.load_model()
       
    
    async def check_page(self) -> Dict[str, Any]:
        """Kiểm tra và xử lý trang"""
        if not self.browser_setup.page:
            raise RuntimeError("Page not initialized, call initialize() first")
    
        await self.browser_setup.page.goto(Config.ORANGE_LOGIN_URL, wait_until="load")
    
        # Xác định captcha
        is_captcha = self._detect_captcha()
        current_url = self.browser_setup.page.url
        
        # Giải captcha nếu có
        if is_captcha:
            print(f"Captcha detected: {current_url}")
            await self.browser_setup.page.wait_for_selector("#image-grid", timeout=10000)
            solved = await self.captcha_handler.solve_captcha()
    
            if solved:
                if await self.captcha_handler.click_continue_button():
                    try:
                        await self.browser_setup.page.wait_for_selector("#login", timeout=20000)
                        print("Login page loaded")
                        await self.form_handler.inject_accept_button_script()
                    except Exception as e:
                        print(f"Wait for login page timeout: {e}")
                    
                    # Gọi API email
                    api_response = await self.form_handler.fill_email_and_submit(self.email)
                    
                    if not api_response:
                        print("No API response -> exiting context early")
                        return {
                            "title": await self.browser_setup.page.title(),
                            "url": current_url,
                            "is_captcha": is_captcha,
                            "is_mobile_connect": False,
                            "success": False
                        }
                    
                    # Kiểm tra mobile connect
                    is_mobile_connect = await self.form_handler.check_mobile_connect(api_response)
                    
                    if is_mobile_connect:
                        await self.form_handler.save_mobile_email(self.email)
                else:
                    return {
                        "title": await self.browser_setup.page.title(),
                        "url": current_url,
                        "is_captcha": is_captcha,
                        "is_mobile_connect": False,
                        "success": False
                    }
        else:
            # Nếu không có captcha, điền email luôn
            api_response = await self.form_handler.fill_email_and_submit(self.email)
            
            if not api_response:
                print("No API response")
                return {
                    "title": await self.browser_setup.page.title(),
                    "url": current_url,
                    "is_captcha": is_captcha,
                    "is_mobile_connect": False,
                    "success": False
                }
            
            # Kiểm tra mobile connect
            is_mobile_connect = await self.form_handler.check_mobile_connect(api_response)
            
            if is_mobile_connect:
                await self.form_handler.save_mobile_email(self.email)
    
        title = await self.browser_setup.page.title()
    
        return {
            "title": title,
            "url": current_url,
            "is_captcha": is_captcha,
            "is_mobile_connect": is_mobile_connect,
            "success": True
        }
    
    def _detect_captcha(self) -> bool:
        """Phát hiện có captcha không"""
        try:
            # Cách 1: Kiểm tra URL
            if self.browser_setup.page:
                # Sử dụng evaluate để kiểm tra URL
                current_url = self.browser_setup.page.url
                parsed = self.image_processor.is_captcha_url(current_url)
                if parsed:
                    return True
            
            # Cách 2: Kiểm tra chuyển hướng đến trang captcha
            return self.browser_setup.page.evaluate(
                "() => location.hostname === 'captcha.orange.fr'"
            )
        except:
            return False
    
    async def run(self) -> Optional[Dict[str, Any]]:
        """Chạy quy trình tự động hóa"""
        try:
            await self.initialize()
            print("check0")
            result = await self.check_page()
            
            return result
        except Exception as e:
            print(f"Unexpected error: {e}")
            await self._handle_error(e)
            return None
        finally:
            await self.browser_setup.cleanup()
    
    async def _handle_error(self, error):
        """Xử lý lỗi"""
        print(f"Error detected: {error}")
        print("Resetting context...")
        
        # Đóng page và context cũ
        try:
            if self.browser_setup.page:
                await self.browser_setup.page.close()
                print("Closed old page")
        except:
            pass
        
        try:
            if self.browser_setup.context:
                await self.browser_setup.context.close()
                print("Closed old context")
        except:
            pass
        
        # Tạo context mới
        await self.browser_setup.create_context()
        await self.browser_setup.setup_page()
        await self.browser_setup.inject_stealth()
        
        print("Created new context")
