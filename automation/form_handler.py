import asyncio
from typing import Optional, Dict, Any

from automation.browser_setup import BrowserSetup
from config import Config

class FormHandler:
    """Xử lý các form trên trang web"""
    
    def __init__(self, browser_setup: BrowserSetup):
        self.browser_setup = browser_setup
    
    async def inject_accept_button_script(self) -> None:
        """Chèn script để click nút chấp nhận cookie"""
        if not self.browser_setup.page:
            return

        accept_script = """
        (() => {
            const clickAcceptButton = () => {
                const btn = document.querySelector('#didomi-notice-agree-button');
                if (btn && btn.offsetParent !== null) {
                    btn.click();
                    return true;
                }
                return false;
            };

            setTimeout(() => {
                if (clickAcceptButton()) {
                    console.log('Clicked accept button');
                } else {
                    const observer = new MutationObserver(() => {
                        if (clickAcceptButton()) {
                            observer.disconnect();
                        }
                    });

                    if (document.body) {
                        observer.observe(document.body, {
                            childList: true,
                            subtree: true
                        });

                        let attempts = 0;
                        const interval = setInterval(() => {
                            attempts++;
                            if (clickAcceptButton() || attempts >= 120) {
                                clearInterval(interval);
                                observer.disconnect();
                            }
                        }, 500);
                    }
                }
            }, 60000);
        })();
        """
        await self.browser_setup.page.evaluate(accept_script)
    
    async def fill_email_and_submit(self, email: str) -> Optional[Dict[str, Any]]:
        """Điền email và submit form"""
        if not self.browser_setup.page:
            print("Page not initialized")
            return None

        if not email:
            print("Email not provided")
            return None

        try:
            # Đợi element xuất hiện
            await self.browser_setup.page.wait_for_selector("#login", timeout=10000)
            email_input = self.browser_setup.page.locator("#login")
            await email_input.wait_for(state="visible", timeout=10000)
            await email_input.fill(email)
            print(f"Filled email: {email}")

            submit_button = self.browser_setup.page.locator("#btnSubmit")
            await submit_button.wait_for(state="visible", timeout=5000)

            # Lắng nghe API response
            response_data = await self._capture_api_response(submit_button)
            return response_data

        except Exception as e:
            print(f"Error filling email form: {e}")
            # Fallback click
            try:
                await submit_button.click()
                print("Clicked submit button (fallback)")
                return None
            except:
                return None
    
    async def _capture_api_response(self, submit_button) -> Optional[Dict[str, Any]]:
        """Capture API response khi submit"""
        try:
            async with self.browser_setup.page.expect_response(
                lambda response: (
                    "orange.fr" in response.url
                    and ("idme" in response.url.lower()
                         or "api" in response.url.lower())
                ),
                timeout=15000,
            ) as response_info:

                await submit_button.click()
                print("Clicked submit button")

            response = await response_info.value
            print(f"API response URL: {response.url}")
            print(f"API response status: {response.status}")

            try:
                response_data = await response.json()
                print(f"API response data: {response_data}")
                return response_data
            except Exception:
                response_text = await response.text()
                print(f"API response text: {response_text}")
                return {
                    "text": response_text,
                    "status": response.status,
                }

        except Exception as e:
            print(f"No API response captured: {e}")
            await submit_button.click()
            print("Clicked submit button (fallback)")
            return None
    
    async def check_mobile_connect(self, api_response: Dict[str, Any]) -> bool:
        """Kiểm tra có phải là mobile connect không"""
        try:
            return (
                api_response.get("data", {})
                .get("mobileConnectScreen", {})
                .get("displayedAccount", {})
                .get("isMobileConnect") or False
            )
        except Exception as e:
            print("Error checking mobile connect:", e)
            return False
    
    async def save_mobile_email(self, email: str) -> None:
        """Lưu email nếu là mobile connect"""
        with open(Config.MOBILE_EMAIL_FILE, "a", encoding="utf-8") as f:
            f.write(email.strip() + "\n")
        print(f"Saved mobile email: {email}")
