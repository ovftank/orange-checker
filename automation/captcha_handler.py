import asyncio
import tempfile
import os
from typing import List, Dict, Optional

from automation.browser_setup import BrowserSetup
from models.image_processor import ImageProcessor
from config import Config

class CaptchaHandler:
    """Xử lý CAPTCHA trên trang web"""
    
    def __init__(self, browser_setup: BrowserSetup, image_processor: ImageProcessor):
        self.browser_setup = browser_setup
        self.image_processor = image_processor
    
    async def has_active_step(self) -> bool:
        """Kiểm tra có bước CAPTCHA đang active không"""
        if not self.browser_setup.page:
            return False

        try:
            active_steps = self.browser_setup.page.locator(".ob1-timeline-step.active")
            return await active_steps.count() > 0
        except Exception:
            return False
    
    async def get_captcha_requirement(self) -> Optional[str]:
        """Lấy yêu cầu của CAPTCHA"""
        if not self.browser_setup.page:
            return None

        try:
            # KHÔNG ĐƯỢC await locator
            timeline_element = self.browser_setup.page.locator(".ob1-timeline-step.active")
            await timeline_element.wait_for(state="attached", timeout=Config.CAPTCHA_TIMEOUT)

            title_element = timeline_element.locator(".ob1-timeline-title")
            if await title_element.count() > 0:
                content = await title_element.first.text_content()
                if content:
                    return content.strip()

            # fallback: đọc trực tiếp text trong step
            content = await timeline_element.text_content()
            if content:
                return content.strip()

        except Exception:
            pass

        return None
    
    async def get_captcha_images(self) -> List[Dict]:
        """Lấy các ảnh CAPTCHA"""
        if not self.browser_setup.page:
            return []

        buttons = await self.browser_setup.page.locator(".captcha_btn__1Pngd").all()
        image_data = []

        for idx, button in enumerate(buttons):
            try:
                image_data.append({"index": idx, "button": button})
            except Exception:
                continue

        return image_data
    
    async def screenshot_element(self, button) -> str:
        """Chụp ảnh element"""
        screenshot_bytes = await button.screenshot()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(screenshot_bytes)
            print("Saved file:", tmp.name)
            print("Exists:", os.path.exists(tmp.name))
            return tmp.name
    
    async def reload_captcha_images(self) -> bool:
        """Tải lại các ảnh CAPTCHA"""
        if not self.browser_setup.page:
            return False

        try:
            reload_button = self.browser_setup.page.locator(".ob1-link-icon").filter(
                has_text="Nouveau jeu d'images"
            )
            await reload_button.wait_for(state="visible", timeout=Config.CAPTCHA_TIMEOUT)
            await reload_button.click()
            print("Reloaded captcha images")
            return True
        except Exception as e:
            print(f"Could not find reload button: {e}")
            return False
    
    async def predict_and_cache_images(self, image_data: List[Dict]) -> None:
        """Dự đoán và cache kết quả các ảnh"""
        if not self.browser_setup.page:
            return

        items_to_process = [
            item for item in image_data 
            if item["index"] not in self.image_processor.image_cache
        ]
        if not items_to_process:
            return

        image_paths = {}
        temp_files = []

        try:
            for item in items_to_process:
                try:
                    image_path =  await self.screenshot_element(item["button"])
                    image_paths[item["index"]] = image_path
                    temp_files.append(image_path)
                except Exception as e:
                    print(f"Error screenshot image {item['index']}: {e}")

            # Xử lý ảnh
            await self.image_processor.process_images_async(image_paths)
            print("check4")

        finally:
            for tmp_file in temp_files:
                try:
                    os.unlink(tmp_file)
                except Exception:
                    pass
    
    async def solve_captcha_step(self, use_cache: bool = False) -> int:
        """Giải một bước CAPTCHA"""
        if not self.browser_setup.page:
            return False

        requirement = await self.get_captcha_requirement()
        if not requirement:
            print("No captcha requirement found")
            return False

        print(f"Captcha requirement: {requirement}")

        image_data = await self.get_captcha_images()
        if not image_data:
            print("No captcha images found")
            return False

        print(f"Found {len(image_data)} captcha images")

        if not use_cache:
            await self.predict_and_cache_images(image_data)

        clicked_count = 0

        for item in image_data:
            class_name = self.image_processor.image_cache.get(item["index"])
            if not class_name:
                continue

            if class_name.lower() in requirement.lower():
                button = item["button"]
                await button.click()
                clicked_count += 1
                print(f"Clicked image {item['index']}")

        print(f"Clicked {clicked_count} images")
        return clicked_count
    
    async def solve_captcha(self) -> bool:
        """Giải toàn bộ CAPTCHA"""
        if not self.browser_setup.page:
            return False

        self.image_processor.image_cache.clear()

        step_count = 0
        max_steps = Config.MAX_STEPS
        max_reloads = Config.MAX_RELOADS

        while await self.has_active_step() and step_count < max_steps:
            step_count += 1
            print(f"\n--- Solving step {step_count} ---")

            use_cache = step_count > 1
            reload_count = 0
            clicked_count = 0

            while reload_count < max_reloads:
                clicked_count = await self.solve_captcha_step(use_cache=use_cache)

                if clicked_count > 0:
                    break

                if reload_count < max_reloads - 1:
                    print("No matching images found, reloading...")
                    self.image_processor.image_cache.clear()
                    if await self.reload_captcha_images():
                        use_cache = False
                        reload_count += 1
                    else:
                        break
                else:
                    break

            if clicked_count == 0:
                print("Failed to solve step after reloads")
                return False

            if not await self.has_active_step():
                print("All steps completed")
                return True

        if step_count >= max_steps:
            print(f"Reached max steps ({max_steps})")
            return False

        return True
    
    async def click_continue_button(self) -> bool:
        """Click nút tiếp tục sau khi giải CAPTCHA"""
        if not self.browser_setup.page:
            return False

        try:
            continue_button = self.browser_setup.page.get_by_role("button", name="Continuer")
            await continue_button.wait_for(state="visible", timeout=Config.CAPTCHA_TIMEOUT)
            await continue_button.click()
            print("Clicked continue button")
            return True
        except Exception as e:
            print(f"Could not find continue button: {e}")
            return False
