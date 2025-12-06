import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
    
)
# from playwright_stealth import Stealth
from python_ghost_cursor.playwright_async import install_mouse_helper
from torch import Tensor, nn
from torchvision import models, transforms
from multiprocessing import Pool, cpu_count
from playwright_stealth import Stealth
import asyncio



class OrangeChecker:
    def __init__(self, headless: bool = False, email: str = "",context=None) -> None:
        self.headless = headless
        self.model_path = "model.pth"
        self.email = email
        self.context: BrowserContext | None = context
        self.page: Page | None = None
        self.model = None
        self.class_names = None
        self.transform_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_cache: dict[int, str] = {}
        self._model_lock = Lock()
        self.response_data =None


    async def setup_context(self) -> None:
        

        self.page =  await self.context.new_page()

        await install_mouse_helper(self.page)

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

        self.context.set_default_timeout(60000)
        self.context.set_default_navigation_timeout(60000)

        self.page.set_default_timeout(60000)
        self.page.set_default_navigation_timeout(60000)

    def _is_captcha_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc == "captcha.orange.fr" and parsed.path == "/":
            query_params = parse_qs(parsed.query)
            if "csid" in query_params and "captchaService" in query_params:
                return True
        return False

    def _load_model(self) -> None:
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, checkpoint["num_classes"])
        )
        setattr(model, "fc", classifier)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.class_names = checkpoint["class_names"]
        self.transform_config = checkpoint.get("transform", {})

    def _preprocess_image(self, image_path: str) -> Tensor:
        if not self.transform_config:
            raise RuntimeError("model not loaded")

        mean = self.transform_config.get("mean", [0.485, 0.456, 0.406])
        std = self.transform_config.get("std", [0.229, 0.224, 0.225])
        input_size = self.transform_config.get("input_size", (75, 75))

        transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img)
        assert isinstance(img_tensor, Tensor), "transform should return Tensor"
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def _predict(self, image_tensor: Tensor) -> tuple[str, float]:
        if not self.model or not self.class_names:
            raise RuntimeError("model not loaded")

        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)

        class_name = self.class_names[top_idx.item()]
        confidence = top_prob.item()
        return class_name, confidence

    async def _get_captcha_requirement(self) -> str | None:
        if not self.page:
            return None

        try:
            timeline_element = await self.page.locator(".ob1-timeline-step.active")
            await timeline_element.wait_for(state="attached", timeout=5000)
            title_element = await timeline_element.locator(".ob1-timeline-title")
            if title_element.count() > 0:
                content = title_element.first.text_content()
                if content:
                    return content.strip()
            else:
                content = timeline_element.text_content()
                if content:
                    return content.strip()
        except Exception:
            pass
        return None

    async def _has_active_step(self) -> bool:
        if not self.page:
            return False

        try:
            active_steps = self.page.locator(".ob1-timeline-step.active")
            return await active_steps.count() > 0
        except Exception:
            return False

    async def _get_captcha_images(self) -> list[dict]:
        if not self.page:
            return []

        buttons = await self.page.locator(".captcha_btn__1Pngd").all()
        image_data = []

        for idx, button in enumerate(buttons):
            try:
                image_data.append({"index": idx, "button": button})
            except Exception:
                continue

        return image_data

    def _screenshot_element(self, button_locator) -> str:
        screenshot_bytes = button_locator.screenshot()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(screenshot_bytes)
            return tmp.name

    def _process_single_image(
        self, image_path: str, index: int
    ) -> tuple[int, str, float] | None:
        if index in self.image_cache:
            return None

        try:
            image_tensor = self._preprocess_image(image_path)

            with self._model_lock:
                class_name, confidence = self._predict(image_tensor)

            return (index, class_name, confidence)
        except Exception as e:
            print(f"error processing image {index}: {e}")
            return None

    async def _predict_and_cache_images(self, image_data: list[dict]) -> None:
        if not self.page:
            return

        await self._load_model()

        items_to_process = [
            item for item in image_data if item["index"] not in self.image_cache
        ]

        if not items_to_process:
            return

        image_paths = {}
        temp_files = []

        try:
            for item in items_to_process:
                try:
                    image_path = await self._screenshot_element(item["button"])
                    image_paths[item["index"]] = image_path
                    temp_files.append(image_path)
                except Exception as e:
                    print(f"error screenshot image {item['index']}: {e}")

            with ThreadPoolExecutor(max_workers=9) as executor:
                futures = {
                    executor.submit(
                        self._process_single_image, image_paths[index], index
                    ): index
                    for index in image_paths.keys()
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        index, class_name, confidence = result
                        self.image_cache[index] = class_name
                        print(
                            f"image {index}: {class_name} (confidence: {confidence:.2f})"
                        )

        finally:
            for tmp_file in temp_files:
                try:
                    os.unlink(tmp_file)
                except Exception:
                    pass

    async def _solve_captcha_step(self, use_cache: bool = False) -> int:
        if not self.page:
            return False

        requirement = await self._get_captcha_requirement()
        if not requirement:
            print("không tìm thấy yêu cầu captcha")
            return False

        print(f"captcha requirement: {requirement}")

        image_data = await self._get_captcha_images()
        if not image_data:
            print("không tìm thấy images captcha")
            return False

        print(f"found {len(image_data)} captcha images")

        if not use_cache:
            self._predict_and_cache_images(image_data)

        clicked_count = 0

        for item in image_data:
            class_name = self.image_cache.get(item["index"])
            if not class_name:
                continue

            if class_name.lower() in requirement.lower():
                button = item["button"]
                button.click()
                clicked_count += 1
                print(f"clicked image {item['index']}")

        print(f"clicked {clicked_count} images")

        return clicked_count

    def _reload_captcha_images(self) -> bool:
        if not self.page:
            return False

        try:
            reload_button = self.page.locator(".ob1-link-icon").filter(
                has_text="Nouveau jeu d'images"
            )
            reload_button.wait_for(state="visible", timeout=5000)
            reload_button.click()
            print("reloaded captcha images")
            return True
        except Exception as e:
            print(f"không tìm thấy reload button: {e}")
            return False

    async def _solve_captcha(self) -> bool:
        if not self.page:
            return False

        self.image_cache.clear()

        step_count = 0
        max_steps = 10
        max_reloads = 3

        while self._has_active_step() and step_count < max_steps:
            step_count += 1
            print(f"\n--- solving step {step_count} ---")

            use_cache = step_count > 1
            reload_count = 0
            clicked_count = 0

            while reload_count < max_reloads:
                clicked_count = await self._solve_captcha_step(use_cache=use_cache)

                if clicked_count > 0:
                    break

                if reload_count < max_reloads - 1:
                    print("no matching images found, reloading...")
                    self.image_cache.clear()
                    if self._reload_captcha_images():
                        use_cache = False
                        reload_count += 1
                    else:
                        break
                else:
                    break

            if clicked_count == 0:
                print("failed to solve step after reloads")
                return False

            if not self._has_active_step():
                print("all steps completed")
                return True

        if step_count >= max_steps:
            print(f"reached max steps ({max_steps})")
            return False

        return True

    async def _click_continue_button(self) -> bool:
        if not self.page:
            return False

        try:
            continue_button = self.page.get_by_role("button", name="Continuer")
            await continue_button.wait_for(state="visible", timeout=5000)
            await continue_button.click()
            print("clicked continue button")
            return True
        except Exception as e:
            print(f"không tìm thấy continue button: {e}")
            return False

    def _inject_accept_button_script(self) -> None:
        if not self.page:
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
                    console.log('clicked accept button');
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
        self.page.evaluate(accept_script)

    async def _fill_email_and_submit(self) -> dict | None:
        if not self.page:
            print("page not initialized")
            return None

        if not self.email:
            print("email not provided")
            return None

        try:
            await self.page.wait_for_selector("#login", timeout=10000)
            email_input = self.page.locator("#login")
            await email_input.wait_for(state="visible", timeout=10000)
            await email_input.fill(self.email)
            print(f"filled email: {self.email}")

            submit_button = self.page.locator("#btnSubmit")
            await submit_button.wait_for(state="visible", timeout=5000)

            try:
                with self.page.expect_response(
                    lambda response: "orange.fr" in response.url
                    and (
                        "idme" in response.url.lower() or "api" in response.url.lower()
                    ),
                    timeout=15000,
                ) as response_info:
                    submit_button.click()
                    print("clicked submit button")

                response = response_info.value
                print(f"api response url: {response.url}")
                print(f"api response status: {response.status}")

                try:
                    response_data = response.json()
                    print(f"api response data: {response_data}")
                    return response_data
                except Exception:
                    response_text = response.text()
                    print(f"api response text: {response_text}")
                    return {"text": response_text, "status": response.status}
            except Exception as e:
                print(f"no api response captured: {e}")
                await submit_button.click()
                print("clicked submit button")
                return None

        except Exception as e:
            print(f"error filling email: {e}")
            return None

    async def check_page(self) -> dict[str, str | bool]:
        if not self.page:
            raise RuntimeError("page chưa được setup, gọi setup_browser() trước")

        await self.page.goto("https://login.orange.fr", wait_until="load")

        try:
            await self.page.wait_for_url(
                self._is_captcha_url, timeout=5000, wait_until="networkidle"
            )
            is_captcha = True
        except Exception:
            is_captcha = False

        current_url = self.page.url
        if not is_captcha:
            is_captcha = self._is_captcha_url(current_url)

        if is_captcha:
            print(f"captcha detected: {current_url}")
            self.page.wait_for_selector("#image-grid", timeout=10000)
            solved = await self._solve_captcha()
            check_mobile_connect= False
            if solved:
                if self._click_continue_button():
                    try:
                        await self.page.wait_for_selector("#login", timeout=10000)
                        print("login page loaded")
                        await self._inject_accept_button_script()
                        print("injected accept button script (will click after 60s)")
                    except Exception as e:
                        print(f"wait for login page timeout: {e}")
                    api_response = await self._fill_email_and_submit()
                        
                    if api_response:
                        if api_response["data"]["mobileConnectScreen"]["displayedAccount"]["isMobileConnect"] :
                            check_mobile_connect = True
                        print(f"captured api response: {api_response}")

        
        if check_mobile_connect:
            save_mail =  self.email.strip()
            with open("mail_is_mobile.txt","a", encoding="utf-8") as f:
                f.write(self.email.strip() + "\r\n")
        title = self.page.title()
        print(f"page title: {title}")
        print(f"current url: {current_url}")

        return {"title": title, "url": current_url, "is_captcha": is_captcha,"is_mobile_connect":check_mobile_connect}

    async def run(self) -> dict[str, str | bool]:
        await self.setup_context()
        return await self.check_page()



async def main() -> None:
    # Đọc mail  
    with open("html.txt") as f:
        mails = [m.strip() for m in f.readlines()]

    async with Stealth().use_async(async_playwright()) as p:
    # p: Playwright = await async_playwright().start()
        browser_count = 3

        browser: Browser = await p.chromium.launch(
            channel="chromium", headless=False
        )

        try:
            context_list: list[OrangeChecker] = []
            for idx in range(browser_count):
                print(f"Đang tạo context {idx + 1}...")

                ctx: BrowserContext = await browser.new_context()
                print(ctx)
                mail = mails[idx] if idx < len(mails) else None
                ctx_wrapper = OrangeChecker(context=ctx, email=mail,)
                context_list.append(ctx_wrapper)


            for idx, ctx_wrapper in enumerate(context_list,1):
                ctx_wrapper.run()

            await asyncio.to_thread(input)

        finally:
            await browser.close()
            await p.stop()

if __name__ == "__main__":
    asyncio.run(main())





