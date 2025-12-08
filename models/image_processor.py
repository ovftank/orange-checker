import os
import tempfile
from typing import Optional, Dict, Tuple
from urllib.parse import parse_qs, urlparse
from config import Config
from models.model_loader import ModelLoader

class ImageProcessor:
    """Xử lý các tác vụ liên quan đến ảnh"""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.image_cache: Dict[int, str] = {}
    
    def is_captcha_url(self, url: str) -> bool:
        """Kiểm tra xem URL có phải là trang CAPTCHA không"""
        parsed = urlparse(url)
        if parsed.netloc == Config.CAPTCHA_DOMAIN and parsed.path == "/":
            query_params = parse_qs(parsed.query)
            if "csid" in query_params and "captchaService" in query_params:
                return True
        return False
    
    def process_single_image(
        self, image_path: str, index: int
    ) -> Optional[Tuple[int, str, float]]:
        """Xử lý một ảnh và trả về kết quả"""
        if index in self.image_cache:
            return None

        try:
            image_tensor = self.model_loader.preprocess_image(image_path)
            with self.model_loader._model_lock:
                class_name, confidence = self.model_loader.predict(image_tensor)
            return (index, class_name, confidence)
        except Exception as e:
            print(f"Error processing image {index}: {e}")
            return None
    
    async def process_images_async(self, image_paths: Dict[int, str]) -> Dict[int, Tuple[str, float]]:
        """Xử lý nhiều ảnh đồng thời"""
        import asyncio
        
        items_to_process = [
            item for item in image_paths.items() 
            if item[0] not in self.image_cache
        ]
        if not items_to_process:
            return {}
        
        results = {}
        
        # Chạy model bằng asyncio.to_thread
        tasks = []
        for index, path in items_to_process:
            tasks.append(
                asyncio.to_thread(self.process_single_image, path, index)
            )

        
        processing_results = await asyncio.gather(*tasks)

        for result in processing_results:
            if result:
                index, class_name, confidence = result
                self.image_cache[index] = class_name
                results[index] = (class_name, confidence)
                print(f"Image {index}: {class_name} (confidence: {confidence:.2f})")

        return results
