import os

class Config:
    MODEL_PATH = "model.pth"
    EMAIL_FILE = "html.txt"
    MOBILE_EMAIL_FILE = "mail_is_mobile.txt"
    
    DEFAULT_TIMEOUT = 60000
    DEFAULT_NAVIGATION_TIMEOUT = 60000
    BROWSER_COUNT = 3
    
    # Cấu hình CAPTCHA
    MAX_STEPS = 10
    MAX_RELOADS = 3
    CAPTCHA_TIMEOUT = 5000
    
    # Cấu hình mạng
    ORANGE_LOGIN_URL = "https://login.orange.fr"
    CAPTCHA_DOMAIN = "captcha.orange.fr"
    
    # Cấu hình Model ML
    MODEL_INPUT_SIZE = (75, 75)
    MODEL_MEAN = [0.485, 0.456, 0.406]
    MODEL_STD = [0.229, 0.224, 0.225]
    MODEL_DROPOUT = 0.5
