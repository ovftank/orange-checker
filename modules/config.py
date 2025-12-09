from dataclasses import dataclass
@dataclass
class Config:
    cookie_click_script = """(() => {
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
            })();""",
    accept_script= """
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





