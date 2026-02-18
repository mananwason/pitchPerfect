"""
Unified device interface for Android (ADB) and iOS (WebDriverAgent).

Android: uses pure-python-adb (ppadb)
iOS: uses pymobiledevice3 + facebook-wda (WebDriverAgent)

iOS setup:
  1. pip install pymobiledevice3 facebook-wda
  2. Install WebDriverAgent on your iPhone via Xcode
  3. Connect iPhone via USB, then in Terminal 1:
     python3 -m pymobiledevice3 developer dvt xcuitest com.facebook.WebDriverAgentRunner.xctrunner
  4. In Terminal 2, forward port 8100:
     python3 -m pymobiledevice3 usbmux forward 8100 8100
  5. WDA is now at http://localhost:8100. Run:
     cd app && python3 main.py
"""

import os
import time
from abc import ABC, abstractmethod


class Device(ABC):
    """Abstract device interface for screen capture and interaction."""

    @abstractmethod
    def get_screen_resolution(self) -> tuple:
        pass

    @abstractmethod
    def capture_screenshot(self, filename: str) -> str:
        pass

    @abstractmethod
    def tap(self, x: int, y: int):
        pass

    @abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
        pass

    @abstractmethod
    def input_text(self, text: str):
        pass

    @abstractmethod
    def open_hinge(self):
        pass


class AndroidDevice(Device):
    """Android device via ADB."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5037):
        from ppadb.client import Client as AdbClient

        adb = AdbClient(host=host, port=port)
        devices = adb.devices()
        if not devices:
            raise ConnectionError("No Android device found via ADB.")
        self.device = devices[0]
        print(f"[Android] Connected to {self.device.serial}")

    def get_screen_resolution(self) -> tuple:
        output = self.device.shell("wm size")
        resolution = output.strip().split(":")[1].strip()
        width, height = map(int, resolution.split("x"))
        return width, height

    def capture_screenshot(self, filename: str) -> str:
        os.makedirs("images", exist_ok=True)
        path = f"images/{filename}.png"
        result = self.device.screencap()
        with open(path, "wb") as f:
            f.write(result)
        return path

    def tap(self, x: int, y: int):
        self.device.shell(f"input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
        self.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def input_text(self, text: str):
        escaped = text.replace(" ", "%s")
        self.device.shell(f'input text "{escaped}"')

    def open_hinge(self):
        package = "co.match.android.matchhinge"
        self.device.shell(
            f"monkey -p {package} -c android.intent.category.LAUNCHER 1"
        )


class IOSDevice(Device):
    """iOS device via WebDriverAgent (facebook-wda).

    Requires WDA running and accessible at wda_url.
    Start WDA with pymobiledevice3:
      Terminal 1: python3 -m pymobiledevice3 developer dvt xcuitest com.facebook.WebDriverAgentRunner.xctrunner
      Terminal 2: python3 -m pymobiledevice3 usbmux forward 8100 8100
    """

    def __init__(self, wda_url: str = "http://localhost:8100"):
        import wda

        self.client = wda.Client(wda_url)
        # Test connection
        try:
            status = self.client.status()
            session_id = status.get("sessionId", "connected")
            print(f"[iOS] Connected via WDA at {wda_url} — session: {session_id}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to WDA at {wda_url}: {e}\n"
                "Make sure WebDriverAgent is running. Start it with:\n"
                "  Terminal 1: python3 -m pymobiledevice3 developer dvt xcuitest com.facebook.WebDriverAgentRunner.xctrunner\n"
                "  Terminal 2: python3 -m pymobiledevice3 usbmux forward 8100 8100"
            )

    def get_screen_resolution(self) -> tuple:
        info = self.client.window_size()
        return int(info.width), int(info.height)

    def capture_screenshot(self, filename: str) -> str:
        os.makedirs("images", exist_ok=True)
        path = f"images/{filename}.png"
        self.client.screenshot(path)
        return path

    def tap(self, x: int, y: int):
        self.client.click(x, y)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
        self.client.swipe(x1, y1, x2, y2, duration=duration_ms / 1000.0)

    def input_text(self, text: str):
        self.client.send_keys(text)

    def open_hinge(self):
        bundle_id = "co.hinge.app"
        self.client.session(bundle_id)


def detect_device() -> Device:
    """
    Auto-detect connected device.
    Override with DEVICE_PLATFORM=android or DEVICE_PLATFORM=ios in .env.
    """
    platform = os.getenv("DEVICE_PLATFORM", "auto").lower()
    wda_url = os.getenv("WDA_URL", "http://localhost:8100")
    device_ip = os.getenv("DEVICE_IP", "127.0.0.1")

    if platform == "ios":
        return IOSDevice(wda_url)

    if platform == "android":
        return AndroidDevice(device_ip)

    # Auto-detect: try iOS first, then Android
    # Try iOS/WDA
    try:
        import wda
        client = wda.Client(wda_url, timeout=3)
        client.status()
        print("[Auto-detect] iOS device found via WDA.")
        return IOSDevice(wda_url)
    except Exception:
        pass

    # Try Android/ADB
    try:
        return AndroidDevice(device_ip)
    except Exception:
        pass

    raise ConnectionError(
        "No device found. Ensure either:\n"
        "  Android: ADB is running (adb devices)\n"
        "  iOS: Start WDA with:\n"
        "    Terminal 1: python3 -m pymobiledevice3 developer dvt xcuitest com.facebook.WebDriverAgentRunner.xctrunner\n"
        "    Terminal 2: python3 -m pymobiledevice3 usbmux forward 8100 8100\n"
        "Or set DEVICE_PLATFORM=android|ios in .env"
    )