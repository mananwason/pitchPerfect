import time
import os
import uuid
from config import DEVICE_IP
from device import detect_device
from llm_vision import analyze_profile
from data_store import store_generated_comment
from calibrate import load_buttons


def capture_full_profile(device, width, height, max_scrolls=5):
    """
    Scroll through the entire Hinge profile and capture a screenshot at each scroll position.
    Returns a list of screenshot file paths.
    
    Hinge profiles typically have 3-6 cards (photos + prompts).
    We scroll down and capture at each position until we've covered the full profile.
    """
    screenshots = []
    swipe_x = int(width * 0.50)
    swipe_y1 = int(height * 0.70)
    swipe_y2 = int(height * 0.30)

    num_scrolls = int(os.getenv("PROFILE_SCROLLS", str(max_scrolls)))

    for scroll_idx in range(num_scrolls):
        # Capture current view
        path = device.capture_screenshot(f"profile_{scroll_idx}")
        screenshots.append(path)
        print(f"  Captured screenshot {scroll_idx + 1}/{num_scrolls}: {path}")

        # Scroll down to reveal more content (skip on last capture)
        if scroll_idx < num_scrolls - 1:
            device.swipe(swipe_x, swipe_y1, swipe_x, swipe_y2, duration_ms=400)
            time.sleep(0.8)

    return screenshots


def scroll_to_target(device, width, height, current_scroll, target_scroll):
    """Scroll back up to the target screenshot position to interact with that content."""
    swipe_x = int(width * 0.50)
    scrolls_needed = current_scroll - target_scroll

    if scrolls_needed > 0:
        # Scroll up
        swipe_y1 = int(height * 0.30)
        swipe_y2 = int(height * 0.70)
        for _ in range(scrolls_needed):
            device.swipe(swipe_x, swipe_y1, swipe_x, swipe_y2, duration_ms=400)
            time.sleep(0.5)
    elif scrolls_needed < 0:
        # Scroll down
        swipe_y1 = int(height * 0.70)
        swipe_y2 = int(height * 0.30)
        for _ in range(abs(scrolls_needed)):
            device.swipe(swipe_x, swipe_y1, swipe_x, swipe_y2, duration_ms=400)
            time.sleep(0.5)


def main():
    # Auto-detect Android or iOS device
    device = detect_device()

    width, height = device.get_screen_resolution()
    print(f"Device resolution: {width}x{height}")

    # Load button coordinates (from calibration file or defaults)
    buttons = load_buttons(width, height)
    like_btn_x, like_btn_y = buttons["like"]
    dislike_btn_x, dislike_btn_y = buttons["dislike"]
    comment_btn_x, comment_btn_y = buttons["comment_field"]
    send_btn_x, send_btn_y = buttons["send"]
    print(f"Buttons: like=({like_btn_x},{like_btn_y}) skip=({dislike_btn_x},{dislike_btn_y})")

    device.open_hinge()
    time.sleep(5)

    num_profiles = int(os.getenv("NUM_PROFILES", "10"))
    num_scrolls = int(os.getenv("PROFILE_SCROLLS", "4"))

    for i in range(num_profiles):
        print(f"\n{'='*50}")
        print(f"Profile {i + 1}/{num_profiles}")
        print(f"{'='*50}")

        # Step 1: Capture the full profile (multiple screenshots)
        print("Capturing full profile...")
        screenshots = capture_full_profile(device, width, height, max_scrolls=num_scrolls)
        print(f"Captured {len(screenshots)} screenshots.")

        # Step 2: Send ALL screenshots to the vision LLM
        print("Analyzing profile with LLM...")
        try:
            result = analyze_profile(screenshots)
            print(f"Decision: {result.get('decision')}")
            print(f"Reasoning: {result.get('reasoning')}")
            if result.get('comment'):
                print(f"Comment: {result.get('comment')}")
            if result.get('target_screenshot'):
                print(f"Target screenshot: {result.get('target_screenshot')}")
        except Exception as e:
            print(f"LLM error: {e}. Skipping this profile.")
            # Scroll back to top and dismiss
            scroll_to_target(device, width, height, num_scrolls - 1, 0)
            time.sleep(0.5)
            device.tap(dislike_btn_x, dislike_btn_y)
            time.sleep(2)
            continue

        decision = result.get("decision", "skip")
        comment = result.get("comment")
        target_screenshot = result.get("target_screenshot", 1)

        # We're currently at the bottom of the profile (last scroll position)
        current_scroll = num_scrolls - 1

        if decision == "like" and comment:
            # Step 3a: Scroll to the section the LLM wants to engage with
            target_idx = max(0, min(target_screenshot - 1, num_scrolls - 1))
            print(f"Scrolling to screenshot {target_screenshot} to engage with that content...")
            scroll_to_target(device, width, height, current_scroll, target_idx)
            current_scroll = target_idx
            time.sleep(0.5)

            # Store for analytics
            store_generated_comment(
                comment_id=str(uuid.uuid4()),
                profile_text=result.get("reasoning", ""),
                generated_comment=comment,
                style_used="vision_llm",
            )

            # Tap like on the visible content
            device.tap(like_btn_x, like_btn_y)
            time.sleep(1)

            # Tap comment field
            device.tap(comment_btn_x, comment_btn_y)
            time.sleep(0.5)

            # Type the comment
            device.input_text(comment)
            time.sleep(0.5)

            # Send
            device.tap(send_btn_x, send_btn_y)
            print("Like sent with comment.")

        elif decision == "like":
            # Scroll back to top for a generic like
            scroll_to_target(device, width, height, current_scroll, 0)
            time.sleep(0.5)
            device.tap(like_btn_x, like_btn_y)
            print("Like sent (no comment).")

        else:
            # Scroll back to top and skip
            scroll_to_target(device, width, height, current_scroll, 0)
            time.sleep(0.5)
            device.tap(dislike_btn_x, dislike_btn_y)
            print("Skipped.")

        time.sleep(2)

    print(f"\nDone. Processed {num_profiles} profiles.")


if __name__ == "__main__":
    main()