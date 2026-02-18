import time
import os
import uuid
from config import DEVICE_IP
from device import detect_device
from llm_vision import analyze_profile
from data_store import store_generated_comment


def main():
    # Auto-detect Android or iOS device
    device = detect_device()

    width, height = device.get_screen_resolution()
    print(f"Device resolution: {width}x{height}")

    # Button coordinates (proportional to screen size)
    like_btn_x = int(width * 0.90)
    like_btn_y = int(height * 0.67)
    dislike_btn_x = int(width * 0.15)
    dislike_btn_y = int(height * 0.85)
    comment_btn_x = int(width * 0.50)
    comment_btn_y = int(height * 0.77)
    send_btn_x = int(width * 0.85)
    send_btn_y = int(height * 0.50)

    # Swipe coordinates
    swipe_x = int(width * 0.50)
    swipe_y1 = int(height * 0.60)
    swipe_y2 = int(height * 0.35)

    device.open_hinge()
    time.sleep(5)

    num_profiles = int(os.getenv("NUM_PROFILES", "10"))

    for i in range(num_profiles):
        print(f"\n--- Profile {i + 1}/{num_profiles} ---")

        # Scroll to see profile content
        device.swipe(swipe_x, swipe_y1, swipe_x, swipe_y2)
        time.sleep(1)

        # Capture screenshot
        screenshot_path = device.capture_screenshot("screen")
        print(f"Screenshot saved: {screenshot_path}")

        # Send screenshot to vision LLM
        try:
            result = analyze_profile(screenshot_path)
            print(f"LLM decision: {result.get('decision')} | reasoning: {result.get('reasoning')}")
        except Exception as e:
            print(f"LLM error: {e}. Skipping this profile.")
            device.tap(dislike_btn_x, dislike_btn_y)
            time.sleep(2)
            continue

        decision = result.get("decision", "skip")
        comment = result.get("comment")

        if decision == "like" and comment:
            print(f"Comment: {comment}")

            # Store for analytics
            store_generated_comment(
                comment_id=str(uuid.uuid4()),
                profile_text=result.get("reasoning", ""),
                generated_comment=comment,
                style_used="vision_llm",
            )

            # Tap like
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
            device.tap(like_btn_x, like_btn_y)
            print("Like sent (no comment).")

        else:
            device.tap(dislike_btn_x, dislike_btn_y)
            print("Skipped.")

        time.sleep(2)

    print(f"\nDone. Processed {num_profiles} profiles.")


if __name__ == "__main__":
    main()