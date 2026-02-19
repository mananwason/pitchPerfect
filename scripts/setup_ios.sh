#!/bin/bash
# PitchPerfect iOS Setup — Installs WebDriverAgent on your iPhone
#
# Prerequisites:
#   - macOS with Xcode installed (including command-line tools)
#   - iPhone connected via USB
#   - Apple Developer account (free is fine)
#
# This script clones WebDriverAgent, opens Xcode for you to sign it,
# and provides the commands to run afterward.

set -e

WDA_DIR="$HOME/.pitchperfect/WebDriverAgent"
VENV_PYTHON="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python3"

echo "=== PitchPerfect iOS Setup ==="
echo ""

# Step 1: Check Xcode
if ! command -v xcodebuild &> /dev/null; then
    echo "ERROR: Xcode is not installed."
    echo "Install it from the Mac App Store, then run: xcode-select --install"
    exit 1
fi
echo "✓ Xcode found: $(xcodebuild -version | head -1)"

# Step 2: Check device
echo ""
echo "Checking for connected iOS devices..."
if ! command -v pymobiledevice3 &> /dev/null && [ -f "$VENV_PYTHON" ]; then
    DEVICE_LIST=$("$VENV_PYTHON" -m pymobiledevice3 usbmux list 2>&1 || true)
else
    DEVICE_LIST=$(python3 -m pymobiledevice3 usbmux list 2>&1 || true)
fi

if echo "$DEVICE_LIST" | grep -q "UDID"; then
    echo "✓ iOS device detected"
else
    echo "WARNING: No iOS device detected via USB. Connect your iPhone and try again."
fi

# Step 3: Clone WebDriverAgent
echo ""
if [ -d "$WDA_DIR" ]; then
    echo "✓ WebDriverAgent already cloned at $WDA_DIR"
else
    echo "Cloning WebDriverAgent..."
    mkdir -p "$(dirname "$WDA_DIR")"
    git clone https://github.com/appium/WebDriverAgent.git "$WDA_DIR"
    echo "✓ Cloned to $WDA_DIR"
fi

# Step 4: Open in Xcode for signing
echo ""
echo "============================================"
echo "  MANUAL STEP REQUIRED: Xcode Signing"
echo "============================================"
echo ""
echo "Opening WebDriverAgent in Xcode..."
echo ""
echo "In Xcode, you need to:"
echo "  1. Select the 'WebDriverAgentRunner' target (left sidebar)"
echo "  2. Go to 'Signing & Capabilities' tab"
echo "  3. Check 'Automatically manage signing'"
echo "  4. Select your Team (Apple ID / Developer account)"
echo "  5. If bundle ID conflicts, change it to something unique like:"
echo "     com.yourname.WebDriverAgentRunner"
echo "  6. Select your connected iPhone as the build target (top bar)"
echo "  7. Press Cmd+U to build and run tests (installs WDA on device)"
echo "  8. On your iPhone: Settings > General > VPN & Device Management"
echo "     > Trust your developer certificate"
echo ""
echo "NOTE: Remember the Bundle ID you used! Default is:"
echo "  com.facebook.WebDriverAgentRunner.xctrunner"
echo ""

open "$WDA_DIR/WebDriverAgent.xcodeproj"

echo "After WDA is installed on your iPhone, run these commands:"
echo ""
echo "  # Terminal 1 (keep running):"
echo "  sudo $VENV_PYTHON -m pymobiledevice3 remote tunneld"
echo ""
echo "  # Terminal 2 (keep running):"
echo "  $VENV_PYTHON -m pymobiledevice3 developer dvt xcuitest com.facebook.WebDriverAgentRunner.xctrunner --tunnel \"\""
echo ""
echo "  # Terminal 3 (keep running):"
echo "  $VENV_PYTHON -m pymobiledevice3 usbmux forward 8100 8100"
echo ""
echo "  # Terminal 4:"
echo "  source .venv/bin/activate && cd app && python3 main.py"
echo ""
echo "If you changed the Bundle ID in Xcode, replace"
echo "  com.facebook.WebDriverAgentRunner.xctrunner"
echo "with your custom Bundle ID in Terminal 2."