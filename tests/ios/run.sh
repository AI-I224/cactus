#!/bin/bash

# iOS test runner script
# Called from tests/run.sh with --ios flag
# Handles: device selection, Xcode project configuration, app build, and test execution

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Model names passed from parent run.sh
MODEL_NAME="$1"
TRANSCRIBE_MODEL_NAME="$2"

# Validate Xcode installation
if [ ! -d "/Applications/Xcode.app" ]; then
    echo ""
    echo "Error: Xcode not installed"
    exit 1
fi

if ! xcode-select -p; then
    echo ""
    echo "Error: Xcode Command Line Tools not installed"
    exit 1
fi

if ! /usr/bin/xcrun --version; then
    echo ""
    echo "Error: Xcode license not accepted"
    exit 1
fi

if ! command -v xcodebuild; then
    echo ""
    echo "Error: xcodebuild not found"
    exit 1
fi

echo ""
echo "Step 2: Selecting iOS device..."

# Collect available simulators
SIMULATORS=$(xcrun simctl list devices available | grep -E "^\s+(iPhone|iPad)" | grep -v "unavailable" | sed 's/^[[:space:]]*//' | while read line; do
    uuid=$(echo "$line" | grep -oE '\([A-F0-9-]{36}\)' | head -1 | tr -d '()')
    if [ -n "$uuid" ]; then
        name=$(echo "$line" | sed -E 's/ \([^)]*\)//g' | xargs)
        echo "${name}|simulator|${uuid}"
    fi
done)

# Collect physical devices (UUID pattern: 00008XXX-XXXXXXXXXXXXXXXX)
XCTRACE_OUTPUT=$(xcrun xctrace list devices 2>&1)

PHYSICAL_DEVICES=$(echo "$XCTRACE_OUTPUT" | awk '
    /== Devices ==/ { in_online=1; in_offline=0; next }
    /== Devices Offline ==/ { in_online=0; in_offline=1; next }
    /== Simulators ==/ { exit }
    /00008[A-F0-9]{3}-[A-F0-9]{16}/ {
        if (in_online || in_offline) {
            status = in_offline ? "offline" : ""
            print $0 "|" status
        }
    }
' | while read line; do
    uuid=$(echo "$line" | grep -oE '00008[A-F0-9]{3}-[A-F0-9]{16}')
    status=$(echo "$line" | awk -F'|' '{print $2}')
    if [ -n "$uuid" ]; then
        name=$(echo "$line" | awk -F'|' '{print $1}' | sed -E 's/ \([0-9]+\.[0-9]+.*$//' | xargs)
        ios_version=$(echo "$line" | awk -F'|' '{print $1}' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        echo "${name} (iOS ${ios_version})|device|${uuid}|${status}"
    fi
done)

# Combine devices (physical first, then simulators)
ALL_DEVICES=$(printf "%s\n%s\n" "$PHYSICAL_DEVICES" "$SIMULATORS" | grep -v '^$')

if [ -z "$ALL_DEVICES" ]; then
    echo ""
    echo "Error: No devices or simulators found"
    echo "Install iOS simulators through Xcode or connect a physical device"
    exit 1
fi

# Display devices for selection
PHYSICAL_COUNT=$(echo "$ALL_DEVICES" | grep -c '|device|')
DEVICE_NUM=0

if [ "$PHYSICAL_COUNT" -gt 0 ]; then
    echo "Devices:"
    while IFS='|' read -r name type uuid status; do
        if [ "$type" = "device" ]; then
            DEVICE_NUM=$((DEVICE_NUM + 1))
            if [ "$status" = "offline" ]; then
                printf "  %2d) %s [offline]\n" "$DEVICE_NUM" "$name"
            else
                printf "  %2d) %s\n" "$DEVICE_NUM" "$name"
            fi
        fi
    done <<< "$ALL_DEVICES"
    echo ""
fi

echo "Simulators:"
while IFS='|' read -r name type uuid status; do
    if [ "$type" = "simulator" ]; then
        DEVICE_NUM=$((DEVICE_NUM + 1))
        printf "  %2d) %s\n" "$DEVICE_NUM" "$name"
    fi
done <<< "$ALL_DEVICES"

echo ""
read -p "Select device number (1-$DEVICE_NUM): " DEVICE_NUMBER

if ! [[ "$DEVICE_NUMBER" =~ ^[0-9]+$ ]] || [ "$DEVICE_NUMBER" -lt 1 ] || [ "$DEVICE_NUMBER" -gt "$DEVICE_NUM" ]; then
    echo ""
    echo "Invalid selection. Please enter a number between 1 and $DEVICE_NUM"
    exit 1
fi

# Parse selected device
SELECTED_LINE=$(echo "$ALL_DEVICES" | sed -n "${DEVICE_NUMBER}p")
DEVICE_NAME=$(echo "$SELECTED_LINE" | cut -d'|' -f1)
DEVICE_TYPE=$(echo "$SELECTED_LINE" | cut -d'|' -f2)
DEVICE_UUID=$(echo "$SELECTED_LINE" | cut -d'|' -f3)
DEVICE_STATUS=$(echo "$SELECTED_LINE" | cut -d'|' -f4)

if [ -z "$DEVICE_UUID" ]; then
    echo ""
    echo "Error: Could not parse device information"
    exit 1
fi

echo ""
if [ "$DEVICE_TYPE" = "simulator" ]; then
    echo "Selected: $DEVICE_NAME (Simulator)"
else
    if [ "$DEVICE_STATUS" = "offline" ]; then
        echo "Selected: $DEVICE_NAME (Device - Offline)"
        echo ""
        echo "Warning: This device is currently offline"
        echo "   Please ensure the device is:"
        echo "   - Connected via USB or network"
        echo "   - Unlocked and trusted"
        echo "   - Has developer mode enabled"
        echo ""
        read -p "Continue anyway? (y/N): " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    else
        echo "Selected: $DEVICE_NAME (Device)"
    fi
fi

# Verify certificates exist for physical devices
if [ "$DEVICE_TYPE" = "device" ] && ! security find-identity -v -p codesigning | grep -q "Apple Development"; then
    echo ""
    echo "Error: No development certificates found"
    echo ""
    echo "To fix this:"
    echo "  1. Open Xcode"
    echo "  2. Go to Settings > Accounts"
    echo "  3. Add your Apple ID"
    echo "  4. Download development certificates"
    exit 1
fi

echo ""
echo "Step 3: Configuring Xcode project..."

XCODEPROJ_PATH="$SCRIPT_DIR/CactusTest/CactusTest.xcodeproj"
TESTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACTUS_ROOT="$PROJECT_ROOT/cactus"

# Configure bundle ID and team for physical devices
BUNDLE_ID="com.cactus.test.${USER}"
echo "Using Bundle ID: $BUNDLE_ID"


if [ "$DEVICE_TYPE" = "device" ]; then
    DEVELOPMENT_TEAM=$(security find-certificate -a -c "Apple Development" -p | openssl x509 -noout -subject | grep -oE 'OU=[A-Z0-9]{10}' | head -1 | cut -d= -f2)
    if [ -z "$DEVELOPMENT_TEAM" ]; then
        echo ""
        echo "Error: Could not extract Team ID from certificate"
        exit 1
    fi
    echo "Using Team ID: $DEVELOPMENT_TEAM"
fi

if ! command -v ruby; then
    echo ""
    echo "Error: Ruby not found. Please install Ruby."
    exit 1
fi

if ! gem list xcodeproj -i; then
    echo "   Installing xcodeproj gem..."
    if ! gem install xcodeproj; then
        echo ""
        echo "Error: Failed to install xcodeproj gem"
        exit 1
    fi
fi

export PROJECT_ROOT TESTS_ROOT CACTUS_ROOT XCODEPROJ_PATH BUNDLE_ID DEVELOPMENT_TEAM
if ! ruby "$SCRIPT_DIR/setup_project.rb"; then
    echo ""
    echo "Error: Failed to setup Xcode project"
    exit 1
fi

echo ""
echo "Step 4: Building iOS test application..."

if [ "$DEVICE_TYPE" = "simulator" ]; then
    IOS_SIM_SDK_PATH=$(xcrun --sdk iphonesimulator --show-sdk-path)
    if [ -z "$IOS_SIM_SDK_PATH" ] || [ ! -d "$IOS_SIM_SDK_PATH" ]; then
        echo ""
        echo "Error: iOS Simulator SDK not found"
        exit 1
    fi

    if ! xcodebuild -project "$XCODEPROJ_PATH" \
         -scheme CactusTest \
         -configuration Release \
         -destination "platform=iOS Simulator,id=$DEVICE_UUID" \
         -derivedDataPath "$SCRIPT_DIR/build" \
         ARCHS=arm64 \
         ONLY_ACTIVE_ARCH=NO \
         IPHONEOS_DEPLOYMENT_TARGET=13.0 \
         SDKROOT="$IOS_SIM_SDK_PATH" \
         PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
         build; then
        echo ""
        echo "Error: Build failed"
        exit 1
    fi

    APP_PATH="$SCRIPT_DIR/build/Build/Products/Release-iphonesimulator/CactusTest.app"
else
    IOS_SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path)
    if [ -z "$IOS_SDK_PATH" ] || [ ! -d "$IOS_SDK_PATH" ]; then
        echo ""
        echo "Error: iOS SDK not found"
        exit 1
    fi

    if ! xcodebuild -project "$XCODEPROJ_PATH" \
         -scheme CactusTest \
         -configuration Release \
         -destination "platform=iOS,id=$DEVICE_UUID" \
         -derivedDataPath "$SCRIPT_DIR/build" \
         -allowProvisioningUpdates \
         ARCHS=arm64 \
         ONLY_ACTIVE_ARCH=NO \
         IPHONEOS_DEPLOYMENT_TARGET=13.0 \
         SDKROOT="$IOS_SDK_PATH" \
         PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
         CODE_SIGN_STYLE="Automatic" \
         build; then
        echo ""
        echo "Error: Build failed"
        exit 1
    fi

    APP_PATH="$SCRIPT_DIR/build/Build/Products/Release-iphoneos/CactusTest.app"
fi

MODEL_DIR=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
TRANSCRIBE_MODEL_DIR=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
MODEL_SRC="$PROJECT_ROOT/weights/$MODEL_DIR"
TRANSCRIBE_MODEL_SRC="$PROJECT_ROOT/weights/$TRANSCRIBE_MODEL_DIR"

echo ""
echo "Copying model weights to app bundle..."
if ! cp -R "$MODEL_SRC" "$APP_PATH/"; then
    echo "Warning: Could not copy model weights"
fi
if ! cp -R "$TRANSCRIBE_MODEL_SRC" "$APP_PATH/"; then
    echo "Warning: Could not copy transcribe model weights"
fi

echo ""
echo "Step 5: Running tests..."
echo "------------------------"

if [ "$DEVICE_TYPE" = "simulator" ]; then
    echo "Installing on: $DEVICE_NAME"

    xcrun simctl boot "$DEVICE_UUID" || true

    if ! xcrun simctl install "$DEVICE_UUID" "$APP_PATH"; then
        echo ""
        echo "Error: Failed to install app on simulator"
        exit 1
    fi

    echo "Launching tests..."

    SIMCTL_CHILD_CACTUS_TEST_MODEL="$MODEL_DIR" \
    SIMCTL_CHILD_CACTUS_TEST_TRANSCRIBE_MODEL="$TRANSCRIBE_MODEL_DIR" \
    xcrun simctl launch --console-pty "$DEVICE_UUID" "$BUNDLE_ID"

    echo ""
else
    echo "Installing on: $DEVICE_NAME"

    if ! xcrun devicectl device install app --device "$DEVICE_UUID" "$APP_PATH"; then
        echo ""
        echo "Error: Failed to install app on device"
        echo "Common issues: device not trusted, code signing failed, or device locked"
        exit 1
    fi

    echo "Launching tests..."
    echo "(Logs will be fetched from device after completion)"

    DEVICECTL_CHILD_CACTUS_TEST_MODEL="$MODEL_DIR" \
    DEVICECTL_CHILD_CACTUS_TEST_TRANSCRIBE_MODEL="$TRANSCRIBE_MODEL_DIR" \
    xcrun devicectl device process launch --device "$DEVICE_UUID" "$BUNDLE_ID" || true

    # Wait for process to complete (poll every 2s, max 5 minutes)
    MAX_WAIT=300
    ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        if xcrun devicectl device info processes --device "$DEVICE_UUID" | grep -q "CactusTest.app/CactusTest"; then
            sleep 2
            ELAPSED=$((ELAPSED + 2))
        else
            break
        fi
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "Warning: Test execution timeout reached (${MAX_WAIT}s)"
    fi

    sleep 1

    echo "Fetching logs from device..."

    TEMP_LOG_DIR=$(mktemp -d)
    TEMP_LOG_FILE="$TEMP_LOG_DIR/cactus_test.log"

    if xcrun devicectl device copy from \
        --device "$DEVICE_UUID" \
        --source "Documents/cactus_test.log" \
        --destination "$TEMP_LOG_FILE" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID"; then

        if [ -f "$TEMP_LOG_FILE" ]; then
            echo ""
            cat "$TEMP_LOG_FILE"
        else
            echo "Warning: Could not find downloaded log file"
        fi

        rm -rf "$TEMP_LOG_DIR"
    else
        echo "Warning: Could not fetch log file from device"
    fi

    echo ""
fi
