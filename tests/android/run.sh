#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TESTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_NAME="$1"
TRANSCRIBE_MODEL_NAME="$2"

if ! command -v adb &> /dev/null; then
    echo ""
    echo "Error: adb not found"
    echo "Install Android SDK Platform Tools or set up your PATH"
    exit 1
fi

echo ""
echo "Step 2: Selecting Android device..."

adb start-server > /dev/null 2>&1


DEVICES=$(adb devices | grep -E "device$|emulator" | grep -v "^List" | awk '{print $1}')

if [ -z "$DEVICES" ]; then
    echo ""
    echo "Error: No Android devices or emulators found"
    echo ""
    echo "Start an emulator: emulator -avd <avd_name>"
    echo "Or connect a physical device with USB debugging enabled"
    exit 1
fi

DEVICE_COUNT=$(echo "$DEVICES" | wc -l | tr -d ' ')

if [ "$DEVICE_COUNT" -eq 1 ]; then
    DEVICE_ID=$(echo "$DEVICES" | head -1)
    echo "Using device: $DEVICE_ID"
else
    echo "Available devices:"
    DEVICE_NUM=0
    while read -r device; do
        DEVICE_NUM=$((DEVICE_NUM + 1))
        DEVICE_MODEL=$(adb -s "$device" shell getprop ro.product.model 2>/dev/null | tr -d '\r')
        DEVICE_ANDROID=$(adb -s "$device" shell getprop ro.build.version.release 2>/dev/null | tr -d '\r')
        if [ -n "$DEVICE_MODEL" ]; then
            printf "  %2d) %s - %s (Android %s)\n" "$DEVICE_NUM" "$device" "$DEVICE_MODEL" "$DEVICE_ANDROID"
        else
            printf "  %2d) %s\n" "$DEVICE_NUM" "$device"
        fi
    done <<< "$DEVICES"

    echo ""
    read -p "Select device number (1-$DEVICE_COUNT): " DEVICE_NUMBER

    if ! [[ "$DEVICE_NUMBER" =~ ^[0-9]+$ ]] || [ "$DEVICE_NUMBER" -lt 1 ] || [ "$DEVICE_NUMBER" -gt "$DEVICE_COUNT" ]; then
        echo ""
        echo "Invalid selection"
        exit 1
    fi

    DEVICE_ID=$(echo "$DEVICES" | sed -n "${DEVICE_NUMBER}p")
    echo ""
    echo "Selected: $DEVICE_ID"
fi

if ! adb -s "$DEVICE_ID" shell echo "test" > /dev/null 2>&1; then
    echo ""
    echo "Error: Device not responding"
    exit 1
fi

echo ""
echo "Step 3: Building Cactus library for Android..."

if ! "$PROJECT_ROOT/android/build.sh"; then
    echo ""
    echo "Error: Failed to build Cactus library"
    exit 1
fi

echo ""
echo "Step 4: Building Android tests..."

ANDROID_TEST_DIR="$SCRIPT_DIR"
ANDROID_BUILD_DIR="$ANDROID_TEST_DIR/build"


if [ -z "$ANDROID_NDK_HOME" ]; then
    if [ -n "$ANDROID_HOME" ]; then
        ANDROID_NDK_HOME=$(ls -d "$ANDROID_HOME/ndk/"* 2>/dev/null | sort -V | tail -1)
    elif [ -d "$HOME/Library/Android/sdk" ]; then
        ANDROID_NDK_HOME=$(ls -d "$HOME/Library/Android/sdk/ndk/"* 2>/dev/null | sort -V | tail -1)
    fi
fi

if [ -z "$ANDROID_NDK_HOME" ] || [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo ""
    echo "Error: Android NDK not found"
    exit 1
fi

CMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
ANDROID_PLATFORM=${ANDROID_PLATFORM:-android-21}
ANDROID_ABI="arm64-v8a"

rm -rf "$ANDROID_BUILD_DIR"
mkdir -p "$ANDROID_BUILD_DIR"

if ! cmake -S "$ANDROID_TEST_DIR" -B "$ANDROID_BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DCMAKE_BUILD_TYPE=Release; then
    echo ""
    echo "Error: Failed to configure tests"
    exit 1
fi

n_cpu=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
if ! cmake --build "$ANDROID_BUILD_DIR" -j "$n_cpu"; then
    echo ""
    echo "Error: Failed to build tests"
    exit 1
fi

TEST_EXECUTABLES=($(find "$ANDROID_BUILD_DIR" -maxdepth 1 -name "test_*" -type f | sort))

if [ ${#TEST_EXECUTABLES[@]} -eq 0 ]; then
    echo ""
    echo "Error: No test executables found"
    exit 1
fi

MODEL_DIR=$(echo "$MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
TRANSCRIBE_MODEL_DIR=$(echo "$TRANSCRIBE_MODEL_NAME" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
MODEL_SRC="$PROJECT_ROOT/weights/$MODEL_DIR"
TRANSCRIBE_MODEL_SRC="$PROJECT_ROOT/weights/$TRANSCRIBE_MODEL_DIR"

DEVICE_TEST_DIR="/data/local/tmp/cactus_tests"
DEVICE_MODEL_DIR="/data/local/tmp/cactus_models"

echo ""
echo "Step 5: Deploying to device..."

adb -s "$DEVICE_ID" shell "mkdir -p $DEVICE_TEST_DIR $DEVICE_MODEL_DIR"

echo "Pushing model weights..."
adb -s "$DEVICE_ID" push "$MODEL_SRC" "$DEVICE_MODEL_DIR/"
adb -s "$DEVICE_ID" push "$TRANSCRIBE_MODEL_SRC" "$DEVICE_MODEL_DIR/"

echo ""
echo "Pushing test executables..."
for test_exe in "${TEST_EXECUTABLES[@]}"; do
    test_name=$(basename "$test_exe")
    adb -s "$DEVICE_ID" push "$test_exe" "$DEVICE_TEST_DIR/"
    adb -s "$DEVICE_ID" shell "chmod +x $DEVICE_TEST_DIR/$test_name"
done

echo ""
echo "Step 6: Running tests..."
echo "------------------------"

for test_exe in "${TEST_EXECUTABLES[@]}"; do
    test_name=$(basename "$test_exe")
    echo ""
    echo "Running $test_name..."

    adb -s "$DEVICE_ID" shell "cd /data/local/tmp && \
        export CACTUS_TEST_MODEL=$DEVICE_MODEL_DIR/$MODEL_DIR && \
        export CACTUS_TEST_TRANSCRIBE_MODEL=$DEVICE_MODEL_DIR/$TRANSCRIBE_MODEL_DIR && \
        $DEVICE_TEST_DIR/$test_name"
done

echo ""
