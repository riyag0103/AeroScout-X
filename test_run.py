import os
import sys

# Mocking parts of the original script to handle the input and basic logic for testing
print("📂 Enter video path: ", end="", flush=True)
video_path = sys.stdin.readline().strip()

if not video_path:
    print("No path provided.")
    sys.exit(1)

if not os.path.exists(video_path):
    print(f"Error: Path '{video_path}' does not exist.")
    sys.exit(1)

print(f"Processing {video_path}...")
# Original script would continue here...
sys.exit(0)
