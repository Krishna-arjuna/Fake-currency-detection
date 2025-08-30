import cv2
import numpy as np
from tkinter import filedialog, Tk

# Hide Tkinter root window
Tk().withdraw()

print("Select the REFERENCE (genuine) currency image...")
ref_path = filedialog.askopenfilename(
    filetypes=[("Image files", "*.jpg;*.png;*.bmp")],
    title="Select Reference Currency Image"
)

if not ref_path:
    print("Reference image not selected.")
    exit()

ref_img = cv2.imread(ref_path)
cv2.imshow("Reference Currency Image", ref_img)

print("Select the TEST currency image...")
test_path = filedialog.askopenfilename(
    filetypes=[("Image files", "*.jpg;*.png;*.bmp")],
    title="Select Test Currency Image"
)

if not test_path:
    print("Test image not selected.")
    exit()

test_img = cv2.imread(test_path)
cv2.imshow("Test Currency Image", test_img)

# Convert to grayscale
ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# Feature detection + description (ORB)
orb = cv2.ORB_create()

ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)
test_keypoints, test_descriptors = orb.detectAndCompute(test_gray, None)

# Feature matching (Brute Force with Hamming distance)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(ref_descriptors, test_descriptors)

# Sort matches by distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
match_img = cv2.drawMatches(ref_gray, ref_keypoints,
                            test_gray, test_keypoints,
                            matches[:50], None, flags=2)

cv2.imshow("Matched Features", match_img)

# Count matches
match_count = len(matches)
print(f"\nNumber of matched features: {match_count}")

# Threshold to decide genuine vs fake
threshold = 25
if match_count > threshold:
    print("✅ The currency is likely GENUINE.")
else:
    print("❌ The currency is likely FAKE.")

cv2.waitKey(0)
cv2.destroyAllWindows()
