# 💵 Fake Currency Detection using Python (OpenCV)

This project detects whether a test currency note is **genuine or fake** by comparing it with a reference (genuine) currency image using **feature matching techniques**.

---

## 📌 Features
- Select **reference (genuine)** and **test currency** images.
- Detects keypoints and extracts descriptors using **ORB (Oriented FAST and Rotated BRIEF)**.
- Matches features between the two images using **Brute Force Matcher** with **Hamming distance**.
- Displays:
  - Reference Image
  - Test Image
  - Matched Features Visualization
- Counts the number of matched features and classifies:
  - ✅ Genuine (if matches > threshold)
  - ❌ Fake (if matches ≤ threshold)

---

## 🛠️ Technologies Used
- **Python 3.x**
- **OpenCV (cv2)**
- **NumPy**
- **Tkinter** (for file dialog)

---

## 📂 Project Structure
Fake-currency-detection/
├── fake_currency.py # Main Python script
├── reference.jpg # Example genuine note
├── test.jpg # Example test note
└── README.md # Project documentation
