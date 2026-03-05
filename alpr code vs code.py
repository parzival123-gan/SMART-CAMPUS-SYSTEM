import cv2
import pytesseract

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'path_to_image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edged = cv2.Canny(blurred, 30, 200)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours based on their area keeping minimum required area as '30' (anything smaller than this will be ignored)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# Loop over our contours
for c in contours:
    # Approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)

    # If our approximated contour has four points, then we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)

# Masking the part other than the number plate
mask = cv2.inRange(gray, 0, 0)
new_image = cv2.bitwise_and(image, image, mask=mask)

# Crop the image - this is the region of interest
(x, y, w, h) = cv2.boundingRect(screenCnt)
cropped = gray[y:y + h, x:x + w]

# Use Tesseract to extract text
text = pytesseract.image_to_string(cropped, config='--psm 11')
print("Detected Number is:", text)

# Display the image
cv2.imshow('image', image)
cv2.imshow('Cropped', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

