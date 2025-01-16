from ultralytics import YOLO # type: ignore
import cv2 # type: ignore
import math 

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes (you may not need the full list if you're only interested in "person")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Variable to keep track of the number of people detected
    people_count = 0

    # Iterate over each result detected
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Get the class of the detected object (e.g., person, car, etc.)
            cls = int(box.cls[0])

            # Only count "person" (class 0)
            if classNames[cls] == "person":
                # Increment people count
                people_count += 1

                # Get coordinates for bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box around person
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence and class name
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print(f"Confidence: {confidence}, Class: {classNames[cls]}")

                # Display label on image
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Show the number of people detected on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"People Count: {people_count}", (10, 40), font, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Webcam', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        print("Number of People are: ",people_count)
        break

cap.release()
cv2.destroyAllWindows()
