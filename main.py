import cv2
from ultralytics import YOLO

# Open video
cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Load YOLO model
model = YOLO("yolov8n.pt")

# Position of red counting line (Y coordinate)
line_position = 300

# Total bag count
bag_count = 0

# Store last Y position of each object
last_y_position = {}

# Store IDs which are already counted
counted_objects = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and track objects (bag related classes only)
    results = model.track(frame, persist=True, classes=[24, 26, 28], verbose=False)

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xywh.cpu()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, obj_id in zip(boxes, ids):

            x, y, w, h = box
            center_y = int(y)

            # Check if we already saw this object before
            if obj_id in last_y_position:

                previous_y = last_y_position[obj_id]

                # If object crosses the red line (top to bottom)
                if previous_y < line_position and center_y >= line_position:

                    if obj_id not in counted_objects:
                        bag_count += 1
                        counted_objects.append(obj_id)
                        print("Bag counted. Total:", bag_count)

            # Update last position
            last_y_position[obj_id] = center_y

            # Draw bounding box
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw horizontal red counting line
    cv2.line(frame, (0, line_position),
         (frame.shape[1], line_position),
         (0, 0, 255), 2)

    # Show total count
    cv2.putText(frame, "Total Bags: " + str(bag_count),
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Bag Counter", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()