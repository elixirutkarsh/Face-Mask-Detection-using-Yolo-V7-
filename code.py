import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("yolov3.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face mask detection model
mask_net = cv2.dnn.readNetFromTensorflow('mask_detection_model.pb')

# Load image
image = cv2.imread('image.jpg')

# Resize image
image = cv2.resize(image, None, fx=0.4, fy=0.4)

# Convert image to blob
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Pass blob through the network
net.setInput(blob)
outs = net.forward(output_layers)

# Get bounding boxes, confidences, and class IDs
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 0:  # Class ID 0 represents "person"
            # Get bounding box coordinates
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])

            # Convert center coordinates to top-left coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Save bounding box coordinates, confidences, and class IDs
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression to remove overlapping bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Iterate over detected objects
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = "No Mask"  # Default label

        # Crop face region
        face_img = image[y:y+h, x:x+w]

        # Convert face image to blob
        blob = cv2.dnn.blobFromImage(face_img, 1, (224, 224), (104, 177, 123))

        # Pass blob through mask detection network
        mask_net.setInput(blob)
        preds = mask_net.forward()

        # Get mask prediction
        if preds[0][0] > preds[0][1]:
            label = "No Mask"
            color = (0, 0, 255)  # Red color for "No Mask" label
        else:
            label = "Mask"
            color = (0, 255, 0)  # Green color for "Mask" label

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Display the image
cv2.imshow("Face Mask Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
