import cv2
import numpy as np

# Load a pre-trained model from TensorFlow's model zoo
net = cv2.dnn_DetectionModel('models/ssd_mobilenet_v3_large_coco_2020_01_14.pb', 
                             'models/ssd_mobilenet_v3_large_coco.pbtxt')
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Label mapping; suppose each product type is mapped to certain labels
PRODUCT_LABELS = {
    73: "Bottle",  # example labels
    45: "Can",
    40: "Others"
}

def detect_items(image):
    classes, confidences, boxes = net.detect(image, confThreshold=0.5)
    return classes, confidences, boxes

def count_items(classes):
    item_count = {}
    for cl in classes:
        label = PRODUCT_LABELS.get(cl[0], "Unknown")
        item_count[label] = item_count.get(label, 0) + 1
    return item_count

def main():
    cap = cv2.VideoCapture(0)  # Using a webcam for demonstration
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect items on the shelf
        classes, confidences, boxes = detect_items(frame)

        # Count detected items
        item_count = count_items(classes)

        # Display results
        for (classid, conf, box) in zip(classes.flatten(), confidences.flatten(), boxes):
            label = PRODUCT_LABELS.get(classid, "Unknown")
            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {conf:.2f}', (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print("Current Stock:", item_count)  # Print current inventory level
        cv2.imshow("SmartShelf Inventory", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
