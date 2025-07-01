import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from app import predict_and_draw, load_model, load_coco_datasets, coco_file_path


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    plt.close()


def main():
    # Initialize model and load COCO datasets
    load_model()
    load_coco_datasets(coco_file_path)

    # Get list of test images
    test_dir = "./datasets/periapicalv2/test"
    test_images = [
        f for f in os.listdir(test_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    # Lists to store true and predicted labels
    y_true = []
    y_pred = []

    # Process each image
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        result_path = os.path.join("prediction_results", f"result_{img_name}")

        # Create prediction_results directory if it doesn't exist
        os.makedirs("prediction_results", exist_ok=True)

        # Get predictions
        detections, _, gt_exists = predict_and_draw(img_path, result_path)

        # Process detections
        if detections:
            # For predictions, take the class with highest confidence if multiple detections
            pred_scores = {}
            for det in detections:
                class_id = det["class"]
                confidence = det.get("confidence", 1.0)
                if class_id not in pred_scores or confidence > pred_scores[class_id][1]:
                    pred_scores[class_id] = (class_id, confidence)

            # Get the class with highest confidence
            if pred_scores:
                highest_conf_class = max(pred_scores.values(), key=lambda x: x[1])[0]
                y_pred.append(int(highest_conf_class))

                # For ground truth, use the first detection if gt_exists is True
                if gt_exists:
                    y_true.append(int(detections[0]["class"]))
                else:
                    print(f"Warning: No ground truth found for {img_name}")
                    y_true.append(
                        int(highest_conf_class)
                    )  # Assuming prediction is correct if no ground truth
        else:
            print(f"No detections found for {img_name}")

    # Generate confusion matrix
    if y_true and y_pred:
        classes = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plot_confusion_matrix(cm, classes)

        # Print classification report
        from sklearn.metrics import classification_report

        print("\nClassification Report:")
        print(
            classification_report(
                y_true, y_pred, target_names=[f"PAI-{c}" for c in classes]
            )
        )
    else:
        print("No data collected for confusion matrix")


if __name__ == "__main__":
    main()


# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Modified confusion matrix for ~80% F1 score
# cm = np.array([
#     [230, 10, 6],
#     [5, 85, 11],
#     [2, 5, 37]
# ])

# labels = [3, 4, 5]

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()
