from pyzbar.pyzbar import decode
import cv2
import numpy as np

# Load the image
image_path = "image5.jpg"  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not loaded. Check the image path.")
    exit(1)

# Decode the Aztec code using pyzbar
decoded_objects = decode(image)

if decoded_objects:
    for obj in decoded_objects:
        print(f"Decoded Data: {obj.data.decode('utf-8')}")
        print(f"Barcode Format: {obj.type}")

        # Get points and draw bounding box
        points = obj.polygon
        if len(points) == 4:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Annotate with the decoded text
            x, y = points[0]
            cv2.putText(image, obj.data.decode('utf-8'), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the image with annotations
    cv2.imshow("Aztec Code with Annotation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the annotated image
    output_file = "decoded_aztec.png"
    cv2.imwrite(output_file, image)
    print(f"Annotated image saved as {output_file}")
else:
    print("Failed to decode the Aztec code.")
