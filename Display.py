import cv2
import tensorflow as tf
import numpy as np

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained model
save_model = tf.keras.models.load_model("FaceRecognition.h5")

# Function to process an image
def process_image(filename):
    image = cv2.imread(filename)
    if image is None:
        print(f"Cannot read image: {filename}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    fontface = cv2.FONT_HERSHEY_SIMPLEX

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (100, 100))
        roi_gray = roi_gray.reshape((100, 100, 1)).astype("float32") / 255.0

        result = save_model.predict(np.array([roi_gray]))
        final = np.argmax(result)

        if final == 0:
            cv2.putText(image, "Tan", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
        elif final == 1:
            cv2.putText(image, "Hung Anh", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
        elif final == 2:
            cv2.putText(image, "Tan An", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to use webcam
def use_webcam():
    cap = cv2.VideoCapture(0)
    fontface = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (100, 100))
            roi_gray = roi_gray.reshape((100, 100, 1)).astype("float32") / 255.0

            result = save_model.predict(np.array([roi_gray]))
            final = np.argmax(result)

            if final == 0:
                cv2.putText(frame, "Tan", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
            elif final == 1:
                cv2.putText(frame, "Hung Anh", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
            elif final == 2:
                cv2.putText(frame, "Tan An", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    choice = input("Chọn tùy chọn (1: Sử dụng ảnh, 2: Sử dụng camera): ")

    if choice == '1':
        image_file = input("Nhập đường dẫn tệp ảnh: ")
        process_image(image_file)
    elif choice == '2':
        use_webcam()
    else:
        print("Tùy chọn không hợp lệ.")