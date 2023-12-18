import streamlit as st
import cv2
import time
import tensorflow as tf
import os

model_path = str(os.path.abspath('../model/fire_classifier.h5').replace('\\', '/'))

model = tf.keras.models.load_model(model_path)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    return img

def main():
    st.title("Fire Ditection System")

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Create a placeholder for the live camera feed
    camera_placeholder = st.empty()

    # Create a placeholder for the prediction text
    prediction_placeholder = st.empty()

    while True:
        _, frame = cap.read()

        # Preprocess the frame for model prediction
        processed_frame = preprocess_image(frame)

        # Predict using the loaded model
        prediction = model.predict(processed_frame)
        # Update the placeholders
        camera_placeholder.image(frame, channels="BGR", caption="Live Camera Feed", use_column_width=True)
        if prediction > 0.6:
            prediction_text = 'fire'
        else:
            prediction_text = 'not fire'
        prediction_placeholder.text(prediction_text)

        # Add a short sleep for smoother display
        time.sleep(0.1)

if __name__ == "__main__":
    main()
