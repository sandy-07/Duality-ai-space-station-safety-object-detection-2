from ultralytics import YOLO
import gradio as gr
from pyngrok import ngrok

# Set your ngrok authtoken
ngrok.set_auth_token("330EwoQIWcrlN3b2mBzodyhsgGR_7qXQxGbaEHWdCxxkHvLFL")  # <-- replace with your token

# Load the trained YOLO model
model = YOLO("yolo_best.pt")

# Prediction function
def predict_image(image):
    results = model.predict(image)
    return results[0].plot()

# Create Gradio interface
demo = gr.Interface(fn=predict_image, inputs="image", outputs="image")

# Open a public URL using ngrok
public_url = ngrok.connect(7860)
print("Public URL:", public_url)

# Launch Gradio app
demo.launch(server_name="0.0.0.0", server_port=7860)
