from ultralytics import YOLO
import gradio as gr

# Load the best model
model = YOLO(yolo_best.pt)  # make sure this file is in the same folder

def predict_image(image)
    results = model.predict(image)
    return results[0].plot()  # returns image with predictions

# Gradio interface
demo = gr.Interface(fn=predict_image, inputs=image, outputs=image)
demo.launch()
