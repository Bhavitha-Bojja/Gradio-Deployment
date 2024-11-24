import gradio as gr
from fastai.vision.all import *
from PIL import Image

# Load your trained model
learn = load_learner('/content/models/butterfly_moth_model.pkl')


# Define the prediction function
def predict_image(image):
    # Use the model to predict the label and probability
    is_butterfly, _, probs = learn.predict(image)
    return f"This is a: {is_butterfly}.", f"Probability it's a butterfly: {probs[0]:.4f}"

# Define the Gradio interface
interface = gr.Interface(fn=predict_image,
                         inputs=gr.Image(type='pil'),  # Image input
                         outputs=[gr.Text(), gr.Text()],  # Two text outputs
                         live=True)

# Launch the Gradio app
interface.launch()