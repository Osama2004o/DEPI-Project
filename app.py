import torch
import streamlit as st
from contextlib import nullcontext
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
LORA_DIR = "/teamspace/studios/this_studio/loradir"

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    # Load LoRA weights you trained
    try:
        pipe.load_lora_weights(LORA_DIR)
        st.success("LoRA adapters loaded successfully.")
    except Exception as e:
        st.warning(f"Error loading LoRA: {e}")
    pipe.to(DEVICE)
    return pipe

def main():
    st.title("Kaggle – LoRA Stable Diffusion Demo")
    st.write("Text-to-image using your fine-tuned LoRA model.")

    prompt = st.text_input("Prompt", "man run on grass")
    num_steps = st.slider("Inference steps", 10, 60, 30)
    guidance_scale = st.slider("Guidance scale", 1.0, 15.0, 7.5)

    if st.button("Generate"):
        pipe = load_pipeline()
        with st.spinner("Generating image..."):
            context = torch.autocast("cuda") if DEVICE == "cuda" else nullcontext()
            with context:
                # The pipeline returns images directly, not a dict with return_dict=True
                image = pipe(
                    prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                ).images[0]

        st.image(image, caption=prompt)

if __name__ == "__main__":
    main()