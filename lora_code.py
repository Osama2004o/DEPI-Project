import os
import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import os
import pandas as pd

# 1. Define the correct paths based on your screenshots
# The images are nested: input -> dataset -> flickr30k_images -> flickr30k_images
dataset_root = "/kaggle/input/flickr-image-dataset"
images_path = os.path.join(dataset_root, "flickr30k_images", "flickr30k_images")

# The captions file is named 'results.csv' and is inside that same folder
captions_file = os.path.join(images_path, "results.csv")

print(f"Images path: {images_path}")
print(f"Captions file: {captions_file}")

# 2. Read the captions file
# This specific 'results.csv' usually uses a pipe '|' delimiter
try:
    captions = pd.read_csv(captions_file, delimiter="|")
    # Sometimes the column names have extra spaces, let's fix that
    captions.columns = [col.strip() for col in captions.columns]
    
    # Rename columns to match our code standard ['image', 'comment_number', 'caption']
    # The file usually has: image_name, comment_number, comment
    captions = captions.rename(columns={
        'image_name': 'image', 
        'comment_number': 'comment_number', 
        'comment': 'caption'
    })
    
except Exception as e:
    print(f"First attempt failed: {e}")
    # Fallback if it's a standard comma-separated file
    captions = pd.read_csv(captions_file)

# 3. Clean and Verify
print(f"Total captions: {len(captions)}")
display(captions.head())

# Check if the first image actually exists to be sure
first_image_name = captions.iloc[0]['image']
first_image_path = os.path.join(images_path, first_image_name)
if os.path.exists(first_image_path):
    print(f"SUCCESS: Found image at {first_image_path}")
else:
    print(f"WARNING: Could not find {first_image_path}. Check paths again.")


# Model ID
model_id = "CompVis/stable-diffusion-v1-4"

# Load Tokenizer
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# Define Image Transforms (Resize to 512x512, Normalize to [-1, 1])
train_transforms = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

class Flickr30kDataset(Dataset):
    def __init__(self, dataframe, image_root, tokenizer, transform):
        self.dataframe = dataframe
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row['image']
        caption = str(row['caption']) # Ensure string
        
        img_path = os.path.join(self.image_root, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)
        except Exception as e:
            # Handle corrupt images by returning a blank tensor (or you could skip)
            print(f"Error loading {img_path}: {e}")
            pixel_values = torch.zeros((3, 512, 512))

        # Tokenize caption
        inputs = self.tokenizer(
            caption, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0)
        }

# Create Dataset and Dataloader
train_dataset = Flickr30kDataset(captions, images_path, tokenizer, train_transforms)

# Batch size 1 or 2 is usually best for free GPUs when fine-tuning
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
print("Dataset prepared.")

# 1. Load Scheduler and Models
noise_scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

# 2. Freeze all models
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# 3. Configure LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    r=4,                 # Rank (lower = fewer params, typically 4, 8, or 16)
    lora_alpha=4,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Apply to attention layers
    lora_dropout=0.01,
    bias="none",
)

# 4. Add LoRA adapters to UNet
unet = get_peft_model(unet, lora_config)

# Print trainable parameters to verify we are NOT fully fine-tuning
unet.print_trainable_parameters()

# Move models to GPU
vae.to(device)
text_encoder.to(device)
unet.to(device)

# 1. Load Base Pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# 2. Load and Apply your LoRA Adapters
# FIX: Use the pipeline's built-in method to load the LoRA weights
try:
    pipe.load_lora_weights(output_dir)
    print("LoRA adapters loaded successfully.")
except Exception as e:
    print(f"Error loading LoRA: {e}")
    # Fallback: If using older diffusers version, sometimes we need to specificy the weight name
    # pipe.load_lora_weights(output_dir, weight_name="adapter_model.bin") 

# 3. Generate Image
prompt = "man run on grass"
try:
    image = pipe(prompt, num_inference_steps=30).images[0]
    
    # Save and Display
    save_path = "/kaggle/working/generated_image_finetuned.png"
    image.save(save_path)
    
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    print(f"Image saved to {save_path}")

except Exception as e:
    print(f"Error generating image: {e}")