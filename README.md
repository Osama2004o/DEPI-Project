# DEPI Project: Text-to-Image Fine-Tuning using LoRA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Diffusers%20%7C%20PEFT-orange)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

## 📌 Project Overview

This project demonstrates a parameter-efficient approach to fine-tuning a **Stable Diffusion** model on the **Flickr30k** dataset.

The goal was to specialize a large text-to-image model to better understand detailed image captions without the massive computational cost of full training. To achieve this, we utilized **LoRA (Low-Rank Adaptation)**, which allows us to train only a tiny fraction of the model's parameters while keeping the original "knowledge" frozen.

---

## 📂 Repository Structure

| File / Directory | Description |
| :--- | :--- |
| `lora-text-to-image.ipynb` | **The Training Core.** This notebook contains the entire pipeline: Data loading, Preprocessing, Model initialization, LoRA configuration, and the Training Loop. (Detailed below). |
| `loradir/` | **The Trained Adapters.** This folder stores the output of our training:<br>• `adapter_model.safetensors`: The learned Low-Rank matrices (Weights).<br>• `adapter_config.json`: The hyperparameters used (Rank, Alpha, Target Modules). |
| `app.py` | **Inference Interface.** A Python script that loads the base Stable Diffusion model, injects our trained weights from `loradir/`, and provides a user interface to generate images. |
| `README.md` | Project documentation. |

---

## 🛠️ The Training Pipeline (`lora-text-to-image.ipynb`)

This notebook serves as the "Laboratory" for the project. Here is a granular breakdown of the technical steps performed within it:

### 1. Environment Setup
We initiated the environment by installing the Hugging Face ecosystem libraries:
* `diffusers`: For handling the diffusion model pipeline.
* `transformers`: For the CLIP text encoder.
* `accelerate`: For hardware optimization.
* `peft`: Specifically for implementing the LoRA technique.

### 2. Data Preparation (Flickr30k)
* **Ingestion:** We loaded the Flickr30k dataset, which is renowned for its high-quality images paired with highly descriptive captions.
* **Cleaning:** We parsed the raw `results.csv` file, handling specific delimiter inconsistencies (the file uses `|` instead of standard commas) to create a clean Dataframe of Image paths and Captions.
* **Preprocessing Class (`Flickr30kDataset`):** We implemented a custom PyTorch Dataset class that performs:
    * *Resizing:* All images are resized to a standard 512x512 resolution.
    * *Normalization:* Pixel values are normalized to the range [-1, 1] to match the model's expected input.
    * *Tokenization:* Text captions are converted into embeddings using the CLIP Tokenizer.

### 3. Model Architecture & Freezing
We loaded the Stable Diffusion v1-4 pipeline, which consists of three main components:
1.  **VAE (Variational Autoencoder):** To compress images into latent space.
2.  **Text Encoder (CLIP):** To understand the semantic meaning of prompts.
3.  **UNet:** The noise predictor (the core component).

> **Crucial Step:** We applied `.requires_grad_(False)` to freeze the entire base model. This ensures we do not destroy the pre-trained weights during backpropagation, saving massive computational resources.

### 4. Injecting LoRA (The "Magic")
Instead of training the frozen UNet, we used the **PEFT** library to inject LoRA adapters.
* **Rank (r=4):** We set the rank of the update matrices to 4, which is extremely lightweight.
* **Target Modules:** We targeted the attention layers of the UNet (`to_k`, `to_q`, `to_v`, `to_out`).
* **Result:** This reduced the number of trainable parameters to **< 0.1%** of the total model size (approx. 800k parameters vs 1 billion), making training feasible on standard GPUs.

### 5. The Training Loop
We implemented a custom training loop where the following happens in each step:
1.  **Encode:** Input images are encoded into Latents using the VAE.
2.  **Noise:** Random Gaussian noise is added to the latents (Forward Diffusion).
3.  **Predict:** The UNet (equipped with LoRA) tries to predict the noise that was added, conditioned on the Text Embeddings.
4.  **Loss Calculation:** We calculate `MSELoss` (Mean Squared Error) between the actual noise and the predicted noise.
5.  **Backpropagation:** Gradients are calculated and applied only to the LoRA matrices ($A$ and $B$).

---

## 🧠 Scientific Concept

### Stable Diffusion vs. GANs
It is important to note that this is a **Latent Diffusion Model (LDM)**, not a GAN. Unlike GANs which rely on a Generator/Discriminator adversarial game, Diffusion models learn to generate data by reversing a gradual noise addition process, resulting in more stable training and diverse outputs.

### Mathematical Explanation of LoRA
Fine-tuning a dense neural network layer involves updating a weight matrix $W_0$ ($d \times k$).
LoRA approximates the weight update $\Delta W$ by decomposing it into two smaller matrices:

$$W_{new} = W_{frozen} + \Delta W = W_{frozen} + (B \times A)$$

Where:
* $B \in \mathbb{R}^{d \times r}$ initialized with zeros.
* $A \in \mathbb{R}^{r \times k}$ initialized with random Gaussian noise.
* Rank $r \ll \min(d, k)$ (e.g., $r=4$).

This structure ensures the training starts exactly at the pre-trained model's performance level and adapts gradually.

---

## 🚀 Usage

### 1. Installation
Ensure you have the required dependencies installed:
```bash
pip install diffusers transformers accelerate peft torch pandas matplotlib
```

### 2. Running the Interface
To launch the application and generate images using the trained LoRA weights:

```bash
python app.py
```

Open the local URL provided in the terminal to interact with the model.

---

## 🤝 Credits
* **Project Team:** DEPI Project Team
* **Dataset:** Flickr30k
* **Base Model:** CompVis/stable-diffusion-v1-4
