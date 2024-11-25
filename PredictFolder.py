import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision.models.segmentation import FCN_ResNet50_Weights
preprocess = FCN_ResNet50_Weights.DEFAULT.transforms()

# Function to overlay segmentation on the original image
def overlay_segmentation(image, mask, alpha=0.5):
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask == 15] = [0, 255, 0] 
    overlay = cv2.addWeighted(image, alpha, mask_rgb, 1 - alpha, 0)
    return overlay

# Load model function
def load_model(model_path):
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model

# Preprocess image for the model
def preprocess_image(image, input_size=(512, 512)):
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Post-process the output mask
def postprocess_mask(output):
    mask = output['out'].squeeze(0).detach().cpu().numpy()  # Remove batch dimension
    mask = np.argmax(mask, axis=0)  # Get class index with max score
    return mask

# Predict segmentation for all images in the folder
def predict_and_save(model, image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        original_image = np.array(image)
        # Preprocess the image
        input_image = preprocess_image(image).to(device)
        # Predict the mask
        with torch.no_grad():
            output = model(input_image)
        # Post-process the output mask
        mask = postprocess_mask(output)
        # Overlay the segmentation mask onto the original image
        overlaid_image = overlay_segmentation(original_image, mask)
        # Save the result
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))
        print(f"Processed and saved: {output_path}")

# Main script
if __name__ == "__main__":
    model_path = 'L:/Jav Segmentation 2024/models/bestmodel.pt'
    image_folder = 'G:/jav folder/OutputFolder/image'
    output_folder = './PredictOutput'

    model = load_model(model_path)
    predict_and_save(model, image_folder, output_folder)
