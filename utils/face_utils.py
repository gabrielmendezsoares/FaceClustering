import os
import cv2
import random
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.preprocessing import StandardScaler

def load_images_from_subdirectories(data_path):
    images = []
    image_paths = []
    
    for subdirectory in os.listdir(data_path):
        subdirectory_path = os.path.join(data_path, subdirectory)
        
        if os.path.isdir(subdirectory_path):
            for filename in os.listdir(subdirectory_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_path = os.path.join(subdirectory_path, filename)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        images.append(image)
                        image_paths.append(image_path)
                    else:
                        print(f'Falha ao carregar a imagem: {image_path}')
    
    return images, image_paths

def extract_features(images, batch_size=32):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
   
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch = [preprocess(image) for image in batch]
        batch_tensor = torch.stack(batch)
        
        with torch.no_grad():
            batch_features = model(batch_tensor).numpy()
        
        features.extend(batch_features)
    
    features = StandardScaler().fit_transform(features)

    return np.array(features)

def visualize_clusters(labels, images, image_paths, data_path, output_path):
    unique_labels = set(labels)
    
    for unique_label in unique_labels:
        label_indices = [i for i, label in enumerate(labels) if label == unique_label]
        
        for i in label_indices:
            original_path = image_paths[i]
            relative_path = os.path.relpath(original_path, start=data_path)
            cluster_subfolder = os.path.join(output_path, f'cluster_{unique_label}', os.path.dirname(relative_path))

            os.makedirs(cluster_subfolder, exist_ok=True)
            
            output_image_path = os.path.join(cluster_subfolder, os.path.basename(image_paths[i]))
            cv2.imwrite(output_image_path, images[i])
        
        print(f'cluster_{unique_label} - {len(label_indices)} imagens salvas.')
