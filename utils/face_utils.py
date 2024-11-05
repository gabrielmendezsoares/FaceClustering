import os
import cv2
import random
import torch
import numpy as np

def load_images_from_subdirectories(data_path):
    images = []
    image_paths = []
    
    for subdirectory in os.listdir(data_path):
        subdirectory_path = os.path.join(data_path, subdirectory)
        
        if os.path.isdir(subdirectory_path):
            for filename in os.listdir(subdirectory_path):
                image_path = os.path.join(subdirectory_path, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    images.append(image)
                    image_paths.append(image_path)
                else:
                    print(f'Falha ao carregar a imagem: {image_path}')
    
    combined = list(zip(images, image_paths))
    
    random.shuffle(combined)
    
    images, image_paths = zip(*combined)
    
    return list(images), list(image_paths)

def extract_features(images, batch_size=32):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.fc = torch.nn.Identity()
    
    model.eval()
    
    features = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch = [cv2.resize(image, (160, 160)) for image in batch]
        batch = np.array(batch).transpose((0, 3, 1, 2)) / 255.0
        batch_tensor = torch.tensor(batch).float()
        
        with torch.no_grad():
            batch_features = model(batch_tensor).numpy()
        
        features.extend(batch_features)
    
    return np.array(features)

def get_next_cluster_dir(output_path, label):
    counter = 1
    cluster_name = f'cluster_{label}_{counter}'
    cluster_dir = os.path.join(output_path, cluster_name)

    while os.path.exists(cluster_dir):
        cluster_dir = os.path.join(output_path, cluster_name)
        counter += 1
    
    return cluster_name, cluster_dir

def visualize_clusters(labels, images, image_paths, output_path):
    unique_labels = set(labels)
    
    for unique_label in unique_labels:
        cluster_name, cluster_dir = get_next_cluster_dir(output_path, unique_label)
        
        os.makedirs(cluster_dir, exist_ok=True)
        
        for i, label in enumerate(labels):
            if label == unique_label:
                image_path = os.path.join(cluster_dir, os.path.basename(image_paths[i]))
                
                cv2.imwrite(image_path, images[i])
        
        print(f'{cluster_name} - {len([l for l in labels if l == unique_label])} images saved in {cluster_dir}.')
