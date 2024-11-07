from sklearn.cluster import DBSCAN
from utils.face_utils import load_images_from_subdirectories, extract_features, visualize_clusters

DATA_PATH = 'data/faces'
OUTPUT_PATH = 'output/clusters'

def main():
    images, image_paths = load_images_from_subdirectories(DATA_PATH)
    features = extract_features(images)
    clustering = DBSCAN(metric='euclidean', eps=1.0, min_samples=2).fit(features)
    labels = clustering.labels_
    
    visualize_clusters(labels, images, image_paths, DATA_PATH, OUTPUT_PATH)

if __name__ == "__main__":
    main()
