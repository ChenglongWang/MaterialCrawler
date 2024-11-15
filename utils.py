import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, TextBox
import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans, DBSCAN


class InputHandler:
    def __init__(self):
        self.target_labels = ""
        self.input_number1 = ""
        self.input_number2 = ""

    def submit_target_labels(self, text):
        self.target_labels = text

    def submit_number1(self, text):
        self.input_number1 = text
        print(f"Input number for point 1: {self.input_number1}")

    def submit_number2(self, text):
        self.input_number2 = text
        print(f"Input number for point 2: {self.input_number2}")


def rect_select_callback(eclick, erelease):
    """Callback function for RectangleSelector."""
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata


def toggle_selector(event):
    """Toggle the RectangleSelector on and off."""
    if event.key in ['Q', 'q', 'enter'] and toggle_selector.RS.active:
        print('RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print('RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


def normalize_image(image):
    """Normalize the image to range [0, 1]."""
    image = np.array(image, dtype=np.float64) / 255
    image = image[:, :, :3]
    return image


def quanti_color_fitting(image, kmeans):
    """Fit KMeans model to the image and return cluster centers and labels."""
    image = normalize_image(image)
    w, h, d = tuple(image.shape)
    assert d == 3
    image_array = np.reshape(image, (w * h, d))

    print("Fitting model on a small sub-sample of the data")
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans.fit(image_array_sample)

    print("Predicting color indices on the full image (k-means)")
    labels = kmeans.predict(image_array)
    cluster_centers = kmeans.cluster_centers_

    return cluster_centers, labels, w, h


def quanti_color(image, kmeans):
    """Predict color labels for the image using KMeans."""
    image = normalize_image(image)
    w, h, d = tuple(image.shape)
    assert d == 3
    image_array = np.reshape(image, (w * h, d))
    labels = kmeans.predict(image_array)
    return labels


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels."""
    return codebook[labels].reshape(w, h, -1)


def close_window(event):
    """Close the window on 'Enter' key press."""
    if event.key == 'enter':
        plt.close()


def adjust_start_end_points(start_pt, end_pt):
    """Adjust start and end points to ensure start_pt is above end_pt."""
    if start_pt[1] < end_pt[1]:
        start_pt, end_pt = end_pt, start_pt
    return start_pt, end_pt


def find_intersecton_points(image, line_coords, line_values):
    """Find intersection points of the curve and the lines."""
    gray_image = np.dot(image[..., :3]*255, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    curve_coords = np.argwhere(gray_image)
    rounded_line_coords = np.round(line_coords).astype(int)

    intersection_points = []
    for y_coord, y_value in zip(rounded_line_coords, line_values):
        mask = curve_coords[:, 0] == y_coord
        intersecting_coords = curve_coords[mask]
        for coord in intersecting_coords:
            intersection_points.append((coord[0], coord[1], y_value))

    return intersection_points


def fuse_points(points, eps=5, min_samples=4):
    """Fuse close points using DBSCAN clustering."""
    coords = np.array([(p[0], p[1]) for p in points])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    unique_labels = set(labels)
    fused_points = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = coords[labels == label]
        centroid = cluster_points.mean(axis=0)
        cluster_values = [points[i][2] for i in range(len(points)) if labels[i] == label]
        avg_value = np.mean(cluster_values)
        fused_points.append((centroid[0], centroid[1], avg_value))

    return fused_points