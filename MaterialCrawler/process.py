import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, TextBox
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from utils import (rect_select_callback, toggle_selector, normalize_image, quanti_color_fitting, 
                   quanti_color, recreate_image, close_window, find_intersecton_points, 
                   fuse_points, InputHandler)


def split_curves(png_filename="plot1.png"):
    """Split the curves in the image based on selected ROI and initial cluster centers."""
    input_handler = InputHandler()

    fig, ax = plt.subplots()
    try:
        original_image = Image.open(png_filename)
    except FileNotFoundError:
        print(f"File {png_filename} not found.")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    ax.imshow(original_image)
    ax.set_title("Select a ROI on the image, and press 'Q' to confirm")
    ax.axis('off')

    toggle_selector.RS = RectangleSelector(ax, rect_select_callback,
                                           useblit=True,
                                           button=[1],  # left mouse button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    # Retrieve rectangle extents directly
    if not hasattr(toggle_selector, 'RS') or not toggle_selector.RS.extents:
        print("No ROI selected.")
        return None

    x1, x2, y1, y2 = toggle_selector.RS.extents
    if x1 == x2 or y1 == y2:
        print("Invalid ROI selected.")
        return None
    coords = [(x1, y1), (x2, y2)]
    print(f"Selected rectangle coordinates: {coords}")

    # Crop the image using the selected coordinates
    cropped_image = original_image.crop((x1, y1, x2, y2))
    fig, ax = plt.subplots()
    ax.imshow(cropped_image)
    ax.set_title("Select initial colors by clicking on the image, and press 'Enter' to confirm")
    ax.axis('off')
    initial_centers = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    if len(initial_centers) == 0:
        print("No initial centers selected.")
        return None

    try:
        initial_pixels = [(cropped_image.getpixel((x, y))[0]/255, cropped_image.getpixel((x, y))[1]/255, cropped_image.getpixel((x, y))[2]/255) for x, y in initial_centers]
    except Exception as e:
        print(f"Error processing initial centers: {e}")
        return None

    k = len(initial_centers)
    print("initial_pixels: ", initial_pixels)
    kmeans = KMeans(n_clusters=k, random_state=0, init=initial_pixels)

    fig, axs = plt.subplots(1, k)
    axbox = plt.axes([0.2, 0.1, 0.7, 0.05])
    text_box = TextBox(axbox, 'Labels: ', initial=" ".join([str(i) for i in range(1, k)]))
    text_box.on_submit(input_handler.submit_target_labels)
    fig.canvas.mpl_connect('key_press_event', close_window)

    centers, labels, w, h = quanti_color_fitting(cropped_image, kmeans)
    cropped_quant_image = recreate_image(centers, labels, w, h)
    for i in range(k):
        masked_image = np.zeros_like(cropped_quant_image)
        label_image = labels.reshape(w, h)
        masked_image[label_image == i] = cropped_quant_image[label_image == i]

        axs[i].imshow(masked_image)
        axs[i].set_title(f"{i}")
        axs[i].axis('off')

    plt.show()

    if input_handler.target_labels == "all":
        print("Use all labels as target labels!!")
        target_labels = [i for i in range(k)]
    elif input_handler.target_labels == "":
        print("No target labels selected.")
        return None
    else:
        target_labels = [int(label.strip()) for label in input_handler.target_labels.strip().split()]
        print("Target labels: ", target_labels, type(target_labels))

    curve_seeds = [centers[label] for label in target_labels]

    labels = quanti_color(original_image, kmeans)
    original_image_np = np.array(original_image)
    w, h = original_image_np.shape[0], original_image_np.shape[1]
    new_original_image = recreate_image(centers, labels, w, h)

    filtered_images = []
    fig, axs = plt.subplots(2, len(curve_seeds)//2)
    for idx, seed in enumerate(curve_seeds):
        filtered_image = np.zeros_like(new_original_image)
        mask = np.all(new_original_image == seed, axis=-1)
        filtered_image[mask] = new_original_image[mask]
        filtered_images.append(filtered_image)
        i, j = idx // 2, idx % 2
        axs[j, i].imshow(filtered_image)
        axs[j, i].axis('off')

    plt.show()
    return filtered_images if filtered_images else None


def draw_mask(png_filename='plot1.png'):
    """Draw a mask on the image and return the vertices."""
    fig, ax = plt.subplots()
    try:
        original_image = Image.open(png_filename)
    except FileNotFoundError:
        print(f"File {png_filename} not found.")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    ax.imshow(original_image)
    ax.axis('off')
    ax.set_title("Draw a mask on the image, and press 'Q' to confirm")

    vertices = []

    def onclick_polygon(event):
        if event.inaxes:
            vertices.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')
            if len(vertices) > 1:
                ax.plot([vertices[-2][0], vertices[-1][0]], [vertices[-2][1], vertices[-1][1]], 'r-')
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick_polygon)
    plt.show()

    if len(vertices) < 3:
        print("Not enough vertices to form a polygon.")
        return None

    fig, ax = plt.subplots()
    ax.imshow(original_image)
    ax.axis('off')
    ax.set_title("Mask")

    polygon = patches.Polygon(vertices, closed=True, edgecolor='r', facecolor='none')
    ax.add_patch(polygon)
    plt.show()
    polygon_path = Path(vertices)

    return vertices


def draw_lines(png_filename, start_y, end_y, n_lines=50):
    """Draw horizontal lines on the image and return their coordinates and values."""
    input_handler = InputHandler()

    try:
        original_image = Image.open(png_filename)
    except FileNotFoundError:
        print(f"File {png_filename} not found.")
        return None, None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None

    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap='gray')
    ax.axis('off')
    ax.set_title("Select y1, y2, x1 ,x2 points on the image!")
    points = plt.ginput(4)
    if len(points) != 4:
        print("4 reference points were not selected.")
        return None, None
    point_y1, point_y2, point_x1, point_x2 = points
    print(f"Selected points: {point_y1}, {point_y2}, {point_x1}, {point_x2}")

    ax.plot(point_y1[0], point_y1[1], 'ro')
    ax.plot(point_y2[0], point_y2[1], 'bo')
    ax.plot(point_x1[0], point_x1[1], 'yo')
    ax.plot(point_x2[0], point_x2[1], 'go')

    axbox1 = plt.axes([0.3, 0.05, 0.1, 0.05])
    text_box1 = TextBox(axbox1, 'Point1:', hovercolor='red')
    text_box1.on_submit(input_handler.submit_y1)

    axbox2 = plt.axes([0.5, 0.05, 0.1, 0.05])
    text_box2 = TextBox(axbox2, 'Point2:', hovercolor='b')
    text_box2.on_submit(input_handler.submit_y2)

    axbox3 = plt.axes([0.7, 0.05, 0.1, 0.05])
    text_box3 = TextBox(axbox3, 'Point3:', hovercolor='y')
    text_box3.on_submit(input_handler.submit_x1)

    axbox4 = plt.axes([0.9, 0.05, 0.1, 0.05])
    text_box4 = TextBox(axbox4, 'Point4:', hovercolor='g')
    text_box4.on_submit(input_handler.submit_x2)

    fig.canvas.mpl_connect('key_press_event', close_window)
    plt.show()

    y_coords = np.linspace(start_y, end_y, n_lines)

    try:
        value1 = float(input_handler.y1)
        value2 = float(input_handler.y2)
    except ValueError:
        print("Invalid numeric inputs for reference points.")
        return None, None

    y1, y2 = point_y1[1], point_y2[1]

    m = (value2 - value1) / (y2 - y1)
    b = value1 - m * y1

    value_start = m * start_y + b
    value_end = m * end_y + b

    y_values = np.linspace(value_start, value_end, n_lines)
    print("y_values: ", value_start, value_end)

    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap='gray')
    ax.plot(point_y1[0], point_y1[1], 'ro')
    ax.text(point_y1[0], point_y1[1], f'{input_handler.y1}', color='r', fontsize=15, ha='left')
    ax.plot(point_y2[0], point_y2[1], 'ro')
    ax.text(point_y2[0], point_y2[1], f'{input_handler.y2}', color='r', fontsize=15, ha='left')
    ax.axis('off')

    for y, value in zip(y_coords, y_values):
        ax.axhline(y=y, color='yellow', linestyle='-', lw=0.5)
        ax.text(point_y1[0], y, f'{value:.2f}', color='black', fontsize=6, ha='right')

    plt.show()
    return y_coords, y_values


def final_adjust_points(filtered_images, line_coords, line_values, polygon_mask: Path):
    """Allow user to manually adjust intersection points."""
    if not filtered_images or not line_coords.any() or not line_values.any() or not polygon_mask:
        print("Invalid inputs to final_adjust_points.")
        return

    def on_modify(event):
        if event.inaxes:
            if event.button == 1:
                masked_intersection_points.append((event.ydata, event.xdata, -1))
                ax.plot(event.xdata, event.ydata, 'g+')
                plt.draw()
            elif event.button == 3:
                if masked_intersection_points:
                    distances = [np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2) for y, x, _ in masked_intersection_points]
                    min_dist_index = distances.index(min(distances))
                    if min(distances) < 10:
                        del masked_intersection_points[min_dist_index]
                        print("Point removed!")
                        redraw()

    def redraw():
        ax.clear()
        ax.imshow(filtered_image)
        ax.axis('off')
        for y, x, _ in masked_intersection_points:
            ax.plot(x, y, 'r+', markersize=4)
        plt.draw()

    for idx, filtered_image in enumerate(filtered_images):
        if filtered_image is None:
            print(f"Filtered image at index {idx} is invalid.")
            continue

        print("Processing Curve: ", idx, end=", ")
        intersection_points = find_intersecton_points(filtered_image, line_coords, line_values)
        intersection_points = fuse_points(intersection_points)

        fig, ax = plt.subplots()
        ax.imshow(filtered_image)
        ax.axis('off')

        masked_intersection_points = []
        for point in intersection_points:
            if polygon_mask.contains_point((point[1], point[0])):
                ax.plot(point[1], point[0], 'r+', markersize=4)
                masked_intersection_points.append(point)

        print("Num Intersection points: ", len(masked_intersection_points))
        cid = fig.canvas.mpl_connect('button_press_event', on_modify)
        plt.show()
        print("Final Num Intersection points: ", len(masked_intersection_points))


if __name__ == "__main__":
    filename = "plot1.png"
    filtered_images = split_curves(filename)
    if not filtered_images:
        print("Failed to split curves.")
        exit(1)

    vertices = draw_mask(filename)
    if not vertices:
        print("Failed to draw mask.")
        exit(1)

    y_coords = [vertex[1] for vertex in vertices]
    max_y, min_y = max(y_coords), min(y_coords)
    line_coords, line_values = draw_lines(filename, max_y, min_y, n_lines=60)
    if line_coords is None or line_values is None:
        print("Failed to draw lines.")
        exit(1)

    polygon_mask = Path(vertices)
    final_adjust_points(filtered_images, line_coords, line_values, polygon_mask)
