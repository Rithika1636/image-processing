import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


# =========================================================
# GLOBAL VARIABLES
# =========================================================

original_image = None
processed_image = None


# =========================================================
# OPERATOR KERNELS
# =========================================================

MEAN_KERNEL = np.ones((3, 3), dtype=float) / 9

GAUSSIAN_KERNEL = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=float) / 16

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=float)

PREWITT_X = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

PREWITT_Y = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=float)

LAPLACIAN_KERNEL = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=float)

SHARPEN_KERNEL = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=float)


# =========================================================
# MATRIX TO STRING
# =========================================================

def matrix_to_string(matrix):
    text = ""

    for row in matrix:
        for value in row:
            text += f"{float(value):8.2f} "
        text += "\n"

    return text


# =========================================================
# DISPLAY IMAGE
# =========================================================

def display_image(image, label):
    if image is None:
        return

    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(rgb)

    pil_image.thumbnail((430, 270))

    photo = ImageTk.PhotoImage(pil_image)

    label.config(image=photo)
    label.image = photo


# =========================================================
# GET CENTER 3x3 PIXEL MATRIX
# =========================================================

def get_pixel_matrix(gray):

    height, width = gray.shape

    if height < 3 or width < 3:
        resized = cv2.resize(gray, (3, 3))
        return resized.astype(float)

    center_y = height // 2
    center_x = width // 2

    matrix = gray[
        center_y - 1:center_y + 2,
        center_x - 1:center_x + 2
    ]

    return matrix.astype(float)


# =========================================================
# CALCULATE MATRIX
# =========================================================

def calculate_matrix(pixel_matrix, kernel):

    return np.sum(pixel_matrix * kernel)


# =========================================================
# SHOW MATRIX
# =========================================================

def show_matrix(operator_name, kernel, pixel_matrix, result_text):

    operator_label.config(
        text=f"OPERATOR: {operator_name}"
    )

    kernel_text.delete("1.0", tk.END)
    kernel_text.insert(
        tk.END,
        matrix_to_string(kernel)
    )

    pixel_text.delete("1.0", tk.END)
    pixel_text.insert(
        tk.END,
        matrix_to_string(pixel_matrix)
    )

    result_matrix_text.delete("1.0", tk.END)
    result_matrix_text.insert(
        tk.END,
        result_text
    )


# =========================================================
# CLEAR MATRIX
# =========================================================

def clear_matrix():

    operator_label.config(
        text="OPERATOR: -"
    )

    kernel_text.delete("1.0", tk.END)
    pixel_text.delete("1.0", tk.END)
    result_matrix_text.delete("1.0", tk.END)


# =========================================================
# UPLOAD IMAGE
# =========================================================

def upload_image():

    global original_image
    global processed_image

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    if not file_path:
        return

    original_image = cv2.imread(file_path)

    if original_image is None:
        messagebox.showerror(
            "Error",
            "Unable to read image."
        )
        return

    processed_image = original_image.copy()

    display_image(
        original_image,
        original_label
    )

    display_image(
        original_image,
        result_label
    )

    clear_matrix()

    status_label.config(
        text="Image uploaded successfully ✓"
    )


# =========================================================
# APPLY OPERATOR
# =========================================================

def apply_operator():

    global processed_image

    if original_image is None:

        messagebox.showwarning(
            "Warning",
            "Please upload an image first."
        )

        return

    operator = operator_var.get()

    gray = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2GRAY
    )

    pixel_matrix = get_pixel_matrix(gray)


    # =====================================================
    # MEAN FILTER
    # =====================================================

    if operator == "Mean Filter":

        processed_image = cv2.blur(
            original_image,
            (3, 3)
        )

        result = calculate_matrix(
            pixel_matrix,
            MEAN_KERNEL
        )

        result_text = (
            "Pixel × Kernel Calculation\n\n"
            f"Mean Result = {result:.2f}\n\n"
            "Formula:\n"
            "Sum(Pixel × Kernel)\n"
            "-------------------\n"
            "        9"
        )

        show_matrix(
            "Mean Filter",
            MEAN_KERNEL,
            pixel_matrix,
            result_text
        )


    # =====================================================
    # GAUSSIAN BLUR
    # =====================================================

    elif operator == "Gaussian Blur":

        processed_image = cv2.GaussianBlur(
            original_image,
            (3, 3),
            0
        )

        result = calculate_matrix(
            pixel_matrix,
            GAUSSIAN_KERNEL
        )

        result_text = (
            "Gaussian Matrix Calculation\n\n"
            f"Result = {result:.2f}\n\n"
            "Kernel Sum = 1.00\n"
            "Used for image smoothing."
        )

        show_matrix(
            "Gaussian Blur",
            GAUSSIAN_KERNEL,
            pixel_matrix,
            result_text
        )


    # =====================================================
    # SOBEL
    # =====================================================

    elif operator == "Sobel":

        sobel_x = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        sobel_y = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        processed_image = cv2.addWeighted(
            sobel_x,
            0.5,
            sobel_y,
            0.5,
            0
        )

        gx = calculate_matrix(
            pixel_matrix,
            SOBEL_X
        )

        gy = calculate_matrix(
            pixel_matrix,
            SOBEL_Y
        )

        magnitude = np.sqrt(
            gx ** 2 + gy ** 2
        )

        result_text = (
            "SOBEL MATRIX CALCULATION\n\n"
            f"Gx = {gx:.2f}\n"
            f"Gy = {gy:.2f}\n\n"
            f"Gradient Magnitude = {magnitude:.2f}\n\n"
            "Gx Kernel:\n"
            f"{matrix_to_string(SOBEL_X)}\n"
            "Gy Kernel:\n"
            f"{matrix_to_string(SOBEL_Y)}"
        )

        show_matrix(
            "Sobel Operator",
            SOBEL_X,
            pixel_matrix,
            result_text
        )


    # =====================================================
    # PREWITT
    # =====================================================

    elif operator == "Prewitt":

        px = cv2.filter2D(
            gray,
            cv2.CV_64F,
            PREWITT_X
        )

        py = cv2.filter2D(
            gray,
            cv2.CV_64F,
            PREWITT_Y
        )

        px = cv2.convertScaleAbs(px)
        py = cv2.convertScaleAbs(py)

        processed_image = cv2.addWeighted(
            px,
            0.5,
            py,
            0.5,
            0
        )

        gx = calculate_matrix(
            pixel_matrix,
            PREWITT_X
        )

        gy = calculate_matrix(
            pixel_matrix,
            PREWITT_Y
        )

        magnitude = np.sqrt(
            gx ** 2 + gy ** 2
        )

        result_text = (
            "PREWITT MATRIX CALCULATION\n\n"
            f"Gx = {gx:.2f}\n"
            f"Gy = {gy:.2f}\n\n"
            f"Gradient Magnitude = {magnitude:.2f}\n\n"
            "Gx Kernel:\n"
            f"{matrix_to_string(PREWITT_X)}\n"
            "Gy Kernel:\n"
            f"{matrix_to_string(PREWITT_Y)}"
        )

        show_matrix(
            "Prewitt Operator",
            PREWITT_X,
            pixel_matrix,
            result_text
        )


    # =====================================================
    # LAPLACIAN
    # =====================================================

    elif operator == "Laplacian":

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        processed_image = cv2.convertScaleAbs(
            laplacian
        )

        result = calculate_matrix(
            pixel_matrix,
            LAPLACIAN_KERNEL
        )

        result_text = (
            "LAPLACIAN CALCULATION\n\n"
            f"Result = {result:.2f}\n\n"
            "Kernel:\n"
            f"{matrix_to_string(LAPLACIAN_KERNEL)}\n"
            "Used for edge detection."
        )

        show_matrix(
            "Laplacian Operator",
            LAPLACIAN_KERNEL,
            pixel_matrix,
            result_text
        )


    # =====================================================
    # CANNY
    # =====================================================

    elif operator == "Canny":

        processed_image = cv2.Canny(
            gray,
            100,
            200
        )

        result_text = (
            "CANNY EDGE DETECTION\n\n"
            "Step 1: Gaussian Blur\n"
            "Step 2: Gradient Calculation\n"
            "Step 3: Non-Maximum Suppression\n"
            "Step 4: Double Threshold\n"
            "Step 5: Edge Tracking\n\n"
            "Lower Threshold = 100\n"
            "Upper Threshold = 200"
        )

        show_matrix(
            "Canny Edge Detection",
            GAUSSIAN_KERNEL,
            pixel_matrix,
            result_text
        )


    # =====================================================
    # SHARPENING
    # =====================================================

    elif operator == "Sharpening":

        processed_image = cv2.filter2D(
            original_image,
            -1,
            SHARPEN_KERNEL
        )

        result = calculate_matrix(
            pixel_matrix,
            SHARPEN_KERNEL
        )

        result_text = (
            "SHARPENING CALCULATION\n\n"
            f"Center Pixel Result = {result:.2f}\n\n"
            "Kernel:\n"
            f"{matrix_to_string(SHARPEN_KERNEL)}"
        )

        show_matrix(
            "Sharpening",
            SHARPEN_KERNEL,
            pixel_matrix,
            result_text
        )


    # =====================================================
    # THRESHOLDING
    # =====================================================

    elif operator == "Thresholding":

        threshold_value = 127

        _, processed_image = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )

        center_pixel = pixel_matrix[1, 1]

        if center_pixel > threshold_value:
            result = 255
        else:
            result = 0

        threshold_kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=float)

        result_text = (
            "BINARY THRESHOLD CALCULATION\n\n"
            f"Center Pixel = {center_pixel:.0f}\n"
            f"Threshold = {threshold_value}\n\n"
            f"Result = {result}\n\n"
            "Pixel > 127 → 255\n"
            "Pixel ≤ 127 → 0"
        )

        show_matrix(
            "Thresholding",
            threshold_kernel,
            pixel_matrix,
            result_text
        )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    display_image(
        processed_image,
        result_label
    )

    status_label.config(
        text=f"{operator} applied successfully ✓"
    )


# =========================================================
# SAVE RESULT
# =========================================================

def save_result():

    if processed_image is None:

        messagebox.showwarning(
            "Warning",
            "No processed image available."
        )

        return

    file_path = filedialog.asksaveasfilename(
        title="Save Processed Image",
        defaultextension=".jpg",
        filetypes=[
            ("JPEG Image", "*.jpg"),
            ("PNG Image", "*.png")
        ]
    )

    if file_path:

        cv2.imwrite(
            file_path,
            processed_image
        )

        messagebox.showinfo(
            "Success",
            "Processed image saved successfully!"
        )


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()

root.title(
    "Image Processing Operator Dashboard"
)

root.geometry(
    "1250x900"
)

root.minsize(
    1000,
    750
)

root.configure(
    bg="#202124"
)


# =========================================================
# TITLE
# =========================================================

title_label = tk.Label(
    root,
    text="IMAGE PROCESSING OPERATOR ",
    font=("Arial", 23, "bold"),
    bg="#202124",
    fg="white"
)

title_label.pack(
    pady=15
)


# =========================================================
# CONTROL FRAME
# =========================================================

control_frame = tk.Frame(
    root,
    bg="#202124"
)

control_frame.pack(
    pady=5
)


upload_button = tk.Button(
    control_frame,
    text="📁 UPLOAD IMAGE",
    font=("Arial", 11, "bold"),
    command=upload_image,
    padx=15,
    pady=8
)

upload_button.grid(
    row=0,
    column=0,
    padx=8
)


tk.Label(
    control_frame,
    text="Select Operator:",
    font=("Arial", 11, "bold"),
    bg="#202124",
    fg="white"
).grid(
    row=0,
    column=1,
    padx=5
)


operator_var = tk.StringVar()

operator_var.set(
    "Gaussian Blur"
)


operators = [
    "Mean Filter",
    "Gaussian Blur",
    "Sobel",
    "Prewitt",
    "Laplacian",
    "Canny",
    "Sharpening",
    "Thresholding"
]


operator_menu = tk.OptionMenu(
    control_frame,
    operator_var,
    *operators
)

operator_menu.config(
    width=18,
    font=("Arial", 11)
)

operator_menu.grid(
    row=0,
    column=2,
    padx=8
)


apply_button = tk.Button(
    control_frame,
    text="⚙ APPLY OPERATOR",
    font=("Arial", 11, "bold"),
    command=apply_operator,
    padx=15,
    pady=8
)

apply_button.grid(
    row=0,
    column=3,
    padx=8
)


# =========================================================
# IMAGE DISPLAY FRAME
# =========================================================

image_frame = tk.Frame(
    root,
    bg="#202124"
)

image_frame.pack(
    fill="x",
    padx=15,
    pady=10
)


# =========================================================
# ORIGINAL IMAGE
# =========================================================

original_box = tk.Frame(
    image_frame,
    bg="#303134",
    width=500,
    height=320
)

original_box.grid(
    row=0,
    column=0,
    padx=10
)

original_box.grid_propagate(False)


tk.Label(
    original_box,
    text="ORIGINAL IMAGE",
    font=("Arial", 14, "bold"),
    bg="#303134",
    fg="white"
).pack(
    pady=8
)


original_label = tk.Label(
    original_box,
    bg="#303134"
)

original_label.pack(
    expand=True
)


# =========================================================
# PROCESSED IMAGE
# =========================================================

result_box = tk.Frame(
    image_frame,
    bg="#303134",
    width=500,
    height=320
)

result_box.grid(
    row=0,
    column=1,
    padx=10
)

result_box.grid_propagate(False)


tk.Label(
    result_box,
    text="PROCESSED IMAGE",
    font=("Arial", 14, "bold"),
    bg="#303134",
    fg="white"
).pack(
    pady=8
)


result_label = tk.Label(
    result_box,
    bg="#303134"
)

result_label.pack(
    expand=True
)


# =========================================================
# MATRIX SECTION
# =========================================================

matrix_container = tk.Frame(
    root,
    bg="#202124"
)

matrix_container.pack(
    fill="x",
    padx=15,
    pady=5
)


operator_label = tk.Label(
    matrix_container,
    text="OPERATOR: -",
    font=("Arial", 14, "bold"),
    bg="#202124",
    fg="white"
)

operator_label.pack(
    pady=5
)


# IMPORTANT:
# This frame uses GRID only.
# All matrix boxes are children of this frame.

matrix_grid = tk.Frame(
    matrix_container,
    bg="#202124"
)

matrix_grid.pack(
    fill="x"
)

matrix_grid.columnconfigure(
    0,
    weight=1
)

matrix_grid.columnconfigure(
    1,
    weight=1
)

matrix_grid.columnconfigure(
    2,
    weight=1
)


# =========================================================
# KERNEL FRAME
# =========================================================

kernel_frame = tk.Frame(
    matrix_grid,
    bg="#303134"
)

kernel_frame.grid(
    row=0,
    column=0,
    padx=8,
    pady=5,
    sticky="nsew"
)


tk.Label(
    kernel_frame,
    text="KERNEL MATRIX",
    font=("Arial", 12, "bold"),
    bg="#303134",
    fg="white"
).pack(
    pady=5
)


kernel_text = tk.Text(
    kernel_frame,
    width=25,
    height=7,
    font=("Consolas", 11),
    bg="#181818",
    fg="white",
    insertbackground="white"
)

kernel_text.pack(
    padx=8,
    pady=8
)


# =========================================================
# PIXEL FRAME
# =========================================================

pixel_frame = tk.Frame(
    matrix_grid,
    bg="#303134"
)

pixel_frame.grid(
    row=0,
    column=1,
    padx=8,
    pady=5,
    sticky="nsew"
)


tk.Label(
    pixel_frame,
    text="PIXEL MATRIX",
    font=("Arial", 12, "bold"),
    bg="#303134",
    fg="white"
).pack(
    pady=5
)


pixel_text = tk.Text(
    pixel_frame,
    width=25,
    height=7,
    font=("Consolas", 11),
    bg="#181818",
    fg="white",
    insertbackground="white"
)

pixel_text.pack(
    padx=8,
    pady=8
)


# =========================================================
# RESULT FRAME
# =========================================================

result_matrix_frame = tk.Frame(
    matrix_grid,
    bg="#303134"
)

result_matrix_frame.grid(
    row=0,
    column=2,
    padx=8,
    pady=5,
    sticky="nsew"
)


tk.Label(
    result_matrix_frame,
    text="MATRIX RESULT",
    font=("Arial", 12, "bold"),
    bg="#303134",
    fg="white"
).pack(
    pady=5
)


result_matrix_text = tk.Text(
    result_matrix_frame,
    width=38,
    height=7,
    font=("Consolas", 10),
    bg="#181818",
    fg="white",
    insertbackground="white"
)

result_matrix_text.pack(
    padx=8,
    pady=8
)


# =========================================================
# SAVE BUTTON
# =========================================================

save_button = tk.Button(
    root,
    text="💾 SAVE PROCESSED IMAGE",
    font=("Arial", 11, "bold"),
    command=save_result,
    padx=20,
    pady=8
)

save_button.pack(
    pady=8
)


# =========================================================
# STATUS
# =========================================================

status_label = tk.Label(
    root,
    text="Upload an image to start processing",
    font=("Arial", 11),
    bg="#202124",
    fg="white"
)

status_label.pack(
    pady=5
)


# =========================================================
# START APPLICATION
# =========================================================

root.mainloop() 