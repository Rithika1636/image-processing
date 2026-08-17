import streamlit as st
import cv2
import numpy as np

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Image Processing Operator Dashboard",
    page_icon="🖼️",
    layout="wide"
)

# ---------------------------------------------------------
# KERNELS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def get_pixel_matrix(gray):

    h, w = gray.shape

    if h < 3 or w < 3:
        return cv2.resize(gray, (3, 3)).astype(float)

    cy = h // 2
    cx = w // 2

    return gray[
        cy - 1:cy + 2,
        cx - 1:cx + 2
    ].astype(float)


def calculate_matrix(pixel_matrix, kernel):
    return np.sum(pixel_matrix * kernel)


def show_matrix(matrix):
    return np.round(matrix, 2)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🖼️ IMAGE PROCESSING OPERATOR DASHBOARD")

st.write(
    "Upload an image and apply different image processing operators."
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ CONTROL PANEL")

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

operator = st.sidebar.selectbox(
    "Select Operator",
    [
        "Mean Filter",
        "Gaussian Blur",
        "Sobel",
        "Prewitt",
        "Laplacian",
        "Canny",
        "Sharpening",
        "Thresholding"
    ]
)


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

if uploaded_file is None:

    st.info("Please upload an image from the sidebar.")

    st.markdown("""
    ### Available Operators

    1. Mean Filter
    2. Gaussian Blur
    3. Sobel
    4. Prewitt
    5. Laplacian
    6. Canny
    7. Sharpening
    8. Thresholding
    """)

else:

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    original = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    pixel_matrix = get_pixel_matrix(gray)

    # Default values
    processed = original.copy()
    kernel = MEAN_KERNEL
    result_text = ""


    # -----------------------------------------------------
    # MEAN FILTER
    # -----------------------------------------------------

    if operator == "Mean Filter":

        kernel = MEAN_KERNEL

        processed = cv2.blur(
            original,
            (3, 3)
        )

        result = calculate_matrix(
            pixel_matrix,
            kernel
        )

        result_text = (
            "MEAN FILTER CALCULATION\n\n"
            f"Result = {result:.2f}\n\n"
            "Formula:\n"
            "Sum(Pixel × Kernel)"
        )


    # -----------------------------------------------------
    # GAUSSIAN BLUR
    # -----------------------------------------------------

    elif operator == "Gaussian Blur":

        kernel = GAUSSIAN_KERNEL

        processed = cv2.GaussianBlur(
            original,
            (3, 3),
            0
        )

        result = calculate_matrix(
            pixel_matrix,
            kernel
        )

        result_text = (
            "GAUSSIAN BLUR CALCULATION\n\n"
            f"Result = {result:.2f}\n\n"
            "Kernel Sum = 1.00\n"
            "Used for image smoothing."
        )


    # -----------------------------------------------------
    # SOBEL
    # -----------------------------------------------------

    elif operator == "Sobel":

        kernel = SOBEL_X

        gx_image = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        gy_image = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        gx_image = cv2.convertScaleAbs(gx_image)
        gy_image = cv2.convertScaleAbs(gy_image)

        processed = cv2.addWeighted(
            gx_image,
            0.5,
            gy_image,
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
            "SOBEL CALCULATION\n\n"
            f"Gx = {gx:.2f}\n"
            f"Gy = {gy:.2f}\n\n"
            f"Gradient Magnitude = "
            f"{magnitude:.2f}\n\n"
            "Gx Kernel:\n"
            f"{SOBEL_X}\n\n"
            "Gy Kernel:\n"
            f"{SOBEL_Y}"
        )


    # -----------------------------------------------------
    # PREWITT
    # -----------------------------------------------------

    elif operator == "Prewitt":

        kernel = PREWITT_X

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

        processed = cv2.addWeighted(
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
            "PREWITT CALCULATION\n\n"
            f"Gx = {gx:.2f}\n"
            f"Gy = {gy:.2f}\n\n"
            f"Gradient Magnitude = "
            f"{magnitude:.2f}"
        )


    # -----------------------------------------------------
    # LAPLACIAN
    # -----------------------------------------------------

    elif operator == "Laplacian":

        kernel = LAPLACIAN_KERNEL

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        processed = cv2.convertScaleAbs(
            laplacian
        )

        result = calculate_matrix(
            pixel_matrix,
            kernel
        )

        result_text = (
            "LAPLACIAN CALCULATION\n\n"
            f"Result = {result:.2f}\n\n"
            "Used for edge detection."
        )


    # -----------------------------------------------------
    # CANNY
    # -----------------------------------------------------

    elif operator == "Canny":

        kernel = GAUSSIAN_KERNEL

        processed = cv2.Canny(
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


    # -----------------------------------------------------
    # SHARPENING
    # -----------------------------------------------------

    elif operator == "Sharpening":

        kernel = SHARPEN_KERNEL

        processed = cv2.filter2D(
            original,
            -1,
            kernel
        )

        result = calculate_matrix(
            pixel_matrix,
            kernel
        )

        result_text = (
            "SHARPENING CALCULATION\n\n"
            f"Result = {result:.2f}"
        )


    # -----------------------------------------------------
    # THRESHOLDING
    # -----------------------------------------------------

    elif operator == "Thresholding":

        kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=float)

        threshold = 127

        _, processed = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        center_pixel = pixel_matrix[1, 1]

        result = (
            255
            if center_pixel > threshold
            else 0
        )

        result_text = (
            "THRESHOLD CALCULATION\n\n"
            f"Center Pixel = {center_pixel:.0f}\n"
            f"Threshold = {threshold}\n\n"
            f"Result = {result}\n\n"
            "Pixel > 127 → 255\n"
            "Pixel ≤ 127 → 0"
        )


    # -----------------------------------------------------
    # IMAGE DISPLAY
    # -----------------------------------------------------

    st.subheader(
        f"🔹 {operator}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### ORIGINAL IMAGE")

        original_rgb = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            original_rgb,
            use_container_width=True
        )

    with col2:

        st.markdown("### PROCESSED IMAGE")

        if len(processed.shape) == 2:

            st.image(
                processed,
                use_container_width=True
            )

        else:

            processed_rgb = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                processed_rgb,
                use_container_width=True
            )


    # -----------------------------------------------------
    # MATRIX DISPLAY
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        f"🔢 OPERATOR: {operator}"
    )

    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown("### KERNEL MATRIX")

        st.dataframe(
            show_matrix(kernel),
            hide_index=True,
            use_container_width=True
        )


    with c2:

        st.markdown("### PIXEL MATRIX")

        st.dataframe(
            show_matrix(pixel_matrix),
            hide_index=True,
            use_container_width=True
        )


    with c3:

        st.markdown("### MATRIX RESULT")

        st.code(
            result_text,
            language="text"
        )


    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    st.divider()

    success, encoded = cv2.imencode(
        ".png",
        processed
    )

    if success:

        st.download_button(
            "💾 DOWNLOAD PROCESSED IMAGE",
            encoded.tobytes(),
            "processed_image.png",
            "image/png"
        )

    st.success(
        f"{operator} applied successfully ✓"
    )
