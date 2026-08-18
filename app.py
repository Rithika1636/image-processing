import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image Processing ",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #6C2BD9;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555555;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: bold;
    color: #E91E63;
    border-bottom: 3px solid #E91E63;
    padding-bottom: 5px;
    margin-top: 20px;
}

.info-box {
    background-color: #EEF4FF;
    border-left: 6px solid #4A90E2;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}

.about-box {
    background-color: #FFF4E6;
    border-left: 6px solid #FF9800;
    padding: 15px;
    border-radius: 10px;
}

.matrix-box {
    background-color: #F3E5F5;
    border-left: 6px solid #9C27B0;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🖼️ IMAGE PROCESSING </div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Interactive Image Processing using Python, OpenCV, Pillow and NumPy</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Processing Controls")

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)


# ============================================================
# FILTER LIST
# ============================================================

filter_options = [

    "Original",

    "Grayscale",

    "Mean Filter",

    "Gaussian Filter",

    "Median Filter",

    "Blur Filter",

    "Sharpening",

    "Sobel Edge",

    "Prewitt Edge",

    "Laplacian Edge",

    "Canny Edge",

    "Threshold",

    "Adaptive Threshold",

    "Erosion",

    "Dilation",

    "Opening",

    "Closing",

    "Brightness",

    "Contrast",

    "HSV"
]


selected_filter = st.sidebar.selectbox(
    "🎨 Select Image Processing Operation",
    filter_options
)


# ============================================================
# KERNEL SIZE
# ============================================================

kernel_size = st.sidebar.selectbox(
    "🔢 Kernel Size",
    [3, 5, 7]
)


# ============================================================
# SLIDERS
# ============================================================

brightness_value = st.sidebar.slider(
    "☀️ Brightness",
    -100,
    100,
    0
)

contrast_value = st.sidebar.slider(
    "🌈 Contrast",
    0.5,
    3.0,
    1.0,
    0.1
)


# ============================================================
# ABOUT INFORMATION
# ============================================================

filter_information = {

    "Original": {
        "about": "Displays the original uploaded image without applying any processing.",
        "process": "The input image is directly displayed as it is."
    },

    "Grayscale": {
        "about": "Converts a colour image into a single-channel grayscale image.",
        "process": "Each pixel is converted from RGB/BGR representation into an intensity value."
    },

    "Mean Filter": {
        "about": "Mean filtering is a smoothing technique used to reduce noise.",
        "process": "A kernel is moved over the image and the average value of neighbouring pixels is calculated."
    },

    "Gaussian Filter": {
        "about": "Gaussian filtering smooths an image while giving higher importance to nearby pixels.",
        "process": "A Gaussian kernel is convolved with the image to reduce noise and smooth edges."
    },

    "Median Filter": {
        "about": "Median filtering is useful for removing salt-and-pepper noise.",
        "process": "The neighbouring pixel values are sorted and the middle value is assigned to the center pixel."
    },

    "Blur Filter": {
        "about": "Blur filtering reduces image details and high-frequency noise.",
        "process": "Neighbouring pixels are averaged to produce a smoother image."
    },

    "Sharpening": {
        "about": "Sharpening enhances edges and fine details in an image.",
        "process": "A sharpening kernel is applied to increase differences between neighbouring pixels."
    },

    "Sobel Edge": {
        "about": "Sobel operator detects edges by calculating image intensity gradients.",
        "process": "Horizontal and vertical Sobel kernels are applied and combined to detect strong edges."
    },

    "Prewitt Edge": {
        "about": "Prewitt operator is an edge detection technique.",
        "process": "Prewitt horizontal and vertical kernels calculate intensity changes in the X and Y directions."
    },

    "Laplacian Edge": {
        "about": "Laplacian detects edges using the second derivative of the image.",
        "process": "The Laplacian kernel identifies rapid changes in pixel intensity."
    },

    "Canny Edge": {
        "about": "Canny is a multi-stage edge detection algorithm.",
        "process": "It performs noise reduction, gradient calculation, non-maximum suppression and hysteresis thresholding."
    },

    "Threshold": {
        "about": "Thresholding converts a grayscale image into a binary image.",
        "process": "Pixels above a selected threshold become white and pixels below it become black."
    },

    "Adaptive Threshold": {
        "about": "Adaptive thresholding works well when illumination is not uniform.",
        "process": "A different threshold value is calculated for different local regions of the image."
    },

    "Erosion": {
        "about": "Erosion removes pixels from object boundaries.",
        "process": "A kernel moves across the image and shrinks bright regions."
    },

    "Dilation": {
        "about": "Dilation expands bright regions in an image.",
        "process": "The kernel increases the size of foreground objects."
    },

    "Opening": {
        "about": "Opening is useful for removing small objects and noise.",
        "process": "Opening performs erosion followed by dilation."
    },

    "Closing": {
        "about": "Closing is useful for filling small holes and gaps.",
        "process": "Closing performs dilation followed by erosion."
    },

    "Brightness": {
        "about": "Brightness adjustment makes the image lighter or darker.",
        "process": "A constant value is added or subtracted from the image pixel intensities."
    },

    "Contrast": {
        "about": "Contrast adjustment changes the difference between dark and bright regions.",
        "process": "Pixel values are multiplied by a contrast factor."
    },

    "HSV": {
        "about": "HSV represents colour using Hue, Saturation and Value.",
        "process": "The BGR/RGB image is converted into the HSV colour space."
    }
}


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_image(image, operation, ksize):

    img = image.copy()

    kernel = np.ones((ksize, ksize), np.uint8)

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    if operation == "Original":
        return img

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    elif operation == "Grayscale":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return gray

    # --------------------------------------------------------
    # MEAN FILTER
    # --------------------------------------------------------

    elif operation == "Mean Filter":

        result = cv2.blur(img, (ksize, ksize))

        return result

    # --------------------------------------------------------
    # GAUSSIAN FILTER
    # --------------------------------------------------------

    elif operation == "Gaussian Filter":

        result = cv2.GaussianBlur(
            img,
            (ksize, ksize),
            0
        )

        return result

    # --------------------------------------------------------
    # MEDIAN FILTER
    # --------------------------------------------------------

    elif operation == "Median Filter":

        result = cv2.medianBlur(
            img,
            ksize
        )

        return result

    # --------------------------------------------------------
    # BLUR FILTER
    # --------------------------------------------------------

    elif operation == "Blur Filter":

        result = cv2.blur(
            img,
            (ksize, ksize)
        )

        return result

    # --------------------------------------------------------
    # SHARPENING
    # --------------------------------------------------------

    elif operation == "Sharpening":

        sharpening_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        result = cv2.filter2D(
            img,
            -1,
            sharpening_kernel
        )

        return result

    # --------------------------------------------------------
    # SOBEL
    # --------------------------------------------------------

    elif operation == "Sobel Edge":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        sobel_x = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=ksize
        )

        sobel_y = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=ksize
        )

        magnitude = cv2.magnitude(
            sobel_x.astype(np.float32),
            sobel_y.astype(np.float32)
        )

        result = cv2.convertScaleAbs(
            magnitude
        )

        return result

    # --------------------------------------------------------
    # PREWITT
    # --------------------------------------------------------

    elif operation == "Prewitt Edge":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        prewitt_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32)

        prewitt_y = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.float32)

        gx = cv2.filter2D(
            gray,
            cv2.CV_32F,
            prewitt_x
        )

        gy = cv2.filter2D(
            gray,
            cv2.CV_32F,
            prewitt_y
        )

        magnitude = cv2.magnitude(
            gx,
            gy
        )

        result = cv2.convertScaleAbs(
            magnitude
        )

        return result

    # --------------------------------------------------------
    # LAPLACIAN
    # --------------------------------------------------------

    elif operation == "Laplacian Edge":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        result = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        result = cv2.convertScaleAbs(
            result
        )

        return result

    # --------------------------------------------------------
    # CANNY
    # --------------------------------------------------------

    elif operation == "Canny Edge":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        result = cv2.Canny(
            gray,
            100,
            200
        )

        return result

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    elif operation == "Threshold":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        _, result = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY
        )

        return result

    # --------------------------------------------------------
    # ADAPTIVE THRESHOLD
    # --------------------------------------------------------

    elif operation == "Adaptive Threshold":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        return result

    # --------------------------------------------------------
    # EROSION
    # --------------------------------------------------------

    elif operation == "Erosion":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        result = cv2.erode(
            gray,
            kernel,
            iterations=1
        )

        return result

    # --------------------------------------------------------
    # DILATION
    # --------------------------------------------------------

    elif operation == "Dilation":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        result = cv2.dilate(
            gray,
            kernel,
            iterations=1
        )

        return result

    # --------------------------------------------------------
    # OPENING
    # --------------------------------------------------------

    elif operation == "Opening":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        result = cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            kernel
        )

        return result

    # --------------------------------------------------------
    # CLOSING
    # --------------------------------------------------------

    elif operation == "Closing":

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        result = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel
        )

        return result

    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    elif operation == "Brightness":

        result = cv2.convertScaleAbs(
            img,
            alpha=1.0,
            beta=brightness_value
        )

        return result

    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    elif operation == "Contrast":

        result = cv2.convertScaleAbs(
            img,
            alpha=contrast_value,
            beta=0
        )

        return result

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    elif operation == "HSV":

        result = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        )

        return result

    return img


# ============================================================
# NO IMAGE MESSAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Please upload an image from the sidebar to start image processing."
    )

    st.markdown(
        """
        <div class="info-box">

        <b>How to use:</b><br><br>

        1. Upload an image.<br>
        2. Select an image processing operation.<br>
        3. Select kernel size when required.<br>
        4. View the original and processed image.<br>
        5. Check the About and Process explanation.<br>
        6. View the matrix representation.<br>
        7. Download the processed image.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

pil_image = Image.open(uploaded_file).convert("RGB")

image_array = np.array(pil_image)

img = cv2.cvtColor(
    image_array,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# PROCESS IMAGE
# ============================================================

processed_image = process_image(
    img,
    selected_filter,
    kernel_size
)


# ============================================================
# CONVERT FOR DISPLAY
# ============================================================

if len(processed_image.shape) == 2:

    display_processed = processed_image

else:

    display_processed = cv2.cvtColor(
        processed_image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📊 Image Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Width",
        f"{img.shape[1]} px"
    )

with col2:
    st.metric(
        "Height",
        f"{img.shape[0]} px"
    )

with col3:
    st.metric(
        "Channels",
        img.shape[2]
    )

with col4:
    st.metric(
        "Operation",
        selected_filter
    )


# ============================================================
# ORIGINAL AND PROCESSED IMAGE
# ============================================================

st.markdown(
    '<div class="section-title">🖼️ Image Result</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("📥 Original Image")

    st.image(
        pil_image,
        use_container_width=True
    )

with col2:

    st.subheader("📤 Processed Image")

    st.image(
        display_processed,
        use_container_width=True
    )


# ============================================================
# ABOUT
# ============================================================

st.markdown(
    '<div class="section-title">📚 About Selected Operation</div>',
    unsafe_allow_html=True
)

information = filter_information[selected_filter]

st.markdown(
    f"""
    <div class="about-box">

    <h4>🔹 About {selected_filter}</h4>

    <p>{information["about"]}</p>

    <h4>🔄 Process</h4>

    <p>{information["process"]}</p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KERNEL MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">🔢 Kernel / Matrix</div>',
    unsafe_allow_html=True
)

if selected_filter in [
    "Mean Filter",
    "Gaussian Filter",
    "Median Filter",
    "Blur Filter",
    "Erosion",
    "Dilation",
    "Opening",
    "Closing"
]:

    matrix = np.ones(
        (kernel_size, kernel_size),
        dtype=int
    )

    st.markdown(
        """
        <div class="matrix-box">
        <b>Kernel Matrix used for processing:</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        matrix,
        use_container_width=True
    )


elif selected_filter == "Sharpening":

    matrix = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    st.markdown(
        """
        <div class="matrix-box">
        <b>Sharpening Kernel:</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        matrix,
        use_container_width=True
    )


elif selected_filter == "Prewitt Edge":

    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    prewitt_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    col1, col2 = st.columns(2)

    with col1:

        st.write("Prewitt X Kernel")

        st.dataframe(
            prewitt_x,
            use_container_width=True
        )

    with col2:

        st.write("Prewitt Y Kernel")

        st.dataframe(
            prewitt_y,
            use_container_width=True
        )


elif selected_filter == "Sobel Edge":

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    col1, col2 = st.columns(2)

    with col1:

        st.write("Sobel X Kernel")

        st.dataframe(
            sobel_x,
            use_container_width=True
        )

    with col2:

        st.write("Sobel Y Kernel")

        st.dataframe(
            sobel_y,
            use_container_width=True
        )


elif selected_filter == "Laplacian Edge":

    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    st.dataframe(
        laplacian_kernel,
        use_container_width=True
    )


else:

    st.info(
        "ℹ️ This operation does not use a fixed convolution kernel."
    )


# ============================================================
# PIXEL MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">🔬 Image Pixel Matrix</div>',
    unsafe_allow_html=True
)

st.write(
    "The image is represented internally as a NumPy array of pixel values."
)

if len(processed_image.shape) == 2:

    pixel_matrix = processed_image[:10, :10]

else:

    pixel_matrix = processed_image[:10, :10, 0]


st.write(
    "First 10 × 10 pixel values:"
)

st.dataframe(
    pixel_matrix,
    use_container_width=True
)


# ============================================================
# DOWNLOAD PROCESSED IMAGE
# ============================================================

st.markdown(
    '<div class="section-title">⬇️ Download Result</div>',
    unsafe_allow_html=True
)


if len(processed_image.shape) == 2:

    download_image = Image.fromarray(
        processed_image
    )

else:

    rgb_result = cv2.cvtColor(
        processed_image,
        cv2.COLOR_BGR2RGB
    )

    download_image = Image.fromarray(
        rgb_result
    )


buffer = io.BytesIO()

download_image.save(
    buffer,
    format="PNG"
)

buffer.seek(0)


st.download_button(
    label="⬇️ Download Processed Image",
    data=buffer,
    file_name="processed_image.png",
    mime="image/png"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">

    <b>🖼️ Image Processing Dashboard</b><br>

    Developed using Python + Streamlit + OpenCV + Pillow + NumPy

    </div>
    """,
    unsafe_allow_html=True
)
