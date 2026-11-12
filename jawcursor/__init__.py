"""jawcursor: gaze moves the cursor, a jaw clench clicks."""

import os

# Without this, OpenCV's Media Foundation backend can take many seconds to
# open a webcam on Windows. It has to be set before cv2 is first imported.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

__version__ = "0.1.0"
