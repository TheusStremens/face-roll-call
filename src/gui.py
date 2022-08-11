import numpy as np
import cv2


class GUI:
    def __init__(self, window_name="Face Call Roll"):
        self.canvas = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
        self.background = cv2.imread("background.jpg", -1)
        self.button_img = cv2.imread("button2.png", -1)
        self.button_img = cv2.resize(self.button_img, (0, 0), fx=0.5, fy=0.5)
        self.define_buttons_config()
        self.draw_buttons()
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_click)

    def define_buttons_config(self):
        self.button_height, self.button_width, _ = self.button_img.shape
        self.bg_width = int(self.background.shape[1])
        proportion = int(self.bg_width / 3)

        offset = int(proportion + (self.button_width/2))
        offset_last = self.bg_width - self.button_width - 10
        height_offset = 20

        self.button_call_pos = (10, height_offset)
        self.button_register_pos = (offset, height_offset)
        self.button_class_file_pos = (offset_last, height_offset)

    def draw_buttons(self):
        temp_img = self.background.copy()
        overlay = self.transparent_overlay(temp_img, self.button_img, self.button_call_pos)
        overlay = self.transparent_overlay(temp_img, self.button_img, self.button_register_pos)
        overlay = self.transparent_overlay(temp_img, self.button_img, self.button_class_file_pos)
        opacity = 1.0
        cv2.addWeighted(
            overlay, opacity, self.background, 1 - opacity, 0, self.background
        )

    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, r, d):
        x1, y1 = pt1
        x2, y2 = pt2
        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)

    def update_frame(self, frame):
        self.canvas = frame.copy()

    def display(self):
        cv2.imshow(self.window_name, self.background)

    def draw_face_detection(self, bbox, landmarks):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        self.draw_rounded_rectangle(self.canvas, p1, p2, (0, 255, 0), 2, 15, 10)
        for landmark in landmarks:
            p = (int(landmark[0]), int(landmark[1]))
            cv2.circle(self.canvas, center=p, radius=3, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        # Draw the webcam image in the middle of the background.
        rows, cols, _ = self.canvas.shape
        height, width, _ = self.background.shape
        origin_x = int((width / 2) - (cols / 2))
        origin_y = int((height / 2) - (rows / 2))
        self.background[
            origin_y : origin_y + rows, origin_x : origin_x + cols
        ] = self.canvas
        rows, cols, _ = self.canvas.shape
        self.background[
            origin_y : origin_y + rows, origin_x : origin_x + cols
        ] = self.canvas

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("click")
            from tkinter import Tk
            from tkinter.filedialog import askopenfilename

            Tk().withdraw()  # don't open full image
            filename = askopenfilename()
            print(filename)

    def transparent_overlay(self, src, overlay, pos=(0, 0), scale=1):
        overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
        h, w, _ = overlay.shape  # Size of foreground
        rows, cols, _ = src.shape  # Size of background Image
        y, x = pos[0], pos[1]  # Position of foreground/overlay image
        # loop over all pixels and apply the blending equation
        for i in range(h):
            for j in range(w):
                if x + i >= rows or y + j >= cols:
                    continue
                alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
                src[x + i][y + j] = (
                    alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
                )
        return src
