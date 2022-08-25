import src.gui_utils as utils
import numpy as np
import cv2
from enum import Enum


class Mode(Enum):
    UNSET = 0
    CALL = 1
    REGISTER = 2
    LOAD_FILE = 3


class GUI:
    def __init__(self, window_name="Face Call Roll"):
        self.canvas = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
        self.background = cv2.imread("resources/background.jpg", -1)
        self.btn_img = cv2.imread("resources/button_crop.png", -1)
        self.btn_img = cv2.resize(self.btn_img, (0, 0), fx=0.5, fy=0.5)
        self.define_buttons_config()
        self.draw_buttons()
        self.draw_log_area()
        self.window_name = window_name
        self.mode = Mode.UNSET
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_click)

    def draw_log_area(self):
        height_offset = 20
        bg_height, bg_width, _ = self.background.shape
        log_height = int(bg_height * 0.2)
        log_width = int(bg_width * 0.7)
        x = int((bg_width/2) - (log_width/2))
        y = bg_height - height_offset - log_height
        p1 = (x, y)
        p2 = (x+log_width, y+log_height)
        cv2.rectangle(self.background, p1, p2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.rectangle(self.background, p1, p2, (209, 140, 5), 5, cv2.LINE_AA)
        cv2.rectangle(self.background, p1, p2, (215, 194, 165), 2, cv2.LINE_AA)

    def define_buttons_config(self):
        self.btn_height, self.btn_width, _ = self.btn_img.shape
        self.bg_width = int(self.background.shape[1])
        proportion = int(self.bg_width / 3)

        offset = int(proportion + (self.btn_width / 2))
        offset_last = self.bg_width - self.btn_width - 10
        height_offset = 20
        btn_size = [self.btn_width, self.btn_height]

        self.btn_call_rect = [10, height_offset, *btn_size]
        self.btn_call_pos = self.btn_call_rect[0:2]

        self.btn_register_rect = [offset, height_offset, *btn_size]
        self.btn_register_pos = self.btn_register_rect[0:2]

        self.btn_class_file_rect = [offset_last, height_offset, *btn_size]
        self.btn_class_file_pos = self.btn_class_file_rect[0:2]

    def draw_buttons(self):
        temp_img = self.background.copy()
        overlay = self.transparent_overlay(
            temp_img, self.btn_img, self.btn_call_pos
        )
        overlay = self.transparent_overlay(
            temp_img, self.btn_img, self.btn_register_pos
        )
        overlay = self.transparent_overlay(
            temp_img, self.btn_img, self.btn_class_file_pos
        )
        opacity = 1.0
        cv2.addWeighted(
            overlay, opacity, self.background, 1 - opacity, 0, self.background
        )

    def draw_buttons_text(self):
        label = "Call"
        text_pos = utils.get_text_position(label, self.btn_call_rect)
        cv2.putText(self.background,
                    label, text_pos,
                    0,
                    1.0,
                    (255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        label = "Register"
        text_pos = utils.get_text_position(label, self.btn_register_rect)
        cv2.putText(self.background,
                    label, text_pos,
                    0,
                    1.0,
                    (255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        label = "Class File"
        text_pos = utils.get_text_position(label, self.btn_class_file_rect)
        cv2.putText(self.background,
                    label, text_pos,
                    0,
                    1.0,
                    (255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

    def update_frame(self, frame):
        self.canvas = frame.copy()

    def display(self):
        self.draw_buttons_text()
        cv2.imshow(self.window_name, self.background)

    def draw_face_detection(self, bbox, landmarks, color=(255, 255, 0)):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        utils.draw_rounded_rectangle(self.canvas, p1, p2, color, 2, 15, 10)
        for landmark in landmarks:
            p = (int(landmark[0]), int(landmark[1]))
            cv2.circle(
                self.canvas,
                center=p,
                radius=3,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # Draw the webcam image in the middle of the background.
        rows, cols, _ = self.canvas.shape
        height, width, _ = self.background.shape
        origin_x = int((width / 2) - (cols / 2))
        origin_y = int((height / 2) - (rows / 2))
        self.background[
            origin_y : origin_y + rows, origin_x : origin_x + cols
        ] = self.canvas
        cv2.rectangle(self.background, (origin_x, origin_y), (origin_x + cols, origin_y + rows), (209, 140, 5), 5, cv2.LINE_AA)
        cv2.rectangle(self.background, (origin_x, origin_y), (origin_x + cols, origin_y + rows), (215, 194, 165), 2, cv2.LINE_AA)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.check_click_inside_buttons(x, y)

    def check_click_inside_buttons(self, x, y):
        if utils.rect_contains(self.btn_call_rect, (x, y)):
            self.mode = Mode.CALL
        elif utils.rect_contains(self.btn_register_rect, (x, y)):
            self.mode = Mode.REGISTER
        if utils.rect_contains(self.btn_class_file_rect, (x, y)):
            self.mode = Mode.LOAD_FILE
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
