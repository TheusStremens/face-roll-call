import cv2
from src.face_handler import FaceHandler
from src.gui import GUI


face_handler = FaceHandler()
gui_window = GUI()
cam = cv2.VideoCapture(0)
key = 0

while key != 27:
    check, frame = cam.read()

    if not check:
        continue

    gui_window.update_frame(frame)
    face_handler.update_frame(frame)
    bboxes, landmarks = face_handler.detect()
    if bboxes is None:
        print("No face detected")
        continue
    if len(bboxes) > 1:
        print("Only one person is allowed to be in front of the camera")
        continue
    embedding = face_handler.describe()
    gui_window.draw_face_detection(bboxes[0], landmarks[0])
    gui_window.display()
    key = cv2.waitKey(1)
