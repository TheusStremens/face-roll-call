import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


class FaceHandler:
    def __init__(self, _margin=20):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Using {device} device.")
        self.detector = MTCNN(
            margin=_margin, keep_all=True, select_largest=False, device=device
        )
        self.recognizer = InceptionResnetV1(pretrained="vggface2").eval()

    def update_frame(self, frame):
        self.frame = self.preprocess_frame(frame)

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return frame

    def detect(self):
        self.bboxes, _, self.landmarks = self.detector.detect(self.frame, landmarks=True)
        return self.bboxes, self.landmarks

    def describe(self):
        face_tensor = self.detector.extract(self.frame, self.bboxes, None)
        embedding = self.recognizer(face_tensor[0].unsqueeze(0))
        return embedding
