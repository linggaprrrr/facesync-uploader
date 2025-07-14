import os
import torch
from dotenv import load_dotenv
from facenet_pytorch import InceptionResnetV1


# Ambil dari environment
load_dotenv()
API_BASE = os.getenv("BASE_URL")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
