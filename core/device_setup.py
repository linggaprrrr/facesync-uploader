import os
import torch
from dotenv import load_dotenv
from facenet_pytorch import InceptionResnetV1, MTCNN


# Ambil dari environment
load_dotenv()
API_BASE = os.getenv('API_BASE', 'https://api.ownize.app')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
# Check device
if torch.cuda.is_available():
    print(f"✅ Using CUDA - GPU: {torch.cuda.get_device_name()}")
else:
    print("⚠️ Using CPU")