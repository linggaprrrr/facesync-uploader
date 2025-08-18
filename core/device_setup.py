import os
import torch
from dotenv import load_dotenv
from facenet_pytorch import InceptionResnetV1


# Ambil dari environment
load_dotenv()
API_BASE = os.getenv('API_BASE', 'http://localhost:8001')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Check device
if torch.cuda.is_available():
    print(f"✅ Using CUDA - GPU: {torch.cuda.get_device_name()}")
else:
    print("⚠️ Using CPU")