# utils/face_processor.py

import asyncio
import aiohttp
import aiofiles
import cv2
import numpy as np
import torch
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
from pathlib import Path
import json
from collections import deque
import threading
from queue import Queue, Empty
import multiprocessing as mp

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Ultra-optimized configuration"""
    # Parallel processing
    max_parallel_detection: int = 4  # Parallel face detection threads
    max_parallel_embedding: int = 2  # Parallel embedding generation
    max_parallel_upload: int = 3     # Parallel upload connections
    
    # Batching
    detection_batch_size: int = 10   # Process 10 images at once for detection
    embedding_batch_size: int = 20   # Generate embeddings for 20 faces at once
    upload_batch_size: int = 15      # Upload 15 files per request
    
    # Streaming
    enable_streaming: bool = True    # Stream results as they're ready
    stream_buffer_size: int = 5      # Buffer size for streaming
    
    # GPU optimization
    use_gpu_batch: bool = torch.cuda.is_available()
    gpu_batch_size: int = 32        # GPU batch size for embeddings
    
    # Network optimization
    connection_pool_size: int = 20   # HTTP connection pool
    upload_timeout: int = 30         # Upload timeout in seconds
    enable_compression: bool = True  # Compress upload data
    
    # Queue sizes
    detection_queue_size: int = 100
    embedding_queue_size: int = 100
    upload_queue_size: int = 50

class StreamingFaceProcessor:
    """Ultra-fast streaming face processor with parallel everything"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # Queues for pipeline stages
        self.detection_queue = Queue(maxsize=config.detection_queue_size)
        self.embedding_queue = Queue(maxsize=config.embedding_queue_size)
        self.upload_queue = Queue(maxsize=config.upload_queue_size)
        
        # Thread pools for parallel processing
        self.detection_pool = ThreadPoolExecutor(max_workers=config.max_parallel_detection)
        self.embedding_pool = ThreadPoolExecutor(max_workers=config.max_parallel_embedding)
        
        # Async event loop for uploads
        self.upload_loop = None
        self.upload_thread = None
        
        # Performance tracking
        self.stats = {
            'files_processed': 0,
            'faces_detected': 0,
            'embeddings_generated': 0,
            'files_uploaded': 0,
            'total_time': 0,
            'detection_time': 0,
            'embedding_time': 0,
            'upload_time': 0
        }
        
        # Initialize components
        self._init_detector()
        self._init_embedding_model()
        self._start_upload_thread()
        
    def _init_detector(self):
        """Initialize optimized face detector"""
        from utils.face_detector import HighPerformanceYuNetDetector
        self.detector = HighPerformanceYuNetDetector()
        logger.info("✅ Face detector initialized")
        
    def _init_embedding_model(self):
        """Initialize embedding model with GPU support"""
        from core.device_setup import device, resnet
        self.device = device
        self.resnet = resnet
        
        # Enable GPU optimizations if available
        if self.config.use_gpu_batch and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("✅ GPU optimizations enabled")
            
    def _start_upload_thread(self):
        """Start async upload thread"""
        def run_upload_loop():
            self.upload_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.upload_loop)
            self.upload_loop.run_until_complete(self._upload_worker())
            
        self.upload_thread = threading.Thread(target=run_upload_loop, daemon=True)
        self.upload_thread.start()
        logger.info("✅ Upload thread started")

    # ===== STAGE 1: PARALLEL FACE DETECTION =====
    
    def process_files_streaming(self, file_paths: List[str], allowed_paths: List[str]):
        """Process files with streaming pipeline"""
        start_time = time.time()
        
        # Submit all files for detection in parallel
        detection_futures = []
        for file_path in file_paths:
            future = self.detection_pool.submit(self._detect_faces_optimized, file_path)
            detection_futures.append((file_path, future))
            
        logger.info(f"🚀 Submitted {len(file_paths)} files for parallel detection")
        
        # Stream results as they complete
        for file_path, future in detection_futures:
            try:
                result = future.result(timeout=5)
                if result:
                    # Immediately queue for embedding generation
                    self.embedding_queue.put(result)
                    logger.debug(f"✅ Detection complete: {Path(file_path).name}")
            except Exception as e:
                logger.error(f"❌ Detection failed for {file_path}: {e}")
                
        elapsed = time.time() - start_time
        logger.info(f"⚡ Detection stage completed in {elapsed:.2f}s")
        
    def _detect_faces_optimized(self, file_path: str) -> Optional[Dict]:
        """Optimized face detection with caching"""
        try:
            start = time.time()
            
            # Fast image loading
            img = self._fast_imread(file_path)
            if img is None:
                return None
                
            # Detect faces
            success, faces = self.detector.detect_and_validate(img)
            if not success or not faces:
                return None
                
            # Extract face crops in parallel
            face_crops = self._extract_face_crops_parallel(img, faces)
            
            detection_time = time.time() - start
            self.stats['detection_time'] += detection_time
            self.stats['faces_detected'] += len(faces)
            
            return {
                'file_path': file_path,
                'faces': faces,
                'face_crops': face_crops,
                'detection_time': detection_time
            }
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None
            
    def _fast_imread(self, file_path: str) -> Optional[np.ndarray]:
        """Ultra-fast image reading with memory mapping"""
        try:
            # Use memory mapping for large files
            return cv2.imread(file_path, cv2.IMREAD_COLOR)
        except:
            return None
            
    def _extract_face_crops_parallel(self, img: np.ndarray, faces: List) -> List[np.ndarray]:
        """Extract face crops in parallel"""
        face_crops = []
        h, w = img.shape[:2]
        
        for face in faces:
            x, y, w_box, h_box, _ = face
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(x + w_box, w), min(y + h_box, h)
            
            if x2 > x1 and y2 > y1:
                face_crop = img[y1:y2, x1:x2]
                face_crops.append(face_crop)
                
        return face_crops

    # ===== STAGE 2: BATCH GPU EMBEDDING GENERATION =====
    
    async def _embedding_worker(self):
        """Worker for batch embedding generation"""
        batch = []
        last_process_time = time.time()
        
        while True:
            try:
                # Collect batch
                while len(batch) < self.config.embedding_batch_size:
                    try:
                        item = self.embedding_queue.get(timeout=0.1)
                        batch.append(item)
                    except Empty:
                        break
                        
                # Process batch if ready or timeout
                current_time = time.time()
                if batch and (len(batch) >= self.config.embedding_batch_size or 
                            current_time - last_process_time > 1.0):
                    
                    # Generate embeddings for entire batch on GPU
                    self._process_embedding_batch(batch)
                    batch = []
                    last_process_time = current_time
                    
            except Exception as e:
                logger.error(f"Embedding worker error: {e}")
                
    def _process_embedding_batch(self, batch: List[Dict]):
        """Process batch of faces for embedding generation"""
        try:
            start = time.time()
            
            # Collect all face crops
            all_face_crops = []
            file_mapping = []
            
            for item in batch:
                for i, face_crop in enumerate(item['face_crops']):
                    all_face_crops.append(face_crop)
                    file_mapping.append((item['file_path'], i))
                    
            if not all_face_crops:
                return
                
            # Batch preprocessing
            face_tensors = self._batch_preprocess_faces(all_face_crops)
            
            # GPU batch embedding generation
            with torch.no_grad():
                if self.config.use_gpu_batch:
                    # Process in GPU batches
                    embeddings = []
                    for i in range(0, len(face_tensors), self.config.gpu_batch_size):
                        batch_tensor = face_tensors[i:i + self.config.gpu_batch_size]
                        batch_embeddings = self.resnet(batch_tensor.to(self.device))
                        embeddings.append(batch_embeddings.cpu().numpy())
                    embeddings = np.vstack(embeddings) if embeddings else np.array([])
                else:
                    # CPU processing
                    embeddings = self.resnet(face_tensors).numpy()
                    
            # Group embeddings by file
            results_by_file = {}
            for (file_path, face_idx), embedding in zip(file_mapping, embeddings):
                if file_path not in results_by_file:
                    results_by_file[file_path] = []
                results_by_file[file_path].append(embedding.tolist())
                
            # Queue for upload
            for file_path, embeddings_list in results_by_file.items():
                self.upload_queue.put({
                    'file_path': file_path,
                    'embeddings': embeddings_list
                })
                
            elapsed = time.time() - start
            self.stats['embedding_time'] += elapsed
            self.stats['embeddings_generated'] += len(embeddings)
            
            logger.info(f"⚡ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Embedding batch error: {e}")
            
    def _batch_preprocess_faces(self, face_crops: List[np.ndarray]) -> torch.Tensor:
        """Batch preprocessing with vectorized operations"""
        processed = []
        
        for face_crop in face_crops:
            # Fast resize using cv2
            face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            processed.append(face_rgb)
            
        # Vectorized normalization
        faces_array = np.stack(processed)
        faces_tensor = torch.from_numpy(faces_array).permute(0, 3, 1, 2).float()
        faces_tensor = (faces_tensor / 255.0 - 0.5) / 0.5
        
        return faces_tensor

    # ===== STAGE 3: ASYNC PARALLEL UPLOADS =====
    
    async def _upload_worker(self):
        """Async worker for parallel uploads"""
        
        # Create connection pool
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            limit_per_host=10,
            ttl_dns_cache=300
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.upload_timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            upload_tasks = []
            batch = []
            last_upload_time = time.time()
            
            while True:
                try:
                    # Collect upload batch
                    while len(batch) < self.config.upload_batch_size:
                        try:
                            item = await asyncio.wait_for(
                                asyncio.get_event_loop().run_in_executor(
                                    None, self.upload_queue.get, True, 0.1
                                ), 
                                timeout=0.1
                            )
                            batch.append(item)
                        except (asyncio.TimeoutError, Empty):
                            break
                            
                    # Upload batch if ready
                    current_time = time.time()
                    if batch and (len(batch) >= self.config.upload_batch_size or 
                                current_time - last_upload_time > 0.5):
                        
                        # Create upload task
                        task = asyncio.create_task(
                            self._upload_batch_async(session, batch.copy())
                        )
                        upload_tasks.append(task)
                        batch = []
                        last_upload_time = current_time
                        
                        # Clean completed tasks
                        upload_tasks = [t for t in upload_tasks if not t.done()]
                        
                        # Limit concurrent uploads
                        if len(upload_tasks) >= self.config.max_parallel_upload:
                            await asyncio.gather(*upload_tasks)
                            upload_tasks = []
                            
                except Exception as e:
                    logger.error(f"Upload worker error: {e}")
                    await asyncio.sleep(0.1)
                    
    async def _upload_batch_async(self, session: aiohttp.ClientSession, batch: List[Dict]):
        """Async batch upload with compression"""
        try:
            start = time.time()
            
            # Prepare multipart data
            data = aiohttp.FormData()
            
            for item in batch:
                file_path = item['file_path']
                embeddings = item['embeddings']
                
                # Add file
                with open(file_path, 'rb') as f:
                    data.add_field('files', f.read(), 
                                 filename=Path(file_path).name,
                                 content_type='image/jpeg')
                    
                # Add embeddings (compressed if enabled)
                if self.config.enable_compression:
                    import gzip
                    embeddings_json = json.dumps(embeddings)
                    compressed = gzip.compress(embeddings_json.encode())
                    data.add_field('embeddings', compressed,
                                 content_type='application/gzip')
                else:
                    data.add_field('embeddings', json.dumps(embeddings))
                    
            # Send request
            async with session.post('http://api.example.com/upload', data=data) as response:
                result = await response.json()
                
                elapsed = time.time() - start
                self.stats['upload_time'] += elapsed
                self.stats['files_uploaded'] += len(batch)
                
                logger.info(f"⚡ Uploaded {len(batch)} files in {elapsed:.2f}s")
                
                return result
                
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None


class TurboQueueProcessor:
    """Ultra-fast queue processor with zero-wait streaming"""
    
    def __init__(self):
        self.config = OptimizationConfig()
        self.processor = StreamingFaceProcessor(self.config)
        
        # Priority queue for different file types
        self.priority_queue = []
        self.normal_queue = []
        
        # Processing state
        self.is_processing = False
        self.stop_requested = False
        
    def add_file(self, file_path: str, priority: bool = False):
        """Add file with priority support"""
        if priority:
            self.priority_queue.append(file_path)
        else:
            self.normal_queue.append(file_path)
            
        # Immediately start processing if not already running
        if not self.is_processing:
            self._start_processing()
            
    def _start_processing(self):
        """Start immediate processing"""
        if self.is_processing:
            return
            
        self.is_processing = True
        
        # Process in separate thread
        thread = threading.Thread(target=self._process_queues, daemon=True)
        thread.start()
        
    def _process_queues(self):
        """Process queues with streaming"""
        while not self.stop_requested:
            batch = []
            
            # Get batch from queues (priority first)
            while len(batch) < self.config.detection_batch_size:
                if self.priority_queue:
                    batch.append(self.priority_queue.pop(0))
                elif self.normal_queue:
                    batch.append(self.normal_queue.pop(0))
                else:
                    break
                    
            if batch:
                # Process batch immediately
                self.processor.process_files_streaming(batch, [])
            else:
                # No files, wait briefly
                time.sleep(0.05)
                if not self.priority_queue and not self.normal_queue:
                    self.is_processing = False
                    break

