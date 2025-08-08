# utils/separated_uploader.py

import asyncio
import aiohttp
import json
import logging
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from uuid import UUID
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from core.device_setup import API_BASE

logger = logging.getLogger(__name__)

@dataclass
class UploadConfig:
    """Configuration for separated uploads"""
    api_base_url: str = API_BASE
    max_concurrent_data: int = 3
    max_concurrent_files: int = 2
    batch_size_data: int = 15
    batch_size_files: int = 8
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout_data: int = 30
    timeout_files: int = 120
    status_check_interval: float = 2.0
    max_status_checks: int = 30

@dataclass
class UploadResult:
    """Upload result tracking"""
    success: bool
    photo_id: Optional[UUID] = None
    original_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    file_path: Optional[str] = None

class SeparatedUploadManager:
    """
    Manager for separated upload system (data + files)
    Compatible with existing face detection workflow
    """
    
    def __init__(self, api_base_url: str = API_BASE):
        self.config = UploadConfig(api_base_url=api_base_url)
        self.session: Optional[aiohttp.ClientSession] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=20,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=300)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'SeparatedUploadManager/1.0',
                'Accept': 'application/json'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        self._executor.shutdown(wait=True)
    
    def upload_photos_sync(self, 
                          photos_data: List[Dict[str, Any]], 
                          progress_callback: Optional[Callable] = None) -> List[UploadResult]:
        """
        Synchronous wrapper for async upload
        
        Args:
            photos_data: List of photo data with file_path, unit_id, outlet_id, photo_type_id, faces_data
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of UploadResult objects
        """
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async upload
            return loop.run_until_complete(
                self.upload_photos_async(photos_data, progress_callback)
            )
            
        except Exception as e:
            logger.error(f"❌ Sync upload wrapper error: {e}")
            return [
                UploadResult(
                    success=False,
                    error_message=f"Upload failed: {str(e)}",
                    file_path=photo.get('file_path')
                )
                for photo in photos_data
            ]
    
    async def upload_photos_async(self, 
                                 photos_data: List[Dict[str, Any]], 
                                 progress_callback: Optional[Callable] = None) -> List[UploadResult]:
        """
        Main async upload method using separated data/file approach
        """
        total_items = len(photos_data)
        logger.info(f"🚀 Starting separated upload: {total_items} photos")
        
        if progress_callback:
            progress_callback("Starting separated upload...", 0, total_items)
        
        start_time = time.time()
        
        async with self:
            try:
                # STEP 1: Upload face data and metadata (fast)
                if progress_callback:
                    progress_callback("Uploading face data...", 0, total_items)
                
                data_start = time.time()
                photo_ids = await self._upload_data_batch(photos_data, progress_callback)
                data_time = time.time() - data_start
                
                if not photo_ids:
                    logger.error("❌ Data upload failed completely")
                    return [
                        UploadResult(
                            success=False,
                            error_message="Data upload failed",
                            file_path=photo.get('file_path')
                        )
                        for photo in photos_data
                    ]
                
                logger.info(f"✅ Data upload completed in {data_time:.2f}s: {len(photo_ids)} photos")
                
                # STEP 2: Upload files (slower)
                if progress_callback:
                    progress_callback("Uploading files...", len(photo_ids), total_items)
                
                file_start = time.time()
                await self._upload_files_batch(photos_data, photo_ids, progress_callback)
                file_time = time.time() - file_start
                
                logger.info(f"✅ File upload completed in {file_time:.2f}s")
                
                # STEP 3: Wait for processing completion
                if progress_callback:
                    progress_callback("Processing files...", len(photo_ids), total_items)
                
                status_start = time.time()
                results = await self._wait_for_completion(photo_ids, photos_data, progress_callback)
                status_time = time.time() - status_start
                
                total_time = time.time() - start_time
                successful = len([r for r in results if r.success])
                
                logger.info(f"🎯 Upload summary:")
                logger.info(f"   Total time: {total_time:.2f}s")
                logger.info(f"   Data: {data_time:.2f}s, Files: {file_time:.2f}s, Status: {status_time:.2f}s")
                logger.info(f"   Success rate: {successful}/{total_items} ({successful/total_items*100:.1f}%)")
                
                if progress_callback:
                    progress_callback("Upload completed!", successful, total_items)
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Async upload error: {e}")
                return [
                    UploadResult(
                        success=False,
                        error_message=f"Upload failed: {str(e)}",
                        processing_time=time.time() - start_time,
                        file_path=photo.get('file_path')
                    )
                    for photo in photos_data
                ]
    
    async def _upload_data_batch(self, 
                               photos_data: List[Dict[str, Any]], 
                               progress_callback: Optional[Callable] = None) -> List[str]:
        """Upload face data and metadata in batches"""
        
        # Split into batches
        batches = [
            photos_data[i:i + self.config.batch_size_data]
            for i in range(0, len(photos_data), self.config.batch_size_data)
        ]
        
        logger.info(f"📊 Uploading data in {len(batches)} batches")
        
        all_photo_ids = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_data)
        
        async def upload_data_batch(batch_items: List[Dict], batch_num: int):
            async with semaphore:
                try:
                    logger.info(f"📤 Data batch {batch_num}: {len(batch_items)} items")
                    
                    # Prepare batch payload
                    photos_payload = []
                    for item in batch_items:
                        photo_data = {
                            "unit_code": item["unit_id"],      # Send as unit_code  
                            "outlet_code": item["outlet_id"],  # Send as outlet_code
                            "photo_type_code": item["photo_type_id"],  # Send as photo_type_code
                            "filename": Path(item["file_path"]).name,
                            "faces": [
                                {
                                    "embedding": face["embedding"],
                                    "bbox": face["bbox"],
                                    "confidence": face["confidence"]
                                }
                                for face in item["faces_data"]
                            ]
                        }
                        photos_payload.append(photo_data)
                    
                    batch_payload = {"photos": photos_payload}
                    
                    # Upload data
                    url = f"{self.config.api_base_url}/faces/upload-data"
                    
                    async with self.session.post(
                        url,
                        json=batch_payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_data)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            photo_ids = [str(pid) for pid in result.get("photo_ids", [])]
                            
                            logger.info(f"✅ Data batch {batch_num}: {len(photo_ids)} photos")
                            
                            if progress_callback:
                                progress_callback(
                                    f"Data batch {batch_num} completed",
                                    len(all_photo_ids) + len(photo_ids),
                                    len(photos_data)
                                )
                            
                            return photo_ids
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ Data batch {batch_num} failed: {response.status} - {error_text}")
                            return []
                            
                except Exception as e:
                    logger.error(f"❌ Data batch {batch_num} error: {e}")
                    return []
        
        # Execute all batches concurrently
        tasks = [
            upload_data_batch(batch_items, i + 1)
            for i, batch_items in enumerate(batches)
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect photo IDs
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                all_photo_ids.extend(batch_result)
        
        logger.info(f"📊 Data upload completed: {len(all_photo_ids)} total photo IDs")
        return all_photo_ids
    
    async def _upload_files_batch(self, 
                                photos_data: List[Dict[str, Any]], 
                                photo_ids: List[str],
                                progress_callback: Optional[Callable] = None):
        """Upload files to storage in batches"""
        
        if len(photos_data) != len(photo_ids):
            logger.error(f"❌ Mismatch: {len(photos_data)} items vs {len(photo_ids)} photo IDs")
            return
        
        # Split into file upload batches
        batch_size = self.config.batch_size_files
        batches = []
        
        for i in range(0, len(photos_data), batch_size):
            batch_items = photos_data[i:i + batch_size]
            batch_photo_ids = photo_ids[i:i + batch_size]
            batches.append((batch_items, batch_photo_ids))
        
        logger.info(f"📁 Uploading files in {len(batches)} batches")
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        
        async def upload_file_batch(batch_items: List[Dict], 
                                  batch_photo_ids: List[str], 
                                  batch_num: int):
            async with semaphore:
                try:
                    logger.info(f"📁 File batch {batch_num}: {len(batch_items)} files")
                    
                    # Prepare multipart data
                    data = aiohttp.FormData()
                    
                    # Add photo IDs
                    for photo_id in batch_photo_ids:
                        data.add_field('photo_ids', photo_id)
                    
                    # Add files
                    for item in batch_items:
                        try:
                            file_path = item['file_path']
                            filename = Path(file_path).name
                            
                            async with aiofiles.open(file_path, 'rb') as f:
                                file_content = await f.read()
                                data.add_field(
                                    'files',
                                    file_content,
                                    filename=filename,
                                    content_type='image/jpeg'
                                )
                        except Exception as e:
                            logger.error(f"❌ Failed to read file {item['file_path']}: {e}")
                            continue
                    
                    # Upload files
                    url = f"{self.config.api_base_url}/faces/upload-files"
                    
                    async with self.session.post(
                        url,
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_files)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"✅ File batch {batch_num} uploaded")
                            
                            if progress_callback:
                                progress_callback(
                                    f"File batch {batch_num} completed",
                                    batch_num * batch_size,
                                    len(photos_data)
                                )
                            
                            return result
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ File batch {batch_num} failed: {response.status} - {error_text}")
                            return []
                            
                except Exception as e:
                    logger.error(f"❌ File batch {batch_num} error: {e}")
                    return []
        
        # Execute all file batches concurrently
        tasks = [
            upload_file_batch(batch_items, batch_photo_ids, i + 1)
            for i, (batch_items, batch_photo_ids) in enumerate(batches)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"📁 File upload completed")
    
    async def _wait_for_completion(self, 
                                 photo_ids: List[str],
                                 photos_data: List[Dict[str, Any]],
                                 progress_callback: Optional[Callable] = None) -> List[UploadResult]:
        """Wait for all uploads to complete and get final URLs"""
        
        logger.info(f"⏳ Monitoring completion for {len(photo_ids)} photos")
        
        # Create file path mapping
        file_path_map = {
            photo_ids[i]: photos_data[i]['file_path'] 
            for i in range(min(len(photo_ids), len(photos_data)))
        }
        
        final_results = []
        pending_ids = set(photo_ids)
        check_count = 0
        
        while pending_ids and check_count < self.config.max_status_checks:
            check_count += 1
            logger.info(f"🔍 Status check {check_count}: {len(pending_ids)} pending")
            
            try:
                # Check status in batches
                url = f"{self.config.api_base_url}/faces/batch-upload-status"
                check_ids = list(pending_ids)
                
                async with self.session.post(
                    url,
                    json={"photo_ids": check_ids},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        photos_status = result.get("photos", [])
                        
                        newly_completed = []
                        
                        for photo_status in photos_status:
                            photo_id = photo_status["photo_id"]
                            status = photo_status["status"]
                            
                            if status == "completed":
                                # Photo completed successfully
                                upload_result = UploadResult(
                                    success=True,
                                    photo_id=UUID(photo_id),
                                    original_url=photo_status.get("original_url"),
                                    thumbnail_url=photo_status.get("thumbnail_url"),
                                    file_path=file_path_map.get(photo_id)
                                )
                                final_results.append(upload_result)
                                newly_completed.append(photo_id)
                                
                            elif status == "failed":
                                # Photo failed processing
                                upload_result = UploadResult(
                                    success=False,
                                    photo_id=UUID(photo_id),
                                    error_message="File processing failed",
                                    file_path=file_path_map.get(photo_id)
                                )
                                final_results.append(upload_result)
                                newly_completed.append(photo_id)
                        
                        # Remove completed photos
                        for photo_id in newly_completed:
                            pending_ids.discard(photo_id)
                        
                        if newly_completed:
                            logger.info(f"✅ {len(newly_completed)} photos completed")
                            
                            if progress_callback:
                                progress_callback(
                                    f"Processing: {len(final_results)} completed",
                                    len(final_results),
                                    len(photo_ids)
                                )
                    
                    else:
                        logger.warning(f"⚠️ Status check failed: {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Status check error: {e}")
            
            # Wait before next check
            if pending_ids:
                await asyncio.sleep(self.config.status_check_interval)
        
        # Handle remaining pending photos as timeouts
        for photo_id in pending_ids:
            upload_result = UploadResult(
                success=False,
                photo_id=UUID(photo_id),
                error_message="Processing timeout",
                file_path=file_path_map.get(photo_id)
            )
            final_results.append(upload_result)
        
        logger.info(f"⏳ Status monitoring completed: {len(final_results)} total results")
        return final_results


# ===== INTEGRATION FUNCTIONS =====

def create_separated_upload_manager(api_base_url: str = API_BASE) -> SeparatedUploadManager:
    """Factory function to create upload manager"""
    return SeparatedUploadManager(api_base_url)


def upload_photos_with_separated_system(photos_data: List[Dict[str, Any]], 
                                       api_base_url: str = API_BASE,
                                       progress_callback: Optional[Callable] = None) -> List[UploadResult]:
    """
    High-level function to upload photos using separated system
    
    Args:
        photos_data: List of dictionaries with:
            - file_path: str
            - unit_id: str  
            - outlet_id: str
            - photo_type_id: str
            - faces_data: List[Dict] with embedding, bbox, confidence
        api_base_url: Backend API base URL
        progress_callback: Optional progress callback function
        
    Returns:
        List of UploadResult objects
    """
    manager = SeparatedUploadManager(api_base_url)
    return manager.upload_photos_sync(photos_data, progress_callback)


# ===== COMPATIBILITY WRAPPER =====

class CompatibilityWrapper:
    """
    Wrapper to make separated upload system compatible with existing 
    face detection workflow and function signatures
    """
    
    def __init__(self, api_base_url: str = API_BASE):
        self.api_base_url = api_base_url
        self.upload_manager = SeparatedUploadManager(api_base_url)
    
    def process_batch_faces_and_upload_separated(self, 
                                               files_list: List[str], 
                                               allowed_paths: List[str],
                                               progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
        """
        Drop-in replacement for existing batch upload functions
        
        Args:
            files_list: List of file paths to process
            allowed_paths: List of allowed base paths
            progress_callback: Optional progress callback
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            from utils.face_detector import process_faces_in_image_optimized
            
            logger.info(f"🚀 Processing {len(files_list)} files with separated upload")
            
            # Process files and extract face data
            photos_data = []
            processing_errors = 0
            
            for i, file_path in enumerate(files_list):
                try:
                    if progress_callback:
                        progress_callback(f"Processing faces in {Path(file_path).name}", i, len(files_list))
                    
                    # Process faces in image
                    faces = process_faces_in_image_optimized(file_path)
                    
                    if not faces:
                        processing_errors += 1
                        logger.warning(f"⚠️ No faces detected in {file_path}")
                        continue
                    
                    # Parse path codes
                    relative_path = self._get_relative_path(file_path, allowed_paths)
                    if not relative_path:
                        processing_errors += 1
                        logger.warning(f"⚠️ Invalid path for {file_path}")
                        continue
                    
                    unit_code, outlet_code, photo_type_code = self._parse_codes_from_path(relative_path)
                    if not all([unit_code, outlet_code, photo_type_code]):
                        processing_errors += 1
                        logger.warning(f"⚠️ Path parsing failed for {file_path}")
                        continue
                    
                    # Convert codes to IDs (implement based on your system)
                    unit_id, outlet_id, photo_type_id = self._resolve_codes_to_ids(
                        unit_code, outlet_code, photo_type_code
                    )
                    
                    photo_data = {
                        'file_path': file_path,
                        'unit_id': unit_id,
                        'outlet_id': outlet_id,
                        'photo_type_id': photo_type_id,
                        'faces_data': faces
                    }
                    photos_data.append(photo_data)
                    
                    logger.info(f"✅ Processed {Path(file_path).name}: {len(faces)} faces")
                    
                except Exception as e:
                    processing_errors += 1
                    logger.error(f"❌ Processing error for {file_path}: {e}")
            
            if not photos_data:
                message = f"No valid photos to upload. {processing_errors} processing errors."
                logger.error(f"❌ {message}")
                return False, message
            
            logger.info(f"📊 Ready to upload {len(photos_data)} photos ({processing_errors} errors)")
            
            # Use separated upload
            if progress_callback:
                progress_callback("Starting separated upload...", len(photos_data), len(files_list))
            
            results = self.upload_manager.upload_photos_sync(photos_data, progress_callback)
            
            # Process results
            successful = len([r for r in results if r.success])
            failed = len([r for r in results if not r.success])
            
            success_rate = (successful / len(results)) * 100 if results else 0
            
            if successful > 0:
                message = f"Separated upload: {successful}/{len(results)} successful ({success_rate:.1f}%), {processing_errors} processing errors"
                logger.info(f"✅ {message}")
                return True, message
            else:
                message = f"Separated upload failed: 0/{len(results)} successful, {processing_errors} processing errors"
                logger.error(f"❌ {message}")
                return False, message
                
        except Exception as e:
            logger.error(f"❌ Compatibility wrapper error: {e}")
            return False, f"Separated upload failed: {str(e)}"
    
    def _get_relative_path(self, file_path: str, allowed_paths: List[str]) -> Optional[str]:
        """Get relative path from allowed paths"""
        file_path_obj = Path(file_path).resolve()
        
        for root in allowed_paths:
            root_path = Path(root).resolve()
            try:
                relative = file_path_obj.relative_to(root_path)
                return str(relative)
            except ValueError:
                continue
        
        return None
    
    def _parse_codes_from_path(self, relative_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse codes from relative path"""
        try:
            parts = Path(relative_path).parts
            if len(parts) < 4:
                return None, None, None

            unit_code = parts[0].split("_")[0]
            outlet_code = parts[1].split("_")[0]
            photo_type_code = parts[2].split("_")[0]

            return unit_code, outlet_code, photo_type_code
        except Exception as e:
            logger.error(f"❌ Path parsing error: {e}")
            return None, None, None
    
    def _resolve_codes_to_ids(self, unit_code: str, outlet_code: str, photo_type_code: str) -> Tuple[str, str, str]:
        """
        Convert codes to UUIDs
        
        TODO: Implement proper code-to-ID resolution based on your system.
        This could be done via:
        1. API call to get ID mappings
        2. Local cache/database lookup
        3. Configuration file mapping
        
        For now, returning codes as IDs (you'll need to implement proper resolution)
        """
        # TEMPORARY: Return codes as IDs
        # In production, you should resolve these to actual UUIDs
        logger.debug(f"🔧 Code resolution: {unit_code} -> {outlet_code} -> {photo_type_code}")
        
        # Example implementation (replace with your actual logic):
        # try:
        #     # Option 1: API call
        #     response = requests.get(f"{self.api_base_url}/resolve-codes", 
        #                           params={"unit": unit_code, "outlet": outlet_code, "type": photo_type_code})
        #     if response.status_code == 200:
        #         data = response.json()
        #         return data["unit_id"], data["outlet_id"], data["photo_type_id"]
        # except:
        #     pass
        
        # Option 2: Local mapping (implement based on your system)
        # CODE_TO_ID_MAP = {
        #     ("01", "01", "01"): ("uuid1", "uuid2", "uuid3"),
        #     # ... more mappings
        # }
        # return CODE_TO_ID_MAP.get((unit_code, outlet_code, photo_type_code), (unit_code, outlet_code, photo_type_code))
        
        return unit_code, outlet_code, photo_type_code


# ===== GLOBAL INSTANCES =====

# Create global compatibility wrapper for easy integration
_global_wrapper = None

def get_separated_upload_wrapper(api_base_url: str = API_BASE) -> CompatibilityWrapper:
    """Get or create global compatibility wrapper"""
    global _global_wrapper
    if _global_wrapper is None or _global_wrapper.api_base_url != api_base_url:
        _global_wrapper = CompatibilityWrapper(api_base_url)
    return _global_wrapper


def process_batch_faces_and_upload_separated(files_list: List[str], 
                                           allowed_paths: List[str],
                                           api_base_url: str = API_BASE,
                                           progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
    """
    Global function that can be used as drop-in replacement for existing batch upload functions
    
    This function provides the same interface as your existing batch upload functions
    but uses the new separated upload system internally.
    
    Args:
        files_list: List of file paths to process
        allowed_paths: List of allowed base paths  
        api_base_url: Backend API base URL
        progress_callback: Optional progress callback
        
    Returns:
        Tuple[bool, str]: (success, message)
    """
    wrapper = get_separated_upload_wrapper(api_base_url)
    return wrapper.process_batch_faces_and_upload_separated(files_list, allowed_paths, progress_callback)


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    """Example usage of separated upload system"""
    
    # Example 1: Direct usage
    photos_data = [
        {
            'file_path': '/path/to/photo1.jpg',
            'unit_id': 'unit-uuid-1',
            'outlet_id': 'outlet-uuid-1',
            'photo_type_id': 'type-uuid-1',
            'faces_data': [
                {
                    'embedding': [0.1] * 512,  # 512-dim embedding
                    'bbox': {'x': 100, 'y': 100, 'w': 50, 'h': 50},
                    'confidence': 0.95
                }
            ]
        }
    ]
    
    def progress_callback(message: str, current: int, total: int):
        print(f"Progress: {message} ({current}/{total})")
    
    results = upload_photos_with_separated_system(
        photos_data, 
        API_BASE, 
        progress_callback
    )
    
    for result in results:
        if result.success:
            print(f"✅ {result.file_path}: {result.original_url}")
        else:
            print(f"❌ {result.file_path}: {result.error_message}")
    
    # Example 2: Compatibility mode (drop-in replacement)
    files_list = ['/path/to/photo1.jpg', '/path/to/photo2.jpg']
    allowed_paths = ['/allowed/base/path']
    
    success, message = process_batch_faces_and_upload_separated(
        files_list, 
        allowed_paths, 
        API_BASE,
        progress_callback
    )
    
    print(f"Batch upload result: {success} - {message}")