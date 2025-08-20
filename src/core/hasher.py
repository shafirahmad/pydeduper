"""File hashing functionality for PyDeduper."""

import hashlib
from pathlib import Path
from typing import Optional, BinaryIO
import logging

from ..config import Config, HashAlgorithm

logger = logging.getLogger(__name__)


class FileHasher:
    """Handles file hashing operations."""
    
    def __init__(self, config: Config = None):
        """Initialize the hasher with configuration."""
        self.config = config or Config()
        self.chunk_size = self.config.CHUNK_SIZE
        self.algorithm = self.config.DEFAULT_HASH_ALGORITHM
    
    def _get_hash_object(self):
        """Get the appropriate hash object based on the configured algorithm."""
        algorithm_map = {
            HashAlgorithm.MD5: hashlib.md5,
            HashAlgorithm.SHA1: hashlib.sha1,
            HashAlgorithm.SHA256: hashlib.sha256,
            HashAlgorithm.SHA512: hashlib.sha512
        }
        return algorithm_map[self.algorithm]()
    
    def hash_file(self, file_path: Path) -> Optional[str]:
        """
        Calculate the hash of a complete file.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            Hexadecimal hash string or None if error
        """
        try:
            hash_obj = self._get_hash_object()
            
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.chunk_size):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except (IOError, OSError) as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None
    
    def quick_hash(self, file_path: Path) -> Optional[str]:
        """
        Calculate a quick hash using only the first N bytes of a file.
        Useful for initial filtering before computing full hashes.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            Hexadecimal hash string or None if error
        """
        try:
            hash_obj = self._get_hash_object()
            
            with open(file_path, 'rb') as f:
                chunk = f.read(self.config.QUICK_HASH_SIZE)
                if chunk:
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except (IOError, OSError) as e:
            logger.error(f"Error quick hashing file {file_path}: {e}")
            return None
    
    def hash_file_partial(self, file_path: Path, positions: list = None) -> Optional[str]:
        """
        Calculate hash using samples from specific positions in the file.
        This provides a balance between speed and accuracy.
        
        Args:
            file_path: Path to the file to hash
            positions: List of (offset, size) tuples for sampling
            
        Returns:
            Hexadecimal hash string or None if error
        """
        if positions is None:
            # Default sampling strategy: beginning, middle, and end
            try:
                file_size = file_path.stat().st_size
                if file_size <= self.config.QUICK_HASH_SIZE * 3:
                    # File is small, just hash it completely
                    return self.hash_file(file_path)
                
                positions = [
                    (0, self.config.QUICK_HASH_SIZE),  # Beginning
                    (file_size // 2, self.config.QUICK_HASH_SIZE),  # Middle
                    (max(0, file_size - self.config.QUICK_HASH_SIZE), self.config.QUICK_HASH_SIZE)  # End
                ]
            except OSError:
                return None
        
        try:
            hash_obj = self._get_hash_object()
            
            with open(file_path, 'rb') as f:
                for offset, size in positions:
                    f.seek(offset)
                    chunk = f.read(size)
                    if chunk:
                        hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except (IOError, OSError) as e:
            logger.error(f"Error partial hashing file {file_path}: {e}")
            return None
    
    def hash_stream(self, stream: BinaryIO) -> str:
        """
        Calculate hash from a binary stream.
        
        Args:
            stream: Binary stream to hash
            
        Returns:
            Hexadecimal hash string
        """
        hash_obj = self._get_hash_object()
        
        while chunk := stream.read(self.chunk_size):
            hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """
        Verify that a file matches an expected hash.
        
        Args:
            file_path: Path to the file to verify
            expected_hash: Expected hash value
            
        Returns:
            True if hash matches, False otherwise
        """
        actual_hash = self.hash_file(file_path)
        return actual_hash == expected_hash if actual_hash else False


class HashCache:
    """Simple in-memory cache for file hashes to avoid recomputation."""
    
    def __init__(self):
        """Initialize the cache."""
        self._cache = {}
        self._quick_cache = {}
    
    def get_hash(self, file_path: Path, mtime: float = None) -> Optional[str]:
        """Get cached hash if available and still valid."""
        key = str(file_path)
        if key in self._cache:
            cached_mtime, cached_hash = self._cache[key]
            if mtime is None or cached_mtime == mtime:
                return cached_hash
        return None
    
    def set_hash(self, file_path: Path, hash_value: str, mtime: float = None):
        """Store hash in cache."""
        key = str(file_path)
        self._cache[key] = (mtime, hash_value)
    
    def get_quick_hash(self, file_path: Path, mtime: float = None) -> Optional[str]:
        """Get cached quick hash if available and still valid."""
        key = str(file_path)
        if key in self._quick_cache:
            cached_mtime, cached_hash = self._quick_cache[key]
            if mtime is None or cached_mtime == mtime:
                return cached_hash
        return None
    
    def set_quick_hash(self, file_path: Path, hash_value: str, mtime: float = None):
        """Store quick hash in cache."""
        key = str(file_path)
        self._quick_cache[key] = (mtime, hash_value)
    
    def clear(self):
        """Clear all cached hashes."""
        self._cache.clear()
        self._quick_cache.clear()
    
    def remove(self, file_path: Path):
        """Remove specific file from cache."""
        key = str(file_path)
        self._cache.pop(key, None)
        self._quick_cache.pop(key, None)