"""Configuration settings for PyDeduper."""

import os
from pathlib import Path
from enum import Enum


class HashAlgorithm(Enum):
    """Available hash algorithms."""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"


class Config:
    """Configuration settings for PyDeduper."""
    
    # File reading settings
    CHUNK_SIZE = 8192  # Size of chunks to read when hashing files
    
    # Hash settings
    DEFAULT_HASH_ALGORITHM = HashAlgorithm.SHA256
    QUICK_HASH_SIZE = 1024  # Size to read for quick hash (first N bytes)
    
    # Database settings
    DEFAULT_DB_PATH = Path("data/pydeduper.db")
    
    # Scanning settings
    MAX_SCAN_DEPTH = None  # None for unlimited depth
    FOLLOW_SYMLINKS = False
    IGNORE_HIDDEN = False
    
    # Performance settings
    BATCH_SIZE = 1000  # Number of files to process in a batch
    
    # Output settings
    VERBOSE = False
    DEBUG = False
    
    # File patterns to ignore
    IGNORE_PATTERNS = [
        "*.tmp",
        "~*",
        "*.cache",
        "Thumbs.db",
        ".DS_Store"
    ]
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        config = cls()
        
        if "PYDEDUPER_DB_PATH" in os.environ:
            config.DEFAULT_DB_PATH = Path(os.environ["PYDEDUPER_DB_PATH"])
        
        if "PYDEDUPER_CHUNK_SIZE" in os.environ:
            config.CHUNK_SIZE = int(os.environ["PYDEDUPER_CHUNK_SIZE"])
        
        if "PYDEDUPER_HASH_ALGORITHM" in os.environ:
            algo = os.environ["PYDEDUPER_HASH_ALGORITHM"].lower()
            config.DEFAULT_HASH_ALGORITHM = HashAlgorithm(algo)
        
        if "PYDEDUPER_VERBOSE" in os.environ:
            config.VERBOSE = os.environ["PYDEDUPER_VERBOSE"].lower() in ("true", "1", "yes")
        
        return config