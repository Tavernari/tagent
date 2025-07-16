"""
Memory Persistence Backend for TAgent Pipeline System.

This module implements robust memory persistence for pipeline execution with multiple
storage backends including file-based, database, and Redis storage options.
"""

import json
import pickle
import shutil
import sqlite3
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from .state import PipelineMemory
from .models import PersistenceManagerSummary


logger = logging.getLogger(__name__)


class StorageBackendType(Enum):
    """Types of storage backends."""
    FILE_JSON = "file_json"
    FILE_PICKLE = "file_pickle"
    SQLITE = "sqlite"
    REDIS = "redis"
    MEMORY = "memory"


@dataclass
class PersistenceConfig:
    """Configuration for persistence backend."""
    backend_type: StorageBackendType
    base_path: str = "./pipeline_memory"
    backup_enabled: bool = True
    backup_count: int = 5
    compression_enabled: bool = False
    encryption_enabled: bool = False
    retention_days: int = 30
    auto_cleanup: bool = True
    connection_params: Dict[str, Any] = field(default_factory=dict)


class PersistenceError(Exception):
    """Base exception for persistence errors."""
    pass


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.backup_enabled = config.backup_enabled
        self.backup_count = config.backup_count
        self.retention_days = config.retention_days
        self.auto_cleanup = config.auto_cleanup
    
    @abstractmethod
    async def save(self, pipeline_id: str, data: Dict[str, Any]) -> bool:
        """Save pipeline memory data."""
        pass
    
    @abstractmethod
    async def load(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Load pipeline memory data."""
        pass
    
    @abstractmethod
    async def delete(self, pipeline_id: str) -> bool:
        """Delete pipeline memory data."""
        pass
    
    @abstractmethod
    async def list_pipelines(self) -> List[str]:
        """List all stored pipeline IDs."""
        pass
    
    @abstractmethod
    async def exists(self, pipeline_id: str) -> bool:
        """Check if pipeline data exists."""
        pass
    
    async def cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        if not self.auto_cleanup:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        pipelines = await self.list_pipelines()
        
        for pipeline_id in pipelines:
            if await self._is_pipeline_old(pipeline_id, cutoff_date):
                await self.delete(pipeline_id)
                logger.info(f"Cleaned up old pipeline data: {pipeline_id}")
    
    @abstractmethod
    async def _is_pipeline_old(self, pipeline_id: str, cutoff_date: datetime) -> bool:
        """Check if pipeline data is older than cutoff date."""
        pass
    
    def _create_backup_path(self, pipeline_id: str) -> Path:
        """Create backup path for pipeline."""
        return self.base_path / "backups" / pipeline_id
    
    async def _create_backup(self, pipeline_id: str, data: Dict[str, Any]):
        """Create backup of pipeline data."""
        if not self.backup_enabled:
            return
        
        backup_dir = self._create_backup_path(pipeline_id)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"backup_{timestamp}.json"
        
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)
        
        # Clean up old backups
        await self._cleanup_old_backups(backup_dir)
    
    async def _cleanup_old_backups(self, backup_dir: Path):
        """Clean up old backup files."""
        if not backup_dir.exists():
            return
        
        backup_files = list(backup_dir.glob("backup_*.json"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent backups
        for backup_file in backup_files[self.backup_count:]:
            backup_file.unlink()
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, 'dict'):
            return obj.dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class FileStorageBackend(StorageBackend):
    """File-based storage backend supporting JSON and pickle formats."""
    
    def __init__(self, config: PersistenceConfig):
        super().__init__(config)
        self.use_json = config.backend_type == StorageBackendType.FILE_JSON
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "pipelines").mkdir(exist_ok=True)
        (self.base_path / "backups").mkdir(exist_ok=True)
        (self.base_path / "temp").mkdir(exist_ok=True)
    
    async def save(self, pipeline_id: str, data: Dict[str, Any]) -> bool:
        """Save data to file."""
        try:
            file_path = self._get_pipeline_path(pipeline_id)
            temp_path = self.base_path / "temp" / f"{pipeline_id}.tmp"
            
            # Create backup first
            if file_path.exists():
                existing_data = await self.load(pipeline_id)
                if existing_data:
                    await self._create_backup(pipeline_id, existing_data)
            
            # Save to temporary file first
            if self.use_json:
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2, default=self._json_serializer)
            else:
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
            
            # Atomic move to final location
            shutil.move(str(temp_path), str(file_path))
            
            logger.debug(f"Saved pipeline data: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pipeline data {pipeline_id}: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    async def load(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Load data from file."""
        try:
            file_path = self._get_pipeline_path(pipeline_id)
            
            if not file_path.exists():
                return None
            
            if self.use_json:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            logger.debug(f"Loaded pipeline data: {pipeline_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load pipeline data {pipeline_id}: {e}")
            return None
    
    async def delete(self, pipeline_id: str) -> bool:
        """Delete pipeline data and backups."""
        try:
            file_path = self._get_pipeline_path(pipeline_id)
            backup_dir = self._create_backup_path(pipeline_id)
            
            # Delete main file
            if file_path.exists():
                file_path.unlink()
            
            # Delete backup directory
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            
            logger.debug(f"Deleted pipeline data: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete pipeline data {pipeline_id}: {e}")
            return False
    
    async def list_pipelines(self) -> List[str]:
        """List all stored pipeline IDs."""
        try:
            pipeline_dir = self.base_path / "pipelines"
            extension = ".json" if self.use_json else ".pkl"
            
            pipeline_files = list(pipeline_dir.glob(f"*{extension}"))
            pipeline_ids = [f.stem for f in pipeline_files]
            
            return pipeline_ids
            
        except Exception as e:
            logger.error(f"Failed to list pipelines: {e}")
            return []
    
    async def exists(self, pipeline_id: str) -> bool:
        """Check if pipeline data exists."""
        file_path = self._get_pipeline_path(pipeline_id)
        return file_path.exists()
    
    async def _is_pipeline_old(self, pipeline_id: str, cutoff_date: datetime) -> bool:
        """Check if pipeline data is older than cutoff date."""
        file_path = self._get_pipeline_path(pipeline_id)
        
        if not file_path.exists():
            return False
        
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        return file_mtime < cutoff_date
    
    def _get_pipeline_path(self, pipeline_id: str) -> Path:
        """Get file path for pipeline."""
        extension = ".json" if self.use_json else ".pkl"
        return self.base_path / "pipelines" / f"{pipeline_id}{extension}"


class SQLiteStorageBackend(StorageBackend):
    """SQLite database storage backend."""
    
    def __init__(self, config: PersistenceConfig):
        super().__init__(config)
        self.db_path = self.base_path / "pipelines.db"
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        asyncio.create_task(self._initialize_database())
    
    async def _initialize_database(self):
        """Initialize SQLite database schema."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create pipelines table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipelines (
                    pipeline_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pipelines_updated_at 
                ON pipelines(updated_at)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.debug("SQLite database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise PersistenceError(f"Database initialization failed: {e}")
    
    async def save(self, pipeline_id: str, data: Dict[str, Any]) -> bool:
        """Save data to SQLite database."""
        try:
            # Create backup first
            existing_data = await self.load(pipeline_id)
            if existing_data:
                await self._create_backup(pipeline_id, existing_data)
            
            data_json = json.dumps(data, default=self._json_serializer)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO pipelines (pipeline_id, data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (pipeline_id, data_json))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Saved pipeline data to SQLite: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pipeline data to SQLite {pipeline_id}: {e}")
            return False
    
    async def load(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Load data from SQLite database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT data FROM pipelines WHERE pipeline_id = ?
            ''', (pipeline_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                data = json.loads(result[0])
                logger.debug(f"Loaded pipeline data from SQLite: {pipeline_id}")
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load pipeline data from SQLite {pipeline_id}: {e}")
            return None
    
    async def delete(self, pipeline_id: str) -> bool:
        """Delete pipeline data from SQLite database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM pipelines WHERE pipeline_id = ?
            ''', (pipeline_id,))
            
            conn.commit()
            conn.close()
            
            # Delete backup directory
            backup_dir = self._create_backup_path(pipeline_id)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            
            logger.debug(f"Deleted pipeline data from SQLite: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete pipeline data from SQLite {pipeline_id}: {e}")
            return False
    
    async def list_pipelines(self) -> List[str]:
        """List all stored pipeline IDs."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT pipeline_id FROM pipelines')
            results = cursor.fetchall()
            conn.close()
            
            pipeline_ids = [row[0] for row in results]
            return pipeline_ids
            
        except Exception as e:
            logger.error(f"Failed to list pipelines from SQLite: {e}")
            return []
    
    async def exists(self, pipeline_id: str) -> bool:
        """Check if pipeline data exists in SQLite database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 1 FROM pipelines WHERE pipeline_id = ? LIMIT 1
            ''', (pipeline_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Failed to check pipeline existence in SQLite {pipeline_id}: {e}")
            return False
    
    async def _is_pipeline_old(self, pipeline_id: str, cutoff_date: datetime) -> bool:
        """Check if pipeline data is older than cutoff date."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT updated_at FROM pipelines WHERE pipeline_id = ?
            ''', (pipeline_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                updated_at = datetime.fromisoformat(result[0])
                return updated_at < cutoff_date
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check pipeline age in SQLite {pipeline_id}: {e}")
            return False


class MemoryStorageBackend(StorageBackend):
    """In-memory storage backend for testing and development."""
    
    def __init__(self, config: PersistenceConfig):
        super().__init__(config)
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.timestamps: Dict[str, datetime] = {}
    
    async def save(self, pipeline_id: str, data: Dict[str, Any]) -> bool:
        """Save data to memory."""
        try:
            self.storage[pipeline_id] = data.copy()
            self.timestamps[pipeline_id] = datetime.now()
            
            logger.debug(f"Saved pipeline data to memory: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pipeline data to memory {pipeline_id}: {e}")
            return False
    
    async def load(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Load data from memory."""
        try:
            data = self.storage.get(pipeline_id)
            if data:
                logger.debug(f"Loaded pipeline data from memory: {pipeline_id}")
                return data.copy()
            return None
            
        except Exception as e:
            logger.error(f"Failed to load pipeline data from memory {pipeline_id}: {e}")
            return None
    
    async def delete(self, pipeline_id: str) -> bool:
        """Delete pipeline data from memory."""
        try:
            if pipeline_id in self.storage:
                del self.storage[pipeline_id]
            if pipeline_id in self.timestamps:
                del self.timestamps[pipeline_id]
            
            logger.debug(f"Deleted pipeline data from memory: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete pipeline data from memory {pipeline_id}: {e}")
            return False
    
    async def list_pipelines(self) -> List[str]:
        """List all stored pipeline IDs."""
        return list(self.storage.keys())
    
    async def exists(self, pipeline_id: str) -> bool:
        """Check if pipeline data exists in memory."""
        return pipeline_id in self.storage
    
    async def _is_pipeline_old(self, pipeline_id: str, cutoff_date: datetime) -> bool:
        """Check if pipeline data is older than cutoff date."""
        timestamp = self.timestamps.get(pipeline_id)
        if timestamp:
            return timestamp < cutoff_date
        return False


class PipelineMemoryManager:
    """Central memory management for pipeline execution."""
    
    def __init__(self, config: PersistenceConfig = None):
        self.config = config or PersistenceConfig(StorageBackendType.FILE_JSON)
        self.storage_backend = self._create_storage_backend()
        self.active_pipelines: Dict[str, PipelineMemory] = {}
        self.shared_memory: Dict[str, Any] = {}
        
        # Start cleanup task
        if self.config.auto_cleanup:
            asyncio.create_task(self._periodic_cleanup())
    
    def _create_storage_backend(self) -> StorageBackend:
        """Create storage backend based on configuration."""
        backend_type = self.config.backend_type
        
        if backend_type in [StorageBackendType.FILE_JSON, StorageBackendType.FILE_PICKLE]:
            return FileStorageBackend(self.config)
        elif backend_type == StorageBackendType.SQLITE:
            return SQLiteStorageBackend(self.config)
        elif backend_type == StorageBackendType.MEMORY:
            return MemoryStorageBackend(self.config)
        else:
            raise ValueError(f"Unsupported storage backend: {backend_type}")
    
    def get_pipeline_memory(self, pipeline_id: str) -> PipelineMemory:
        """Get or create memory space for a pipeline."""
        if pipeline_id not in self.active_pipelines:
            self.active_pipelines[pipeline_id] = PipelineMemory(pipeline_id)
        return self.active_pipelines[pipeline_id]
    
    async def persist_memory(self, pipeline_id: str, memory: PipelineMemory = None) -> bool:
        """Persist memory to storage backend."""
        if memory is None:
            memory = self.active_pipelines.get(pipeline_id)
        
        if memory is None:
            logger.warning(f"No memory found for pipeline: {pipeline_id}")
            return False
        
        serialized_data = memory.serialize()
        success = await self.storage_backend.save(pipeline_id, serialized_data)
        
        if success:
            logger.debug(f"Persisted memory for pipeline: {pipeline_id}")
        else:
            logger.error(f"Failed to persist memory for pipeline: {pipeline_id}")
        
        return success
    
    async def restore_memory(self, pipeline_id: str) -> Optional[PipelineMemory]:
        """Restore memory from storage backend."""
        data = await self.storage_backend.load(pipeline_id)
        
        if data:
            memory = PipelineMemory.deserialize(data)
            self.active_pipelines[pipeline_id] = memory
            logger.debug(f"Restored memory for pipeline: {pipeline_id}")
            return memory
        
        logger.debug(f"No stored memory found for pipeline: {pipeline_id}")
        return None
    
    async def delete_memory(self, pipeline_id: str) -> bool:
        """Delete memory for a pipeline."""
        # Remove from active pipelines
        if pipeline_id in self.active_pipelines:
            del self.active_pipelines[pipeline_id]
        
        # Delete from storage
        success = await self.storage_backend.delete(pipeline_id)
        
        if success:
            logger.debug(f"Deleted memory for pipeline: {pipeline_id}")
        else:
            logger.error(f"Failed to delete memory for pipeline: {pipeline_id}")
        
        return success
    
    async def list_stored_pipelines(self) -> List[str]:
        """List all pipelines with stored memory."""
        return await self.storage_backend.list_pipelines()
    
    async def memory_exists(self, pipeline_id: str) -> bool:
        """Check if memory exists for a pipeline."""
        return await self.storage_backend.exists(pipeline_id)
    
    def get_shared_memory(self, key: str, default: Any = None) -> Any:
        """Get shared memory value."""
        return self.shared_memory.get(key, default)
    
    def set_shared_memory(self, key: str, value: Any):
        """Set shared memory value."""
        self.shared_memory[key] = value
    
    async def cleanup_old_memory(self):
        """Clean up old memory data."""
        await self.storage_backend.cleanup_old_data()
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_old_memory()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def get_manager_summary(self) -> PersistenceManagerSummary:
        """Get summary of memory manager state."""
        return PersistenceManagerSummary(
            backend_type=self.config.backend_type.value,
            base_path=self.config.base_path,
            backup_enabled=self.config.backup_enabled,
            retention_days=self.config.retention_days,
            auto_cleanup=self.config.auto_cleanup,
            active_pipelines=list(self.active_pipelines.keys()),
            shared_memory_keys=list(self.shared_memory.keys())
        )