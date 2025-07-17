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
import time
import uuid

from pydantic import BaseModel

from .state import PipelineMemory
from .models import (
    PersistenceManagerSummary, Checkpoint, PipelineMemoryState,
    ExecutionHistoryEvent, AuditLog
)


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


class CheckpointNotFoundError(PersistenceError):
    """Raised when a requested checkpoint is not found."""
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
    async def save(self, key: str, data: Dict[str, Any]) -> bool:
        """Save data."""
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data."""
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all stored keys with a given prefix."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if data exists."""
        pass
    
    async def cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        if not self.auto_cleanup:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        keys = await self.list_keys()
        
        for key in keys:
            if await self._is_key_old(key, cutoff_date):
                await self.delete(key)
                logger.info(f"Cleaned up old data: {key}")
    
    @abstractmethod
    async def _is_key_old(self, key: str, cutoff_date: datetime) -> bool:
        """Check if data is older than cutoff date."""
        pass
    
    def _create_backup_path(self, key: str) -> Path:
        """Create backup path for a key."""
        return self.base_path / "backups" / key
    
    async def _create_backup(self, key: str, data: Dict[str, Any]):
        """Create backup of data."""
        if not self.backup_enabled:
            return
        
        backup_dir = self._create_backup_path(key)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"backup_{timestamp}.json"
        
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)
        
        await self._cleanup_old_backups(backup_dir)
    
    async def _cleanup_old_backups(self, backup_dir: Path):
        """Clean up old backup files."""
        if not backup_dir.exists():
            return
        
        backup_files = list(backup_dir.glob("backup_*.json"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
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
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class FileStorageBackend(StorageBackend):
    """File-based storage backend supporting JSON and pickle formats."""
    
    def __init__(self, config: PersistenceConfig):
        super().__init__(config)
        self.use_json = config.backend_type == StorageBackendType.FILE_JSON
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        (self.base_path / "data").mkdir(exist_ok=True)
        (self.base_path / "backups").mkdir(exist_ok=True)
        (self.base_path / "temp").mkdir(exist_ok=True)
    
    async def save(self, key: str, data: Dict[str, Any]) -> bool:
        try:
            file_path = self._get_path(key)
            temp_path = self.base_path / "temp" / f"{key}.tmp"
            
            if file_path.exists():
                existing_data = await self.load(key)
                if existing_data:
                    await self._create_backup(key, existing_data)
            
            if self.use_json:
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2, default=self._json_serializer)
            else:
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
            
            shutil.move(str(temp_path), str(file_path))
            logger.debug(f"Saved data for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save data for key {key}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            file_path = self._get_path(key)
            if not file_path.exists():
                return None
            
            if self.use_json:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            logger.debug(f"Loaded data for key: {key}")
            return data
        except Exception as e:
            logger.error(f"Failed to load data for key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        try:
            file_path = self._get_path(key)
            backup_dir = self._create_backup_path(key)
            
            if file_path.exists():
                file_path.unlink()
            
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            
            logger.debug(f"Deleted data for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete data for key {key}: {e}")
            return False
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        try:
            data_dir = self.base_path / "data"
            extension = ".json" if self.use_json else ".pkl"
            
            files = list(data_dir.glob(f"{prefix}*{extension}"))
            keys = [f.stem for f in files]
            return keys
        except Exception as e:
            logger.error(f"Failed to list keys with prefix '{prefix}': {e}")
            return []
    
    async def exists(self, key: str) -> bool:
        file_path = self._get_path(key)
        return file_path.exists()
    
    async def _is_key_old(self, key: str, cutoff_date: datetime) -> bool:
        file_path = self._get_path(key)
        if not file_path.exists():
            return False
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        return file_mtime < cutoff_date
    
    def _get_path(self, key: str) -> Path:
        extension = ".json" if self.use_json else ".pkl"
        # Sanitize key to prevent directory traversal
        safe_key = Path(key).name
        return self.base_path / "data" / f"{safe_key}{extension}"


class SQLiteStorageBackend(StorageBackend):
    # ... (implementation can be adapted if needed, for now focusing on File backend)
    pass


class MemoryStorageBackend(StorageBackend):
    # ... (implementation can be adapted if needed)
    pass


class CheckpointManager:
    """Manages pipeline checkpoints for recovery."""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend

    async def create_checkpoint(
        self, 
        pipeline_id: str, 
        checkpoint_type: str,
        state: PipelineMemoryState,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """Create a checkpoint with metadata."""
        checkpoint_id = f"{pipeline_id}_{checkpoint_type}_{int(time.time())}"
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            pipeline_id=pipeline_id,
            checkpoint_type=checkpoint_type,
            state=state,
            metadata=metadata or {},
        )
        
        await self.storage.save(f"checkpoint_{checkpoint_id}", checkpoint.model_dump())
        await self._update_checkpoint_index(pipeline_id, checkpoint_id)
        return checkpoint

    async def restore_from_checkpoint(
        self, 
        checkpoint_id: str
    ) -> PipelineMemoryState:
        """Restore pipeline state from checkpoint."""
        checkpoint_data = await self.storage.load(f"checkpoint_{checkpoint_id}")
        if not checkpoint_data:
            raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")
        
        return PipelineMemoryState(**checkpoint_data['state'])

    async def list_checkpoints(self, pipeline_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a pipeline."""
        index_data = await self.storage.load(f"checkpoint_index_{pipeline_id}")
        return index_data.get('checkpoints', []) if index_data else []

    async def _update_checkpoint_index(self, pipeline_id: str, checkpoint_id: str):
        """Update the index of checkpoints for a pipeline."""
        index_key = f"checkpoint_index_{pipeline_id}"
        index_data = await self.storage.load(index_key) or {"checkpoints": []}
        index_data["checkpoints"].append(checkpoint_id)
        await self.storage.save(index_key, index_data)


class HistoryManager:
    """Tracks and manages pipeline execution history."""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend

    async def record_event(self, event: ExecutionHistoryEvent):
        """Record an execution history event."""
        history_key = f"history_{event.pipeline_id}"
        history_data = await self.storage.load(history_key) or {"events": []}
        history_data["events"].append(event.model_dump())
        await self.storage.save(history_key, history_data)

    async def get_execution_history(self, pipeline_id: str, limit: int = 100) -> List[ExecutionHistoryEvent]:
        """Get execution history for a pipeline."""
        history_data = await self.storage.load(f"history_{pipeline_id}")
        if not history_data:
            return []
        
        events_data = history_data.get('events', [])
        events = [ExecutionHistoryEvent(**e) for e in events_data]
        return sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]


class AuditManager:
    """Manages audit trails for persistence operations."""

    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend

    async def log_event(self, event_name: str, pipeline_id: str, details: Dict[str, Any]):
        """Log an audit event."""
        log = AuditLog(
            pipeline_id=pipeline_id,
            event_name=event_name,
            details=details,
        )
        audit_key = f"audit_{pipeline_id}"
        audit_data = await self.storage.load(audit_key) or {"logs": []}
        audit_data["logs"].append(log.model_dump())
        await self.storage.save(audit_key, audit_data)


class PipelineMemoryManager:
    """Central memory management for pipeline execution."""
    
    def __init__(self, config: PersistenceConfig = None):
        self.config = config or PersistenceConfig(backend_type=StorageBackendType.FILE_JSON)
        self.storage_backend = self._create_storage_backend()
        self.active_pipelines: Dict[str, PipelineMemory] = {}
        
        self.checkpoint_manager = CheckpointManager(self.storage_backend)
        self.history_manager = HistoryManager(self.storage_backend)
        self.audit_manager = AuditManager(self.storage_backend)

        if self.config.auto_cleanup:
            asyncio.create_task(self._periodic_cleanup())
    
    def _create_storage_backend(self) -> StorageBackend:
        """Create storage backend based on configuration."""
        backend_type = self.config.backend_type
        
        if backend_type in [StorageBackendType.FILE_JSON, StorageBackendType.FILE_PICKLE]:
            return FileStorageBackend(self.config)
        elif backend_type == StorageBackendType.SQLITE:
            # return SQLiteStorageBackend(self.config)
            raise NotImplementedError("SQLite backend not fully implemented yet.")
        elif backend_type == StorageBackendType.MEMORY:
            # return MemoryStorageBackend(self.config)
            raise NotImplementedError("Memory backend not fully implemented yet.")
        else:
            raise ValueError(f"Unsupported storage backend: {backend_type}")
    
    def get_pipeline_memory(self, pipeline_id: str) -> PipelineMemory:
        """Get or create memory space for a pipeline."""
        if pipeline_id not in self.active_pipelines:
            self.active_pipelines[pipeline_id] = PipelineMemory(pipeline_id)
        return self.active_pipelines[pipeline_id]
    
    async def persist_memory(self, pipeline_id: str, memory: PipelineMemory) -> bool:
        """Persist memory to storage backend."""
        serialized_data = memory.serialize()
        success = await self.storage_backend.save(f"state_{pipeline_id}", serialized_data)
        
        if success:
            await self.audit_manager.log_event(
                'state_saved',
                pipeline_id,
                {'state_size': len(json.dumps(serialized_data, default=str))}
            )
            logger.debug(f"Persisted memory for pipeline: {pipeline_id}")
        else:
            logger.error(f"Failed to persist memory for pipeline: {pipeline_id}")
        
        return success
    
    async def restore_memory(self, pipeline_id: str) -> Optional[PipelineMemory]:
        """Restore memory from storage backend."""
        data = await self.storage_backend.load(f"state_{pipeline_id}")
        
        if data:
            memory = PipelineMemory.deserialize(data)
            self.active_pipelines[pipeline_id] = memory
            await self.audit_manager.log_event('state_loaded', pipeline_id, {})
            logger.debug(f"Restored memory for pipeline: {pipeline_id}")
            return memory
        
        logger.debug(f"No stored memory found for pipeline: {pipeline_id}")
        return None
    
    async def delete_memory(self, pipeline_id: str) -> bool:
        """Delete memory for a pipeline."""
        if pipeline_id in self.active_pipelines:
            del self.active_pipelines[pipeline_id]
        
        success = await self.storage_backend.delete(f"state_{pipeline_id}")
        
        if success:
            logger.debug(f"Deleted memory for pipeline: {pipeline_id}")
        else:
            logger.error(f"Failed to delete memory for pipeline: {pipeline_id}")
        
        return success
    
    async def list_stored_pipelines(self) -> List[str]:
        """List all pipelines with stored memory."""
        keys = await self.storage_backend.list_keys(prefix="state_")
        return [key.replace("state_", "") for key in keys]
    
    async def memory_exists(self, pipeline_id: str) -> bool:
        """Check if memory exists for a pipeline."""
        return await self.storage_backend.exists(f"state_{pipeline_id}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.storage_backend.cleanup_old_data()
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
            shared_memory_keys=[] # This was removed from here
        )
