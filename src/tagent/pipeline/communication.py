"""
Inter-Pipeline Communication for TAgent Pipeline System.

This module implements robust inter-pipeline communication system for data sharing,
event broadcasting, shared memory spaces, and pipeline registry and discovery.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Priority levels for inter-pipeline messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class EventType(Enum):
    """Types of pipeline events."""
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    DATA_SHARED = "data_shared"
    CUSTOM = "custom"


class MessageType(Enum):
    """Types of inter-pipeline messages."""
    DATA = "data"
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    NOTIFICATION = "notification"


class PipelineMessage(BaseModel):
    """Message sent between pipelines."""
    message_id: str = Field(description="Unique message identifier")
    from_pipeline: str = Field(description="Source pipeline ID")
    to_pipeline: str = Field(description="Target pipeline ID")
    message_type: MessageType = Field(description="Type of message")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority")
    subject: str = Field(default="", description="Message subject")
    content: Any = Field(description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Message expiration time")
    reply_to: Optional[str] = Field(default=None, description="Message ID to reply to")
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.expires_at is not None and datetime.now() > self.expires_at


class PipelineEvent(BaseModel):
    """Event broadcast by pipelines."""
    event_id: str = Field(description="Unique event identifier") 
    event_type: EventType = Field(description="Type of event")
    source_pipeline: str = Field(description="Pipeline that generated the event")
    subject: str = Field(default="", description="Event subject")
    data: Any = Field(description="Event data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    tags: List[str] = Field(default_factory=list, description="Event tags for filtering")


class PipelineInfo(BaseModel):
    """Information about a registered pipeline."""
    pipeline_id: str = Field(description="Unique pipeline identifier")
    pipeline_name: str = Field(description="Human-readable pipeline name")
    description: str = Field(default="", description="Pipeline description")
    status: str = Field(description="Current pipeline status")
    capabilities: List[str] = Field(default_factory=list, description="Pipeline capabilities")
    endpoints: Dict[str, str] = Field(default_factory=dict, description="Available endpoints")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    registered_at: datetime = Field(default_factory=datetime.now, description="Registration timestamp")
    last_seen: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")


class EventSubscriber(BaseModel):
    """Event subscriber information."""
    subscriber_id: str = Field(description="Subscriber identifier")
    pipeline_id: str = Field(description="Pipeline ID of subscriber")
    event_patterns: List[str] = Field(description="Event patterns to match")
    callback_info: Dict[str, Any] = Field(description="Callback information")
    active: bool = Field(default=True, description="Whether subscription is active")
    subscribed_at: datetime = Field(default_factory=datetime.now, description="Subscription timestamp")
    
    async def matches_event(self, event: PipelineEvent) -> bool:
        """Check if event matches subscription patterns."""
        for pattern in self.event_patterns:
            if pattern == "*" or pattern == event.event_type.value:
                return True
            if pattern.endswith("*") and event.event_type.value.startswith(pattern[:-1]):
                return True
            if pattern in event.tags:
                return True
        return False


@dataclass
class SharedMemorySpace:
    """Shared memory space for related pipelines."""
    space_id: str
    name: str = ""
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    locks: Dict[str, asyncio.Lock] = field(default_factory=dict)
    authorized_pipelines: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    max_size: int = 1000  # Maximum number of entries
    
    async def write(self, key: str, value: Any, pipeline_id: str, overwrite: bool = True) -> bool:
        """Write data to shared space with access control."""
        if not self._is_authorized(pipeline_id):
            logger.warning(f"Unauthorized write attempt from {pipeline_id} to space {self.space_id}")
            return False
        
        async with self._get_lock(key):
            if not overwrite and key in self.data:
                return False
            
            # Check size limits
            if len(self.data) >= self.max_size and key not in self.data:
                logger.warning(f"Shared space {self.space_id} is full")
                return False
            
            self.data[key] = value
            self._log_access("write", key, pipeline_id, {"value_type": type(value).__name__})
            return True
    
    async def read(self, key: str, pipeline_id: str) -> Any:
        """Read data from shared space with access control."""
        if not self._is_authorized(pipeline_id):
            logger.warning(f"Unauthorized read attempt from {pipeline_id} to space {self.space_id}")
            return None
        
        async with self._get_lock(key):
            value = self.data.get(key)
            self._log_access("read", key, pipeline_id, {"found": value is not None})
            return value
    
    async def delete(self, key: str, pipeline_id: str) -> bool:
        """Delete data from shared space with access control."""
        if not self._is_authorized(pipeline_id):
            logger.warning(f"Unauthorized delete attempt from {pipeline_id} to space {self.space_id}")
            return False
        
        async with self._get_lock(key):
            if key in self.data:
                del self.data[key]
                self._log_access("delete", key, pipeline_id)
                return True
            return False
    
    async def list_keys(self, pipeline_id: str) -> List[str]:
        """List all keys in shared space."""
        if not self._is_authorized(pipeline_id):
            return []
        
        return list(self.data.keys())
    
    def _is_authorized(self, pipeline_id: str) -> bool:
        """Check if pipeline is authorized to access this space."""
        return not self.authorized_pipelines or pipeline_id in self.authorized_pipelines
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for key."""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]
    
    def _log_access(self, operation: str, key: str, pipeline_id: str, metadata: Dict[str, Any] = None):
        """Log access to shared space."""
        log_entry = {
            "operation": operation,
            "key": key,
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.access_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-500:]


class MessageQueue:
    """High-performance message queue with priority and filtering."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues: Dict[MessagePriority, deque] = {
            priority: deque() for priority in MessagePriority
        }
        self.message_index: Dict[str, PipelineMessage] = {}
        self.lock = asyncio.Lock()
        
    async def put(self, message: PipelineMessage) -> bool:
        """Put message in queue with priority handling."""
        async with self.lock:
            # Check if queue is full
            total_size = sum(len(q) for q in self.queues.values())
            if total_size >= self.max_size:
                # Remove oldest low priority message
                if self.queues[MessagePriority.LOW]:
                    old_msg = self.queues[MessagePriority.LOW].popleft()
                    if old_msg.message_id in self.message_index:
                        del self.message_index[old_msg.message_id]
                else:
                    logger.warning("Message queue is full, dropping message")
                    return False
            
            # Add message to appropriate priority queue
            self.queues[message.priority].append(message)
            self.message_index[message.message_id] = message
            return True
    
    async def get(self, timeout: Optional[float] = None) -> Optional[PipelineMessage]:
        """Get next message from queue with priority handling."""
        start_time = datetime.now()
        
        while True:
            async with self.lock:
                # Check priority queues in order
                for priority in [MessagePriority.URGENT, MessagePriority.HIGH, 
                               MessagePriority.NORMAL, MessagePriority.LOW]:
                    if self.queues[priority]:
                        message = self.queues[priority].popleft()
                        if message.message_id in self.message_index:
                            del self.message_index[message.message_id]
                        
                        # Check if message has expired
                        if message.is_expired():
                            continue
                        
                        return message
            
            # Check timeout
            if timeout is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    return None
            
            await asyncio.sleep(0.01)  # Brief pause before retry
    
    async def get_for_pipeline(self, pipeline_id: str, timeout: Optional[float] = None) -> Optional[PipelineMessage]:
        """Get next message for specific pipeline."""
        start_time = datetime.now()
        
        while True:
            async with self.lock:
                # Check all priority queues for messages to this pipeline
                for priority in [MessagePriority.URGENT, MessagePriority.HIGH, 
                               MessagePriority.NORMAL, MessagePriority.LOW]:
                    queue = self.queues[priority]
                    for i, message in enumerate(queue):
                        if message.to_pipeline == pipeline_id:
                            # Remove from queue
                            del queue[i]
                            if message.message_id in self.message_index:
                                del self.message_index[message.message_id]
                            
                            # Check if message has expired
                            if message.is_expired():
                                break  # Continue searching
                            
                            return message
            
            # Check timeout
            if timeout is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    return None
            
            await asyncio.sleep(0.01)  # Brief pause before retry
    
    async def size(self) -> int:
        """Get total number of messages in queue."""
        async with self.lock:
            return sum(len(q) for q in self.queues.values())


class PipelineCommunicator:
    """Handles communication between pipelines with advanced features."""
    
    def __init__(self, max_queue_size: int = 10000):
        # Message handling
        self.message_queue = MessageQueue(max_queue_size)
        self.reply_handlers: Dict[str, Callable] = {}
        
        # Event handling
        self.event_subscribers: Dict[str, List[EventSubscriber]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=1000)
        
        # Pipeline registry
        self.active_pipelines: Dict[str, PipelineInfo] = {}
        self.pipeline_heartbeats: Dict[str, datetime] = {}
        
        # Shared memory spaces
        self.shared_spaces: Dict[str, SharedMemorySpace] = {}
        
        # Communication statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "events_broadcast": 0,
            "events_delivered": 0,
            "pipelines_registered": 0,
            "active_subscriptions": 0
        }
    
    async def register_pipeline(self, pipeline_info: PipelineInfo):
        """Register a pipeline for communication."""
        self.active_pipelines[pipeline_info.pipeline_id] = pipeline_info
        self.pipeline_heartbeats[pipeline_info.pipeline_id] = datetime.now()
        self.stats["pipelines_registered"] += 1
        
        # Broadcast pipeline registration event
        await self.broadcast_event(PipelineEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.PIPELINE_STARTED,
            source_pipeline=pipeline_info.pipeline_id,
            subject="Pipeline Registered",
            data={"pipeline_name": pipeline_info.pipeline_name},
            tags=["registration", "lifecycle"]
        ))
        
        logger.info(f"Pipeline '{pipeline_info.pipeline_name}' registered with ID {pipeline_info.pipeline_id}")
    
    async def unregister_pipeline(self, pipeline_id: str):
        """Unregister a pipeline from communication."""
        if pipeline_id in self.active_pipelines:
            pipeline_info = self.active_pipelines[pipeline_id]
            del self.active_pipelines[pipeline_id]
            
            if pipeline_id in self.pipeline_heartbeats:
                del self.pipeline_heartbeats[pipeline_id]
            
            # Remove event subscriptions
            for event_type, subscribers in self.event_subscribers.items():
                self.event_subscribers[event_type] = [
                    sub for sub in subscribers if sub.pipeline_id != pipeline_id
                ]
            
            # Broadcast pipeline unregistration event
            await self.broadcast_event(PipelineEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.PIPELINE_COMPLETED,
                source_pipeline=pipeline_id,
                subject="Pipeline Unregistered",
                data={"pipeline_name": pipeline_info.pipeline_name},
                tags=["unregistration", "lifecycle"]
            ))
            
            logger.info(f"Pipeline '{pipeline_info.pipeline_name}' unregistered")
    
    async def send_message(
        self,
        from_pipeline: str,
        to_pipeline: str,
        content: Any,
        message_type: MessageType = MessageType.DATA,
        priority: MessagePriority = MessagePriority.NORMAL,
        subject: str = "",
        expires_in_seconds: Optional[int] = None,
        reply_to: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Send direct message between pipelines."""
        message_id = str(uuid.uuid4())
        expires_at = None
        if expires_in_seconds:
            expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        
        message = PipelineMessage(
            message_id=message_id,
            from_pipeline=from_pipeline,
            to_pipeline=to_pipeline,
            message_type=message_type,
            priority=priority,
            subject=subject,
            content=content,
            metadata=metadata or {},
            expires_at=expires_at,
            reply_to=reply_to
        )
        
        success = await self.message_queue.put(message)
        if success:
            self.stats["messages_sent"] += 1
            logger.debug(f"Message sent from {from_pipeline} to {to_pipeline}: {subject}")
        else:
            logger.warning(f"Failed to queue message from {from_pipeline} to {to_pipeline}")
        
        return message_id
    
    async def receive_message(self, pipeline_id: str, timeout: Optional[float] = None) -> Optional[PipelineMessage]:
        """Receive next message for pipeline."""
        message = await self.message_queue.get_for_pipeline(pipeline_id, timeout)
        if message:
            self.stats["messages_received"] += 1
            
            # Update heartbeat
            self.pipeline_heartbeats[pipeline_id] = datetime.now()
            
            # Handle reply requests
            if message.reply_to and pipeline_id in self.reply_handlers:
                await self.reply_handlers[pipeline_id](message)
        
        return message
    
    async def broadcast_event(self, event: PipelineEvent):
        """Broadcast event to all subscribers."""
        self.event_history.append(event)
        self.stats["events_broadcast"] += 1
        
        # Find matching subscribers
        delivered_count = 0
        for event_type_pattern, subscribers in self.event_subscribers.items():
            for subscriber in subscribers:
                if subscriber.active and await subscriber.matches_event(event):
                    try:
                        # Here you would normally call the callback
                        # For now, we'll just log the delivery
                        logger.debug(f"Event {event.event_id} delivered to {subscriber.pipeline_id}")
                        delivered_count += 1
                    except Exception as e:
                        logger.error(f"Error delivering event to {subscriber.pipeline_id}: {e}")
        
        self.stats["events_delivered"] += delivered_count
        logger.debug(f"Event {event.event_type.value} broadcast to {delivered_count} subscribers")
    
    def subscribe_to_events(
        self,
        pipeline_id: str,
        event_patterns: List[str],
        callback_info: Dict[str, Any] = None
    ) -> str:
        """Subscribe to events from other pipelines."""
        subscription_id = str(uuid.uuid4())
        
        subscriber = EventSubscriber(
            subscriber_id=subscription_id,
            pipeline_id=pipeline_id,
            event_patterns=event_patterns,
            callback_info=callback_info or {}
        )
        
        # Add to all matching event types
        for pattern in event_patterns:
            self.event_subscribers[pattern].append(subscriber)
        
        self.stats["active_subscriptions"] += 1
        logger.info(f"Pipeline {pipeline_id} subscribed to events: {event_patterns}")
        
        return subscription_id
    
    def unsubscribe_from_events(self, subscription_id: str, pipeline_id: str):
        """Unsubscribe from events."""
        removed_count = 0
        for event_type, subscribers in self.event_subscribers.items():
            original_count = len(subscribers)
            self.event_subscribers[event_type] = [
                sub for sub in subscribers 
                if not (sub.subscriber_id == subscription_id and sub.pipeline_id == pipeline_id)
            ]
            removed_count += original_count - len(self.event_subscribers[event_type])
        
        if removed_count > 0:
            self.stats["active_subscriptions"] -= 1
            logger.info(f"Pipeline {pipeline_id} unsubscribed from events")
    
    async def create_shared_space(
        self,
        space_id: str,
        name: str = "",
        description: str = "",
        authorized_pipelines: Set[str] = None,
        max_size: int = 1000
    ) -> SharedMemorySpace:
        """Create a new shared memory space."""
        if space_id in self.shared_spaces:
            raise ValueError(f"Shared space '{space_id}' already exists")
        
        space = SharedMemorySpace(
            space_id=space_id,
            name=name,
            description=description,
            authorized_pipelines=authorized_pipelines or set(),
            max_size=max_size
        )
        
        self.shared_spaces[space_id] = space
        logger.info(f"Created shared space '{space_id}' with max size {max_size}")
        
        return space
    
    def get_shared_space(self, space_id: str) -> Optional[SharedMemorySpace]:
        """Get shared memory space by ID."""
        return self.shared_spaces.get(space_id)
    
    def list_shared_spaces(self) -> List[str]:
        """List all shared memory space IDs."""
        return list(self.shared_spaces.keys())
    
    def discover_pipelines(self, capability: Optional[str] = None) -> List[PipelineInfo]:
        """Discover active pipelines, optionally filtered by capability."""
        pipelines = list(self.active_pipelines.values())
        
        if capability:
            pipelines = [p for p in pipelines if capability in p.capabilities]
        
        return pipelines
    
    def get_pipeline_info(self, pipeline_id: str) -> Optional[PipelineInfo]:
        """Get information about a specific pipeline."""
        return self.active_pipelines.get(pipeline_id)
    
    async def health_check(self, max_age_seconds: int = 300):
        """Remove inactive pipelines based on heartbeat."""
        cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)
        inactive_pipelines = []
        
        for pipeline_id, last_seen in self.pipeline_heartbeats.items():
            if last_seen < cutoff_time:
                inactive_pipelines.append(pipeline_id)
        
        for pipeline_id in inactive_pipelines:
            await self.unregister_pipeline(pipeline_id)
            logger.warning(f"Pipeline {pipeline_id} removed due to inactivity")
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get comprehensive communication statistics."""
        return {
            **self.stats,
            "active_pipelines": len(self.active_pipelines),
            "shared_spaces": len(self.shared_spaces),
            "queue_size": asyncio.create_task(self.message_queue.size()),
            "event_history_size": len(self.event_history),
            "total_subscriptions": sum(len(subs) for subs in self.event_subscribers.values())
        }
    
    async def cleanup(self):
        """Cleanup communication resources."""
        # Clear all data structures
        self.active_pipelines.clear()
        self.pipeline_heartbeats.clear()
        self.event_subscribers.clear()
        self.shared_spaces.clear()
        self.event_history.clear()
        
        # Reset statistics
        self.stats = {key: 0 for key in self.stats}
        
        logger.info("Communication system cleaned up")