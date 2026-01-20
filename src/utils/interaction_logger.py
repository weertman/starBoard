# src/utils/interaction_logger.py
"""
User Interaction Logging for starBoard.

Logs UI interactions to session-unique CSV files for behavioral analytics.
Designed to be non-blocking with background I/O and spam prevention.
"""
from __future__ import annotations

import atexit
import csv
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Event Categories and Spam Policies
# ---------------------------------------------------------------------------
# Immediate: logged right away (low frequency, high value)
# Debounced: wait N ms after last event before logging (prevents rapid-fire)
# Throttled: log at most once per N ms (for continuous actions)

IMMEDIATE_EVENTS = frozenset({
    "button_click",
    "dialog_open",
    "dialog_close",
    "tab_switch",
    "decision_save",
    "file_open",
    "app_start",
    "app_exit",
})

DEBOUNCE_EVENTS = {
    "combo_change": 200,       # ms
    "checkbox_toggle": 150,
    "spinbox_change": 300,
    "date_change": 300,
    "list_selection": 200,     # list widget selection
    "annotation_change": 400,  # annotation widget value changes
}

THROTTLE_EVENTS = {
    "slider_change": 400,      # ms
    "scroll": 500,
    "resize": 500,
    "offset_change": 500,      # numeric offset spinboxes
    "spin_change": 400,        # spinbox changes
}


# ---------------------------------------------------------------------------
# Event Data Structure
# ---------------------------------------------------------------------------
@dataclass
class InteractionEvent:
    """A single user interaction event."""
    timestamp: str
    event_type: str
    widget: str
    tab: str
    value: str
    context: str  # JSON string

    def to_row(self) -> list:
        return [
            self.timestamp,
            self.event_type,
            self.widget,
            self.tab,
            self.value,
            self.context,
        ]

    @staticmethod
    def csv_header() -> list:
        return ["timestamp", "event_type", "widget", "tab", "value", "context"]


# ---------------------------------------------------------------------------
# Debounce/Throttle State
# ---------------------------------------------------------------------------
@dataclass
class _DebouncedEvent:
    """Tracks a pending debounced event."""
    event: InteractionEvent
    scheduled_at: float  # time.monotonic() when debounce started
    delay_ms: int


@dataclass
class _ThrottleState:
    """Tracks throttle state for a widget."""
    last_logged: float  # time.monotonic()
    pending: Optional[InteractionEvent] = None


# ---------------------------------------------------------------------------
# Interaction Logger (Singleton)
# ---------------------------------------------------------------------------
class InteractionLogger:
    """
    Thread-safe interaction logger with background CSV writing.
    
    Usage:
        logger = InteractionLogger.get()
        logger.log("button_click", "btn_refresh", value="clicked")
    """
    _instance: Optional["InteractionLogger"] = None
    _lock = threading.Lock()

    def __init__(self):
        # Prevent direct instantiation
        raise RuntimeError("Use InteractionLogger.get() instead")

    @classmethod
    def get(cls) -> "InteractionLogger":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = object.__new__(cls)
                    instance._init_internal()
                    cls._instance = instance
        return cls._instance

    def _init_internal(self):
        """Internal initialization (called once)."""
        self._session_id: str = ""
        self._log_path: Optional[Path] = None
        self._current_tab: str = ""
        
        # Event queue for background writer
        self._queue: queue.Queue[Optional[InteractionEvent]] = queue.Queue()
        
        # Debounce/throttle state (widget -> state)
        self._debounce_pending: Dict[str, _DebouncedEvent] = {}
        self._throttle_state: Dict[str, _ThrottleState] = {}
        self._spam_lock = threading.Lock()
        
        # Background writer thread
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Debounce timer thread
        self._timer_thread: Optional[threading.Thread] = None
        
        # Track if initialized
        self._initialized = False

    def initialize(self, session_id: str, logs_dir: Path) -> None:
        """
        Initialize the logger for this session.
        
        Args:
            session_id: Unique session identifier (e.g., "20251222-143522-a1b2c3")
            logs_dir: Directory to store log files (e.g., archive/logs/)
        """
        if self._initialized:
            return
        
        self._session_id = session_id
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-unique CSV file
        filename = f"interactions_{session_id}.csv"
        self._log_path = logs_dir / filename
        
        # Write CSV header if file is new
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(InteractionEvent.csv_header())
        
        # Start background threads
        self._running = True
        
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="InteractionLogger-Writer",
            daemon=True,
        )
        self._writer_thread.start()
        
        self._timer_thread = threading.Thread(
            target=self._timer_loop,
            name="InteractionLogger-Timer",
            daemon=True,
        )
        self._timer_thread.start()
        
        # Register shutdown hook
        atexit.register(self.shutdown)
        
        self._initialized = True
        
        # Log app start
        self.log("app_start", "application", value="started")

    def shutdown(self) -> None:
        """Gracefully shutdown the logger, flushing all pending events."""
        if not self._running:
            return
        
        # Log app exit
        self._enqueue_immediate(InteractionEvent(
            timestamp=self._now(),
            event_type="app_exit",
            widget="application",
            tab=self._current_tab,
            value="exiting",
            context="{}",
        ))
        
        # Flush any pending debounced events
        with self._spam_lock:
            for key, debounced in list(self._debounce_pending.items()):
                self._queue.put(debounced.event)
            self._debounce_pending.clear()
            
            # Flush pending throttled events
            for key, state in list(self._throttle_state.items()):
                if state.pending:
                    self._queue.put(state.pending)
            self._throttle_state.clear()
        
        # Signal writer to stop
        self._running = False
        self._queue.put(None)  # Poison pill
        
        # Wait for writer to finish (with timeout)
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2.0)

    def set_current_tab(self, tab_name: str) -> None:
        """Update the current tab context."""
        self._current_tab = tab_name

    def log(
        self,
        event_type: str,
        widget: str,
        *,
        value: str = "",
        context: Optional[Dict[str, Any]] = None,
        tab: Optional[str] = None,
    ) -> None:
        """
        Log a user interaction event.
        
        Args:
            event_type: Category of event (button_click, combo_change, etc.)
            widget: Widget identifier (btn_refresh, cmb_query, etc.)
            value: New value after interaction (optional)
            context: Additional context as dict (optional, will be JSON-encoded)
            tab: Override current tab context (optional)
        """
        if not self._initialized:
            return
        
        event = InteractionEvent(
            timestamp=self._now(),
            event_type=event_type,
            widget=widget,
            tab=tab or self._current_tab,
            value=str(value) if value is not None else "",
            context=json.dumps(context) if context else "{}",
        )
        
        # Route based on event type
        if event_type in IMMEDIATE_EVENTS:
            self._enqueue_immediate(event)
        elif event_type in DEBOUNCE_EVENTS:
            self._handle_debounced(event, widget, DEBOUNCE_EVENTS[event_type])
        elif event_type in THROTTLE_EVENTS:
            self._handle_throttled(event, widget, THROTTLE_EVENTS[event_type])
        else:
            # Default: treat as immediate
            self._enqueue_immediate(event)

    def _now(self) -> str:
        """Current timestamp in ISO format with milliseconds."""
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.") + \
               f"{datetime.now().microsecond // 1000:03d}"

    def _enqueue_immediate(self, event: InteractionEvent) -> None:
        """Enqueue an event for immediate writing."""
        self._queue.put(event)

    def _handle_debounced(self, event: InteractionEvent, widget: str, delay_ms: int) -> None:
        """Handle a debounced event - only log after delay_ms of inactivity."""
        with self._spam_lock:
            # Reset/start debounce timer for this widget
            self._debounce_pending[widget] = _DebouncedEvent(
                event=event,
                scheduled_at=time.monotonic(),
                delay_ms=delay_ms,
            )

    def _handle_throttled(self, event: InteractionEvent, widget: str, interval_ms: int) -> None:
        """Handle a throttled event - log at most once per interval_ms."""
        now = time.monotonic()
        interval_sec = interval_ms / 1000.0
        
        with self._spam_lock:
            state = self._throttle_state.get(widget)
            
            if state is None:
                # First event for this widget - log immediately
                self._throttle_state[widget] = _ThrottleState(last_logged=now)
                self._queue.put(event)
            elif (now - state.last_logged) >= interval_sec:
                # Enough time passed - log immediately
                state.last_logged = now
                state.pending = None
                self._queue.put(event)
            else:
                # Too soon - store as pending (will be flushed later)
                state.pending = event

    def _timer_loop(self) -> None:
        """Background loop to flush debounced events after their delays."""
        while self._running:
            time.sleep(0.05)  # Check every 50ms
            
            now = time.monotonic()
            to_flush: list[InteractionEvent] = []
            
            with self._spam_lock:
                # Check debounced events
                expired_keys = []
                for key, debounced in self._debounce_pending.items():
                    elapsed_ms = (now - debounced.scheduled_at) * 1000
                    if elapsed_ms >= debounced.delay_ms:
                        to_flush.append(debounced.event)
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._debounce_pending[key]
                
                # Check throttled pending events (flush if interval passed)
                for key, state in self._throttle_state.items():
                    if state.pending:
                        # Throttle interval is in the THROTTLE_EVENTS dict
                        interval_ms = THROTTLE_EVENTS.get(state.pending.event_type, 500)
                        if (now - state.last_logged) >= (interval_ms / 1000.0):
                            to_flush.append(state.pending)
                            state.pending = None
                            state.last_logged = now
            
            # Enqueue flushed events
            for event in to_flush:
                self._queue.put(event)

    def _writer_loop(self) -> None:
        """Background loop to write events to CSV."""
        batch: list[InteractionEvent] = []
        last_flush = time.monotonic()
        flush_interval = 1.0  # Flush at least every second
        
        while True:
            try:
                # Wait for event with timeout
                event = self._queue.get(timeout=0.5)
                
                if event is None:
                    # Poison pill - flush and exit
                    if batch:
                        self._write_batch(batch)
                    break
                
                batch.append(event)
                
                # Flush if batch is large or enough time passed
                now = time.monotonic()
                if len(batch) >= 10 or (now - last_flush) >= flush_interval:
                    self._write_batch(batch)
                    batch = []
                    last_flush = now
                    
            except queue.Empty:
                # Timeout - flush any pending batch
                if batch:
                    self._write_batch(batch)
                    batch = []
                    last_flush = time.monotonic()

    def _write_batch(self, batch: list[InteractionEvent]) -> None:
        """Write a batch of events to the CSV file."""
        if not batch or not self._log_path:
            return
        
        try:
            with open(self._log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for event in batch:
                    writer.writerow(event.to_row())
        except Exception:
            # Fail silently - logging should never crash the app
            pass


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def get_interaction_logger() -> InteractionLogger:
    """Get the interaction logger singleton."""
    return InteractionLogger.get()

