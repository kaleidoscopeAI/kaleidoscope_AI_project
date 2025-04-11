#!/usr/bin/env python3
"""
Kaleidoscope AI – Cube ⇄ Ollama Bridge (sandbox-safe)
=====================================================
This single-file FastAPI backend now boots even in constrained runtimes that
lack optional stdlib modules **ssl** (needed by `httpx` / `anyio`) and
**micropip** (sometimes imported by Pyodide front-ends).

We create lightweight stubs for those modules so downstream imports work
without crashing. On a normal CPython install with full stdlib these stubs are
ignored.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Graceful stubs for missing modules (ssl, micropip)
# ---------------------------------------------------------------------------
import sys, types

# ---- ssl stub -------------------------------------------------------------
if "ssl" not in sys.modules:
    ssl_stub = types.ModuleType("ssl")

    class _DummySSLContext:  # minimal API used by httpx / anyio
        def __init__(self, *_, **__):
            pass
        def load_default_certs(self, *_, **__):
            pass
        def load_verify_locations(self, *_, **__):
            pass
        def set_ciphers(self, *_):
            pass

    # Common symbols expected by libraries
    ssl_stub.SSLContext = _DummySSLContext
    ssl_stub.create_default_context = lambda *a, **k: _DummySSLContext()
    ssl_stub.PROTOCOL_TLS_CLIENT = 2  # arbitrary int constant
    ssl_stub.CERT_REQUIRED = 2
    sys.modules["ssl"] = ssl_stub

# ---- micropip stub --------------------------------------------------------
if "micropip" not in sys.modules:
    micropip_stub = types.ModuleType("micropip")
    async def _noop_install(*_a, **_kw):
        return None
    micropip_stub.install = _noop_install  # type: ignore[attr-defined]
    sys.modules["micropip"] = micropip_stub

# ---------------------------------------------------------------------------
# Actual backend implementation
# ---------------------------------------------------------------------------
import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set # Added Set for ConnectionManager originally

import httpx # Ensure httpx is imported
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status # Added status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse # Added FileResponse
from fastapi.staticfiles import StaticFiles # Added StaticFiles
from starlette.websockets import WebSocketState

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    # Assuming llm_client.py is in the same directory orPYTHONPATH
    from llm_client import LLMClient
except ImportError:  # pragma: no cover
    LLMClient = None

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate" # Used by basic call if LLMClient fails
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_BINARY = os.getenv("OLLAMA_BINARY", "ollama") # Not used in this script directly
KAI_SECRET = os.getenv("KAI_SECRET", "change-me") # Used for WebSocket auth
MAX_WS_CONNECTIONS = int(os.getenv("MAX_WS_CONNECTIONS", 100))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", 60)) # Not used in WebSocketManager below, but could be added back

# Configure Logging (Simple Stream Handler)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("quantum-bridge")

# Create static directory if needed (for frontend files if not using /cube route)
# STATIC_DIR = Path("static")
# STATIC_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Kaleidoscope AI – Cube/LLM Bridge", version="0.4.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional advanced LLM client
# ---------------------------------------------------------------------------
llm_client: Optional["LLMClient"]
if LLMClient is not None:
    logger.info("LLMClient found, initializing...")
    try:
        llm_client = LLMClient(
            # Configuration now primarily from LLMClient reading env vars itself
            # We just ensure it gets initialized if the class is available
            # model=OLLAMA_MODEL, # LLMClient reads OLLAMA_MODEL from env
            # provider="ollama", # LLMClient reads LLM_PROVIDER from env
            # endpoint=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}", # LLMClient constructs its endpoint
        )
        logger.info(f"LLMClient initialized successfully for model {llm_client.config.get('model')}")
    except Exception as e:
         logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
         llm_client = None
else:
    logger.warning("LLMClient class not found (llm_client.py missing or import failed). LLM features unavailable.")
    llm_client = None

# ---------------------------------------------------------------------------
# System state dataclass
# ---------------------------------------------------------------------------
class SystemState:
    def __init__(self) -> None:
        self.is_paused = False
        self.timestamps: Dict[str, List[float]] = defaultdict(list) # Used for rate limiting if re-added
        self.ollama_proc: Optional[asyncio.subprocess.Process] = None # Not currently used to start Ollama
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "failed_requests": 0,
            "ws_connections": 0,
            "health_checks": 0,
            "resource_usage": {},
            "boot": datetime.utcnow().isoformat() + "Z",
        }
        self.last_metric_update = time.time()

    async def update_metrics(self, current_connections: int):
         """Update system metrics periodically"""
         now = time.time()
         if now - self.last_metric_update < 5: # Update every 5 seconds
              return

         self.last_metric_update = now
         self.metrics["ws_connections"] = current_connections

         if psutil:
              self.metrics["resource_usage"] = {
                   "backend_cpu_percent": psutil.cpu_percent(),
                   "backend_memory_percent": psutil.virtual_memory().percent,
                   "disk_usage_percent": psutil.disk_usage('/').percent
              }
         else:
              self.metrics["resource_usage"] = {"status": "psutil not installed"}

    async def check_ollama_running(self) -> bool:
         """Check if the configured Ollama instance is responsive."""
         try:
              async with httpx.AsyncClient() as client:
                   response = await client.get(
                        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/", # Check base endpoint
                        timeout=2.0
                   )
                   # Ollama root usually returns "Ollama is running"
                   return response.status_code == 200 and "Ollama is running" in response.text
         except Exception as e:
              logger.warning(f"Ollama check failed: {e}")
              return False

state = SystemState()

# ---------------------------------------------------------------------------
# WebSocket Manager (Simpler Version from last script)
# ---------------------------------------------------------------------------
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_lock = asyncio.Lock() # Use asyncio lock for async context

    async def connect(self, websocket: WebSocket):
        async with self.connection_lock:
            if len(self.active_connections) >= MAX_WS_CONNECTIONS:
                await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Max connections reached")
                logger.warning(f"Connection rejected: Max ({MAX_WS_CONNECTIONS}) connections reached.")
                return False
            await websocket.accept()
            self.active_connections.append(websocket)
            state.metrics["ws_connections"] = len(self.active_connections) # Update metric
            logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
            return True

    async def disconnect(self, websocket: WebSocket):
        async with self.connection_lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                state.metrics["ws_connections"] = len(self.active_connections) # Update metric
                logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
         # Check state before sending
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(json.dumps(message))
                return True
            except Exception as e:
                 logger.error(f"Failed to send message to client: {e}. Disconnecting.")
                 # Disconnect if sending fails
                 await self.disconnect(websocket)
                 # Ensure websocket is closed gracefully if possible
                 try:
                      await websocket.close()
                 except:
                      pass
                 return False
        else:
            logger.warning("Attempted to send message to disconnected client.")
            await self.disconnect(websocket) # Clean up if state mismatch
            return False

    async def broadcast(self, message: dict):
        # Use asyncio.gather for concurrent broadcasting
        disconnected_clients = []
        async with self.connection_lock:
             # Iterate over a copy in case the list is modified during iteration
             tasks = [self.send_personal_message(message, connection) for connection in self.active_connections]
             results = await asyncio.gather(*tasks, return_exceptions=True)

             # Check results to find clients that failed to send (and were potentially disconnected)
             # Note: send_personal_message now handles disconnect, so we just log here
             for i, result in enumerate(results):
                  if isinstance(result, Exception) or result is False:
                       # Connection likely failed and was removed by send_personal_message
                       logger.warning(f"Broadcast failed for one client (likely disconnected).")


manager = WebSocketManager()

# ---------------------------------------------------------------------------
# WebSocket Authentication Dependency
# ---------------------------------------------------------------------------
async def ws_auth(websocket: WebSocket = Depends()): # Correct way to use Depends with WebSockets
    """Dependency function to authenticate WebSocket connections."""
    token = websocket.headers.get("x-kai-token") or websocket.query_params.get("token")
    if token != KAI_SECRET:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        # Raising HTTPException might not work as expected in WebSocket dependencies directly closing
        # Log and return None or let the close handle it.
        logger.warning(f"WebSocket connection denied: Invalid token '{token}'.")
        return None # Indicate failure
    return token # Return token if valid, can be used in endpoint if needed

# ---------------------------------------------------------------------------
# WebSocket Endpoint with LLM Integration
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket): # Remove Depends(ws_auth) here, call manually
    """Main WebSocket endpoint handling client communication and LLM interaction."""
    # Perform authentication manually after accepting preliminary connection attempt
    token = websocket.headers.get("x-kai-token") or websocket.query_params.get("token")
    if token != KAI_SECRET:
         # Close immediately if token is wrong before full connect
         await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
         logger.warning(f"WebSocket connection rejected: Invalid token '{token}'.")
         return

    # If token is valid, proceed with connection via manager
    if not await manager.connect(websocket):
        # Connection failed (e.g., max connections)
        return

    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Client {client_id} connected successfully.")

    try:
        # Send welcome message or initial state if needed
        await manager.send_personal_message({"type": "system", "msg": "Connected."}, websocket)

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            state.metrics["total_requests"] += 1
            logger.debug(f"Received message from {client_id}: {message}")

            if state.is_paused:
                await manager.send_personal_message(
                    {"type": "system", "msg": "System is currently paused, request ignored."}, websocket
                )
                continue

            msg_type = message.get("type")
            if msg_type == "chat":
                received_prompt = message.get("msg", "")
                if not received_prompt:
                    await manager.send_personal_message(
                        {"type": "error", "msg": "Empty chat message received"}, websocket
                    )
                    continue

                if llm_client is not None:
                    logger.info(f"Processing chat prompt from {client_id}...")
                    try:
                        # Call LLMClient's complete method asynchronously
                        ollama_response = await llm_client.complete(prompt=received_prompt)
                        response = {
                            "type": "chat",
                            "role": "assistant",
                            "msg": ollama_response
                        }
                        await manager.send_personal_message(response, websocket)
                        logger.info(f"Sent LLM response to {client_id}.")
                    except Exception as e:
                        state.metrics["failed_requests"] += 1
                        logger.error(f"LLMClient completion error: {e}", exc_info=True)
                        await manager.send_personal_message(
                            {"type": "error", "msg": f"LLM error: {str(e)}"}, websocket
                        )
                else:
                    # Fallback if LLMClient isn't available
                    await manager.send_personal_message(
                        {"type": "error", "msg": "LLM client not initialized or available."}, websocket
                    )

            elif msg_type == "cube_state":
                # Broadcast cube state to all clients (excluding sender?)
                # For now, simple broadcast
                logger.debug(f"Broadcasting cube_state from {client_id}")
                await manager.broadcast(message)

            elif msg_type == "command":
                # Handle commands - simple broadcast for now
                logger.debug(f"Broadcasting command from {client_id}: {message.get('command')}")
                await manager.broadcast(message)

            else:
                 logger.warning(f"Received unknown message type '{msg_type}' from {client_id}")
                 await manager.send_personal_message({"type":"error", "msg":f"Unknown message type: {msg_type}"}, websocket)


    except WebSocketDisconnect as e:
        logger.info(f"Client {client_id} disconnected (code: {e.code}).")
    except json.JSONDecodeError:
         logger.error(f"Invalid JSON received from {client_id}. Disconnecting.")
         await manager.send_personal_message({"type":"error", "msg":"Invalid JSON format"}, websocket)
    except Exception as e:
        state.metrics["failed_requests"] += 1
        logger.error(f"Unexpected WebSocket error for client {client_id}: {e}", exc_info=True)
        # Try to inform client before disconnecting, might fail
        try:
             await manager.send_personal_message(
                  {"type": "error", "msg": f"Server error: {str(e)}"}, websocket
             )
        except:
             pass # Ignore send error during exception handling
    finally:
        await manager.disconnect(websocket)

# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/system/status")
async def get_system_status():
    """Get current system status and metrics."""
    await state.update_metrics(len(manager.active_connections)) # Update metrics before reporting
    ollama_running = await state.check_ollama_running()
    status_report = {
        "ollama_running": ollama_running,
        "is_paused": state.is_paused,
        "ws_connections": state.metrics["ws_connections"],
        "metrics": state.metrics, # Includes resource usage if psutil installed
        "ollama_model": OLLAMA_MODEL,
        "ollama_endpoint": f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
    }
    return status_report

@app.post("/api/system/pause")
async def pause_system():
    """Pause message processing."""
    if not state.is_paused:
        state.is_paused = True
        await manager.broadcast({"type": "system", "msg": "System paused by admin."})
        logger.info("System paused.")
    return {"status": "paused"}

@app.post("/api/system/resume")
async def resume_system():
    """Resume message processing."""
    if state.is_paused:
        state.is_paused = False
        await manager.broadcast({"type": "system", "msg": "System resumed by admin."})
        logger.info("System resumed.")
    return {"status": "resumed"}

@app.post("/api/system/reset")
async def reset_system():
    """Reset system state (metrics) and notify clients."""
    logger.info("Resetting system state.")
    # Reset metrics - consider if other state needs reset too
    state.metrics["total_requests"] = 0
    state.metrics["failed_requests"] = 0
    # state.timestamps.clear() # Clear rate limits if re-added
    await manager.broadcast({"type": "system", "action": "reset", "msg": "System reset initiated."})
    return {"status": "reset"}

# ---------------------------------------------------------------------------
# Serve Frontend - Using path from last user script
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse) # Changed route to root
async def serve_cube_frontend():
    """Serves the main frontend HTML file."""
    # Define path relative to this script file
    script_dir = Path(__file__).parent
    html_path = script_dir / "frontend" / "advanced-quantum-cube.html" # Path from user script
    logger.info(f"Attempting to serve frontend from: {html_path}")
    if html_path.exists():
        return FileResponse(html_path)
    else:
         logger.error(f"Frontend HTML file not found at expected location: {html_path}")
         raise HTTPException(status_code=404, detail=f"Frontend HTML not found at {html_path}. Create the file or adjust the path.")

# Serve other static files if needed (e.g., CSS, JS linked by the HTML)
# app.mount("/frontend", StaticFiles(directory=script_dir / "frontend"), name="frontend_static")


# ---------------------------------------------------------------------------
# Main execution block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if uvicorn is installed
    try:
         import uvicorn
    except ImportError:
         print("Error: uvicorn is required to run this server.")
         print("Please install it: pip install uvicorn")
         sys.exit(1)

    logger.info(f"Starting Kaleidoscope AI Bridge Server on http://0.0.0.0:8000")
    logger.info(f"Configured Ollama endpoint: http://{OLLAMA_HOST}:{OLLAMA_PORT}")
    logger.info(f"Using Ollama model: {OLLAMA_MODEL}")
    logger.info(f"WebSocket connections require token: {'YES' if KAI_SECRET else 'NO (Warning: Insecure!)'}")

    # Run the FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
