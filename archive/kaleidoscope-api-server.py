#!/usr/bin/env python3
"""
Kaleidoscope AI - API Server
=======================================
Provides a FastAPI-based REST API for the Kaleidoscope AI system, enabling:
- Processing of input data through SuperNodes
- Application generation from descriptions
- Code reconstruction and optimization
- Sandbox execution of generated applications
- System monitoring and management
"""

import os
import sys
import json
import logging
import asyncio
import uuid
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Kaleidoscope integration
from kaleidoscope_integration import (
    KaleidoscopeIntegrator, 
    KaleidoscopeAPI,
    SystemCapabilities,
    ReconstructionConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_api.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("KaleidoscopeAPI")

# Define request/response models
class ProcessInputRequest(BaseModel):
    """Request for processing input data"""
    data: Dict[str, Any]
    node_id: Optional[str] = None
    include_details: bool = False

class GenerateAppRequest(BaseModel):
    """Request for generating an application"""
    description: str
    output_dir: Optional[str] = None

class ReconstructCodeRequest(BaseModel):
    """Request for reconstructing code"""
    input_path: str
    config: Optional[Dict[str, Any]] = None

class RunApplicationRequest(BaseModel):
    """Request for running an application"""
    app_dir: str
    app_config: Dict[str, Any]

class StatusRequest(BaseModel):
    """Request for system status"""
    include_details: bool = False

# Create FastAPI app
app = FastAPI(
    title="Kaleidoscope AI API",
    description="API for interacting with the Kaleidoscope AI system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize integrator
integrator = None
api = None

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global integrator, api
    
    try:
        # Load configuration
        config_path = os.environ.get("KALEIDOSCOPE_CONFIG", "config.json")
        
        # Initialize integrator
        integrator = KaleidoscopeIntegrator(config_path=config_path if os.path.exists(config_path) else None)
        
        # Initialize system
        node_count = int(os.environ.get("KALEIDOSCOPE_NODES", "0"))
        await integrator.initialize_system(node_count=node_count)
        
        # Create API wrapper
        api = KaleidoscopeAPI(integrator)
        
        logger.info("Kaleidoscope API initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Kaleidoscope API: {e}")
        # Continue running to allow manual initialization

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the system"""
    global integrator
    
    if integrator:
        await integrator.shutdown()
        logger.info("Kaleidoscope API shutdown complete")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Kaleidoscope AI API",
        "version": "1.0.0",
        "status": "operational" if integrator and api else "initializing"
    }

@app.post("/api/process")
async def process_input(request: ProcessInputRequest):
    """Process input data through SuperNodes"""
    if not integrator or not api:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    result = await integrator.process_input(
        request.data,
        request.node_id
    )
    
    return result

@app.post("/api/generate")
async def generate_application(request: GenerateAppRequest, background_tasks: BackgroundTasks):
    """Generate an application from a description"""
    if not integrator or not api:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Generate application
    task_id = str(uuid.uuid4())
    
    # Create a task response
    response = {
        "task_id": task_id,
        "status": "processing",
        "description": request.description[:100] + "..." if len(request.description) > 100 else request.description
    }
    
    # Start generation in background
    background_tasks.add_task(
        _generate_application_task,
        task_id,
        request.description,
        request.output_dir
    )
    
    return response

async def _generate_application_task(task_id: str, description: str, output_dir: Optional[str] = None):
    """Background task for application generation"""
    try:
        # Update task status in some storage (e.g., a database or in-memory dict)
        # For simplicity, we'll just log the status changes
        logger.info(f"Task {task_id}: Starting application generation")
        
        # Generate application
        result = await integrator.generate_application(description, output_dir)
        
        # Update task status
        logger.info(f"Task {task_id}: Application generation completed")
        
        # Store result (in a real implementation, store in a database)
        # task_storage[task_id] = {"status": "completed", "result": result}
        
    except Exception as e:
        logger.error(f"Task {task_id}: Error generating application: {e}")
        # task_storage[task_id] = {"status": "error", "error": str(e)}

@app.get("/api/generate/{task_id}")
async def get_generation_status(task_id: str):
    """Get application generation status"""
    if not integrator or not api:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # In a real implementation, get task status from storage
    # For now, return a placeholder
    return {
        "task_id": task_id,
        "status": "unknown"
    }

@app.post("/api/reconstruct")
async def reconstruct_code(request: ReconstructCodeRequest):
    """Reconstruct and improve code"""
    if not integrator or not api:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Create ReconstructionConfig if provided
    config = None
    if request.config:
        config = ReconstructionConfig(**request.config)
    
    # Reconstruct code
    result = await integrator.reconstruct_code(
        request.input_path,
        config
    )
    
    return result

@app.post("/api/run")
async def run_application(request: RunApplicationRequest):
    """Run an application in the sandbox"""
    if not integrator or not api:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Run application
    result = await integrator.run_application(
        request.app_dir,
        request.app_config
    )
    
    return result

@app.get("/api/status")
async def get_status(include_details: bool = False):
    """Get system status"""
    if not integrator:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get status
    status = integrator.get_system_status()
    
    # Remove detailed information if not requested
    if not include_details:
        # Remove detailed metrics and errors
        if "error_stats" in status:
            status["error_stats"] = {
                "recent_errors": status["error_stats"].get("recent_errors", 0)
            }
    
    return status

@app.post("/api/initialize")
async def initialize_system(node_count: int = 0):
    """Initialize the system manually"""
    global integrator, api
    
    if integrator:
        # System already initialized
        return {
            "status": "already_initialized"
        }
    
    try:
        # Load configuration
        config_path = os.environ.get("KALEIDOSCOPE_CONFIG", "config.json")
        
        # Initialize integrator
        integrator = KaleidoscopeIntegrator(config_path=config_path if os.path.exists(config_path) else None)
        
        # Initialize system
        await integrator.initialize_system(node_count=node_count)
        
        # Create API wrapper
        api = KaleidoscopeAPI(integrator)
        
        return {
            "status": "success",
            "node_count": node_count
        }
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/nodes")
async def get_nodes():
    """Get all nodes in the system"""
    if not integrator:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get nodes
    nodes = {}
    for node_id, node in integrator.nodes.items():
        nodes[node_id] = {
            "id": node.id,
            "hostname": node.hostname,
            "roles": node.roles,
            "status": node.status,
            "capabilities": list(node.capabilities)
        }
    
    return {
        "node_count": len(nodes),
        "nodes": nodes
    }

@app.get("/api/supernodes")
async def get_supernodes():
    """Get all SuperNodes in the system"""
    if not integrator or not integrator.supernode_manager:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get SuperNode IDs
    supernode_ids = integrator.supernode_manager.list_nodes()
    
    # Get details for each SuperNode
    supernodes = {}
    for supernode_id in supernode_ids:
        status = integrator.supernode_manager.get_node_status(supernode_id)
        if status:
            supernodes[supernode_id] = {
                "id": supernode_id,
                "status": status.get("status", "unknown"),
                "pattern_count": status.get("pattern_count", 0),
                "insight_count": status.get("insight_count", 0),
                "perspective_count": status.get("perspective_count", 0),
                "processing_count": status.get("processing_count", 0),
                "uptime": status.get("uptime", 0)
            }
    
    return {
        "supernode_count": len(supernodes),
        "supernodes": supernodes
    }

@app.get("/api/node/{node_id}")
async def get_node(node_id: str):
    """Get details for a specific node"""
    if not integrator:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    if node_id not in integrator.nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Get node
    node = integrator.nodes[node_id]
    
    # Get SuperNode ID if available
    supernode_id = integrator.node_supernode_map.get(node_id)
    
    # Get quantum node ID if available
    quantum_node_id = None
    if node.node_sync and node.node_sync.quantum_node_id:
        quantum_node_id = node.node_sync.quantum_node_id
    
    # Create response
    response = {
        "id": node.id,
        "hostname": node.hostname,
        "ip_address": node.ip_address,
        "port": node.port,
        "roles": node.roles,
        "status": node.status,
        "capabilities": list(node.capabilities),
        "resources": node.resources,
        "metrics": node.metrics,
        "supernode_id": supernode_id,
        "quantum_node_id": quantum_node_id
    }
    
    # Add node sync if available
    if node.node_sync:
        response["node_sync"] = {
            "coherence": node.node_sync.coherence,
            "entanglement_count": node.node_sync.entanglement_count,
            "teleportation_success_rate": node.node_sync.teleportation_success_rate,
            "last_sync_time": node.node_sync.last_sync_time,
            "sync_attempts": node.node_sync.sync_attempts,
            "sync_failures": node.node_sync.sync_failures,
            "state_fidelity": node.node_sync.state_fidelity,
            "quantum_operations": node.node_sync.quantum_operations
        }
    
    return response

@app.get("/api/supernode/{supernode_id}")
async def get_supernode(supernode_id: str):
    """Get details for a specific SuperNode"""
    if not integrator or not integrator.supernode_manager:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get node from manager
    node = integrator.supernode_manager.get_node(supernode_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"SuperNode {supernode_id} not found")
    
    # Get node status
    status = integrator.supernode_manager.get_node_status(supernode_id)
    
    # Get insights and perspectives
    insights = node.get_insights()
    perspectives = node.get_perspectives()
    recommendations = node.get_recommendations()
    
    # Create response
    response = {
        "id": supernode_id,
        "status": status,
        "insights_count": len(insights),
        "perspectives_count": len(perspectives),
        "recommendations_count": len(recommendations),
        "insights": insights[:10],  # Limit to 10 insights
        "perspectives": perspectives[:5],  # Limit to 5 perspectives
        "recommendations": recommendations[:3]  # Limit to 3 recommendations
    }
    
    return response

@app.get("/api/supernode/{supernode_id}/insights")
async def get_supernode_insights(supernode_id: str, limit: int = 20, offset: int = 0):
    """Get insights for a specific SuperNode"""
    if not integrator or not integrator.supernode_manager:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get node from manager
    node = integrator.supernode_manager.get_node(supernode_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"SuperNode {supernode_id} not found")
    
    # Get insights
    all_insights = node.get_insights()
    
    # Apply pagination
    insights = all_insights[offset:offset + limit]
    
    return {
        "supernode_id": supernode_id,
        "insights_count": len(all_insights),
        "limit": limit,
        "offset": offset,
        "insights": insights
    }

@app.get("/api/supernode/{supernode_id}/perspectives")
async def get_supernode_perspectives(supernode_id: str, limit: int = 10, offset: int = 0):
    """Get perspectives for a specific SuperNode"""
    if not integrator or not integrator.supernode_manager:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get node from manager
    node = integrator.supernode_manager.get_node(supernode_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"SuperNode {supernode_id} not found")
    
    # Get perspectives
    all_perspectives = node.get_perspectives()
    
    # Apply pagination
    perspectives = all_perspectives[offset:offset + limit]
    
    return {
        "supernode_id": supernode_id,
        "perspectives_count": len(all_perspectives),
        "limit": limit,
        "offset": offset,
        "perspectives": perspectives
    }

@app.get("/api/supernode/{supernode_id}/insight/{insight_id}")
async def get_supernode_insight(supernode_id: str, insight_id: str):
    """Get a specific insight from a SuperNode"""
    if not integrator or not integrator.supernode_manager:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get node from manager
    node = integrator.supernode_manager.get_node(supernode_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"SuperNode {supernode_id} not found")
    
    # Get specific insight
    insight = node.get_insight(insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail=f"Insight {insight_id} not found")
    
    return insight

@app.get("/api/supernode/{supernode_id}/perspective/{perspective_id}")
async def get_supernode_perspective(supernode_id: str, perspective_id: str):
    """Get a specific perspective from a SuperNode"""
    if not integrator or not integrator.supernode_manager:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get node from manager
    node = integrator.supernode_manager.get_node(supernode_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"SuperNode {supernode_id} not found")
    
    # Get specific perspective
    perspective = node.get_perspective(perspective_id)
    if not perspective:
        raise HTTPException(status_code=404, detail=f"Perspective {perspective_id} not found")
    
    return perspective

@app.post("/api/insights/search")
async def search_insights(query: Dict[str, Any]):
    """Search for insights across all SuperNodes"""
    if not integrator or not integrator.supernode_manager:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get all SuperNode IDs
    supernode_ids = integrator.supernode_manager.list_nodes()
    
    # Search criteria
    search_type = query.get("type")
    min_confidence = query.get("min_confidence", 0.0)
    min_importance = query.get("min_importance", 0.0)
    min_novelty = query.get("min_novelty", 0.0)
    limit = query.get("limit", 20)
    
    # Collect matching insights
    matching_insights = []
    
    for supernode_id in supernode_ids:
        node = integrator.supernode_manager.get_node(supernode_id)
        if not node:
            continue
        
        # Get all insights
        insights = node.get_insights()
        
        # Filter insights
        for insight in insights:
            matches = True
            
            if search_type and insight.get("type") != search_type:
                matches = False
            
            if insight.get("confidence", 0.0) < min_confidence:
                matches = False
            
            if insight.get("importance", 0.0) < min_importance:
                matches = False
            
            if insight.get("novelty", 0.0) < min_novelty:
                matches = False
            
            if matches:
                # Add SuperNode ID
                insight["supernode_id"] = supernode_id
                matching_insights.append(insight)
    
    # Sort by importance (descending)
    matching_insights.sort(key=lambda x: x.get("importance", 0.0), reverse=True)
    
    # Apply limit
    matching_insights = matching_insights[:limit]
    
    return {
        "count": len(matching_insights),
        "insights": matching_insights
    }

@app.post("/api/quantum/entanglement")
async def request_entanglement(node_ids: List[str]):
    """Request quantum entanglement between nodes"""
    if not integrator or not integrator.quantum_sync_protocol:
        raise HTTPException(status_code=503, detail="Quantum synchronization not enabled")
    
    # Check if nodes exist
    for node_id in node_ids:
        if node_id not in integrator.nodes:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Request entanglement
    # This is a simplified placeholder - actual implementation would
    # depend on the quantum sync protocol implementation
    success = False
    try:
        # TODO: Implement actual entanglement request
        # For now, we'll simulate with a delay
        await asyncio.sleep(1)
        success = True
    except Exception as e:
        logger.error(f"Error requesting entanglement: {e}")
    
    return {
        "status": "success" if success else "error",
        "node_ids": node_ids
    }

@app.get("/api/topology")
async def get_topology():
    """Get system topology graph"""
    if not integrator or not integrator.topology_optimizer:
        raise HTTPException(status_code=503, detail="System still initializing")
    
    # Get topology graph
    graph = integrator.topology_optimizer.topology_graph
    
    # Convert to node-link format for JSON serialization
    import networkx as nx
    graph_data = nx.node_link_data(graph)
    
    # Add node details
    for node in graph_data["nodes"]:
        node_id = node["id"]
        if node_id in integrator.nodes:
            topology_node = integrator.nodes[node_id]
            node["hostname"] = topology_node.hostname
            node["roles"] = topology_node.roles
            node["status"] = topology_node.status
            
            # Add SuperNode ID if available
            if node_id in integrator.node_supernode_map:
                node["supernode_id"] = integrator.node_supernode_map[node_id]
    
    # Add topology metrics
    metrics = integrator.topology_optimizer.analyze_topology()
    
    return {
        "graph": graph_data,
        "metrics": metrics
    }

# Command-line interface
def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kaleidoscope AI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--nodes", type=int, default=0, help="Number of nodes (0 for auto)")
    args = parser.parse_args()
    
    # Set environment variables for configuration
    if args.config:
        os.environ["KALEIDOSCOPE_CONFIG"] = args.config
    
    if args.nodes:
        os.environ["KALEIDOSCOPE_NODES"] = str(args.nodes)
    
    # Run server
    uvicorn.run(
        "kaleidoscope_api:app",
        host=args.host,
        port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main()
