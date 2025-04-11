#!/usr/bin/env python3
"""
Quantum Visualization Renderer
=============================

This module provides a standalone visualization server for the
Quantum Kaleidoscope system. It creates a Flask application that
can render the visualization interface and serve data from the
core system.

Usage:
    python quantum_visualization.py --port 8080 --api-port 8000
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import threading
import traceback
import requests
from flask import Flask, render_template, jsonify, request, send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("QuantumVisualizerNew")

class VisualizationServer:
    """Server for quantum kaleidoscope visualization."""
    
    def __init__(self, api_url="http://localhost:8000", port=8080):
        """
        Initialize the visualization server.
        
        Args:
            api_url: URL of the main quantum kaleidoscope API
            port: Port to run the visualization server on
        """
        self.api_url = api_url
        self.port = port
        self.app = Flask(__name__, static_folder="static", template_folder="templates")
        self.last_data_fetch = 0
        self.data_cache = None
        self.data_cache_ttl = 2  # seconds
        self.setup_routes()
        
        # Create template and static directories if they don't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "static", "js"), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "static", "css"), exist_ok=True)
        
        # Generate necessary template and static files
        self._generate_static_files()
        
        logger.info(f"Visualization server initialized to connect to API at {api_url}")
    
    def setup_routes(self):
        """Set up Flask routes for the visualization server."""
        
        @self.app.route('/')
        def index():
            """Render the main visualization page."""
            return render_template('visualization.html')
        
        @self.app.route('/api/visualization-data')
        def visualization_data():
            """
            Get visualization data from the main API.
            This acts as a proxy to the main API with caching.
            """
            current_time = time.time()
            if self.data_cache and current_time - self.last_data_fetch < self.data_cache_ttl:
                # Use cached data
                return jsonify(self.data_cache)
            
            try:
                # Fetch data from main API
                response = requests.get(f"{self.api_url}/api/visualization")
                if response.status_code == 200:
                    data = response.json()
                    self.data_cache = data
                    self.last_data_fetch = current_time
                    return jsonify(data)
                else:
                    logger.error(f"Error fetching visualization data: {response.status_code}")
                    return jsonify({
                        "error": f"API returned status {response.status_code}",
                        "timestamp": current_time
                    }), 500
            except Exception as e:
                logger.error(f"Error fetching visualization data: {e}")
                return jsonify({
                    "error": str(e),
                    "timestamp": current_time
                }), 500
        
        @self.app.route('/api/system-status')
        def system_status():
            """Get system status from the main API."""
            try:
                response = requests.get(f"{self.api_url}/api/status")
                if response.status_code == 200:
                    return jsonify(response.json())
                else:
                    logger.error(f"Error fetching system status: {response.status_code}")
                    return jsonify({
                        "error": f"API returned status {response.status_code}",
                        "status": "unknown"
                    }), 500
            except Exception as e:
                logger.error(f"Error fetching system status: {e}")
                return jsonify({
                    "error": str(e),
                    "status": "unknown"
                }), 500
        
        @self.app.route('/api/node/<node_id>')
        def node_details(node_id):
            """Get detailed information about a specific node."""
            try:
                response = requests.get(f"{self.api_url}/api/node/{node_id}")
                if response.status_code == 200:
                    return jsonify(response.json())
                elif response.status_code == 404:
                    return jsonify({"error": "Node not found"}), 404
                else:
                    logger.error(f"Error fetching node details: {response.status_code}")
                    return jsonify({
                        "error": f"API returned status {response.status_code}"
                    }), 500
            except Exception as e:
                logger.error(f"Error fetching node details: {e}")
                return jsonify({
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/insights')
        def get_insights():
            """Get system insights, optionally filtered by node ID."""
            node_id = request.args.get('node_id')
            limit = request.args.get('limit', 10)
            
            try:
                # Construct URL with query parameters
                insights_url = f"{self.api_url}/api/insights?limit={limit}"
                if node_id:
                    insights_url += f"&node_id={node_id}"
                
                response = requests.get(insights_url)
                if response.status_code == 200:
                    return jsonify(response.json())
                else:
                    logger.error(f"Error fetching insights: {response.status_code}")
                    return jsonify({
                        "error": f"API returned status {response.status_code}",
                        "insights": []
                    }), 500
            except Exception as e:
                logger.error(f"Error fetching insights: {e}")
                return jsonify({
                    "error": str(e),
                    "insights": []
                }), 500
    
    def _generate_static_files(self):
        """Generate necessary template and static files for the visualization."""
        # Generate the main HTML template
        self._generate_html_template()
        
        # Generate JavaScript files
        self._generate_js_files()
        
        # Generate CSS files
        self._generate_css_files()
    
    def _generate_html_template(self):
        """Generate the HTML template for the visualization page."""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Kaleidoscope Visualization</title>
    <link rel="stylesheet" href="/static/css/visualization.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/stats.js/r17/Stats.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Quantum Kaleidoscope Visualization</h1>
            <div class="controls">
                <button id="btnRefresh">Refresh</button>
                <button id="btnAutoRefresh">Auto Refresh</button>
                <select id="colorScheme">
                    <option value="energy">Color by Energy</option>
                    <option value="stability">Color by Stability</option>
                    <option value="connections">Color by Connections</option>
                    <option value="type">Color by Type</option>
                </select>
                <button id="btnSettings">Settings</button>
            </div>
        </header>
        
        <div class="main-content">
            <div class="visualization-container" id="visualizationContainer"></div>
            
            <aside class="sidebar">
                <div class="panel node-info-panel">
                    <h2>Node Information</h2>
                    <div id="nodeInfo">
                        <p>Select a node to see details</p>
                    </div>
                </div>
                
                <div class="panel insights-panel">
                    <h2>Insights</h2>
                    <div id="insightsContainer">
                        <p>Loading insights...</p>
                    </div>
                </div>
                
                <div class="panel system-info-panel">
                    <h2>System Status</h2>
                    <div id="systemStatus">
                        <p>Loading system status...</p>
                    </div>
                </div>
            </aside>
        </div>
        
        <!-- Settings Modal -->
        <div id="settingsModal" class="modal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <h2>Visualization Settings</h2>
                <div class="settings-form">
                    <div class="form-group">
                        <label for="nodeSize">Node Size:</label>
                        <input type="range" id="nodeSize" min="0.5" max="5" step="0.1" value="2">
                        <span id="nodeSizeValue">2.0</span>
                    </div>
                    <div class="form-group">
                        <label for="edgeThickness">Edge Thickness:</label>
                        <input type="range" id="edgeThickness" min="0.5" max="5" step="0.1" value="1">
                        <span id="edgeThicknessValue">1.0</span>
                    </div>
                    <div class="form-group">
                        <label for="minEdgeStrength">Min Edge Strength:</label>
                        <input type="range" id="minEdgeStrength" min="0.1" max="0.9" step="0.05" value="0.2">
                        <span id="minEdgeStrengthValue">0.2</span>
                    </div>
                    <div class="form-group">
                        <label for="rotationSpeed">Rotation Speed:</label>
                        <input type="range" id="rotationSpeed" min="0" max="0.01" step="0.001" value="0.001">
                        <span id="rotationSpeedValue">0.001</span>
                    </div>
                    <div class="form-group">
                        <label for="visualizationMode">Visualization Mode:</label>
                        <select id="visualizationMode">
                            <option value="default">Default</option>
                            <option value="temporal">Temporal Trails</option>
                            <option value="clusters">Cluster Highlighting</option>
                        </select>
                    </div>
                    <button id="applySettings">Apply Settings</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="/static/js/visualization-renderer.js"></script>
    <script src="/static/js/ui-controller.js"></script>
</body>
</html>
"""
        template_path = os.path.join(os.path.dirname(__file__), "templates", "visualization.html")
        with open(template_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML template at {template_path}")
    
    def _generate_js_files(self):
        """Generate JavaScript files for the visualization."""
        # Main visualization renderer
        visualization_js = """
// Quantum Kaleidoscope Visualization Renderer
// This script handles the 3D visualization using Three.js

class QuantumVisualizationRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        
        // Visualization state
        this.nodes = {};
        this.connections = [];
        this.selectedNode = null;
        
        // Rendering parameters
        this.params = {
            nodeSize: 2.0,
            edgeThickness: 1.0,
            minEdgeStrength: 0.2,
            rotationSpeed: 0.001,
            visualizationMode: 'default',
            colorScheme: 'energy'
        };
        
        // Setup scene
        this.setupScene();
        
        // Initialize object groups
        this.nodeGroup = new THREE.Group();
        this.scene.add(this.nodeGroup);
        
        this.edgeGroup = new THREE.Group();
        this.scene.add(this.edgeGroup);
        
        this.temporalTrailGroup = new THREE.Group();
        this.scene.add(this.temporalTrailGroup);
        
        // Setup interaction
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Add event listeners
        window.addEventListener('resize', this.onWindowResize.bind(this));
        this.container.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.container.addEventListener('click', this.onMouseClick.bind(this));
        
        // Start animation loop
        this.animate();
        
        // Setup performance monitor
        this.stats = new Stats();
        this.stats.dom.style.position = 'absolute';
        this.stats.dom.style.top = '0px';
        this.stats.dom.style.left = '0px';
        this.container.appendChild(this.stats.dom);
        
        // Setup event emitter for selected node
        this.onNodeSelected = null;
        
        console.log("Quantum visualization renderer initialized");
    }
    
    setupScene() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a1a);
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(75, this.width / this.height, 0.1, 1000);
        this.camera.position.z = 50;
        
        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
    }
    
    onWindowResize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(this.width, this.height);
    }
    
    onMouseMove(event) {
        // Calculate mouse position in normalized device coordinates (-1 to +1)
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / this.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / this.height) * 2 + 1;
    }
    
    onMouseClick(event) {
        // Check for node intersection
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.nodeGroup.children);
        
        if (intersects.length > 0) {
            const nodeObject = intersects[0].object;
            this.selectNode(nodeObject.userData.id);
        } else {
            this.clearNodeSelection();
        }
    }
    
    selectNode(nodeId) {
        // Clear previous selection
        this.clearNodeSelection();
        
        // Set new selection
        this.selectedNode = nodeId;
        
        // Highlight the selected node
        if (this.nodes[nodeId]) {
            const nodeMesh = this.nodes[nodeId].mesh;
            nodeMesh.material.emissive.set(0x444444);
            
            // Emit selection event
            if (this.onNodeSelected) {
                this.onNodeSelected(nodeId, this.nodes[nodeId].data);
            }
        }
    }
    
    clearNodeSelection() {
        if (this.selectedNode && this.nodes[this.selectedNode]) {
            // Reset emissive color
            this.nodes[this.selectedNode].mesh.material.emissive.set(0x000000);
        }
        
        this.selectedNode = null;
        
        // Emit deselection event
        if (this.onNodeSelected) {
            this.onNodeSelected(null, null);
        }
    }
    
    updateData(visualizationData) {
        if (!visualizationData || !visualizationData.nodes) {
            console.error("Invalid visualization data");
            return;
        }
        
        // Clear current visualization
        this.clearVisualization();
        
        // Create mapping of node IDs for edge creation
        const nodeMap = {};
        visualizationData.nodes.forEach(node => {
            nodeMap[node.id] = node;
        });
        
        // Create nodes
        visualizationData.nodes.forEach(node => {
            this.createNode(node);
        });
        
        // Create edges
        if (visualizationData.connections) {
            visualizationData.connections.forEach(connection => {
                // Skip if either node doesn't exist
                if (!nodeMap[connection.source] || !nodeMap[connection.target]) {
                    return;
                }
                
                // Skip if strength is below threshold
                if (connection.strength < this.params.minEdgeStrength) {
                    return;
                }
                
                this.createEdge(
                    nodeMap[connection.source],
                    nodeMap[connection.target],
                    connection.strength
                );
            });
        }
        
        // Create temporal trails if mode is enabled
        if (this.params.visualizationMode === 'temporal') {
            this.createTemporalTrails(visualizationData.nodes);
        }
        
        console.log(`Visualization updated with ${visualizationData.nodes.length} nodes and ${this.edgeGroup.children.length} connections`);
    }
    
    clearVisualization() {
        // Clear node group
        while (this.nodeGroup.children.length > 0) {
            const object = this.nodeGroup.children[0];
            
            // Dispose geometry and material
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(m => m.dispose());
                } else {
                    object.material.dispose();
                }
            }
            
            this.nodeGroup.remove(object);
        }
        
        // Clear edge group
        while (this.edgeGroup.children.length > 0) {
            const object = this.edgeGroup.children[0];
            
            // Dispose geometry and material
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(m => m.dispose());
                } else {
                    object.material.dispose();
                }
            }
            
            this.edgeGroup.remove(object);
        }
        
        // Clear temporal trails
        while (this.temporalTrailGroup.children.length > 0) {
            const object = this.temporalTrailGroup.children[0];
            
            // Dispose geometry and material
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(m => m.dispose());
                } else {
                    object.material.dispose();
                }
            }
            
            this.temporalTrailGroup.remove(object);
        }
        
        // Reset nodes dictionary
        this.nodes = {};
        this.connections = [];
        this.selectedNode = null;
    }
    
    createNode(nodeData) {
        // Calculate node size based on energy and scale parameter
        const baseSize = 0.5 + (nodeData.energy * 0.5);
        const scaledSize = baseSize * this.params.nodeSize;
        
        // Calculate node color based on selected scheme
        const color = this.getNodeColor(nodeData);
        
        // Create node material
        const material = new THREE.MeshPhongMaterial({
            color: color,
            specular: 0x444444,
            shininess: 30 * nodeData.stability,
            transparent: true,
            opacity: 0.9
        });
        
        // Create node geometry and mesh
        const geometry = new THREE.SphereGeometry(scaledSize, 16, 16);
        const mesh = new THREE.Mesh(geometry, material);
        
        // Set position
        if (nodeData.position && nodeData.position.length >= 3) {
            mesh.position.set(
                nodeData.position[0],
                nodeData.position[1],
                nodeData.position[2]
            );
        }
        
        // Set name and user data for interaction
        mesh.name = `node-${nodeData.id}`;
        mesh.userData = { id: nodeData.id, type: 'node', data: nodeData };
        
        // Add to node group
        this.nodeGroup.add(mesh);
        
        // Store reference for later access
        this.nodes[nodeData.id] = {
            mesh: mesh,
            data: nodeData
        };
        
        return mesh;
    }
    
    createEdge(sourceNode, targetNode, strength) {
        // Get node positions
        const sourcePos = sourceNode.position;
        const targetPos = targetNode.position;
        
        // Edge thickness based on strength and scale parameter
        const thickness = 0.5 + (strength * 2.5 * this.params.edgeThickness);
        
        // Create line material
        const material = new THREE.LineBasicMaterial({
            color: 0x88aaff,
            transparent: true,
            opacity: Math.min(0.8, strength),
            linewidth: 1  // Note: WebGL has a limitation on line width
        });
        
        // Create line geometry
        const geometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(sourcePos[0], sourcePos[1], sourcePos[2]),
            new THREE.Vector3(targetPos[0], targetPos[1], targetPos[2])
        ]);
        
        // Create line
        const line = new THREE.Line(geometry, material);
        line.name = `edge-${sourceNode.id}-${targetNode.id}`;
        line.userData = {
            source: sourceNode.id,
            target: targetNode.id,
            strength: strength
        };
        
        // Add to edge group
        this.edgeGroup.add(line);
        
        // Add to connections list
        this.connections.push({
            source: sourceNode.id,
            target: targetNode.id,
            strength: strength,
            line: line
        });
        
        return line;
    }
    
    createTemporalTrails(nodes) {
        // Create temporal trails for nodes with history
        nodes.forEach(node => {
            if (node.temporalTrail && node.temporalTrail.length > 1) {
                // Create points for the trail
                const points = [];
                node.temporalTrail.forEach(position => {
                    points.push(new THREE.Vector3(position[0], position[1], position[2]));
                });
                
                // Create line geometry and material
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({
                    color: this.getNodeColor(node),
                    transparent: true,
                    opacity: 0.5
                });
                
                // Create line
                const trail = new THREE.Line(geometry, material);
                trail.name = `trail-${node.id}`;
                
                // Add to temporal trail group
                this.temporalTrailGroup.add(trail);
            }
        });
    }
    
    getNodeColor(nodeData) {
        // Different color schemes based on node properties
        switch (this.params.colorScheme) {
            case 'energy':
                // Gradient from blue (low) to red (high) based on energy
                const h = (1 - nodeData.energy) * 0.6; // 0.6 = blue, 0 = red
                return new THREE.Color().setHSL(h, 0.8, 0.5);
                
            case 'stability':
                // Gradient from yellow (low) to green (high) based on stability
                const s = 0.3 + (nodeData.stability * 0.3); // Hue between 0.3 and 0.6
                return new THREE.Color().setHSL(s, 0.8, 0.5);
                
            case 'connections':
                // Gradient from purple (few) to cyan (many) based on connection count
                const connectionCount = nodeData.numConnections || 0;
                const maxConnections = 20; // Assume this is maximum for normalization
                const connNorm = Math.min(1.0, connectionCount / maxConnections);
                return new THREE.Color().setHSL(0.7 - (connNorm * 0.2), 0.8, 0.5);
                
            case 'type':
                // Different colors based on node type
                if (nodeData.metadata && nodeData.metadata.type) {
                    if (nodeData.metadata.type === 'text') {
                        return new THREE.Color(0x4488ff); // Blue for text nodes
                    } else if (nodeData.metadata.type === 'auto_generated') {
                        return new THREE.Color(0x44aa44); // Green for auto-generated nodes
                    }
                }
                // Default color for unknown types
                return new THREE.Color(0xaa44aa); // Purple
                
            default:
                // Default color scheme
                return new THREE.Color(0x4477ff);
        }
    }
    
    updateParams(newParams) {
        // Update parameters
        for (const key in newParams) {
            if (key in this.params) {
                this.params[key] = newParams[key];
            }
        }
    }
    
    animate() {
        requestAnimationFrame(this.animate.bind(this));
        
        // Update stats
        this.stats.update();
        
        // Update controls
        this.controls.update();
        
        // Rotate the scene slowly if rotation speed > 0
        if (this.params.rotationSpeed > 0) {
            this.nodeGroup.rotation.y += this.params.rotationSpeed;
            this.edgeGroup.rotation.y += this.params.rotationSpeed;
            this.temporalTrailGroup.rotation.y += this.params.rotationSpeed;
        }
        
        // Highlight hovered node
        this.highlightHoveredNode();
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
    }
    
    highlightHoveredNode() {
        // Check for node intersection
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.nodeGroup.children);
        
        // Reset all nodes to normal state
        for (const nodeId in this.nodes) {
            if (nodeId !== this.selectedNode) {
                this.nodes[nodeId].mesh.material.emissive.set(0x000000);
            }
        }
        
        // Highlight hovered node
        if (intersects.length > 0 && intersects[0].object.userData.id !== this.selectedNode) {
            intersects[0].object.material.emissive.set(0x222222);
        }
    }
}

// Initialize the visualization when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded, initializing visualization");
});
"""
        
        # UI Controller
        ui_controller_js = """
// UI Controller for Quantum Kaleidoscope Visualization
// This script handles UI interactions and data fetching

class UIController {
    constructor() {
        // Initialize the visualization renderer
        this.renderer = new QuantumVisualizationRenderer('visualizationContainer');
        
        // UI elements
        this.elements = {
            btnRefresh: document.getElementById('btnRefresh'),
            btnAutoRefresh: document.getElementById('btnAutoRefresh'),
            colorScheme: document.getElementById('colorScheme'),
            btnSettings: document.getElementById('btnSettings'),
            settingsModal: document.getElementById('settingsModal'),
            closeModal: document.querySelector('.close'),
            applySettings: document.getElementById('applySettings'),
            nodeInfo: document.getElementById('nodeInfo'),
            insightsContainer: document.getElementById('insightsContainer'),
            systemStatus: document.getElementById('systemStatus'),
            
            // Settings elements
            nodeSize: document.getElementById('nodeSize'),
            nodeSizeValue: document.getElementById('nodeSizeValue'),
            edgeThickness: document.getElementById('edgeThickness'),
            edgeThicknessValue: document.getElementById('edgeThicknessValue'),
            minEdgeStrength: document.getElementById('minEdgeStrength'),
            minEdgeStrengthValue: document.getElementById('minEdgeStrengthValue'),
            rotationSpeed: document.getElementById('rotationSpeed'),
            rotationSpeedValue: document.getElementById('rotationSpeedValue'),
            visualizationMode: document.getElementById('visualizationMode')
        };
        
        // Data state
        this.autoRefresh = false;
        this.refreshIntervalId = null;
        this.refreshInterval = 5000; // 5 seconds
        
        // Register event handlers
        this.registerEventHandlers();
        
        // Set up node selection callback
        this.renderer.onNodeSelected = this.onNodeSelected.bind(this);
        
        // Initial data load
        this.loadVisualizationData();
        this.loadSystemStatus();
        this.loadInsights();
        
        console.log("UI Controller initialized");
    }
    
    registerEventHandlers() {
        // Refresh button
        this.elements.btnRefresh.addEventListener('click', () => {
            this.loadVisualizationData();
            this.loadSystemStatus();
        });
        
        // Auto refresh toggle
        this.elements.btnAutoRefresh.addEventListener('click', () => {
            this.toggleAutoRefresh();
        });
        
        // Color scheme selector
        this.elements.colorScheme.addEventListener('change', () => {
            this.renderer.updateParams({
                colorScheme: this.elements.colorScheme.value
            });
            this.loadVisualizationData();
        });
        
        // Settings button
        this.elements.btnSettings.addEventListener('click', () => {
            this.elements.settingsModal.style.display = 'block';
        });
        
        // Close modal button
        this.elements.closeModal.addEventListener('click', () => {
            this.elements.settingsModal.style.display = 'none';
        });
        
        // Close modal when clicking outside
        window.addEventListener('click', (event) => {
            if (event.target === this.elements.settingsModal) {
                this.elements.settingsModal.style.display = 'none';
            }
        });
        
        // Apply settings button
        this.elements.applySettings.addEventListener('click', () => {
            this.applySettings();
            this.elements.settingsModal.style.display = 'none';
        });
        
        // Settings sliders - update displayed values
        this.elements.nodeSize.addEventListener('input', () => {
            this.elements.nodeSizeValue.textContent = this.elements.nodeSize.value;
        });
        
        this.elements.edgeThickness.addEventListener('input', () => {
            this.elements.edgeThicknessValue.textContent = this.elements.edgeThickness.value;
        });
        
        this.elements.minEdgeStrength.addEventListener('input', () => {
            this.elements.minEdgeStrengthValue.textContent = this.elements.minEdgeStrength.value;
        });
        
        this.elements.rotationSpeed.addEventListener('input', () => {
            this.elements.rotationSpeedValue.textContent = this.elements.rotationSpeed.value;
        });
    }
    
    toggleAutoRefresh() {
        this.autoRefresh = !this.autoRefresh;
        
        if (this.autoRefresh) {
            this.elements.btnAutoRefresh.textContent = 'Stop Auto Refresh';
            this.elements.btnAutoRefresh.classList.add('active');
            
            // Start refresh interval
            this.refreshIntervalId = setInterval(() => {
                this.loadVisualizationData();
                this.loadSystemStatus();
            }, this.refreshInterval);
        } else {
            this.elements.btnAutoRefresh.textContent = 'Auto Refresh';
            this.elements.btnAutoRefresh.classList.remove('active');
            
            // Clear refresh interval
            if (this.refreshIntervalId) {
                clearInterval(this.refreshIntervalId);
                this.refreshIntervalId = null;
            }
        }
    }
    
    applySettings() {
        // Get values from UI
        const newParams = {
            nodeSize: parseFloat(this.elements.nodeSize.value),
            edgeThickness: parseFloat(this.elements.edgeThickness.value),
            minEdgeStrength: parseFloat(this.elements.minEdgeStrength.value),
            rotationSpeed: parseFloat(this.elements.rotationSpeed.value),
            visualizationMode: this.elements.visualizationMode.value,
            colorScheme: this.elements.colorScheme.value
        };
        
        // Update renderer params
        this.renderer.updateParams(newParams);
        
        // Reload visualization with new params
        this.loadVisualizationData();
        
        console.log("Applied new visualization settings:", newParams);
    }
    
    loadVisualizationData() {
        fetch('/api/visualization-data')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                this.renderer.updateData(data);
            })
            .catch(error => {
                console.error('Error loading visualization data:', error);
            });
    }
    
    loadSystemStatus() {
        fetch('/api/system-status')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                this.updateSystemStatus(data);
            })
            .catch(error => {
                console.error('Error loading system status:', error);
                this.elements.systemStatus.innerHTML = `<p class="error">Error loading status: ${error.message}</p>`;
            });
    }
    
    loadInsights(nodeId = null) {
        let url = '/api/insights?limit=5';
        if (nodeId) {
            url += `&node_id=${nodeId}`;
        }
        
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                this.updateInsights(data);
            })
            .catch(error => {
                console.error('Error loading insights:', error);
                this.elements.insightsContainer.innerHTML = `<p class="error">Error loading insights: ${error.message}</p>`;
            });
    }
    
    onNodeSelected(nodeId, nodeData) {
        if (nodeId && nodeData) {
            // Load node-specific insights
            this.loadInsights(nodeId);
            
            // Update node info panel
            this.updateNodeInfo(nodeData);
        } else {
            // Clear node info
            this.elements.nodeInfo.innerHTML = '<p>Select a node to see details</p>';
            
            // Load general insights
            this.loadInsights();
        }
    }
    
    updateNodeInfo(nodeData) {
        // Format node properties for display
        let html = `
            <h3>Node ID: ${nodeData.id.substring(0, 8)}...</h3>
            <div class="node-properties">
                <div class="property">
                    <span class="property-name">Energy:</span>
                    <span class="property-value">${nodeData.energy.toFixed(2)}</span>
                    <div class="property-bar" style="width: ${nodeData.energy * 100}%; background-color: #4477ff;"></div>
                </div>
                <div class="property">
                    <span class="property-name">Stability:</span>
                    <span class="property-value">${nodeData.stability.toFixed(2)}</span>
                    <div class="property-bar" style="width: ${nodeData.stability * 100}%; background-color: #44aa44;"></div>
                </div>
                <div class="property">
                    <span class="property-name">Connections:</span>
                    <span class="property-value">${nodeData.numConnections}</span>
                </div>
            </div>
        `;
        
        // Add metadata if available
        if (nodeData.metadata && Object.keys(nodeData.metadata).length > 0) {
            html += '<h4>Metadata</h4><div class="metadata">';
            
            for (const [key, value] of Object.entries(nodeData.metadata)) {
                // Truncate long text values
                let displayValue = value;
                if (typeof value === 'string' && value.length > 50) {
                    displayValue = value.substring(0, 50) + '...';
                }
                
                html += `
                    <div class="metadata-item">
                        <span class="metadata-key">${key}:</span>
                        <span class="metadata-value">${displayValue}</span>
                    </div>
                `;
            }
            
            html += '</div>';
        }
        
        this.elements.nodeInfo.innerHTML = html;
    }
    
    updateSystemStatus(statusData) {
        if (!statusData) {
            this.elements.systemStatus.innerHTML = '<p>No status data available</p>';
            return;
        }
        
        let html = '<table class="status-table">';
        
        // Basic status properties
        if (statusData.status) {
            html += `<tr><td>Status:</td><td>${statusData.status}</td></tr>`;
        }
        
        if (statusData.uptime_formatted) {
            html += `<tr><td>Uptime:</td><td>${statusData.uptime_formatted}</td></tr>`;
        }
        
        if (statusData.node_count !== undefined) {
            html += `<tr><td>Nodes:</td><td>${statusData.node_count}</td></tr>`;
        }
        
        if (statusData.processed_texts !== undefined) {
            html += `<tr><td>Processed Texts:</td><td>${statusData.processed_texts}</td></tr>`;
        }
        
        if (statusData.simulation_steps !== undefined) {
            html += `<tr><td>Simulation Steps:</td><td>${statusData.simulation_steps}</td></tr>`;
        }
        
        if (statusData.insights_generated !== undefined) {
            html += `<tr><td>Insights Generated:</td><td>${statusData.insights_generated}</td></tr>`;
        }
        
        if (statusData.auto_generation_running !== undefined) {
            const autoGenStatus = statusData.auto_generation_running ? 'Running' : 'Stopped';
            html += `<tr><td>Auto-Generation:</td><td>${autoGenStatus}</td></tr>`;
        }
        
        html += '</table>';
        
        this.elements.systemStatus.innerHTML = html;
    }
    
    updateInsights(data) {
        if (!data || !data.insights || data.insights.length === 0) {
            this.elements.insightsContainer.innerHTML = '<p>No insights available</p>';
            return;
        }
        
        let html = '';
        
        data.insights.forEach(insight => {
            const date = new Date(insight.timestamp * 1000).toLocaleString();
            
            html += `
                <div class="insight-item">
                    <div class="insight-header">
                        <span class="insight-type">${insight.type}</span>
                        <span class="insight-time">${date}</span>
                    </div>
                    <div class="insight-content">${insight.content}</div>
                </div>
            `;
        });
        
        this.elements.insightsContainer.innerHTML = html;
    }
}

// Initialize the UI controller when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log("Initializing UI Controller");
    window.uiController = new UIController();
});
"""
        
        # Save JavaScript files
        js_dir = os.path.join(os.path.dirname(__file__), "static", "js")
        
        with open(os.path.join(js_dir, "visualization-renderer.js"), 'w') as f:
            f.write(visualization_js)
            
        with open(os.path.join(js_dir, "ui-controller.js"), 'w') as f:
            f.write(ui_controller_js)
            
        logger.info(f"Generated JavaScript files in {js_dir}")
    
    def _generate_css_files(self):
        """Generate CSS files for the visualization."""
        visualization_css = """
/* Quantum Kaleidoscope Visualization Styles */

:root {
    --background-color: #0a0a1a;
    --panel-color: #111133;
    --text-color: #ccccff;
    --highlight-color: #4466ff;
    --secondary-color: #6644ff;
    --button-color: #2233cc;
    --button-hover-color: #3344ee;
    --success-color: #44cc44;
    --warning-color: #ccaa44;
    --error-color: #cc4444;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Arial', sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
    line-height: 1.6;
}

.container {
    width: 100%;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

header {
    background-color: var(--panel-color);
    padding: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

h1 {
    font-size: 1.5rem;
    color: var(--highlight-color);
    margin: 0;
}

h2 {
    font-size: 1.2rem;
    color: var(--highlight-color