// ========== Core Quantum Visualization Engine ==========

/**
 * Quantum Kaleidoscope Visualization Engine
 * 
 * This complex visualization engine uses three.js to render quantum-inspired
 * data structures in 3D space with advanced physics simulations.
 */
class QuantumVisualizationEngine {
    constructor(containerId, options = {}) {
        // Initialize container
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container element with ID '${containerId}' not found.`);
        }
        
        // Configuration options with defaults
        this.config = {
            cubeSize: options.cubeSize || 30,
            nodeDensity: options.nodeDensity || 8,
            energyLevel: options.energyLevel || 0.6,
            connectionThreshold: options.connectionThreshold || 5,
            rotationSpeed: options.rotationSpeed || 0.002,
            nodeColor: options.nodeColor || 0xf72585,
            backgroundColor: options.backgroundColor || 0x000020,
            quantumJitter: options.quantumJitter || 0.2,
            entanglementStrength: options.entanglementStrength || 0.7,
            phaseCoherence: options.phaseCoherence || 0.8,
            wireframe: options.wireframe !== undefined ? options.wireframe : true,
            glow: options.glow !== undefined ? options.glow : true,
            maxNodes: options.maxNodes || 500,
            maxConnections: options.maxConnections || 1000
        };
        
        // Initialize stats
        this.stats = {
            fps: 60,
            nodes: 0,
            connections: 0,
            energy: this.config.energyLevel,
            phase: 0
        };

        // Initialize data structures
        this.nodes = new Map();
        this.connections = new Map();
        this.selectedNode = null;
        this.hoveredNode = null;
        this.entangledNodes = new Set();
        this.nodeGroups = new Map();
        this.simulationTime = 0;
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
        
        // Initialize quantum state arrays for simulation
        this.quantumStates = [];
        this.phaseShifts = [];
        this.waveFunction = new Float32Array(128);
        
        // Initialize Three.js setup
        this._initThreeJS();
        
        // Initialize event handlers
        this._initEventHandlers();
        
        // Initialize physics simulation
        this._initPhysicsSimulation();
        
        // Generate initial visualization
        this._generateInitialVisualization();
        
        // Start animation loop
        this._startAnimationLoop();
        
        console.log("Quantum Visualization Engine initialized");
    }
    
    // Initialize Three.js environment
    _initThreeJS() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.config.backgroundColor);
        
        // Add fog for depth effect
        this.scene.fog = new THREE.FogExp2(this.config.backgroundColor, 0.0025);
        
        // Create camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.z = this.config.cubeSize * 2.5;
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
        
        // Set up post-processing for glow effect
        this._setupPostProcessing();
        
        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);
        
        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Add point light at center for dramatic effect
        const pointLight = new THREE.PointLight(this.config.nodeColor, 0.5, 50);
        pointLight.position.set(0, 0, 0);
        this.scene.add(pointLight);
        
        // Create container group for nodes
        this.nodesGroup = new THREE.Group();
        this.scene.add(this.nodesGroup);
        
        // Create container group for connections
        this.connectionsGroup = new THREE.Group();
        this.scene.add(this.connectionsGroup);
        
        // Create bounding box to visualize the quantum field
        this._createBoundingBox();
        
        // Set up orbit controls for interactive rotation
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Setup raycaster for node selection
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        // Handle window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
            this.composer.setSize(this.container.clientWidth, this.container.clientHeight);
        });
    }
    
    // Set up post-processing effects
    _setupPostProcessing() {
        this.composer = new THREE.EffectComposer(this.renderer);
        
        // Add render pass
        const renderPass = new THREE.RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        
        // Add bloom pass for glow effect
        this.bloomPass = new THREE.UnrealBloomPass(
            new THREE.Vector2(this.container.clientWidth, this.container.clientHeight),
            1.5,  // strength
            0.4,  // radius
            0.85   // threshold
        );
        this.composer.addPass(this.bloomPass);
        
        // Add optional FXAA pass for antialiasing
        const fxaaPass = new THREE.ShaderPass(THREE.FXAAShader);
        fxaaPass.uniforms.resolution.value.set(
            1 / this.container.clientWidth,
            1 / this.container.clientHeight
        );
        this.composer.addPass(fxaaPass);
    }
    
    // Create bounding box visualization
    _createBoundingBox() {
        const size = this.config.cubeSize;
        const geometry = new THREE.BoxGeometry(size, size, size);
        const edges = new THREE.EdgesGeometry(geometry);
        const material = new THREE.LineBasicMaterial({
            color: this.config.nodeColor,
            transparent: true,
            opacity: 0.15
        });
        
        this.boundingBox = new THREE.LineSegments(edges, material);
        this.scene.add(this.boundingBox);
    }
    
    // Initialize event handlers
    _initEventHandlers() {
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            // Calculate normalized mouse coordinates
            const rect = this.container.getBoundingClientRect();
            this.mouse.x = ((event.clientX - rect.left) / this.container.clientWidth) * 2 - 1;
            this.mouse.y = -((event.clientY - rect.top) / this.container.clientHeight) * 2 + 1;
        });
        
        this.renderer.domElement.addEventListener('click', (event) => {
            this._handleClick(event);
        });
        
        this.renderer.domElement.addEventListener('contextmenu', (event) => {
            event.preventDefault();
            this._handleContextMenu(event);
        });
    }
    
    // Initialize physics simulation
    _initPhysicsSimulation() {
        // Setup quantum state simulation parameters
        for (let i = 0; i < 128; i++) {
            // Initial phase is random
            this.phaseShifts.push(Math.random() * Math.PI * 2);
            
            // Quantum states initialized with random values
            this.quantumStates.push({
                amplitude: Math.random(),
                phase: Math.random() * Math.PI * 2,
                frequency: 0.5 + Math.random() * 2,
                damping: 0.9 + Math.random() * 0.1
            });
        }
        
        // Initialize wave function visualization
        this._initWaveFunction();
    }
    
    // Initialize wave function visualization
    _initWaveFunction() {
        const canvas = document.getElementById("wave-function");
        if (!canvas) return;
        
        this.waveContext = canvas.getContext("2d");
        
        // Set initial wave function to Gaussian shape
        for (let i = 0; i < this.waveFunction.length; i++) {
            const x = (i / this.waveFunction.length) * 2 - 1;
            this.waveFunction[i] = Math.exp(-x * x * 5) * Math.cos(x * 20);
        }
    }
    
    // Generate initial visualization
    _generateInitialVisualization() {
        // Calculate number of nodes based on density
        const numNodes = Math.ceil(Math.pow(this.config.nodeDensity, 3));
        const size = this.config.cubeSize / 2;  // Half size for node distribution
        
        // Initialize phase vector for quantum coherence
        const phaseVector = new THREE.Vector3(
            Math.random() * 2 - 1,
            Math.random() * 2 - 1,
            Math.random() * 2 - 1
        ).normalize();
        
        // Generate nodes
        for (let i = 0; i < numNodes; i++) {
            // Create node at random position within bounding box
            const position = new THREE.Vector3(
                (Math.random() * 2 - 1) * size,
                (Math.random() * 2 - 1) * size,
                (Math.random() * 2 - 1) * size
            );
            
            // Calculate phase based on position and phase vector (for coherence)
            const phase = (position.dot(phaseVector) / size + 1) * Math.PI;
            
            // Generate energy level with Gaussian distribution around configured value
            const energy = Math.max(0.1, Math.min(1.0, 
                this.config.energyLevel + (Math.random() * 0.4 - 0.2)
            ));
            
            // Create node
            this.createNode({
                position: position,
                energy: energy,
                phase: phase,
                stability: 0.7 + Math.random() * 0.3
            });
        }
        
        // Generate connections between close nodes
        this._generateConnections();
        
        // Update stats
        this.stats.nodes = this.nodes.size;
        this.stats.connections = this.connections.size;
        
        // Update UI displays
        this._updateStatsDisplay();
        this._updateChartsDisplay();
    }
    
    // Generate connections between nodes
    _generateConnections() {
        // Get all node IDs
        const nodeIds = Array.from(this.nodes.keys());
        
        // Calculate connections
        for (let i = 0; i < nodeIds.length; i++) {
            const nodeId1 = nodeIds[i];
            const node1 = this.nodes.get(nodeId1);
            
            for (let j = i + 1; j < nodeIds.length; j++) {
                const nodeId2 = nodeIds[j];
                const node2 = this.nodes.get(nodeId2);
                
                // Calculate distance between nodes
                const distance = node1.data.position.distanceTo(node2.data.position);
                
                // Connect nodes if they are close enough
                if (distance < this.config.connectionThreshold) {
                    // Calculate connection strength based on distance and energy
                    const strength = Math.max(0.1, 
                        (1 - distance / this.config.connectionThreshold) * 
                        (node1.data.energy + node2.data.energy) / 2
                    );
                    
                    // Create connection
                    this.createConnection(nodeId1, nodeId2, strength);
                }
            }
        }
    }
    
    // Create a new node with the given properties
    createNode(properties = {}) {
        if (this.nodes.size >= this.config.maxNodes) {
            console.warn(`Maximum node limit (${this.config.maxNodes}) reached.`);
            return null;
        }
        
        // Generate unique ID
        const nodeId = `node-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
        
        // Set default properties
        const nodeData = {
            id: nodeId,
            position: properties.position || new THREE.Vector3(0, 0, 0),
            energy: properties.energy !== undefined ? properties.energy : this.config.energyLevel,
            phase: properties.phase !== undefined ? properties.phase : Math.random() * Math.PI * 2,
            stability: properties.stability !== undefined ? properties.stability : 0.8,
            connections: new Set(),
            metadata: properties.metadata || {},
            createdAt: Date.now(),
            lastUpdated: Date.now()
        };
        
        // Calculate node size based on energy
        const nodeSize = 0.2 + nodeData.energy * 0.8;
        
        // Create node geometry
        const geometry = new THREE.SphereGeometry(nodeSize, 16, 16);
        
        // Create node material
        const material = new THREE.MeshPhongMaterial({
            color: this._getNodeColor(nodeData.energy, nodeData.phase),
            emissive: new THREE.Color(this._getNodeColor(nodeData.energy, nodeData.phase)).multiplyScalar(0.3),
            transparent: true,
            opacity: 0.8,
            wireframe: this.config.wireframe
        });
        
        // Create mesh
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(nodeData.position);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Set custom properties on mesh
        mesh.userData.id = nodeId;
        mesh.userData.type = 'node';
        
        // Add to scene
        this.nodesGroup.add(mesh);
        
        // Store node data and mesh
        this.nodes.set(nodeId, {
            mesh: mesh,
            data: nodeData
        });
        
        // Update stats
        this.stats.nodes = this.nodes.size;
        this._updateStatsDisplay();
        
        return nodeId;
    }
    
    // Create connection between two nodes
    createConnection(nodeId1, nodeId2, strength = 0.5) {
        if (this.connections.size >= this.config.maxConnections) {
            console.warn(`Maximum connection limit (${this.config.maxConnections}) reached.`);
            return null;
        }
        
        // Check if both nodes exist
        if (!this.nodes.has(nodeId1) || !this.nodes.has(nodeId2)) {
            console.warn("Cannot create connection - one or both nodes don't exist");
            return null;
        }
        
        // Check if it's not a self-connection
        if (nodeId1 === nodeId2) {
            console.warn("Cannot create self-connection");
            return null;
        }
        
        // Generate connection ID
        const connId = [nodeId1, nodeId2].sort().join('_');
        
        // Check if connection already exists
        if (this.connections.has(connId)) {
            return connId;
        }
        
        // Get node data
        const node1 = this.nodes.get(nodeId1);
        const node2 = this.nodes.get(nodeId2);
        
        // Create connection properties
        const connectionData = {
            id: connId,
            sourceId: nodeId1,
            targetId: nodeId2,
            strength: strength,
            createdAt: Date.now(),
            lastUpdated: Date.now()
        };
        
        // Add connection to nodes
        node1.data.connections.add(nodeId2);
        node2.data.connections.add(nodeId1);
        
        // Calculate connection color
        const connectionColor = new THREE.Color(this.config.nodeColor)
            .lerp(new THREE.Color(0xffffff), strength * 0.5);
        
        // Create line geometry
        const points = [
            node1.mesh.position.clone(),
            node2.mesh.position.clone()
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        // Create line material
        const material = new THREE.LineBasicMaterial({
            color: connectionColor,
            transparent: true,
            opacity: strength * 0.7,
            linewidth: 1
        });
        
        // Create line
        const line = new THREE.Line(geometry, material);
        
        // Set custom properties on line
        line.userData.id = connId;
        line.userData.type = 'connection';
        
        // Add to scene
        this.connectionsGroup.add(line);
        
        // Store connection data and line
        this.connections.set(connId, {
            line: line,
            data: connectionData
        });
        
        // Update stats
        this.stats.connections = this.connections.size;
        this._updateStatsDisplay();
        
        return connId;
    }
    
    // Calculate node color based on energy and phase
    _getNodeColor(energy, phase) {
        if (typeof this.config.nodeColor === 'function') {
            return this.config.nodeColor(energy, phase);
        }
        
        // Base color
        const baseColor = new THREE.Color(this.config.nodeColor);
        
        // Modify hue based on phase
        const hsl = {};
        baseColor.getHSL(hsl);
        hsl.h = (hsl.h + phase / (Math.PI * 2) * 0.2) % 1;
        
        // Modify lightness based on energy
        hsl.l = Math.min(0.9, Math.max(0.3, 0.4 + energy * 0.5));
        
        // Create modified color
        const newColor = new THREE.Color().setHSL(hsl.h, hsl.s, hsl.l);
        return newColor.getHex();
    }
    
    // Start animation loop
    _startAnimationLoop() {
        // Store start time
        this.startTime = Date.now();
        
        // Animation loop function
        const animate = (timestamp) => {
            requestAnimationFrame(animate);
            
            // Calculate delta time and update simulation time
            const deltaTime = this.lastFrameTime === 0 ? 0 : (timestamp - this.lastFrameTime) / 1000;
            this.lastFrameTime = timestamp;
            this.simulationTime += deltaTime;
            
            // Update FPS counter
            this.frameCount++;
            if (timestamp - this.lastFpsUpdate > 1000) {
                this.stats.fps = Math.round(this.frameCount * 1000 / (timestamp - this.lastFpsUpdate));
                this.lastFpsUpdate = timestamp;
                this.frameCount = 0;
                
                // Update stats display
                document.getElementById('fps-value').textContent = this.stats.fps;
            }
            
            // Update simulation
            this._updateSimulation(deltaTime);
            
            // Update controls
            this.controls.update();
            
            // Update wave function visualization
            this._updateWaveFunction(deltaTime);
            
            // Find hovered node
            this._updateHoveredNode();
            
            // Render scene
            if (this.config.glow) {
                this.composer.render();
            } else {
                this.renderer.render(this.scene, this.camera);
            }
        };
        
        // Start animation loop
        animate(0);
    }
    
    // Update quantum simulation
    _updateSimulation(deltaTime) {
        // Rotate bounding box
        this.boundingBox.rotation.x += this.config.rotationSpeed * 0.3 * deltaTime;
        this.boundingBox.rotation.y += this.config.rotationSpeed * 0.5 * deltaTime;
        
        // Update quantum state
        for (let i = 0; i < this.quantumStates.length; i++) {
            const state = this.quantumStates[i];
            
            // Update phase
            state.phase += state.frequency * deltaTime;
            
            // Apply damping to amplitude
            state.amplitude *= Math.pow(state.damping, deltaTime);
            
            // Occasionally add energy to maintain dynamics
            if (Math.random() < 0.01 * deltaTime) {
                state.amplitude = Math.min(1.0, state.amplitude + Math.random() * 0.2);
            }
        }
        
        // Update nodes
        for (const [nodeId, node] of this.nodes.entries()) {
            // Apply quantum jitter
            if (this.config.quantumJitter > 0) {
                const jitterAmount = this.config.quantumJitter * deltaTime;
                
                node.mesh.position.x += (Math.random() * 2 - 1) * jitterAmount;
                node.mesh.position.y += (Math.random() * 2 - 1) * jitterAmount;
                node.mesh.position.z += (Math.random() * 2 - 1) * jitterAmount;
                
                // Keep within bounds
                this._keepNodeInBounds(node.mesh);
            }
            
            // Update node phase
            node.data.phase += deltaTime * (0.5 + node.data.energy * 2.0);
            while (node.data.phase > Math.PI * 2) node.data.phase -= Math.PI * 2;
            
            // Update node color based on energy and phase
            node.mesh.material.color.set(this._getNodeColor(node.data.energy, node.data.phase));
            node.mesh.material.emissive.set(new THREE.Color(this._getNodeColor(node.data.energy, node.data.phase)).multiplyScalar(0.3));
            
            // Pulse size based on phase
            const basePulse = Math.sin(node.data.phase) * 0.1 + 1.0;
            const sizeScale = 0.2 + node.data.energy * 0.8 * basePulse;
            node.mesh.scale.set(sizeScale, sizeScale, sizeScale);
            
            // Update node data
            node.data.lastUpdated = Date.now();
        }
        
        // Update connections
        for (const [connId, connection] of this.connections.entries()) {
            // Get node positions
            const sourceId = connection.data.sourceId;
            const targetId = connection.data.targetId;
            
            // Check if nodes still exist
            if (!this.nodes.has(sourceId) || !this.nodes.has(targetId)) {
                // Remove connection if nodes don't exist
                this.removeConnection(connId);
                continue;
            }
            
            const sourcePos = this.nodes.get(sourceId).mesh.position;
            const targetPos = this.nodes.get(targetId).mesh.position;
            
            // Update line geometry to match node positions
            const points = [sourcePos, targetPos];
            connection.line.geometry.setFromPoints(points);
            
            // Pulse connection opacity based on phase difference
            const sourcePhase = this.nodes.get(sourceId).data.phase;
            const targetPhase = this.nodes.get(targetId).data.phase;
            const phaseDiff = Math.abs(sourcePhase - targetPhase) % (Math.PI * 2);
            const normDiff = phaseDiff / (Math.PI);
            
            // Higher opacity for nodes in phase
            const phaseCoherence = 1 - Math.min(normDiff, 1);
            
            // Pulse effect
            const pulse = 0.7 + Math.sin(this.simulationTime * 5) * 0.3;
            
            // Update connection opacity
            connection.line.material.opacity = 
                connection.data.strength * 0.7 * 
                (this.config.phaseCoherence * phaseCoherence + (1 - this.config.phaseCoherence)) *
                pulse;
            
            // Update connection data
            connection.data.lastUpdated = Date.now();
        }
        
        // Apply entanglement effects
        if (this.entangledNodes.size > 0) {
            this._applyEntanglementEffects(deltaTime);
        }
        
        // Update system phase (global property)
        this.stats.phase = (this.stats.phase + deltaTime * 0.5) % (Math.PI * 2);
        document.getElementById('phase-value').textContent = this.stats.phase.toFixed(2);
    }
    
    // Keep node within bounds of visualization
    _keepNodeInBounds(nodeMesh) {
        const size = this.config.cubeSize / 2;
        
        // X bounds
        if (nodeMesh.position.x > size) {
            nodeMesh.position.x = size;
        } else if (nodeMesh.position.x < -size) {
            nodeMesh.position.x = -size;
        }
        
        // Y bounds
        if (nodeMesh.position.y > size) {
            nodeMesh.position.y = size;
        } else if (nodeMesh.position.y < -size) {
            nodeMesh.position.y = -size;
        }
        
        // Z bounds
        if (nodeMesh.position.z > size) {
            nodeMesh.position.z = size;
        } else if (nodeMesh.position.z < -size) {
            nodeMesh.position.z = -size;
        }
    }
    
    // Apply entanglement effects to entangled nodes
    _applyEntanglementEffects(deltaTime) {
        // If no entangled nodes, do nothing
        if (this.entangledNodes.size === 0) return;
        
        // Get array of entangled node IDs
        const entangledNodeIds = Array.from(this.entangledNodes);
        
        // Calculate average phase
        let avgPhase = 0;
        for (const nodeId of entangledNodeIds) {
            if (this.nodes.has(nodeId)) {
                avgPhase += this.nodes.get(nodeId).data.phase;
            }
        }
        avgPhase /= this.entangledNodes.size;
        
        // Apply entanglement effects
        for (const nodeId of entangledNodeIds) {
            if (!this.nodes.has(nodeId)) continue;
            
            const node = this.nodes.get(nodeId);
            
            // Phase synchronization - adjust node phase toward average
            const phaseDiff = avgPhase - node.data.phase;
            node.data.phase += phaseDiff * this.config.entanglementStrength * deltaTime * 5;
            
            // Energy osmosis - nodes share energy
            const avgEnergy = this.stats.energy;
            const energyDiff = avgEnergy - node.data.energy;
            node.data.energy += energyDiff * this.config.entanglementStrength * deltaTime;
        }
    }
    
    // Update wave function visualization
    _updateWaveFunction(deltaTime) {
        if (!this.waveContext) return;
        
        const canvas = this.waveContext.canvas;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        this.waveContext.clearRect(0, 0, width, height);
        
        // Draw background
        this.waveContext.fillStyle = 'rgba(0, 0, 20, 0.3)';
        this.waveContext.fillRect(0, 0, width, height);
        
        // Update wave function
        for (let i = 0; i < this.waveFunction.length; i++) {
            // Each point is affected by quantum states
            let sum = 0;
            for (let j = 0; j < Math.min(5, this.quantumStates.length); j++) {
                const state = this.quantumStates[j];
                const x = (i / this.waveFunction.length) * Math.PI * 4;
                sum += state.amplitude * Math.sin(x * state.frequency + state.phase);
            }
            
            // Normalize and update
            this.waveFunction[i] = this.waveFunction[i] * 0.95 + sum * 0.05;
        }
        
        // Draw wave function
        this.waveContext.beginPath();
        this.waveContext.strokeStyle = `rgba(76, 201, 240, 0.8)`;
        this.waveContext.lineWidth = 2;
        
        for (let i = 0; i < this.waveFunction.length; i++) {
            const x = (i / this.waveFunction.length) * width;
            const y = height / 2 + this.waveFunction[i] * height / 3;
            
            if (i === 0) {
                this.waveContext.moveTo(x, y);
            } else {
                this.waveContext.lineTo(x, y);
            }
        }
        
        this.waveContext.stroke();
        
        // Draw probability envelope
        this.waveContext.beginPath();
        this.waveContext.strokeStyle = `rgba(247, 37, 133, 0.5)`;
        this.waveContext.lineWidth = 1;
        
        for (let i = 0; i < this.waveFunction.length; i++) {
            const x = (i / this.waveFunction.length) * width;
            const amplitude = Math.abs(this.waveFunction[i]);
            const y = height / 2 + amplitude * height / 3;
            
            if (i === 0) {
                this.waveContext.moveTo(x, y);
            } else {
                this.waveContext.lineTo(x, y);
            }
        }
        
        this.waveContext.stroke();
        
        // Draw zero line
        this.waveContext.beginPath();
        this.waveContext.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        this.waveContext.lineWidth = 1;
        this.waveContext.moveTo(0, height / 2);
        this.waveContext.lineTo(width, height / 2);
        this.waveContext.stroke();
    }
    
    // Update hovered node detection
    _updateHoveredNode() {
        // Clear previous hover state
        if (this.hoveredNode && this.nodes.has(this.hoveredNode)) {
            const node = this.nodes.get(this.hoveredNode);
            // Reset emissive color if not selected
            if (this.hoveredNode !== this.selectedNode) {
                node.mesh.material.emissive.set(new THREE.Color(this._getNodeColor(node.data.energy, node.data.phase)).multiplyScalar(0.3));
            }
        }
        
        // Update raycaster with current mouse position
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        // Find intersections with nodes
        const intersects = this.raycaster.intersectObjects(this.nodesGroup.children);
        
        // Set new hovered node
        if (intersects.length > 0) {
            const object = intersects[0].object;
            if (object.userData.type === 'node') {
                this.hoveredNode = object.userData.id;
                
                // Highlight hovered node if not selected
                if (this.hoveredNode !== this.selectedNode) {
                    const node = this.nodes.get(this.hoveredNode);
                    node.mesh.material.emissive.set(new THREE.Color(0xffffff).multiplyScalar(0.5));
                }
            }
        } else {
            this.hoveredNode = null;
        }
    }
    
    // Handle node click
    _handleClick(event) {
        // Check if we have a hovered node
        if (this.hoveredNode) {
            this.selectNode(this.hoveredNode);
        } else {
            this.clearSelection();
        }
    }
    
    // Handle context menu
    _handleContextMenu(event) {
        event.preventDefault();
        
        // Get mouse position
        const rect = this.container.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        
        // Show context menu
        const contextMenu = document.getElementById('context-menu');
        if (contextMenu) {
            contextMenu.style.display = 'block';
            contextMenu.style.left = `${mouseX}px`;
            contextMenu.style.top = `${mouseY}px`;
            
            // Store current 3D position for adding nodes
            this.contextMenuPosition = this._getMousePosition3D(mouseX, mouseY);
            
            // Set up event listener to hide menu when clicking elsewhere
            const hideContextMenu = () => {
                contextMenu.style.display = 'none';
                document.removeEventListener('click', hideContextMenu);
            };
            
            // Delay adding the event listener to avoid immediate closing
            setTimeout(() => {
                document.addEventListener('click', hideContextMenu);
            }, 10);
        }
    }
    
    // Get 3D position from 2D mouse coordinates
    _getMousePosition3D(x, y) {
        // Calculate normalized mouse coordinates
        const rect = this.container.getBoundingClientRect();
        const mouseX = ((x / rect.width) * 2) - 1;
        const mouseY = -((y / rect.height) * 2) + 1;
        
        // Create raycaster
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), this.camera);
        
        // Calculate intersection with bounding box plane
        const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1).normalize(), 0);
        const position = new THREE.Vector3();
        raycaster.ray.intersectPlane(plane, position);
        
        // Keep within bounds
        const size = this.config.cubeSize / 2;
        position.x = Math.max(-size, Math.min(size, position.x));
        position.y = Math.max(-size, Math.min(size, position.y));
        position.z = Math.max(-size, Math.min(size, position.z));
        
        return position;
    }
    
    // Select a node
    selectNode(nodeId) {
        // Clear previous selection
        this.clearSelection();
        
        // Set new selection
        if (this.nodes.has(nodeId)) {
            this.selectedNode = nodeId;
            const node = this.nodes.get(nodeId);
            
            // Highlight selected node
            node.mesh.material.emissive.set(new THREE.Color(0xffffff));
            
            // Display node details
            this._showNodeDetails(nodeId);
        }
    }
    
    // Clear node selection
    clearSelection() {
        // Reset previous selection
        if (this.selectedNode && this.nodes.has(this.selectedNode)) {
            const node = this.nodes.get(this.selectedNode);
            node.mesh.material.emissive.set(new THREE.Color(this._getNodeColor(node.data.energy, node.data.phase)).multiplyScalar(0.3));
        }
        
        this.selectedNode = null;
        
        // Clear node details display
        this._hideNodeDetails();
    }
    
    // Show node details in UI
    _showNodeDetails(nodeId) {
        // Get node data
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        // Show modal with node details
        this._showModal({
            title: `Node Details: ${nodeId.substring(0, 8)}...`,
            content: `
                <div>
                    <p><strong>Energy:</strong> ${node.data.energy.toFixed(2)}</p>
                    <p><strong>Phase:</strong> ${node.data.phase.toFixed(2)}</p>
                    <p><strong>Stability:</strong> ${node.data.stability.toFixed(2)}</p>
                    <p><strong>Connections:</strong> ${node.data.connections.size}</p>
                    <p><strong>Position:</strong> 
                        (${node.data.position.x.toFixed(2)}, 
                         ${node.data.position.y.toFixed(2)}, 
                         ${node.data.position.z.toFixed(2)})</p>
                    <p><strong>Created:</strong> ${new Date(node.data.createdAt).toLocaleTimeString()}</p>
                </div>
            `,
            actions: [
                {
                    label: 'Entangle',
                    callback: () => this.entangleNode(nodeId)
                },
                {
                    label: 'Boost Energy',
                    callback: () => this.boostNodeEnergy(nodeId)
                },
                {
                    label: 'Delete',
                    callback: () => this.removeNode(nodeId)
                }
            ]
        });
    }
    
    // Hide node details
    _hideNodeDetails() {
        // Hide modal
        this._hideModal();
    }
    
    // Show modal dialog
    _showModal(options = {}) {
        const modal = document.getElementById('modal');
        const title = document.querySelector('.modal-title');
        const body = document.getElementById('modal-body');
        const closeBtn = document.querySelector('.modal-close');
        const cancelBtn = document.getElementById('modal-cancel');
        const confirmBtn = document.getElementById('modal-confirm');
        const footer = document.querySelector('.modal-footer');
        
        // Set title and content
        title.textContent = options.title || 'Quantum Visualization';
        body.innerHTML = options.content || '';
        
        // Set up actions
        footer.innerHTML = '';
        
        if (options.actions && Array.isArray(options.actions)) {
            options.actions.forEach(action => {
                const button = document.createElement('button');
                button.className = 'modal-btn';
                button.textContent = action.label;
                button.onclick = () => {
                    this._hideModal();
                    if (typeof action.callback === 'function') {
                        action.callback();
                    }
                };
                footer.appendChild(button);
            });
        } else {
            // Default buttons
            cancelBtn.onclick = () => this._hideModal();
            confirmBtn.onclick = () => {
                this._hideModal();
                if (typeof options.onConfirm === 'function') {
                    options.onConfirm();
                }
            };
            footer.appendChild(cancelBtn);
            footer.appendChild(confirmBtn);
        }
        
        // Set up close button
        closeBtn.onclick = () => this._hideModal();
        
        // Show modal
        modal.classList.add('visible');
    }
    
    // Hide modal dialog
    _hideModal() {
        const modal = document.getElementById('modal');
        modal.classList.remove('visible');
    }
    
    // Show toast notification
    _showToast(message, duration = 3000) {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.classList.add('visible');
        
        // Hide after duration
        setTimeout(() => {
            toast.classList.remove('visible');
        }, duration);
    }
    
    // Update stats display
    _updateStatsDisplay() {
        document.getElementById('nodes-value').textContent = this.stats.nodes;
        document.getElementById('connections-value').textContent = this.stats.connections;
        document.getElementById('energy-stat').textContent = `${Math.round(this.stats.energy * 100)}%`;
    }
    
    // Update charts display
    _updateChartsDisplay() {
        // Update probability chart
        this._updateProbabilityChart();
        
        // Update evolution chart
        this._updateEvolutionChart();
        
        // Update energy chart
        this._updateEnergyChart();
        
        // Update metrics chart
        this._updateMetricsChart();
    }
    
    // Update probability chart
    _updateProbabilityChart() {
        const canvas = document.getElementById('probabilityChart');
        if (!canvas) return;
        
        if (!this.probabilityChart) {
            // Create chart if it doesn't exist
            this.probabilityChart = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: Array.from({length: 10}, (_, i) => `t-${9-i}`),
                    datasets: [{
                        label: 'Probability Amplitude',
                        data: Array.from({length: 10}, () => Math.random()),
                        borderColor: '#4cc9f0',
                        backgroundColor: 'rgba(76, 201, 240, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    }
                }
            });
        } else {
            // Update existing chart
            const data = this.probabilityChart.data.datasets[0].data;
            data.shift();
            data.push(this.stats.energy * (0.8 + Math.random() * 0.4));
            this.probabilityChart.update();
        }
    }
    
    // Update evolution chart
    _updateEvolutionChart() {
        const canvas = document.getElementById('evolutionChart');
        if (!canvas) return;
        
        if (!this.evolutionChart) {
            // Create chart if it doesn't exist
            this.evolutionChart = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: Array.from({length: 10}, (_, i) => `t-${9-i}`),
                    datasets: [{
                        label: 'Node Count',
                        data: Array(10).fill(this.stats.nodes),
                        borderColor: '#f72585',
                        backgroundColor: 'rgba(247, 37, 133, 0.1)',
                        tension: 0.4,
                        fill: true,
                        yAxisID: 'y'
                    }, {
                        label: 'Connections',
                        data: Array(10).fill(this.stats.connections),
                        borderColor: '#7209b7',
                        backgroundColor: 'rgba(114, 9, 183, 0.1)',
                        tension: 0.4,
                        fill: true,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            position: 'left',
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        },
                        y1: {
                            beginAtZero: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    }
                }
            });
        } else {
            // Update existing chart
            const nodesData = this.evolutionChart.data.datasets[0].data;
            nodesData.shift();
            nodesData.push(this.stats.nodes);
            
            const connectionsData = this.evolutionChart.data.datasets[1].data;
            connectionsData.shift();
            connectionsData.push(this.stats.connections);
            
            this.evolutionChart.update();
        }
    }
    
    // Update energy chart
    _updateEnergyChart() {
        const canvas = document.getElementById('energyChart');
        if (!canvas) return;
        
        // Collect energy data from nodes
        const energyData = Array(10).fill(0);
        let totalNodes = this.nodes.size;
        
        if (totalNodes > 0) {
            for (const [_, node] of this.nodes.entries()) {
                const energyBin = Math.min(9, Math.floor(node.data.energy * 10));
                energyData[energyBin]++;
            }
            
            // Convert to percentages
            for (let i = 0; i < energyData.length; i++) {
                energyData[i] = (energyData[i] / totalNodes) * 100;
            }
        }
        
        if (!this.energyChart) {
            // Create chart if it doesn't exist
            this.energyChart = new Chart(canvas, {
                type: 'bar',
                data: {
                    labels: Array.from({length: 10}, (_, i) => `${i*10}-${(i+1)*10}%`),
                    datasets: [{
                        label: 'Energy Distribution',
                        data: energyData,
                        backgroundColor: [
                            'rgba(114, 9, 183, 0.7)',
                            'rgba(86, 11, 173, 0.7)',
                            'rgba(58, 12, 163, 0.7)',
                            'rgba(63, 55, 201, 0.7)',
                            'rgba(67, 97, 238, 0.7)',
                            'rgba(72, 149, 239, 0.7)',
                            'rgba(76, 201, 240, 0.7)',
                            'rgba(103, 232, 249, 0.7)',
                            'rgba(159, 243, 233, 0.7)',
                            'rgba(217, 250, 211, 0.7)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    }
                }
            });
        } else {
            // Update existing chart
            this.energyChart.data.datasets[0].data = energyData;
            this.energyChart.update();
        }
    }
    
    // Update metrics chart
    _updateMetricsChart() {
        const canvas = document.getElementById('metricsChart');
        if (!canvas) return;
        
        // Calculate metrics
        const coherence = this.config.phaseCoherence;
        const entanglement = this.entangledNodes.size / Math.max(1, this.nodes.size);
        const stability = Array.from(this.nodes.values()).reduce((sum, node) => sum + node.data.stability, 0) 
                         / Math.max(1, this.nodes.size);
        const complexity = this.connections.size / Math.max(1, this.nodes.size);
        
        if (!this.metricsChart) {
            // Create chart if it doesn't exist
            this.metricsChart = new Chart(canvas, {
                type: 'radar',
                data: {
                    labels: ['Coherence', 'Entanglement', 'Stability', 'Complexity'],
                    datasets: [{
                        label: 'System Metrics',
                        data: [coherence, entanglement, stability, complexity],
                        backgroundColor: 'rgba(247, 37, 133, 0.2)',
                        borderColor: 'rgba(247, 37, 133, 0.7)',
                        pointBackgroundColor: 'rgba(247, 37, 133, 1)',
                        pointBorderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                                display: false
                            },
                            pointLabels: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            angleLines: {
                                color: 'rgba(255, 255, 255, 0.2)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    }
                }
            });
        } else {
            // Update existing chart
            this.metricsChart.data.datasets[0].data = [coherence, entanglement, stability, complexity];
            this.metricsChart.update();
        }
    }
    
    // Public API Methods
    
    // Add a new node at specified position
    addNodeAt(position, properties = {}) {
        // Create node with specified position
        const nodeId = this.createNode({
            position: position,
            energy: properties.energy !== undefined ? properties.energy : this.config.energyLevel,
            phase: properties.phase !== undefined ? properties.phase : Math.random() * Math.PI * 2,
            stability: properties.stability !== undefined ? properties.stability : 0.8,
            metadata: properties.metadata || {}
        });
        
        if (nodeId) {
            this._showToast(`Node created: ${nodeId.substring(0, 8)}...`);
            return nodeId;
        }
        
        return null;
    }
    
    // Remove a node
    removeNode(nodeId) {
        if (!this.nodes.has(nodeId)) {
            console.warn(`Node ${nodeId} does not exist`);
            return false;
        }
        
        // Get node
        const node = this.nodes.get(nodeId);
        
        // Remove from scene
        this.nodesGroup.remove(node.mesh);
        
        // Remove any connections to this node
        for (const otherNodeId of node.data.connections) {
            // Get connection ID
            const connId = [nodeId, otherNodeId].sort().join('_');
            
            // Remove connection
            if (this.connections.has(connId)) {
                const connection = this.connections.get(connId);
                this.connectionsGroup.remove(connection.line);
                this.connections.delete(connId);
            }
            
            // Update other node's connections
            if (this.nodes.has(otherNodeId)) {
                this.nodes.get(otherNodeId).data.connections.delete(nodeId);
            }
        }
        
        // Remove node
        this.nodes.delete(nodeId);
        
        // Update selected node if it was removed
        if (this.selectedNode === nodeId) {
            this.selectedNode = null;
            this._hideNodeDetails();
        }
        
        // Update hovered node if it was removed
        if (this.hoveredNode === nodeId) {
            this.hoveredNode = null;
        }
        
        // Update entangled nodes if it was in the set
        if (this.entangledNodes.has(nodeId)) {
            this.entangledNodes.delete(nodeId);
        }
        
        // Update stats
        this.stats.nodes = this.nodes.size;
        this.stats.connections = this.connections.size;
        this._updateStatsDisplay();
        
        this._showToast(`Node removed: ${nodeId.substring(0, 8)}...`);
        return true;
    }
    
    // Remove a connection
    removeConnection(connId) {
        if (!this.connections.has(connId)) {
            console.warn(`Connection ${connId} does not exist`);
            return false;
        }
        
        // Get connection
        const connection = this.connections.get(connId);
        
        // Get node IDs
        const [nodeId1, nodeId2] = connId.split('_');
        
        // Remove from scene
        this.connectionsGroup.remove(connection.line);
        
        // Update nodes
        if (this.nodes.has(nodeId1)) {
            this.nodes.get(nodeId1).data.connections.delete(nodeId2);
        }
        
        if (this.nodes.has(nodeId2)) {
            this.nodes.get(nodeId2).data.connections.delete(nodeId1);
        }
        
        // Remove connection
        this.connections.delete(connId);
        
        // Update stats
        this.stats.connections = this.connections.size;
        this._updateStatsDisplay();
        
        return true;
    }
    
    // Entangle a node
    entangleNode(nodeId) {
        if (!this.nodes.has(nodeId)) {
            console.warn(`Node ${nodeId} does not exist`);
            return false;
        }
        
        // Add to entangled set
        this.entangledNodes.add(nodeId);
        
        // Visual effect - make node pulse
        const node = this.nodes.get(nodeId);
        const originalMaterial = node.mesh.material.clone();
        
        // Create pulsing animation
        const pulseEffect = (time) => {
            if (!this.nodes.has(nodeId)) return;
            
            const pulseFactor = 1 + Math.sin(time * 5) * 0.3;
            node.mesh.scale.set(pulseFactor, pulseFactor, pulseFactor);
            
            // Pulsing color
            const emissiveIntensity = 0.3 + Math.sin(time * 5) * 0.2;
            node.mesh.material.emissive.copy(new THREE.Color(this._getNodeColor(node.data.energy, node.data.phase)));
            node.mesh.material.emissive.multiplyScalar(emissiveIntensity);
            
            requestAnimationFrame(pulseEffect);
        };
        
        // Start animation
        requestAnimationFrame(pulseEffect);
        
        this._showToast(`Node entangled: ${nodeId.substring(0, 8)}...`);
        return true;
    }
    
    // Boost node energy
    boostNodeEnergy(nodeId) {
        if (!this.nodes.has(nodeId)) {
            console.warn(`Node ${nodeId} does not exist`);
            return false;
        }
        
        // Get node
        const node = this.nodes.get(nodeId);
        
        // Boost energy
        node.data.energy = Math.min(1.0, node.data.energy + 0.2);
        
        // Visual effect - flash node
        const originalColor = node.mesh.material.color.clone();
        node.mesh.material.color.set(0xffffff);
        
        setTimeout(() => {
            if (this.nodes.has(nodeId)) {
                node.mesh.material.color.set(this._getNodeColor(node.data.energy, node.data.phase));
            }
        }, 300);
        
        this._showToast(`Node energy boosted: ${nodeId.substring(0, 8)}...`);
        return true;
    }
    
    // Clear all nodes and connections
    clearAll() {
        // Remove all connections
        for (const [connId, connection] of this.connections.entries()) {
            this.connectionsGroup.remove(connection.line);
        }
        this.connections.clear();
        
        // Remove all nodes
        for (const [nodeId, node] of this.nodes.entries()) {
            this.nodesGroup.remove(node.mesh);
        }
        this.nodes.clear();
        
        // Clear selection
        this.selectedNode = null;
        this.hoveredNode = null;
        this.entangledNodes.clear();
        
        // Update stats
        this.stats.nodes = 0;
        this.stats.connections = 0;
        this._updateStatsDisplay();
        
        this._showToast("Cleared all nodes and connections");
        return true;
    }
    
    // Process text input
    processText(text) {
        if (!text || text.trim() === '') return null;
        
        // Hash the text for deterministic output
        const hash = this._hashString(text);
        
        // Create metadata
        const metadata = {
            type: "text",
            content: text.length > 50 ? text.substring(0, 50) + "..." : text,
            timestamp: Date.now()
        };
        
        // Calculate a deterministic position based on hash
        const position = new THREE.Vector3(
            ((hash[0] / 255) * 2 - 1) * this.config.cubeSize / 2,
            ((hash[1] / 255) * 2 - 1) * this.config.cubeSize / 2,
            ((hash[2] / 255) * 2 - 1) * this.config.cubeSize / 2
        );
        
        // Calculate energy and phase based on text
        const energy = 0.3 + (text.length % 100) / 100 * 0.7;
        const phase = (hash[3] / 255) * Math.PI * 2;
        
        // Create node
        const nodeId = this.createNode({
            position: position,
            energy: energy,
            phase: phase,
            stability: 0.7 + (hash[4] / 255) * 0.3,
            metadata: metadata
        });
        
        if (!nodeId) return null;
        
        // Find close nodes to connect with
        this._connectToNearbyNodes(nodeId, 3);
        
        // Create quantum burst effect
        this._createQuantumBurst(position, 5);
        
        // Update charts
        this._updateChartsDisplay();
        
        this._showToast(`Text processed: ${text.length} characters`);
        return nodeId;
    }
    
    // Connect a node to nearby nodes
    _connectToNearbyNodes(nodeId, maxConnections = 3) {
        if (!this.nodes.has(nodeId)) return;
        
        const node = this.nodes.get(nodeId);
        const nodePos = node.data.position;
        
        // Calculate distances to all other nodes
        const distances = [];
        for (const [otherNodeId, otherNode] of this.nodes.entries()) {
            if (otherNodeId !== nodeId) {
                const distance = nodePos.distanceTo(otherNode.data.position);
                distances.push({
                    id: otherNodeId,
                    distance: distance
                });
            }
        }
        
        // Sort by distance
        distances.sort((a, b) => a.distance - b.distance);
        
        // Connect to closest nodes
        const connectCount = Math.min(maxConnections, distances.length);
        for (let i = 0; i < connectCount; i++) {
            const otherNodeId = distances[i].id;
            const distance = distances[i].distance;
            
            // Only connect if within threshold
            if (distance < this.config.connectionThreshold * 1.5) {
                // Connection strength inversely proportional to distance
                const strength = Math.max(0.2, 1 - (distance / (this.config.connectionThreshold * 1.5)));
                this.createConnection(nodeId, otherNodeId, strength);
            }
        }
    }
    
    // Create quantum burst effect at position
    _createQuantumBurst(position, numParticles = 10) {
        const size = this.config.cubeSize / 4;
        const particleSize = 0.2;
        const duration = 1.0;
        const particles = [];
        
        // Create particles
        for (let i = 0; i < numParticles; i++) {
            const direction = new THREE.Vector3(
                Math.random() * 2 - 1,
                Math.random() * 2 - 1,
                Math.random() * 2 - 1
            ).normalize();
            
            const velocity = direction.clone().multiplyScalar(size * 2);
            
            const geometry = new THREE.SphereGeometry(particleSize, 8, 8);
            const material = new THREE.MeshPhongMaterial({
                color: this.config.nodeColor,
                emissive: this.config.nodeColor,
                emissiveIntensity: 0.5,
                transparent: true,
                opacity: 1.0
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.copy(position);
            
            this.scene.add(mesh);
            
            particles.push({
                mesh: mesh,
                velocity: velocity,
                startTime: this.simulationTime
            });
        }
        
        // Animate particles
        const animateParticles = () => {
            let allDone = true;
            
            for (let i = 0; i < particles.length; i++) {
                const particle = particles[i];
                const elapsed = this.simulationTime - particle.startTime;
                
                if (elapsed < duration) {
                    allDone = false;
                    
                    // Update position
                    particle.mesh.position.add(
                        particle.velocity.clone().multiplyScalar(0.03)
                    );
                    
                    // Update scale and opacity
                    const progress = elapsed / duration;
                    const scale = 1.0 - progress;
                    particle.mesh.scale.set(scale, scale, scale);
                    particle.mesh.material.opacity = 1.0 - progress;
                } else if (particle.mesh.parent) {
                    // Remove completed particle
                    this.scene.remove(particle.mesh);
                }
            }
            
            if (!allDone) {
                requestAnimationFrame(animateParticles);
            }
        };
        
        // Start animation
        requestAnimationFrame(animateParticles);
    }
    
    // Create a hash from a string
    _hashString(str) {
        let hash = new Uint8Array(16);
        
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash[i % 16] = (hash[i % 16] + char) % 256;
        }
        
        return hash;
    }
    
    // Simulate quantum analysis on text
    simulateQuantumAnalysis(text) {
        // Update quantum states based on text
        const hash = this._hashString(text);
        
        for (let i = 0; i < Math.min(hash.length, this.quantumStates.length); i++) {
            const state = this.quantumStates[i];
            state.amplitude = hash[i] / 255;
            state.phase = (hash[i] / 255) * Math.PI * 2;
        }
        
        // Update wave function
        for (let i = 0; i < this.waveFunction.length; i++) {
            let sum = 0;
            for (let j = 0; j < Math.min(5, this.quantumStates.length); j++) {
                const state = this.quantumStates[j];
                const x = (i / this.waveFunction.length) * Math.PI * 4;
                sum += state.amplitude * Math.sin(x * state.frequency + state.phase);
            }
            
            this.waveFunction[i] = sum;
        }
        
        // Return simulated analysis result
        return {
            sentiment: Math.random() * 2 - 1, // -1 to 1
            complexity: 0.1 + (text.length / 1000) * 0.9, // 0.1 to 1
            coherence: hash[0] / 255, // 0 to 1
            quantumEntropy: hash[1] / 255, // 0 to 1
            dominantFrequency: 1 + (hash[2] / 25), // 1 to ~11 Hz
            phaseCoupling: hash[3] / 255 // 0 to 1
        };
    }
    
    // Generate quantum insights from text
    generateQuantumInsights(text) {
        const analysis = this.simulateQuantumAnalysis(text);
        
        // Generate insights based on analysis
        const insights = [];
        
        if (analysis.sentiment > 0.5) {
            insights.push("High positive quantum resonance detected in semantic field");
        } else if (analysis.sentiment < -0.5) {
            insights.push("Strong negative phase interference in conceptual structures");
        }
        
        if (analysis.complexity > 0.7) {
            insights.push("Complex multi-dimensional semantic entanglement present");
        } else if (analysis.complexity < 0.3) {
            insights.push("Simple harmonic information patterns observed");
        }
        
        if (analysis.coherence > 0.8) {
            insights.push("Exceptionally coherent quantum information state");
        } else if (analysis.coherence < 0.2) {
            insights.push("Significant decoherence in semantic superposition");
        }
        
        if (analysis.quantumEntropy > 0.7) {
            insights.push("High entropy quantum state suggests creative potential");
        } else if (analysis.quantumEntropy < 0.3) {
            insights.push("Low entropy indicates structured information patterns");
        }
        
        if (analysis.dominantFrequency > 8) {
            insights.push("High frequency semantic oscillations detected");
        } else if (analysis.dominantFrequency < 3) {
            insights.push("Low frequency conceptual resonance observed");
        }
        
        return insights;
    }
}

// UI Controller to connect visualization engine with DOM elements
class UIController {
    constructor(engine) {
        this.engine = engine;
        this._initUIElements();
        this._setupEventListeners();
    }
    
    _initUIElements() {
        // Control panel toggle
        this.panelToggle = document.getElementById('panel-toggle');
        this.controlPanel = document.getElementById('control-panel');
        
        // Charts panel toggle
        this.chartsToggle = document.getElementById('charts-toggle');
        this.chartsPanel = document.getElementById('charts-panel');
        
        // Sliders
        this.sizeSlider = document.getElementById('size-slider');
        this.densitySlider = document.getElementById('density-slider');
        this.energySlider = document.getElementById('energy-slider');
        this.connectionSlider = document.getElementById('connection-slider');
        this.rotationSlider = document.getElementById('rotation-slider');
        this.jitterSlider = document.getElementById('jitter-slider');
        this.entanglementSlider = document.getElementById('entanglement-slider');
        this.coherenceSlider = document.getElementById('coherence-slider');
        
        // Slider value displays
        this.sizeValue = document.getElementById('size-value');
        this.densityValue = document.getElementById('density-value');
        this.energyValue = document.getElementById('energy-value');
        this.connectionValue = document.getElementById('connection-value');
        this.rotationValue = document.getElementById('rotation-value');
        this.jitterValue = document.getElementById('jitter-value');
        this.entanglementValue = document.getElementById('entanglement-value');
        this.coherenceValue = document.getElementById('coherence-value');
        
        // Color selector
        this.colorBoxes = document.querySelectorAll('.colorbox');
        
        // Control buttons
        this.resetBtn = document.getElementById('reset-btn');
        this.addNodesBtn = document.getElementById('add-nodes-btn');
        this.entangleBtn = document.getElementById('entangle-btn');
        this.explosionBtn = document.getElementById('explosion-btn');
        this.wireframeBtn = document.getElementById('wireframe-btn');
        this.glowBtn = document.getElementById('glow-btn');
        this.saveBtn = document.getElementById('save-btn');
        this.fullscreenBtn = document.getElementById('fullscreen-btn');
        
        // Text input elements
        this.textInput = document.getElementById('text-input');
        this.textClearBtn = document.getElementById('text-clear-btn');
        this.textProcessBtn = document.getElementById('text-process-btn');
        
        // Toolbar buttons
        this.quantumBtn = document.getElementById('quantum-btn');
        this.insightsBtn = document.getElementById('insights-btn');
        this.simulateBtn = document.getElementById('simulate-btn');
        
        // Context menu items
        this.ctxAddNode = document.getElementById('ctx-add-node');
        this.ctxClearArea = document.getElementById('ctx-clear-area');
        this.ctxExplodeFromHere = document.getElementById('ctx-explode-from-here');
        this.ctxCreateCluster = document.getElementById('ctx-create-cluster');
        
        // Loading elements
        this.loading = document.getElementById('loading');
        this.progressFill = document.getElementById('progress-fill');
        this.loadingText = document.getElementById('loading-text');
    }
    
    _setupEventListeners() {
        // Panel toggles
        this.panelToggle.addEventListener('click', () => {
            this.controlPanel.classList.toggle('visible');
        });
        
        this.chartsToggle.addEventListener('click', () => {
            this.chartsPanel.classList.toggle('visible');
        });
        
        // Sliders
        this.sizeSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.sizeValue.textContent = value;
            this.engine.config.cubeSize = value;
            this.engine.boundingBox.scale.set(value/30, value/30, value/30);
        });
        
        this.densitySlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.densityValue.textContent = value;
            this.engine.config.nodeDensity = value;
        });
        
        this.energySlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.energyValue.textContent = value;
            this.engine.config.energyLevel = value / 100;
            this.engine.stats.energy = value / 100;
            document.getElementById('energy-stat').textContent = `${value}%`;
        });
        
        this.connectionSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.connectionValue.textContent = value;
            this.engine.config.connectionThreshold = value;
        });
        
        this.rotationSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.rotationValue.textContent = value;
            this.engine.config.rotationSpeed = value / 1000;
        });
        
        this.jitterSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.jitterValue.textContent = value;
            this.engine.config.quantumJitter = value / 100;
        });
        
        this.entanglementSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.entanglementValue.textContent = value;
            this.engine.config.entanglementStrength = value / 100;
        });
        
        this.coherenceSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.coherenceValue.textContent = value;
            this.engine.config.phaseCoherence = value / 100;
        });
        
        // Color boxes
        this.colorBoxes.forEach(box => {
            box.addEventListener('click', (e) => {
                // Remove active class from all
                this.colorBoxes.forEach(b => b.classList.remove('active'));
                
                // Add active class to clicked
                box.classList.add('active');
                
                // Set color
                const color = box.dataset.color;
                this.engine.config.nodeColor = parseInt(color.replace('#', '0x'));
            });
        });
        
        // Control buttons
        this.resetBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to reset the visualization?')) {
                this.engine.clearAll();
                this.engine._generateInitialVisualization();
            }
        });
        
        this.addNodesBtn.addEventListener('click', () => {
            const numNodes = Math.min(20, this.engine.config.maxNodes - this.engine.nodes.size);
            
            if (numNodes <= 0) {
                this.engine._showToast('Maximum node limit reached');
                return;
            }
            
            const size = this.engine.config.cubeSize / 2;
            
            for (let i = 0; i < numNodes; i++) {
                const position = new THREE.Vector3(
                    (Math.random() * 2 - 1) * size,
                    (Math.random() * 2 - 1) * size,
                    (Math.random() * 2 - 1) * size
                );
                
                this.engine.addNodeAt(position);
            }
            
            this.engine._connectToNearbyNodes(this.engine.nodes.keys().next().value, 5);
            this.engine._showToast(`Added ${numNodes} nodes`);
        });
        
        this.entangleBtn.addEventListener('click', () => {
            // Entangle 3 random nodes
            const nodeIds = Array.from(this.engine.nodes.keys());
            if (nodeIds.length < 3) {
                this.engine._showToast('Need at least 3 nodes to entangle');
                return;
            }
            
            for (let i = 0; i < Math.min(3, nodeIds.length); i++) {
                const randomIndex = Math.floor(Math.random() * nodeIds.length);
                const nodeId = nodeIds[randomIndex];
                nodeIds.splice(randomIndex, 1);
                
                this.engine.entangleNode(nodeId);
            }
        });
        
        this.explosionBtn.addEventListener('click', () => {
            // Create explosion at center
            this.engine._createQuantumBurst(new THREE.Vector3(0, 0, 0), 20);
            this.engine._showToast('Quantum burst created');
        });
        
        this.wireframeBtn.addEventListener('click', () => {
            this.wireframeBtn.classList.toggle('active');
            this.engine.config.wireframe = this.wireframeBtn.classList.contains('active');
            
            // Update all node materials
            for (const [nodeId, node] of this.engine.nodes.entries()) {
                node.mesh.material.wireframe = this.engine.config.wireframe;
            }
        });
        
        this.glowBtn.addEventListener('click', () => {
            this.glowBtn.classList.toggle('active');
            this.engine.config.glow = this.glowBtn.classList.contains('active');
        });
        
        this.saveBtn.addEventListener('click', () => {
            // Take screenshot
            this.engine.renderer.preserveDrawingBuffer = true;
            this.engine.renderer.render(this.engine.scene, this.engine.camera);
            
            const imgData = this.engine.renderer.domElement.toDataURL('image/png');
            
            // Create download link
            const link = document.createElement('a');
            link.href = imgData;
            link.download = `quantum-kaleidoscope-${Date.now()}.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.engine.renderer.preserveDrawingBuffer = false;
            this.engine._showToast('Screenshot saved');
        });
        
        this.fullscreenBtn.addEventListener('click', () => {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        });
        
        // Text input controls
        this.textClearBtn.addEventListener('click', () => {
            this.textInput.value = '';
        });
        
        this.textProcessBtn.addEventListener('click', () => {
            const text = this.textInput.value.trim();
            if (text) {
                this.engine.processText(text);
                this.textInput.value = '';
            }
        });
        
        // Toolbar buttons
        this.quantumBtn.addEventListener('click', () => {
            const text = this.textInput.value.trim();
            if (!text) {
                this.engine._showToast('Please enter text to analyze');
                return;
            }
            
            const analysis = this.engine.simulateQuantumAnalysis(text);
            
            this.engine._showModal({
                title: 'Quantum Analysis',
                content: `
                    <div>
                        <p><strong>Sentiment:</strong> ${analysis.sentiment.toFixed(2)}</p>
                        <p><strong>Complexity:</strong> ${analysis.complexity.toFixed(2)}</p>
                        <p><strong>Coherence:</strong> ${analysis.coherence.toFixed(2)}</p>
                        <p><strong>Quantum Entropy:</strong> ${analysis.quantumEntropy.toFixed(2)}</p>
                        <p><strong>Dominant Frequency:</strong> ${analysis.dominantFrequency.toFixed(2)} Hz</p>
                        <p><strong>Phase Coupling:</strong> ${analysis.phaseCoupling.toFixed(2)}</p>
                    </div>
                `
            });
        });
        
        this.insightsBtn.addEventListener('click', () => {
            const text = this.textInput.value.trim();
            if (!text) {
                this.engine._showToast('Please enter text to analyze');
                return;
            }
            
            const insights = this.engine.generateQuantumInsights(text);
            
            this.engine._showModal({
                title: 'Quantum Insights',
                content: `
                    <div>
                        <ul>
                            ${insights.map(insight => `<li>${insight}</li>`).join('')}
                        </ul>
                    </div>
                `
            });
        });
        
        this.simulateBtn.addEventListener('click', () => {
            this.engine._showToast('Running quantum simulation...');
            
            // Run multiple simulation steps
            for (let i = 0; i < 10; i++) {
                setTimeout(() => {
                    this.engine._updateSimulation(0.1);
                }, i * 100);
            }
        });
        
        // Context menu items
        this.ctxAddNode.addEventListener('click', () => {
            if (this.engine.contextMenuPosition) {
                this.engine.addNodeAt(this.engine.contextMenuPosition);
            }
        });
        
        this.ctxClearArea.addEventListener('click', () => {
            if (this.engine.contextMenuPosition) {
                const position = this.engine.contextMenuPosition;
                const radius = 5;
                
                // Find nodes in radius
                const nodesToRemove = [];
                for (const [nodeId, node] of this.engine.nodes.entries()) {
                    const distance = position.distanceTo(node.mesh.position);
                    if (distance < radius) {
                        nodesToRemove.push(nodeId);
                    }
                }
                
                // Remove nodes
                for (const nodeId of nodesToRemove) {
                    this.engine.removeNode(nodeId);
                }
                
                this.engine._showToast(`Cleared ${nodesToRemove.length} nodes`);
            }
        });
        
        this.ctxExplodeFromHere.addEventListener('click', () => {
            if (this.engine.contextMenuPosition) {
                this.engine._createQuantumBurst(this.engine.contextMenuPosition, 15);
            }
        });
        
        this.ctxCreateCluster.addEventListener('click', () => {
            if (this.engine.contextMenuPosition) {
                const center = this.engine.contextMenuPosition;
                const numNodes = 5;
                const radius = 3;
                
                // Create central node
                const centerId = this.engine.addNodeAt(center, {
                    energy: 0.9,
                    stability: 0.9
                });
                
                // Create surrounding nodes
                for (let i = 0; i < numNodes; i++) {
                    const angle = (i / numNodes) * Math.PI * 2;
                    const x = center.x + Math.cos(angle) * radius;
                    const y = center.y + Math.sin(angle) * radius;
                    const z = center.z + (Math.random() - 0.5) * 2;
                    
                    const position = new THREE.Vector3(x, y, z);
                    const nodeId = this.engine.addNodeAt(position, {
                        energy: 0.7,
                        stability: 0.8
                    });
                    
                    // Connect to center
                    this.engine.createConnection(centerId, nodeId, 0.8);
                    
                    // Connect to neighbors
                    if (i > 0) {
                        const prevId = this.engine.nodes.keys()[i-1];
                        this.engine.createConnection(nodeId, prevId, 0.6);
                    }
                }
                
                this.engine._showToast('Cluster created');
            }
        });
    }
    
    // Initialize loading animation
    initializeLoading() {
        // Show loading screen
        this.loading.style.display = 'flex';
        
        // Simulate loading progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 5;
            if (progress >= 100) {
                progress = 100;
                clearInterval(progressInterval);
                
                // Hide loading screen
                setTimeout(() => {
                    this.loading.style.opacity = 0;
                    setTimeout(() => {
                        this.loading.style.display = 'none';
                    }, 500);
                }, 500);
            }
            
            // Update progress bar
            this.progressFill.style.width = `${progress}%`;
            
            // Update loading text
            if (progress < 30) {
                this.loadingText.textContent = 'Initializing quantum field...';
            } else if (progress < 60) {
                this.loadingText.textContent = 'Calculating phase coherence...';
            } else if (progress < 90) {
                this.loadingText.textContent = 'Stabilizing entangled states...';
            } else {
                this.loadingText.textContent = 'Launching visualization...';
            }
        }, 100);
    }
}

// Initialize the visualization when document is ready
document.addEventListener('DOMContentLoaded', () => {
    // Create the engine
    const engine = new QuantumVisualizationEngine('visualization-container', {
        cubeSize: 30,
        nodeDensity: 8,
        energyLevel: 0.6,
        connectionThreshold: 5,
        nodeColor: 0xf72585
    });
    
    // Create UI controller
    const ui = new UIController(engine);
    
    // Initialize loading animation
    ui.initializeLoading();
});
