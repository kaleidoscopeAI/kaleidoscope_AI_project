<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Kaleidoscope</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Source+Code+Pro&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #3a0ca3;
            --secondary: #4cc9f0;
            --accent: #f72585;
            --success: #06d6a0;
            --warning: #ffd166;
            --danger: #ef476f;
            --dark: #101020;
            --light: #ffffff;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Source Code Pro', monospace;
        }
        
        body {
            overflow: hidden;
            background: radial-gradient(circle at center, #1a1a3a 0%, #000020 100%);
            color: var(--light);
        }
        
        #visualization-container {
            position: fixed;
            width: 100%;
            height: 100%;
            z-index: 1;
        }
        
        #particle-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 2;
            pointer-events: none;
        }
        
        header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background: rgba(16, 16, 32, 0.7);
            backdrop-filter: blur(5px);
            padding: 10px 20px;
            text-align: center;
            z-index: 10;
            border-bottom: 1px solid var(--secondary);
        }
        
        header h1 {
            color: var(--secondary);
            font-size: 1.8rem;
            text-shadow: 0 0 10px rgba(76, 201, 240, 0.5);
        }
        
        /* Control Panel */
        #control-panel {
            position: fixed;
            top: 70px;
            right: 20px;
            width: 330px;
            background: rgba(16, 16, 32, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid var(--secondary);
            padding: 15px;
            color: var(--light);
            z-index: 10;
            box-shadow: 0 0 20px rgba(76, 201, 240, 0.3);
            transform: translateX(350px);
            transition: transform 0.3s ease;
            max-height: calc(100vh - 100px);
            overflow-y: auto;
        }
        
        #control-panel.visible {
            transform: translateX(0);
        }
        
        .panel-section {
            margin-bottom: 20px;
        }
        
        .panel-section h3 {
            margin-bottom: 15px;
            color: var(--secondary);
            font-size: 16px;
            border-bottom: 1px solid rgba(76, 201, 240, 0.3);
            padding-bottom: 8px;
        }
        
        .control-row {
            margin-bottom: 12px;
        }
        
        .control-row label {
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.8);
        }
        
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .slider-container input {
            flex: 1;
        }
        
        .slider-container .value {
            width: 40px;
            text-align: center;
            font-size: 12px;
            color: var(--secondary);
        }
        
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 5px;
            background: rgba(76, 201, 240, 0.2);
            border-radius: 3px;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            height: 15px;
            width: 15px;
            border-radius: 50%;
            background: var(--secondary);
            cursor: pointer;
        }
        
        /* Button Grid */
        .button-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        
        .ctrl-btn {
            background: rgba(58, 12, 163, 0.4);
            border: 1px solid var(--primary);
            color: var(--light);
            padding: 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        
        .ctrl-btn:hover {
            background: var(--primary);
        }
        
        .ctrl-btn.active {
            background: var(--primary);
            border-color: var(--secondary);
        }
        
        /* Color Selectors */
        .colors-row {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        
        .colorbox {
            width: 20px;
            height: 20px;
            display: inline-block;
            border-radius: 3px;
            cursor: pointer;
            border: 2px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.2s;
        }
        
        .colorbox:hover {
            transform: scale(1.1);
            border-color: white;
        }
        
        .colorbox.active {
            border-color: white;
        }
        
        /* Toggle Button */
        #panel-toggle {
            position: fixed;
            top: 70px;
            right: 20px;
            background: rgba(16, 16, 32, 0.8);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 11;
            border: 1px solid var(--secondary);
            color: var(--secondary);
            box-shadow: 0 0 10px rgba(76, 201, 240, 0.3);
        }
        
        /* Stats Panel */
        #stats-panel {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(16, 16, 32, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid var(--secondary);
            padding: 10px 15px;
            color: var(--light);
            font-size: 12px;
            z-index: 10;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            gap: 15px;
            margin-bottom: 4px;
        }
        
        .stat-value {
            color: var(--secondary);
        }
        
        /* Toast Notification */
        #toast {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%) translateY(100px);
            background: rgba(16, 16, 32, 0.9);
            border-left: 3px solid var(--secondary);
            color: var(--light);
            padding: 12px 20px;
            border-radius: 5px;
            font-size: 14px;
            transition: transform 0.3s ease;
            z-index: 100;
        }
        
        #toast.visible {
            transform: translateX(-50%) translateY(0);
        }
        
        /* Context Menu */
        #context-menu {
            position: absolute;
            background: rgba(16, 16, 32, 0.9);
            border: 1px solid var(--secondary);
            border-radius: 5px;
            padding: 5px 0;
            min-width: 150px;
            z-index: 100;
            display: none;
        }
        
        .context-item {
            padding: 8px 12px;
            cursor: pointer;
            font-size: 14px;
            color: var(--light);
        }
        
        .context-item:hover {
            background: rgba(76, 201, 240, 0.2);
        }
        
        .context-divider {
            height: 1px;
            background: rgba(255, 255, 255, 0.1);
            margin: 5px 0;
        }
        
        /* Charts Panel */
        #charts-panel {
            position: fixed;
            top: 70px;
            left: 20px;
            width: 330px;
            background: rgba(16, 16, 32, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid var(--secondary);
            padding: 15px;
            color: var(--light);
            z-index: 10;
            box-shadow: 0 0 20px rgba(76, 201, 240, 0.3);
            transform: translateX(-350px);
            transition: transform 0.3s ease;
            max-height: calc(100vh - 100px);
            overflow-y: auto;
        }
        
        #charts-panel.visible {
            transform: translateX(0);
        }
        
        .chart-container {
            margin-bottom: 20px;
            position: relative;
            height: 200px;
        }
        
        #charts-toggle {
            position: fixed;
            top: 70px;
            left: 20px;
            background: rgba(16, 16, 32, 0.8);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 11;
            border: 1px solid var(--secondary);
            color: var(--secondary);
            box-shadow: 0 0 10px rgba(76, 201, 240, 0.3);
        }
        
        /* Loading Screen */
        #loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, #1a1a3a 0%, #000020 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            transition: opacity 0.5s ease;
        }
        
        #loading h2 {
            color: var(--light);
            margin-bottom: 20px;
            font-size: 24px;
        }
        
        .loading-cube-container {
            width: 100px;
            height: 100px;
            perspective: 800px;
            margin-bottom: 30px;
        }
        
        .loading-cube {
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            transform: translateZ(-50px);
            animation: loading-rotate 3s infinite linear;
        }
        
        .loading-face {
            position: absolute;
            width: 100px;
            height: 100px;
            border: 2px solid var(--secondary);
            background: rgba(76, 201, 240, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--secondary);
            font-size: 24px;
        }
        
        .loading-face:nth-child(1) { transform: rotateY(0deg) translateZ(50px); }
        .loading-face:nth-child(2) { transform: rotateY(90deg) translateZ(50px); }
        .loading-face:nth-child(3) { transform: rotateY(180deg) translateZ(50px); }
        .loading-face:nth-child(4) { transform: rotateY(-90deg) translateZ(50px); }
        .loading-face:nth-child(5) { transform: rotateX(90deg) translateZ(50px); }
        .loading-face:nth-child(6) { transform: rotateX(-90deg) translateZ(50px); }
        
        @keyframes loading-rotate {
            0% { transform: translateZ(-50px) rotateX(0deg) rotateY(0deg); }
            100% { transform: translateZ(-50px) rotateX(360deg) rotateY(360deg); }
        }
        
        #progress-bar {
            width: 300px;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 20px;
        }
        
        #progress-fill {
            height: 100%;
            width: 0;
            background: linear-gradient(90deg, var(--accent), var(--secondary));
            transition: width 0.5s ease;
        }
        
        #loading-text {
            color: rgba(255, 255, 255, 0.7);
            font-size: 14px;
            margin-top: 10px;
            font-family: monospace;
        }
        
        /* Text Input Panel */
        #text-input-panel {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(16, 16, 32, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid var(--secondary);
            padding: 15px;
            width: 600px;
            max-width: calc(100vw - 40px);
            z-index: 10;
        }
        
        #text-input {
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(76, 201, 240, 0.3);
            border-radius: 4px;
            color: var(--light);
            padding: 10px;
            margin-bottom: 10px;
            resize: vertical;
            min-height: 80px;
        }
        
        .input-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        
        .floating-toolbar {
            position: fixed;
            bottom: 130px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            z-index: 5;
        }
        
        .toolbar-btn {
            background: rgba(16, 16, 32, 0.8);
            border: 1px solid var(--primary);
            color: var(--light);
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
            backdrop-filter: blur(10px);
        }
        
        .toolbar-btn:hover {
            background: var(--primary);
            box-shadow: 0 0 15px rgba(58, 12, 163, 0.5);
            transform: translateY(-2px);
        }
        
        /* Wave Function Visualization */
        #wave-function {
            position: fixed;
            bottom: 200px;
            right: 20px;
            width: 300px;
            height: 100px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 5px;
            border: 1px solid var(--secondary);
            z-index: 4;
            opacity: 0.7;
        }

        /* Modal Dialog */
        #modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }
        
        #modal.visible {
            opacity: 1;
            pointer-events: auto;
        }
        
        .modal-content {
            background: rgba(26, 26, 46, 0.95);
            border: 1px solid var(--secondary);
            border-radius: 10px;
            padding: 20px;
            width: 90%;
            max-width: 500px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(76, 201, 240, 0.3);
            padding-bottom: 10px;
        }
        
        .modal-title {
            color: var(--secondary);
            font-size: 18px;
        }
        
        .modal-close {
            color: var(--light);
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
        }
        
        .modal-body {
            color: var(--light);
            margin-bottom: 20px;
            max-height: 60vh;
            overflow-y: auto;
        }
        
        .modal-footer {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }
        
        .modal-btn {
            background: rgba(58, 12, 163, 0.4);
            border: 1px solid var(--primary);
            color: var(--light);
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        
        .modal-btn:hover {
            background: var(--primary);
        }
        
        .modal-btn.primary {
            background: var(--primary);
            border-color: var(--secondary);
        }
        
        .modal-btn.primary:hover {
            background: var(--secondary);
            color: var(--dark);
        }
    </style>
</head>
<body>
    <!-- Loading Screen -->
    <div id="loading">
        <div class="loading-cube-container">
            <div class="loading-cube">
                <div class="loading-face"><i class="fas fa-atom"></i></div>
                <div class="loading-face"><i class="fas fa-brain"></i></div>
                <div class="loading-face"><i class="fas fa-project-diagram"></i></div>
                <div class="loading-face"><i class="fas fa-cube"></i></div>
                <div class="loading-face"><i class="fas fa-network-wired"></i></div>
                <div class="loading-face"><i class="fas fa-microchip"></i></div>
            </div>
        </div>
        <h2>Initializing Quantum Kaleidoscope</h2>
        <div id="progress-bar">
            <div id="progress-fill"></div>
        </div>
        <div id="loading-text">Loading core components...</div>
    </div>

    <!-- Main visualization canvas -->
    <div id="visualization-container"></div>
    <canvas id="particle-canvas"></canvas>
    
    <!-- Header -->
    <header>
        <h1>Quantum Kaleidoscope Visualization</h1>
    </header>
    
    <!-- Control Panel Toggle -->
    <div id="panel-toggle">
        <i class="fas fa-cog"></i>
    </div>
    
    <!-- Charts Toggle -->
    <div id="charts-toggle">
        <i class="fas fa-chart-line"></i>
    </div>
    
    <!-- Control Panel -->
    <div id="control-panel" class="visible">
        <div class="panel-section">
            <h3>Simulation Controls</h3>
            
            <div class="control-row">
                <label>Cube Size</label>
                <div class="slider-container">
                    <input type="range" id="size-slider" min="5" max="50" value="30">
                    <div class="value" id="size-value">30</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Node Density</label>
                <div class="slider-container">
                    <input type="range" id="density-slider" min="1" max="20" value="8">
                    <div class="value" id="density-value">8</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Energy Level</label>
                <div class="slider-container">
                    <input type="range" id="energy-slider" min="0" max="100" value="60">
                    <div class="value" id="energy-value">60</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Connection Threshold</label>
                <div class="slider-container">
                    <input type="range" id="connection-slider" min="1" max="15" value="5">
                    <div class="value" id="connection-value">5</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Rotation Speed</label>
                <div class="slider-container">
                    <input type="range" id="rotation-slider" min="0" max="10" value="2">
                    <div class="value" id="rotation-value">2</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Node Color</label>
                <div class="colors-row">
                    <div class="colorbox active" data-color="#f72585" style="background-color: #f72585;"></div>
                    <div class="colorbox" data-color="#4cc9f0" style="background-color: #4cc9f0;"></div>
                    <div class="colorbox" data-color="#7209b7" style="background-color: #7209b7;"></div>
                    <div class="colorbox" data-color="#06d6a0" style="background-color: #06d6a0;"></div>
                    <div class="colorbox" data-color="#ffd166" style="background-color: #ffd166;"></div>
                </div>
            </div>
        </div>
        
        <div class="panel-section">
            <h3>Actions</h3>
            <div class="button-grid">
                <button class="ctrl-btn" id="reset-btn">Reset</button>
                <button class="ctrl-btn" id="add-nodes-btn">Add Nodes</button>
                <button class="ctrl-btn" id="entangle-btn">Entangle</button>
                <button class="ctrl-btn" id="explosion-btn">Quantum Burst</button>
                <button class="ctrl-btn active" id="wireframe-btn">Wireframe</button>
                <button class="ctrl-btn active" id="glow-btn">Glow</button>
                <button class="ctrl-btn" id="save-btn">Save Image</button>
                <button class="ctrl-btn" id="fullscreen-btn">Fullscreen</button>
            </div>
        </div>
        
        <div class="panel-section">
            <h3>Advanced Physics</h3>
            <div class="control-row">
                <label>Quantum Jitter</label>
                <div class="slider-container">
                    <input type="range" id="jitter-slider" min="0" max="100" value="20">
                    <div class="value" id="jitter-value">20</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Entanglement Strength</label>
                <div class="slider-container">
                    <input type="range" id="entanglement-slider" min="0" max="100" value="70">
                    <div class="value" id="entanglement-value">70</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Phase Coherence</label>
                <div class="slider-container">
                    <input type="range" id="coherence-slider" min="0" max="100" value="80">
                    <div class="value" id="coherence-value">80</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Charts Panel -->
    <div id="charts-panel">
        <div class="panel-section">
            <h3>Quantum State Probability</h3>
            <div class="chart-container">
                <canvas id="probabilityChart"></canvas>
            </div>
        </div>
        
        <div class="panel-section">
            <h3>Node Evolution</h3>
            <div class="chart-container">
                <canvas id="evolutionChart"></canvas>
            </div>
        </div>
        
        <div class="panel-section">
            <h3>Energy Distribution</h3>
            <div class="chart-container">
                <canvas id="energyChart"></canvas>
            </div>
        </div>
        
        <div class="panel-section">
            <h3>System Metrics</h3>
            <div class="chart-container">
                <canvas id="metricsChart"></canvas>
            </div>
        </div>
    </div>
    
    <!-- Stats Panel -->
    <div id="stats-panel">
        <div class="stat-row">
            <div class="stat-label">FPS:</div>
            <div class="stat-value" id="fps-value">60</div>
        </div>
        <div class="stat-row">
            <div class="stat-label">Nodes:</div>
            <div class="stat-value" id="nodes-value">0</div>
        </div>
        <div class="stat-row">
            <div class="stat-label">Connections:</div>
            <div class="stat-value" id="connections-value">0</div>
        </div>
        <div class="stat-row">
            <div class="stat-label">Energy:</div>
            <div class="stat-value" id="energy-stat">60%</div>
        </div>
        <div class="stat-row">
            <div class="stat-label">Phase:</div>
            <div class="stat-value" id="phase-value">0.0</div>
        </div>
    </div>
    
    <!-- Text Input Panel -->
    <div id="text-input-panel">
        <textarea id="text-input" placeholder="Enter text to process through quantum analysis..."></textarea>
        <div class="input-buttons">
            <button class="ctrl-btn" id="text-clear-btn">Clear</button>
            <button class="ctrl-btn" id="text-process-btn">Process Text</button>
        </div>
    </div>
    
    <!-- Floating Toolbar -->
    <div class="floating-toolbar">
        <button class="toolbar-btn" id="quantum-btn">Quantum Analysis</button>
        <button class="toolbar-btn" id="insights-btn">Generate Insights</button>
        <button class="toolbar-btn" id="simulate-btn">Simulate Outcomes</button>
    </div>
    
    <!-- Wave Function Visualization -->
    <canvas id="wave-function"></canvas>
    
    <!-- Context Menu -->
    <div id="context-menu">
        <div class="context-item" id="ctx-add-node">Add Node Here</div>
        <div class="context-item" id="ctx-clear-area">Clear Nearby Nodes</div>
        <div class="context-divider"></div>
        <div class="context-item" id="ctx-explode-from-here">Burst From Here</div>
        <div class="context-item" id="ctx-create-cluster">Create Cluster</div>
    </div>
    
    <!-- Toast Notification -->
    <div id="toast"></div>
    
    <!-- Modal Dialog -->
    <div id="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">Quantum Visualization</div>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body" id="modal-body">
                Modal content here
            </div>
            <div class="modal-footer">
                <button class="modal-btn" id="modal-cancel">Cancel</button>
                <button class="modal-btn primary" id="modal-confirm">Confirm</button>
            </div>
        </div>
    </div>

    <script>
        // ========== Core Quantum Simulation ==========
        
        /**
         * Vector3D class for 3D vector operations.
         */
        class Vector3D {
            constructor(x = 0, y = 0, z = 0) {
                this.x = x;
                this.y = y;
                this.z = z;
            }
            
            add(v) {
                return new Vector3D(this.x + v.x, this.y + v.y, this.z + v.z);
            }
            
            subtract(v) {
                return new Vector3D(this.x - v.x, this.y - v.y,