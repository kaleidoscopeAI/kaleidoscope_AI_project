# Kaleidoscope AI - Quantum Visualization Bridge

An advanced 3D quantum state visualization system with WebSocket communication, AI integration, and real-time collaboration capabilities.

## 🚀 Features

- **Interactive 3D Quantum Cube**: Visualize quantum states, nodes, and connections in 3D space
- **Real-time WebSocket Communication**: Secure bidirectional communication with token validation
- **LLM Integration**: Connect to local LLM models for quantum analysis via Ollama
- **Multi-user Collaboration**: Multiple clients can view and interact with the same visualization
- **Resource Monitoring**: Real-time CPU, memory, and connection statistics
- **Secure API**: Token-based authentication for all WebSocket connections
- **Error Handling**: Comprehensive error management and recovery
- **Cross-Platform**: Works on any device with a modern web browser

## 📋 Requirements

- Python 3.8+ (virtual environment creation is handled automatically)
- Modern web browser with WebGL support
- [Ollama](https://ollama.ai/) (optional, for LLM integration)

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quantum-bridge.git
   cd quantum-bridge
   ```

2. Make the start script executable:
   ```bash
   chmod +x start_quantum_bridge.sh
   ```

3. Configure security token:
   Edit `start_quantum_bridge.sh` and change the `KAI_SECRET` value to a secure token.

## 🚀 Usage

1. Start the server:
   ```bash
   ./start_quantum_bridge.sh
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

3. The visualization will initialize automatically, and you'll see the 3D quantum cube.

## 🎮 Controls

### Interface Controls
- **Cog Icon (Bottom-Right)**: Toggle control panel
- **Sliders**: Adjust quantum cube parameters (size, density, rotation, energy)
- **System Panel (Top-Left)**: Monitor and control system state
- **Right-Click on Node**: Open context menu for node-specific actions

### Control Panel Actions
- **Entangle Nodes**: Create quantum entanglement between eligible nodes
- **Analyze Pattern**: Request LLM analysis of current quantum state
- **Reset**: Reset the quantum cube to default state

### System Controls
- **Pause/Resume**: Pause or resume the visualization
- **Reset System**: Completely reset the system state
- **Switch Model**: Change the LLM model used for analysis

## 🔒 Security

The WebSocket connection is secured using token-based authentication. All clients must provide the token to connect to the backend. The token is configured in the `start_quantum_bridge.sh` script.

Be sure to change the default token (`change-me`) to a secure value in production environments!

## 🧩 Integration with Ollama

For AI-powered quantum analysis, the system can integrate with Ollama:

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Start Ollama with your preferred model:
   ```bash
   ollama run llama3:8b
   ```
3. The quantum bridge will automatically detect and connect to Ollama

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🧪 Advanced Configuration

### Environment Variables

The following environment variables can be configured in `start_quantum_bridge.sh`:

- `HOST`: The host address to bind the server to (default: `0.0.0.0`)
- `PORT`: The port to bind the server to (default: `8000`)
- `OLLAMA_HOST`: The host address of the Ollama server (default: `127.0.0.1`)
- `OLLAMA_PORT`: The port of the Ollama server (default: `11434`)
- `OLLAMA_MODEL`: The default Ollama model to use (default: `llama3:8b`)
- `KAI_SECRET`: The security token for WebSocket connections

### Custom Models

You can add additional LLM models to the dropdown menu by modifying the `quantum_bridge.py` file. Look for the `showModal('Switch LLM Model', ...)` function call and add your model to the options list.

## 🔍 Troubleshooting

### Connection Issues
- Ensure the server is running and accessible
- Check that the KAI_SECRET token matches between client and server
- Verify network connectivity and firewall settings

### Performance Issues
- Reduce the density parameter for better performance on lower-end devices
- Decrease the number of particles if animation is choppy
- Close other resource-intensive applications
