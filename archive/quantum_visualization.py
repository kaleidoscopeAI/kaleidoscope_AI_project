#!/usr/bin/env python3
"""
Quantum Visualization System
============================

Provides a web-based 3D visualization of the Quantum Kaleidoscope data.
"""

import os
import sys
import socket
import json
import threading
import argparse
import logging
import urllib.request
import traceback
import time  # Added missing import

# Configure logging with detailed output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]  # Ensure logs go to stderr
)
logger = logging.getLogger("QuantumVisualization")

class VisualizationServer:
    """HTTP server for visualization."""
    def __init__(self, api_port: int, port: int, host="0.0.0.0"):
        self.api_port = api_port
        self.port = port
        self.host = host
        self.running = False
        self.static_content = self._load_static_content()
    
    def _load_static_content(self):
        """Generate static HTML with embedded Three.js visualization."""
        try:
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Kaleidoscope Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body {{ margin: 0; background: #0a0a1a; }}
        #canvas {{ width: 100%; height: 100vh; }}
    </style>
</head>
<body>
    <div id="canvas"></div>
    <script>
        try {{
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer();
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('canvas').appendChild(renderer.domElement);
            
            camera.position.z = 30;
            const nodesGroup = new THREE.Group();
            scene.add(nodesGroup);
            const connectionsGroup = new THREE.Group();
            scene.add(connectionsGroup);
            
            scene.add(new THREE.AmbientLight(0x404040));
            scene.add(new THREE.DirectionalLight(0xffffff, 0.8));
            
            function fetchData() {{
                fetch('http://localhost:{self.api_port}/api/visualization')
                    .then(response => response.json())
                    .then(data => {{
                        nodesGroup.clear();
                        connectionsGroup.clear();
                        
                        data.nodes.forEach(node => {{
                            const geometry = new THREE.SphereGeometry(0.5 + node.energy, 16, 16);
                            const material = new THREE.MeshPhongMaterial({{ color: 0x4488ff }});
                            const mesh = new THREE.Mesh(geometry, material);
                            mesh.position.set(node.position[0], node.position[1], node.position[2]);
                            nodesGroup.add(mesh);
                        }});
                        
                        data.connections.forEach(conn => {{
                            const source = data.nodes.find(n => n.id === conn.source);
                            const target = data.nodes.find(n => n.id === conn.target);
                            if (source && target) {{
                                const geometry = new THREE.BufferGeometry().setFromPoints([
                                    new THREE.Vector3(source.position[0], source.position[1], source.position[2]),
                                    new THREE.Vector3(target.position[0], target.position[1], target.position[2])
                                ]);
                                const material = new THREE.LineBasicMaterial({{ color: 0xaaaaff, opacity: conn.strength, transparent: true }});
                                connectionsGroup.add(new THREE.Line(geometry, material));
                            }}
                        }});
                    }})
                    .catch(error => console.error('Fetch error:', error));
            }}
            
            function animate() {{
                requestAnimationFrame(animate);
                nodesGroup.rotation.y += 0.001;
                connectionsGroup.rotation.y += 0.001;
                renderer.render(scene, camera);
            }}
            
            fetchData();
            setInterval(fetchData, 5000);
            animate();
            
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        }} catch (e) {{
            console.error('JavaScript error:', e);
        }}
    </script>
</body>
</html>
            """
            logger.debug("Static HTML content generated successfully")
            return {"index.html": (html.encode('utf-8'), "text/html")}
        except Exception as e:
            logger.error(f"Error generating static content: {e}\n{traceback.format_exc()}")
            raise
    
    def start(self):
        """Start the visualization server."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex((self.host, self.port)) == 0:
                    logger.error(f"Port {self.port} is already in use")
                    sys.exit(1)
            self.running = True
            threading.Thread(target=self._server_loop, daemon=True, name="ServerThread").start()
            logger.info(f"Visualization Server started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start server: {e}\n{traceback.format_exc()}")
            sys.exit(1)
    
    def stop(self):
        """Stop the visualization server."""
        self.running = False
        logger.info("Visualization Server stopped")
    
    def _server_loop(self):
        """Main server loop to handle incoming connections."""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            server_socket.settimeout(1.0)
            logger.debug(f"Server listening on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, addr = server_socket.accept()
                    logger.debug(f"Accepted connection from {addr}")
                    threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True, name=f"ClientThread-{addr}").start()
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}\n{traceback.format_exc()}")
            server_socket.close()
            logger.debug("Server socket closed")
        except Exception as e:
            logger.error(f"Server loop failed: {e}\n{traceback.format_exc()}")
            self.running = False
    
    def _handle_client(self, client_socket):
        """Handle individual client requests."""
        try:
            request_data = client_socket.recv(1024).decode('utf-8')
            logger.debug(f"Received request: {request_data[:100]}")
            if "GET /" in request_data or "GET /index.html" in request_data:
                content, content_type = self.static_content["index.html"]
                response = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: {content_type}\r\n"
                    f"Content-Length: {len(content)}\r\n"
                    f"Connection: close\r\n"
                    f"\r\n"
                ).encode('utf-8') + content
                client_socket.send(response)
                logger.debug("Response sent successfully")
            else:
                response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nNot Found".encode('utf-8')
                client_socket.send(response)
                logger.debug("404 response sent")
        except Exception as e:
            logger.error(f"Error handling client: {e}\n{traceback.format_exc()}")
        finally:
            try:
                client_socket.close()
            except Exception as e:
                logger.error(f"Error closing client socket: {e}")

def main():
    """Main entry point for the visualization server."""
    parser = argparse.ArgumentParser(description="Quantum Visualization System")
    parser.add_argument("--api-port", type=int, default=8000, help="Main system API port")
    parser.add_argument("--port", type=int, default=8080, help="Visualization server port")
    
    args = parser.parse_args()
    
    server = VisualizationServer(api_port=args.api_port, port=args.port)
    server.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Visualization Server...")
        server.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Main loop error: {e}\n{traceback.format_exc()}")
        server.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
