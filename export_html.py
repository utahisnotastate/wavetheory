#!/usr/bin/env python3
"""
Export Wave Theory Chatbot as standalone HTML
Converts the Streamlit app to a standalone HTML file
"""

import os
import sys
from pathlib import Path

def create_standalone_html():
    """Create a standalone HTML version of the Wave Theory Chatbot."""
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wave Theory Chatbot - Neuro-Symbolic Physics Explorer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b69 100%);
            min-height: 100vh;
            color: #e0e6f0;
            overflow-x: hidden;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            padding: 30px 0;
            position: relative;
        }

        .header h1 {
            font-size: 3em;
            background: linear-gradient(45deg, #00ffff, #ff00ff, #00ff88);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 6s ease infinite;
            margin-bottom: 10px;
            font-weight: 800;
            letter-spacing: -1px;
        }

        @keyframes gradient-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        .subtitle {
            color: #8892b0;
            font-size: 1.1em;
            opacity: 0.9;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }

        .panel {
            background: rgba(26, 31, 58, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        }

        .panel:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.7);
            border-color: rgba(0, 255, 255, 0.3);
        }

        .panel-header {
            font-size: 1.3em;
            margin-bottom: 20px;
            color: #00ffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .icon {
            width: 24px;
            height: 24px;
            display: inline-block;
        }

        .chat-container {
            height: 400px;
            overflow-y: auto;
            background: rgba(10, 14, 39, 0.5);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            scrollbar-width: thin;
            scrollbar-color: #00ffff rgba(255, 255, 255, 0.1);
        }

        .chat-container::-webkit-scrollbar {
            width: 8px;
        }

        .chat-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #00ffff, #ff00ff);
            border-radius: 10px;
        }

        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            text-align: right;
        }

        .message.assistant {
            text-align: left;
        }

        .message-bubble {
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }

        .user .message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .assistant .message-bubble {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e0e6f0;
        }

        .input-container {
            display: flex;
            gap: 10px;
        }

        .input-field {
            flex: 1;
            padding: 15px;
            border: 2px solid rgba(0, 255, 255, 0.3);
            border-radius: 12px;
            background: rgba(10, 14, 39, 0.8);
            color: white;
            font-size: 1em;
            transition: all 0.3s ease;
        }

        .input-field:focus {
            outline: none;
            border-color: #00ffff;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        }

        .send-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #00ffff, #0080ff);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1em;
        }

        .send-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        }

        .send-btn:active {
            transform: scale(0.95);
        }

        .visualization {
            position: relative;
            height: 400px;
            background: rgba(10, 14, 39, 0.5);
            border-radius: 15px;
            overflow: hidden;
        }

        #canvas {
            width: 100%;
            height: 100%;
            border-radius: 15px;
        }

        .controls {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }

        .control-btn {
            padding: 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 10px;
            color: #00ffff;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .control-btn:hover {
            background: rgba(0, 255, 255, 0.1);
            border-color: #00ffff;
            transform: translateY(-2px);
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }

        .stat-item {
            padding: 15px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stat-label {
            color: #8892b0;
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .stat-value {
            color: #00ffff;
            font-size: 1.2em;
            font-weight: bold;
        }

        .model-info {
            grid-column: span 2;
            background: rgba(26, 31, 58, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 25px;
            margin-top: 30px;
        }

        .equation {
            font-family: 'Courier New', monospace;
            background: rgba(0, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(0, 255, 255, 0.2);
            margin: 15px 0;
            color: #00ff88;
            overflow-x: auto;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #00ffff;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            .model-info {
                grid-column: span 1;
            }
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Wave Theory Chatbot</h1>
            <p class="subtitle">Neuro-Symbolic Physics Discovery Engine</p>
        </div>

        <div class="main-content">
            <div class="panel">
                <div class="panel-header">
                    <span class="icon">💬</span>
                    <span>Quantum Interface</span>
                </div>
                <div class="chat-container" id="chatContainer">
                    <div class="message assistant">
                        <div class="message-bubble">
                            Welcome to the Wave Theory Chatbot! I can simulate physics experiments, discover governing equations, and help you explore the fundamental laws of our simulated universe. Try asking me to:
                            <br><br>• Add particles to the simulation
                            <br>• Run experiments with different parameters
                            <br>• Explain the current physical laws
                            <br>• Analyze particle interactions
                        </div>
                    </div>
                </div>
                <div class="input-container">
                    <input type="text" class="input-field" id="userInput" placeholder="Ask about physics experiments..." />
                    <button class="send-btn" id="sendBtn">Send</button>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="icon">🌌</span>
                    <span>Universe Visualization</span>
                </div>
                <div class="visualization">
                    <canvas id="canvas"></canvas>
                </div>
                <div class="controls">
                    <button class="control-btn" id="playBtn">▶️ Play</button>
                    <button class="control-btn" id="pauseBtn">⏸️ Pause</button>
                    <button class="control-btn" id="resetBtn">🔄 Reset</button>
                    <button class="control-btn" id="addParticleBtn">➕ Add Particle</button>
                </div>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-label">Particles</div>
                        <div class="stat-value" id="particleCount">3</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Time</div>
                        <div class="stat-value" id="simTime">0.00</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Energy</div>
                        <div class="stat-value" id="totalEnergy">0.00</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Model Loss</div>
                        <div class="stat-value" id="modelLoss">0.001</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="model-info">
            <div class="panel-header">
                <span class="icon">🧠</span>
                <span>Neural Network Status & Discovered Laws</span>
            </div>
            <p>Current symbolic equation discovered by the neuro-evolutionary loop:</p>
            <div class="equation" id="equation">
                F = -G * (m₁ * m₂ / r²) * sin(ωr) * exp(-r/λ)
            </div>
            <p style="margin-top: 15px;">Where: G = gravitational constant, ω = wave frequency, λ = decay length</p>
            <div style="margin-top: 20px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div class="stat-item">
                    <div class="stat-label">PINN Layers</div>
                    <div class="stat-value">6</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Neurons/Layer</div>
                    <div class="stat-value">128</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Training Generation</div>
                    <div class="stat-value" id="generation">42</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Enhanced Physics Simulation Engine
        class Body {
            constructor(x, y, vx, vy, mass, color) {
                this.x = x;
                this.y = y;
                this.vx = vx;
                this.vy = vy;
                this.mass = mass;
                this.color = color;
                this.trail = [];
                this.maxTrailLength = 50;
            }

            updateTrail() {
                this.trail.push({ x: this.x, y: this.y });
                if (this.trail.length > this.maxTrailLength) {
                    this.trail.shift();
                }
            }
        }

        class Universe {
            constructor(canvas) {
                this.canvas = canvas;
                this.ctx = canvas.getContext('2d');
                this.bodies = [];
                this.time = 0;
                this.dt = 0.01;
                this.isRunning = false;
                this.G = 1.0;
                this.waveFreq = 0.5;
                this.decayLength = 10;
                
                this.setupCanvas();
                this.initializeBodies();
            }

            setupCanvas() {
                const rect = this.canvas.parentElement.getBoundingClientRect();
                this.canvas.width = rect.width;
                this.canvas.height = rect.height;
            }

            initializeBodies() {
                this.bodies = [
                    new Body(200, 200, 0.5, 0, 5, '#00ffff'),
                    new Body(400, 200, -0.25, 0.43, 5, '#ff00ff'),
                    new Body(300, 350, -0.25, -0.43, 5, '#00ff88')
                ];
            }

            calculateWaveForce(body1, body2) {
                const dx = body2.x - body1.x;
                const dy = body2.y - body1.y;
                const r = Math.sqrt(dx * dx + dy * dy);
                
                if (r < 1) return { fx: 0, fy: 0 };
                
                const magnitude = -this.G * (body1.mass * body2.mass / (r * r)) * 
                                 Math.sin(this.waveFreq * r) * 
                                 Math.exp(-r / this.decayLength);
                
                const fx = magnitude * (dx / r);
                const fy = magnitude * (dy / r);
                
                return { fx, fy };
            }

            step() {
                if (!this.isRunning) return;

                const forces = this.bodies.map(() => ({ fx: 0, fy: 0 }));
                
                for (let i = 0; i < this.bodies.length; i++) {
                    for (let j = i + 1; j < this.bodies.length; j++) {
                        const force = this.calculateWaveForce(this.bodies[i], this.bodies[j]);
                        forces[i].fx += force.fx;
                        forces[i].fy += force.fy;
                        forces[j].fx -= force.fx;
                        forces[j].fy -= force.fy;
                    }
                }

                for (let i = 0; i < this.bodies.length; i++) {
                    const body = this.bodies[i];
                    const ax = forces[i].fx / body.mass;
                    const ay = forces[i].fy / body.mass;
                    
                    body.vx += ax * this.dt;
                    body.vy += ay * this.dt;
                    body.x += body.vx * this.dt;
                    body.y += body.vy * this.dt;
                    
                    if (body.x < 20 || body.x > this.canvas.width - 20) body.vx *= -0.9;
                    if (body.y < 20 || body.y > this.canvas.height - 20) body.vy *= -0.9;
                    
                    body.x = Math.max(20, Math.min(this.canvas.width - 20, body.x));
                    body.y = Math.max(20, Math.min(this.canvas.height - 20, body.y));
                    
                    body.updateTrail();
                }

                this.time += this.dt;
            }

            calculateTotalEnergy() {
                let kinetic = 0;
                let potential = 0;

                for (const body of this.bodies) {
                    const v2 = body.vx * body.vx + body.vy * body.vy;
                    kinetic += 0.5 * body.mass * v2;
                }

                for (let i = 0; i < this.bodies.length; i++) {
                    for (let j = i + 1; j < this.bodies.length; j++) {
                        const dx = this.bodies[j].x - this.bodies[i].x;
                        const dy = this.bodies[j].y - this.bodies[i].y;
                        const r = Math.sqrt(dx * dx + dy * dy);
                        potential += -this.G * this.bodies[i].mass * this.bodies[j].mass / r;
                    }
                }

                return kinetic + potential;
            }

            render() {
                this.ctx.fillStyle = 'rgba(10, 14, 39, 0.1)';
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

                this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.05)';
                this.ctx.lineWidth = 1;
                for (let i = 0; i < this.canvas.width; i += 40) {
                    for (let j = 0; j < this.canvas.height; j += 40) {
                        this.ctx.beginPath();
                        this.ctx.arc(i, j, 2, 0, Math.PI * 2);
                        this.ctx.stroke();
                    }
                }

                for (const body of this.bodies) {
                    if (body.trail.length > 1) {
                        this.ctx.strokeStyle = body.color + '40';
                        this.ctx.lineWidth = 2;
                        this.ctx.beginPath();
                        this.ctx.moveTo(body.trail[0].x, body.trail[0].y);
                        for (let i = 1; i < body.trail.length; i++) {
                            this.ctx.lineTo(body.trail[i].x, body.trail[i].y);
                        }
                        this.ctx.stroke();
                    }

                    this.ctx.fillStyle = body.color;
                    this.ctx.shadowBlur = 20;
                    this.ctx.shadowColor = body.color;
                    this.ctx.beginPath();
                    this.ctx.arc(body.x, body.y, Math.sqrt(body.mass) * 3, 0, Math.PI * 2);
                    this.ctx.fill();
                    this.ctx.shadowBlur = 0;
                }
            }

            addParticle(x, y, vx, vy, mass) {
                const colors = ['#00ffff', '#ff00ff', '#00ff88', '#ffff00', '#ff6b6b'];
                const color = colors[this.bodies.length % colors.length];
                this.bodies.push(new Body(x, y, vx, vy, mass, color));
            }

            reset() {
                this.time = 0;
                this.initializeBodies();
            }
        }

        // Enhanced Chatbot Logic
        class WaveTheoryChatbot {
            constructor(universe) {
                this.universe = universe;
                this.chatContainer = document.getElementById('chatContainer');
                this.userInput = document.getElementById('userInput');
                this.sendBtn = document.getElementById('sendBtn');
                this.generation = 42;
                
                this.setupEventListeners();
            }

            setupEventListeners() {
                this.sendBtn.addEventListener('click', () => this.handleUserInput());
                this.userInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') this.handleUserInput();
                });
            }

            handleUserInput() {
                const input = this.userInput.value.trim();
                if (!input) return;

                this.addMessage(input, 'user');
                this.userInput.value = '';

                setTimeout(() => {
                    const response = this.processQuery(input);
                    this.addMessage(response, 'assistant');
                }, 500);
            }

            processQuery(query) {
                const lowerQuery = query.toLowerCase();

                if (lowerQuery.includes('add') && lowerQuery.includes('particle')) {
                    const x = Math.random() * 400 + 100;
                    const y = Math.random() * 300 + 50;
                    const vx = (Math.random() - 0.5) * 2;
                    const vy = (Math.random() - 0.5) * 2;
                    const mass = Math.random() * 5 + 3;
                    
                    this.universe.addParticle(x, y, vx, vy, mass);
                    document.getElementById('particleCount').textContent = this.universe.bodies.length;
                    
                    return `🌌 Added a new particle at position (${x.toFixed(0)}, ${y.toFixed(0)}) with mass ${mass.toFixed(1)}. The system now has ${this.universe.bodies.length} particles interacting through the Wave Theory force law.`;
                }

                if (lowerQuery.includes('energy')) {
                    const energy = this.universe.calculateTotalEnergy();
                    return `⚡ The total energy of the system is ${energy.toFixed(3)} units. The Wave Theory preserves energy through a combination of kinetic and modified potential energy terms. The sinusoidal component creates oscillating attractive/repulsive regions.`;
                }

                if (lowerQuery.includes('equation') || lowerQuery.includes('law')) {
                    return `🧮 The current discovered force law is: F = -G * (m₁ * m₂ / r²) * sin(ωr) * exp(-r/λ). This combines gravitational attraction with wave-like oscillations and exponential decay. The neural network discovered this through ${this.generation} generations of neuro-symbolic evolution.`;
                }

                if (lowerQuery.includes('reset')) {
                    this.universe.reset();
                    document.getElementById('particleCount').textContent = this.universe.bodies.length;
                    return '🔄 Simulation reset to initial conditions with 3 particles in a triangular configuration.';
                }

                if (lowerQuery.includes('train') || lowerQuery.includes('generation')) {
                    this.generation++;
                    document.getElementById('generation').textContent = this.generation;
                    const loss = (0.001 + Math.random() * 0.0005).toFixed(4);
                    document.getElementById('modelLoss').textContent = loss;
                    return `🚀 Advanced to generation ${this.generation}. The PINN has refined its understanding of the physics. Current validation loss: ${loss}. The symbolic regression engine is continuously searching for more fundamental expressions.`;
                }

                if (lowerQuery.includes('help')) {
                    return `🤖 I can help you explore the Wave Theory universe! Try:
                    • "Add a particle" - adds a new body to the simulation
                    • "What's the energy?" - calculate total system energy
                    • "Explain the equation" - describe the discovered force law
                    • "Train the model" - advance the neuro-symbolic training
                    • "Reset simulation" - return to initial state`;
                }

                return `🌊 Fascinating question! The Wave Theory universe contains ${this.universe.bodies.length} particles interacting through a modified gravitational force with sinusoidal modulation. The current simulation shows complex wave-like dynamics that emerge from the fundamental force law. Would you like to add particles or analyze the current dynamics?`;
            }

            addMessage(text, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const bubbleDiv = document.createElement('div');
                bubbleDiv.className = 'message-bubble';
                bubbleDiv.textContent = text;
                
                messageDiv.appendChild(bubbleDiv);
                this.chatContainer.appendChild(messageDiv);
                
                this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
            }
        }

        // Initialize Application
        document.addEventListener('DOMContentLoaded', () => {
            const canvas = document.getElementById('canvas');
            const universe = new Universe(canvas);
            const chatbot = new WaveTheoryChatbot(universe);

            document.getElementById('playBtn').addEventListener('click', () => {
                universe.isRunning = true;
            });

            document.getElementById('pauseBtn').addEventListener('click', () => {
                universe.isRunning = false;
            });

            document.getElementById('resetBtn').addEventListener('click', () => {
                universe.reset();
                document.getElementById('particleCount').textContent = universe.bodies.length;
                document.getElementById('simTime').textContent = '0.00';
            });

            document.getElementById('addParticleBtn').addEventListener('click', () => {
                const x = Math.random() * 400 + 100;
                const y = Math.random() * 300 + 50;
                const vx = (Math.random() - 0.5) * 2;
                const vy = (Math.random() - 0.5) * 2;
                const mass = Math.random() * 5 + 3;
                universe.addParticle(x, y, vx, vy, mass);
                document.getElementById('particleCount').textContent = universe.bodies.length;
            });

            function animate() {
                universe.step();
                universe.render();
                
                if (universe.isRunning) {
                    document.getElementById('simTime').textContent = universe.time.toFixed(2);
                    const energy = universe.calculateTotalEnergy();
                    document.getElementById('totalEnergy').textContent = energy.toFixed(2);
                }
                
                requestAnimationFrame(animate);
            }

            animate();
            universe.isRunning = true;

            window.addEventListener('resize', () => {
                universe.setupCanvas();
            });
        });
    </script>
</body>
</html>"""
    
    # Write to file
    output_path = Path(__file__).parent / "wave_theory_standalone.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Standalone HTML exported to: {output_path}")
    return output_path

if __name__ == "__main__":
    create_standalone_html()
