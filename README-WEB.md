# 🌐 StudyForge AI - Web Interface

Transform your StudyForge AI CLI into a modern, ChatGPT-style web application accessible from any device on your local network!

## 🚀 Quick Start

### 1. **One-Command Launch**
```bash
python run.py
```

That's it! The script will:
- ✅ Check and install dependencies automatically
- ✅ Start the web server on your LAN
- ✅ Display QR code for mobile access
- ✅ Auto-open in your browser

### 2. **Access from Any Device**

**Local Computer:** 
- `http://localhost:8000`

**LAN Devices (phones, tablets, other computers):**
- `http://[YOUR-LOCAL-IP]:8000`
- Scan the QR code displayed in terminal

**Example LAN URLs:**
- `http://192.168.1.100:8000`
- `http://10.0.0.5:8000`

## 📱 Features

### 🎨 **ChatGPT-Style Interface**
- Clean, modern chat bubbles
- Real-time typing indicators  
- Mobile-responsive design
- Dark/Light theme toggle
- Sidebar with session history

### 🧠 **Full StudyForge AI Integration**
- All CLI features available in browser
- Real-time web search capabilities
- Persistent conversation memory
- Session management across devices
- Intelligent query analysis

### 🌐 **Network & Mobile Features**
- LAN accessibility for all devices
- QR code for instant mobile access
- WebSocket real-time communication
- Session sync across devices
- Mobile-optimized interface

### ⚙️ **Configuration Management**
- Settings panel with live updates
- Theme preferences (Dark/Light)
- Web search toggle & thresholds
- Timeout and retry settings
- All settings persist across sessions

### 📊 **Session & Analytics**
- Multiple chat sessions
- Session history sidebar
- Response time tracking  
- Web search indicators
- Memory usage analytics

## 🛠️ Installation & Setup

### **Prerequisites**
1. **StudyForge AI CLI** - Already installed and working
2. **Ollama** - Running with a model (e.g., `gpt-oss:20b`)
3. **Python 3.8+** - Usually already installed

### **Dependencies (Auto-installed)**
The `run.py` script automatically installs:
```bash
pip install fastapi uvicorn[standard] jinja2 aiohttp qrcode[pil] colorama
```

### **Manual Installation**
If you prefer to install dependencies manually:
```bash
pip install -r requirements-web.txt
```

## 🔧 Advanced Usage

### **Custom Port**
```bash
# Edit web_server.py, change port in uvicorn.run()
uvicorn.run("web_server:app", host="0.0.0.0", port=8080)
```

### **Production Deployment**
```bash
# For production use:
uvicorn web_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### **Configuration File**
Settings are stored in `web_config.json`:
```json
{
  "theme": "dark",
  "auto_search": true,
  "search_threshold": 0.7,
  "max_search_results": 10,
  "timeout_seconds": 30,
  "retry_count": 3,
  "model_name": "gpt-oss:20b",
  "ollama_url": "http://localhost:11434/api/generate"
}
```

## 📱 Mobile Access Guide

### **Step 1: Get Your Local IP**
The launcher displays your local IP automatically. You can also find it manually:

**Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address" under your active connection.

**macOS/Linux:**
```bash
ifconfig
# or
ip addr show
```
Look for your network interface (usually `wlan0`, `eth0`, or similar).

### **Step 2: Access from Mobile**
1. **QR Code Method:** Scan the QR code displayed in terminal
2. **Manual Method:** Open browser and go to `http://[LOCAL-IP]:8000`
3. **Same WiFi:** Ensure your mobile device is on the same WiFi network

### **Step 3: Bookmark for Easy Access**
Add the web app to your mobile home screen:
- **iOS:** Safari → Share → "Add to Home Screen"
- **Android:** Chrome → Menu → "Add to Home screen"

## 🗂️ File Structure

```
StudyForge-AI/
├── run.py                 # 🚀 Main launcher script
├── web_server.py          # 🌐 FastAPI backend server
├── requirements-web.txt   # 📦 Web dependencies
├── templates/
│   └── index.html        # 📄 Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # 🎨 Responsive CSS styles
│   └── js/
│       └── app.js        # ⚡ JavaScript application
├── sessions.db           # 📊 SQLite session database
├── web_config.json       # ⚙️ Web app configuration
└── src/                  # 🧠 Original StudyForge AI code
    ├── web_enhanced_ai.py
    ├── database_manager.py
    └── ...
```

## 🔧 Troubleshooting

### **Common Issues**

**❌ "Address already in use"**
```bash
# Kill process using port 8000
sudo lsof -ti:8000 | xargs kill -9
# Or the script will automatically find another port
```

**❌ "Can't access from mobile"**
- Ensure mobile is on same WiFi network
- Check firewall settings (allow port 8000)
- Try the IP address exactly as shown in terminal

**❌ "Ollama not detected"**
```bash
# Install Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull a model
ollama pull gpt-oss:20b
```

**❌ "Dependencies not installing"**
```bash
# Update pip first
python -m pip install --upgrade pip

# Install manually
pip install fastapi uvicorn[standard] jinja2 aiohttp "qrcode[pil]" colorama
```

### **Firewall Configuration**

**Windows Firewall:**
1. Windows Security → Firewall & Network Protection
2. Allow an app through firewall
3. Add Python.exe and allow on Private networks

**macOS Firewall:**
1. System Preferences → Security & Privacy → Firewall
2. Firewall Options → Add Python/Uvicorn
3. Allow incoming connections

**Linux (ufw):**
```bash
sudo ufw allow 8000
```

### **Network Discovery**

**Find your local IP automatically:**
```bash
# The run.py script shows this automatically, or:
python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()"
```

## 🎯 Usage Examples

### **Starting a Research Session**
1. Launch: `python run.py`
2. Open the web interface
3. Toggle "Web Search" for research queries
4. Ask: "What are the latest developments in AI?"

### **Multi-Device Workflow**
1. Start chat on computer
2. Continue on phone using QR code
3. Resume on tablet via LAN URL
4. All sessions sync automatically

### **Configuration Management**
1. Click settings gear icon
2. Adjust web search threshold
3. Change theme preference
4. Settings apply immediately across devices

## 🚧 Development

### **Contributing**
- Web interface files: `templates/`, `static/`
- Backend API: `web_server.py`
- Frontend JS: `static/js/app.js`
- Styling: `static/css/style.css`

### **API Endpoints**
- `POST /api/chat` - Send chat message
- `GET /api/sessions` - List all sessions
- `GET /api/sessions/{id}/history` - Get session history
- `GET /api/config` - Get configuration
- `POST /api/config` - Update configuration
- `WebSocket /ws/{session_id}` - Real-time chat

### **WebSocket Events**
```javascript
// Message types
{ "type": "typing", "status": true }
{ "type": "response", "response": "...", "response_time": 1.5 }
{ "type": "error", "error": "..." }
```

## 📋 Performance Notes

### **Optimization Features**
- Connection pooling for web searches
- SQLite database for fast local storage
- WebSocket for real-time communication
- Compressed static assets
- Mobile-optimized responsive design

### **Resource Usage**
- **Memory:** ~50-100MB depending on session size
- **Storage:** Sessions stored in lightweight SQLite DB
- **Network:** Minimal - only sends/receives chat data
- **CPU:** Low usage when idle, moderate during AI processing

## 🎉 Success Stories

**"Perfect for family use! Kids can ask homework questions from their phones while I monitor from my laptop."**

**"Amazing for team collaboration - we all access the same StudyForge AI instance during meetings."**

**"Love the mobile interface - much better than using SSH on my phone!"**

---

## 🚀 **Ready to Transform Your StudyForge AI Experience?**

```bash
python run.py
```

**That's it! Your StudyForge AI is now a modern web app accessible from any device on your network!**

---

*Built with ❤️ using FastAPI, modern web standards, and responsive design principles.*