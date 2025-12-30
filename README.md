# AI Web Chat Interface

A minimalistic web UI for interacting with multiple AI model providers (Ollama, OpenAI, Google, OpenRouter, and Claude) with MCP (Model Context Protocol) HTTP SSE support.

## Features

- 🤖 **Multiple AI Providers**: Support for Ollama, OpenAI, Claude, Google, and OpenRouter
- 💬 **Real-time Streaming**: Server-Sent Events (SSE) for streaming responses
- 🔌 **MCP Integration**: HTTP SSE support for Model Context Protocol servers with JSON-RPC 2.0
- 🛠️ **Tool Calling**: Full support for MCP tool discovery and execution across all providers
- 🔍 **QA Agent**: Automatic response validation with configurable follow-up model for data verification
- 📊 **Live Statistics**: Real-time query stats including time-to-first-token, tokens/sec, and total time
- 📋 **Activity Log**: Step-by-step activity tracking in sidebar showing connection, thinking, streaming, and tool states
- 🎨 **Minimalistic UI**: Clean, dark-themed chat interface with compact layout
- 🐳 **Dockerized**: Easy deployment with Docker and Docker Compose
- 🔄 **Live Status**: Real-time MCP connection status monitoring with collapsible details
- ⌨️ **Command History**: Arrow up/down to navigate through previous messages
- 📦 **Collapsible Tools**: Click on tool calls to expand/collapse JSON details
- 🧠 **Thinking Model Support**: Extended timeouts and status updates for reasoning models like qwen3-next

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and configure**:
```bash
cd /home/toor/AI-Web2
cp .env.example .env
# Edit .env and add your API keys
```

2. **Build and run**:
```bash
docker-compose up --build
```

3. **Access the UI**:
   - Open your browser to http://localhost:5005

### Manual Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"
export OLLAMA_BASE_URL="http://localhost:11434"
export MCP_SERVER_URL="http://your-mcp-server"
```

3. **Run the application**:
```bash
python app.py
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic Claude API key
- `GOOGLE_API_KEY`: Your Google AI API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OLLAMA_BASE_URL`: URL for Ollama server (default: http://localhost:11434)
- `OLLAMA_KEEP_ALIVE`: How long to keep model loaded in memory (default: 1h, examples: '5m', '30m', '1h', '-1' for never unload)
- `MCP_SERVER_URL`: URL for your MCP HTTP SSE server (legacy, optional)
- `MCP_AUTH_TOKEN`: Bearer token for MCP server (legacy, optional)
- `MCP_SERVERS`: JSON configuration for multiple named MCP servers (see below)
- `SYSTEM_PROMPT`: Custom system prompt for the AI assistant (optional, default: "You are a helpful AI assistant.")

#### QA Agent Configuration

- `QA_AGENT_ENABLED`: Enable/disable QA agent response validation (default: false)
- `QA_AGENT_MODEL`: Ollama model for follow-up questions (default: gpt-oss:latest)
- `QA_AGENT_MAX_RETRIES`: Maximum retry attempts if response lacks data (default: 1)

### Supported Models

**Ollama**: Configurable list of models (auto-detection available but disabled by default). Default models include:
- llama4:latest
- qwen3-next:latest (thinking model with extended reasoning)
- deepseek-coder:33b
- llama3:latest
- gemma3:12b
- phi3:14b
- qwen3:8b
- granite4:latest
- llama3.2:latest

**OpenAI** (disabled by default, uncomment in code):
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-3.5-turbo

**Claude**:
- claude-opus-4-5-20251101
- claude-haiku-4-5-20251001
- claude-sonnet-4-5-20250929

**Google** (disabled by default, uncomment in code):
- gemini-2.0-flash-exp
- gemini-1.5-pro
- gemini-1.5-flash

**OpenRouter** (disabled by default, uncomment in code):
- anthropic/claude-3.5-sonnet
- openai/gpt-4o
- google/gemini-2.0-flash-exp

## API Endpoints

### GET `/`
Main chat interface

### GET `/api/providers`
Get available providers and their models

### GET `/api/mcp/status`
Check MCP server connection status

### POST `/api/chat`
Send chat messages (streaming SSE response)

**Request body**:
```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "provider": "ollama",
  "model": "llama2"
}
```

### GET `/api/mcp/tools`
Get available MCP tools

### POST `/api/mcp/call`
Call an MCP tool

**Request body**:
```json
{
  "tool": "tool_name",
  "arguments": {}
}
```

## MCP Integration

The application supports connecting to multiple named MCP (Model Context Protocol) servers via HTTP SSE.

### Single MCP Server (Legacy)

```bash
export MCP_SERVER_URL="https://your-mcp-server/mcp/"
export MCP_AUTH_TOKEN="your-bearer-token"
```

### Multiple Named MCP Servers

Configure multiple MCP servers using JSON format:

```bash
export MCP_SERVERS='{"huginn": {"url": "https://huginn.example.com/mcp/", "auth_token": "token1"}, "local": {"url": "http://localhost:8080/mcp/", "auth_token": ""}}'
```

Or in your `.env` file:

```env
MCP_SERVERS={"huginn": {"url": "https://huginn.freakshowindustries.net/mcp/", "auth_token": "your-token"}, "local": {"url": "http://localhost:8080/mcp/", "auth_token": ""}}
```

### MCP Server Configuration Format

Each MCP server requires:
- **name**: A unique identifier for the server (used as the key)
- **url**: The HTTP SSE endpoint URL (**must end with `/`**)
- **auth_token**: Optional Bearer token for authentication

The UI will display the connection status for all configured servers in the header. Click on the MCP status bar to expand and see detailed connection information for each server.

### Tool Calling

The application automatically discovers tools from all configured MCP servers and makes them available to AI models. When a model decides to use a tool:

1. The tool call is displayed in an orange box (click to expand details)
2. The tool is executed via the appropriate MCP server
3. The result is displayed in a green box (click to expand details)
4. The conversation continues with the tool result

**Note**: Some Ollama models (like DeepSeek) don't support tool calling. The application will automatically retry without tools for these models.

### QA Agent

The QA Agent validates model responses and can automatically request clarification when responses lack concrete data. This is useful for ensuring data queries actually return data rather than explanations.

**Configuration:**
```bash
QA_AGENT_ENABLED=true
QA_AGENT_MODEL=gpt-oss:latest
QA_AGENT_MAX_RETRIES=1
```

**How it works:**
1. After the primary model responds, the QA Agent analyzes the response
2. If the response lacks concrete data (numbers, tables, actual values), it triggers a follow-up
3. The follow-up uses the configured QA model to request specific data
4. The additional response is appended to the conversation

## UI Features

### Statistics Bar
Located below the input area, shows real-time query statistics:
- **Status**: Current state (Idle, Thinking, Streaming, Done)
- **Time**: Total elapsed time
- **TTFT**: Time to first token
- **Tokens**: Number of tokens generated
- **Speed**: Tokens per second

### Activity Log
Located in the sidebar, shows step-by-step progress:
- 🚀 Query started
- 🔌 Connecting to model
- 🧠 Model thinking/reasoning
- 📡 Streaming response
- 🔧 Tool calls
- ✅ Tool results
- 🔍 QA Agent actions
- ✓ Completion with stats

## Docker Details

- **Port**: 5005
- **Base Image**: python:3.11-slim
- **Volume Mounts**: Application code mounted for hot-reloading during development

### Building the Docker Image

```bash
docker build -t ai-web-chat .
```

### Running with Docker

```bash
docker run -p 5005:5005 \
  -e OPENAI_API_KEY="your-key" \
  -e ANTHROPIC_API_KEY="your-key" \
  ai-web-chat
```

## Development

The application uses Flask with hot-reloading enabled in development mode. Modify the code and the server will automatically restart.

### Project Structure

```
AI-Web2/
├── app.py                 # Flask backend with API endpoints
├── templates/
│   └── index.html        # Frontend UI
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables
└── README.md            # This file
```

## Troubleshooting

### Ollama Connection Issues
If running Ollama locally, use `host.docker.internal:11434` instead of `localhost:11434` when running in Docker.

### API Key Errors
Ensure all required API keys are set in your `.env` file and the file is in the same directory as `docker-compose.yml`.

### MCP Connection
Verify your MCP server is running and accessible from the Docker container. Check the status indicator in the UI header.

### ROCm GPU Issues (Fedora with AMD Radeon 8060S / gfx1151)

If you're experiencing GPU hangs with certain large models (like deepseek-r1:70b) on Fedora with ROCm:

1. **Install missing libdrm library**:
```bash
sudo dnf install -y libdrm libdrm-devel
```

2. **Check GPU temperature during inference**:
```bash
watch -n 1 rocm-smi --showtemp
```

3. **Update ROCm to latest version**:
```bash
sudo dnf upgrade rocm-*
```

4. **Verify GPU architecture support**:
```bash
rocminfo | grep "Marketing Name"
rocm-smi --showproductname
```

5. **Known Issues**:
   - **deepseek-r1:70b**: Causes GPU hang on gfx1151 (Radeon 8060S) with ROCm 6.4.2
   - **Workaround**: Use alternative models like `gpt-oss:120b`, `mixtral:8x22b`, or `llama4:latest`
   - **Root cause**: Possible ROCm driver compatibility issue with new RDNA 3.5 architecture

6. **Check Ollama logs for GPU errors**:
```bash
sudo journalctl -u ollama -f
```

If GPU hangs persist, the model may be triggering a hardware exception. Use smaller quantizations or alternative models until ROCm support improves for gfx1151.

## License

MIT License - Feel free to use and modify as needed.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
