# AI Web Chat Interface

A minimalistic web UI for interacting with multiple AI model providers (Ollama, OpenAI, Google, OpenRouter, and Claude) with MCP (Model Context Protocol) HTTP SSE support.

## Features

- 🤖 **Multiple AI Providers**: Support for Ollama, OpenAI, Claude, Google, and OpenRouter
- 💬 **Real-time Streaming**: Server-Sent Events (SSE) for streaming responses
- 🔌 **MCP Integration**: HTTP SSE support for Model Context Protocol servers with JSON-RPC 2.0
- 🛠️ **Tool Calling**: Full support for MCP tool discovery and execution across all providers
- 🎨 **Minimalistic UI**: Clean, dark-themed chat interface with compact layout
- 🐳 **Dockerized**: Easy deployment with Docker and Docker Compose
- 🔄 **Live Status**: Real-time MCP connection status monitoring with collapsible details
- ⌨️ **Command History**: Arrow up/down to navigate through previous messages
- 📦 **Collapsible Tools**: Click on tool calls to expand/collapse JSON details

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
- `MCP_SERVER_URL`: URL for your MCP HTTP SSE server (legacy, optional)
- `MCP_AUTH_TOKEN`: Bearer token for MCP server (legacy, optional)
- `MCP_SERVERS`: JSON configuration for multiple named MCP servers (see below)
- `SYSTEM_PROMPT`: Custom system prompt for the AI assistant (optional, default: "You are a helpful AI assistant.")

### Supported Models

**Ollama**: Automatically detects installed models

**OpenAI**:
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-3.5-turbo

**Claude**:
- claude-opus-4-5-20251101
- claude-haiku-4-5-20251001
- claude-sonnet-4-5-20250929
- claude-opus-4-1-20250805
- claude-opus-4-20250514
- claude-sonnet-4-20250514
- claude-3-7-sonnet-20250219
- claude-3-5-sonnet-20241022
- claude-3-5-haiku-20241022
- claude-3-haiku-20240307
- claude-3-opus-20240229

**Google**:
- gemini-2.0-flash-exp
- gemini-1.5-pro
- gemini-1.5-flash

**OpenRouter**:
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

## License

MIT License - Feel free to use and modify as needed.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
