from flask import Flask, render_template, request, jsonify, Response
import json
import os
import requests
from typing import Generator, Optional, Dict, Any, List
import anthropic
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import uuid
import threading
import time
from queue import Queue, Empty
from dataclasses import dataclass
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# MCP Session management
@dataclass
class MCPSession:
    server_name: str
    session_id: str
    response: requests.Response
    tools: List[Dict[str, Any]]
    initialized: bool
    created_at: float
    server_config: Dict[str, Any]
    
mcp_session_lock = threading.Lock()
mcp_active_sessions: Dict[str, MCPSession] = {}  # server_name -> session
mcp_tools_cache = {'tools': [], 'timestamp': 0}

# Environment variables for API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'You are a helpful AI assistant.')

# Legacy single MCP server (for backward compatibility)
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', '')
MCP_AUTH_TOKEN = os.getenv('MCP_AUTH_TOKEN', '')

# Multiple MCP servers configuration
# Format: MCP_SERVERS={"server_name": {"url": "...", "auth_token": "..."}}
MCP_SERVERS = {}
try:
    mcp_servers_json = os.getenv('MCP_SERVERS', '')
    if mcp_servers_json:
        MCP_SERVERS = json.loads(mcp_servers_json)
    # Add legacy server if configured
    if MCP_SERVER_URL:
        MCP_SERVERS['default'] = {
            'url': MCP_SERVER_URL,
            'auth_token': MCP_AUTH_TOKEN
        }
except Exception as e:
    print(f"Error parsing MCP_SERVERS: {e}")
    if MCP_SERVER_URL:
        MCP_SERVERS = {
            'default': {
                'url': MCP_SERVER_URL,
                'auth_token': MCP_AUTH_TOKEN
            }
        }

# Configure Google AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/providers', methods=['GET'])
def get_providers():
    """Get available providers and their models"""
    providers = {
        'ollama': {'name': 'Ollama', 'models': []},
        'openai': {'name': 'OpenAI', 'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']},
        'claude': {'name': 'Claude', 'models': ['claude-opus-4-5-20251101', 'claude-haiku-4-5-20251001', 'claude-sonnet-4-5-20250929', 'claude-opus-4-1-20250805', 'claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219', 'claude-3-5-haiku-20241022', 'claude-3-haiku-20240307', 'claude-3-opus-20240229']},
        'google': {'name': 'Google', 'models': ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash']},
        'openrouter': {'name': 'OpenRouter', 'models': ['anthropic/claude-3.5-sonnet', 'openai/gpt-4o', 'google/gemini-2.0-flash-exp']}
    }
    
    # Get Ollama models if available
    try:
        response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            providers['ollama']['models'] = [m['name'] for m in models]
    except:
        pass
    
    return jsonify(providers)

@app.route('/api/mcp/status', methods=['GET'])
def mcp_status():
    """Check MCP server connection status"""
    if not MCP_SERVERS:
        return jsonify({'connected': False, 'message': 'No MCP servers configured', 'servers': []})
    
    server_statuses = []
    any_connected = False
    
    for server_name, server_config in MCP_SERVERS.items():
        try:
            headers = {'Accept': 'text/event-stream'}
            auth_token = server_config.get('auth_token', '')
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            
            # Try to establish SSE connection
            response = requests.get(server_config['url'], headers=headers, timeout=3, stream=True)
            
            # Status 200 or 400 with mcp-session-id means server is accessible
            # 400 is expected without proper JSON-RPC initialization
            if response.status_code == 200:
                response.close()
                server_statuses.append({'name': server_name, 'connected': True, 'message': 'Connected'})
                any_connected = True
            elif response.status_code == 400 and 'mcp-session-id' in response.headers:
                response.close()
                server_statuses.append({'name': server_name, 'connected': True, 'message': 'Connected'})
                any_connected = True
            elif response.status_code == 401:
                server_statuses.append({'name': server_name, 'connected': False, 'message': 'Auth failed'})
            else:
                server_statuses.append({'name': server_name, 'connected': False, 'message': f'Status {response.status_code}'})
        except requests.exceptions.Timeout:
            server_statuses.append({'name': server_name, 'connected': True, 'message': 'Connected'})
            any_connected = True
        except Exception as e:
            server_statuses.append({'name': server_name, 'connected': False, 'message': str(e)})
    
    return jsonify({
        'connected': any_connected,
        'message': f'{sum(1 for s in server_statuses if s["connected"])} of {len(server_statuses)} connected',
        'servers': server_statuses
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with streaming"""
    data = request.json
    messages = data.get('messages', [])
    provider = data.get('provider', 'ollama')
    model = data.get('model', '')
    use_mcp = data.get('use_mcp', True)
    
    print(f"\n{'='*60}")
    print(f"[CHAT] New request - Provider: {provider}, Model: {model}")
    print(f"[CHAT] Messages count: {len(messages)}")
    print(f"[CHAT] Use MCP: {use_mcp}")
    print(f"[CHAT] Active MCP servers: {list(MCP_SERVERS.keys())}")
    
    # Add system prompt and MCP context if needed
    mcp_context = ""
    if use_mcp and (MCP_SERVERS or MCP_SERVER_URL):
        mcp_context = "\n\nNote: An MCP (Model Context Protocol) server is available for extended capabilities."
        print(f"[CHAT] Adding MCP context to system prompt")
    
    # Add to existing system message or create new one
    has_system = any(msg['role'] == 'system' for msg in messages)
    if has_system:
        for msg in messages:
            if msg['role'] == 'system':
                msg['content'] += mcp_context
                break
    else:
        print(f"[CHAT] Using system prompt: {SYSTEM_PROMPT[:50]}...")
        messages.insert(0, {'role': 'system', 'content': f'{SYSTEM_PROMPT}{mcp_context}'})
    
    def generate() -> Generator[str, None, None]:
        try:
            if provider == 'ollama':
                yield from stream_ollama(messages, model)
            elif provider == 'openai':
                yield from stream_openai(messages, model)
            elif provider == 'claude':
                yield from stream_claude(messages, model)
            elif provider == 'google':
                yield from stream_google(messages, model)
            elif provider == 'openrouter':
                yield from stream_openrouter(messages, model)
            else:
                yield f"data: {json.dumps({'error': 'Unknown provider'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

def stream_ollama(messages, model):
    """Stream responses from Ollama with tool calling support"""
    # Get MCP tools from all servers
    all_tools = []
    server_tool_map = {}  # tool_name -> server_name
    
    for server_name in MCP_SERVERS.keys():
        session = get_or_create_mcp_session(server_name)
        if session and session.tools:
            for tool in session.tools:
                server_tool_map[tool['name']] = server_name
            all_tools.extend(session.tools)
    
    payload = {'model': model, 'messages': messages, 'stream': True}
    if all_tools:
        tools = convert_mcp_tools_to_openai(all_tools)
        payload['tools'] = tools
        print(f"[Ollama] Publishing {len(tools)} tools to model {model}: {[t['function']['name'] for t in tools]}")
    
    print(f"[Ollama] Connecting to {OLLAMA_BASE_URL}/api/chat")
    
    try:
        response = requests.post(
            f'{OLLAMA_BASE_URL}/api/chat',
            json=payload,
            stream=True,
            timeout=300
        )
        
        print(f"[Ollama] Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_body = response.text
            error_msg = f"Ollama returned status {response.status_code}: {error_body}"
            print(f"[Ollama] Error: {error_msg}")
            
            # Check if it's a tool-related error
            if response.status_code == 400 and 'tool' in error_body.lower() and all_tools:
                print(f"[Ollama] Tool error detected, retrying without tools")
                payload.pop('tools', None)
                response = requests.post(
                    f'{OLLAMA_BASE_URL}/api/chat',
                    json=payload,
                    stream=True,
                    timeout=300
                )
                print(f"[Ollama] Retry response status: {response.status_code}")
                if response.status_code != 200:
                    yield f"data: {json.dumps({'error': f'Ollama error: {response.text}'})}\n\n"
                    return
            else:
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                return
            
    except Exception as e:
        error_msg = f"Failed to connect to Ollama: {str(e)}"
        print(f"[Ollama] Connection error: {error_msg}")
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
        return
    
    tool_calls_made = []
    line_count = 0
    
    for line in response.iter_lines():
        if line:
            line_count += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Ollama] JSON decode error on line {line_count}: {e}")
                continue
                
            if 'message' in data:
                msg = data['message']
                content = msg.get('content', '')
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
                
                # Handle tool calls (Ollama uses OpenAI-compatible format)
                if 'tool_calls' in msg:
                    tool_calls_made = msg['tool_calls']
                    
            if data.get('done', False):
                # If we have tool calls, execute them and continue
                if tool_calls_made:
                    tool_results = []
                    for tool_call in tool_calls_made:
                        try:
                            tool_name = tool_call['function']['name']
                            # Ollama might return arguments as dict or string
                            args_raw = tool_call['function']['arguments']
                            if isinstance(args_raw, str):
                                args = json.loads(args_raw)
                            else:
                                args = args_raw
                            
                            server_name = server_tool_map.get(tool_name, list(MCP_SERVERS.keys())[0] if MCP_SERVERS else None)
                            
                            yield f"data: {json.dumps({'tool_call': tool_name, 'server': server_name, 'arguments': args})}\n\n"
                            
                            if server_name:
                                result = call_mcp_tool(server_name, tool_name, args)
                                yield f"data: {json.dumps({'tool_result': tool_name, 'result': result})}\n\n"
                                tool_results.append({
                                    'role': 'tool',
                                    'content': json.dumps(result)
                                })
                        except Exception as e:
                            print(f"Tool call error: {e}")
                            tool_results.append({
                                'role': 'tool',
                                'content': json.dumps({'error': str(e)})
                            })
                    
                    # Continue conversation with tool results
                    messages.extend(tool_results)
                    continue_payload = {'model': model, 'messages': messages, 'stream': True}
                    if tools:
                        continue_payload['tools'] = tools
                    
                    continue_response = requests.post(
                        f'{OLLAMA_BASE_URL}/api/chat',
                        json=continue_payload,
                        stream=True
                    )
                    
                    for cont_line in continue_response.iter_lines():
                        if cont_line:
                            cont_data = json.loads(cont_line)
                            if 'message' in cont_data:
                                cont_content = cont_data['message'].get('content', '')
                                if cont_content:
                                    yield f"data: {json.dumps({'content': cont_content})}\n\n"
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                print(f"[Ollama] Stream complete. Lines processed: {line_count}")
                break

def stream_openai(messages, model):
    """Stream responses from OpenAI with tool calling support"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Get MCP tools from all servers
    all_tools = []
    server_tool_map = {}  # tool_name -> server_name
    
    for server_name in MCP_SERVERS.keys():
        session = get_or_create_mcp_session(server_name)
        if session and session.tools:
            for tool in session.tools:
                server_tool_map[tool['name']] = server_name
            all_tools.extend(session.tools)
    
    tools = None
    if all_tools:
        tools = convert_mcp_tools_to_openai(all_tools)
        print(f"[OpenAI] Publishing {len(tools)} tools to model {model}: {[t['function']['name'] for t in tools]}")
    
    # Create chat completion with tools
    kwargs = {
        'model': model,
        'messages': messages,
        'stream': True
    }
    if tools:
        kwargs['tools'] = tools
        kwargs['tool_choice'] = 'auto'
    
    stream = client.chat.completions.create(**kwargs)
    
    tool_calls = []
    current_tool_call = None
    
    for chunk in stream:
        delta = chunk.choices[0].delta
        
        # Handle content
        if delta.content:
            yield f"data: {json.dumps({'content': delta.content})}\n\n"
        
        # Handle tool calls
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                if tc_delta.index is not None:
                    # New tool call
                    if len(tool_calls) <= tc_delta.index:
                        tool_calls.append({
                            'id': tc_delta.id or '',
                            'name': '',
                            'arguments': ''
                        })
                    current_tool_call = tool_calls[tc_delta.index]
                    
                    if tc_delta.function:
                        if tc_delta.function.name:
                            current_tool_call['name'] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            current_tool_call['arguments'] += tc_delta.function.arguments
        
        # Check if done
        if chunk.choices[0].finish_reason == 'tool_calls':
            # Execute tool calls
            for tool_call in tool_calls:
                try:
                    args = json.loads(tool_call['arguments'])
                    tool_name = tool_call['name']
                    server_name = server_tool_map.get(tool_name, list(MCP_SERVERS.keys())[0] if MCP_SERVERS else None)
                    
                    yield f"data: {json.dumps({'tool_call': tool_name, 'server': server_name, 'arguments': args})}\n\n"
                    
                    if server_name:
                        result = call_mcp_tool(server_name, tool_name, args)
                        yield f"data: {json.dumps({'tool_result': tool_name, 'server': server_name, 'result': result})}\n\n"
                    else:
                        yield f"data: {json.dumps({'error': f'No MCP server found for tool {tool_name}'})}\n\n"
                        continue
                    
                    # Add tool result to messages and continue
                    messages.append({
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [{
                            'id': tool_call['id'],
                            'type': 'function',
                            'function': {
                                'name': tool_call['name'],
                                'arguments': tool_call['arguments']
                            }
                        }]
                    })
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call['id'],
                        'content': json.dumps(result)
                    })
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'Tool call failed: {str(e)}'})}\n\n"
            
            # Continue conversation with tool results
            continue_stream = client.chat.completions.create(**kwargs)
            for chunk in continue_stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            
            break
    
    yield f"data: {json.dumps({'done': True})}\n\n"

def stream_claude(messages, model):
    """Stream responses from Claude with tool calling support"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Get MCP tools from all servers
    all_tools = []
    server_tool_map = {}  # tool_name -> server_name
    
    for server_name in MCP_SERVERS.keys():
        session = get_or_create_mcp_session(server_name)
        if session and session.tools:
            for tool in session.tools:
                server_tool_map[tool['name']] = server_name
            all_tools.extend(session.tools)
    
    tools = None
    if all_tools:
        tools = convert_mcp_tools_to_anthropic(all_tools)
        print(f"[Claude] Publishing {len(tools)} tools to model {model}: {[t['name'] for t in tools]}")
    
    # Convert messages format
    system_message = ""
    claude_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            system_message = msg['content']
        else:
            claude_messages.append(msg)
    
    # Create message with tools
    kwargs = {
        'model': model,
        'max_tokens': 4096,
        'messages': claude_messages
    }
    if system_message:
        kwargs['system'] = system_message
    if tools:
        kwargs['tools'] = tools
    
    while True:
        response = client.messages.create(**kwargs)
        
        # Collect tool calls first
        tool_calls = []
        text_content = ""
        
        for block in response.content:
            if block.type == 'text':
                text_content += block.text
            elif block.type == 'tool_use':
                tool_calls.append(block)
        
        # Stream text content if any
        if text_content:
            yield f"data: {json.dumps({'content': text_content})}\n\n"
        
        # If no tool calls, we're done
        if not tool_calls:
            break
        
        # Execute all tool calls and collect results
        tool_results = []
        for block in tool_calls:
            tool_name = block.name
            server_name = server_tool_map.get(tool_name, list(MCP_SERVERS.keys())[0] if MCP_SERVERS else None)
            
            yield f"data: {json.dumps({'tool_call': tool_name, 'server': server_name, 'arguments': block.input})}\n\n"
            
            try:
                if server_name:
                    result = call_mcp_tool(server_name, tool_name, block.input)
                    yield f"data: {json.dumps({'tool_result': tool_name, 'server': server_name, 'result': result})}\n\n"
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': block.id,
                        'content': json.dumps(result)
                    })
                else:
                    yield f"data: {json.dumps({'error': f'No MCP server found for tool {tool_name}'})}\n\n"
                    return
            except Exception as e:
                yield f"data: {json.dumps({'error': f'Tool call failed: {str(e)}'})}\n\n"
                return
        
        # Add assistant message with tool calls and user message with all tool results
        claude_messages.append({
            'role': 'assistant',
            'content': response.content
        })
        claude_messages.append({
            'role': 'user',
            'content': tool_results
        })
        kwargs['messages'] = claude_messages
    
    yield f"data: {json.dumps({'done': True})}\n\n"

def stream_google(messages, model):
    """Stream responses from Google with tool calling support"""
    from google.generativeai.types import Tool
    
    # Get MCP tools from all servers
    all_tools = []
    server_tool_map = {}  # tool_name -> server_name
    
    for server_name in MCP_SERVERS.keys():
        session = get_or_create_mcp_session(server_name)
        if session and session.tools:
            for tool in session.tools:
                server_tool_map[tool['name']] = server_name
            all_tools.extend(session.tools)
    
    tools_param = None
    if all_tools:
        function_declarations = convert_mcp_tools_to_google(all_tools)
        tools_param = [Tool(function_declarations=function_declarations)]
        print(f"[Google] Publishing {len(function_declarations)} tools to model {model}: {[f.name for f in function_declarations]}")
    
    # Create model with tools
    model_kwargs = {}
    if tools_param:
        model_kwargs['tools'] = tools_param
    
    genai_model = genai.GenerativeModel(model, **model_kwargs)
    
    # Convert messages to Google format
    chat_history = []
    prompt = ""
    for msg in messages:
        if msg['role'] == 'system':
            continue
        elif msg['role'] == 'user':
            if not chat_history:
                prompt = msg['content']
            else:
                chat_history.append({'role': 'user', 'parts': [msg['content']]})
        elif msg['role'] == 'assistant':
            chat_history.append({'role': 'model', 'parts': [msg['content']]})
    
    chat = genai_model.start_chat(history=chat_history if chat_history else None)
    
    while True:
        response = chat.send_message(prompt, stream=True)
        
        has_tool_calls = False
        for chunk in response:
            if chunk.text:
                yield f"data: {json.dumps({'content': chunk.text})}\n\n"
            
            # Check for function calls
            if hasattr(chunk, 'parts'):
                for part in chunk.parts:
                    if hasattr(part, 'function_call'):
                        has_tool_calls = True
                        fc = part.function_call
                        args = dict(fc.args)
                        tool_name = fc.name
                        server_name = server_tool_map.get(tool_name, list(MCP_SERVERS.keys())[0] if MCP_SERVERS else None)
                        
                        yield f"data: {json.dumps({'tool_call': tool_name, 'server': server_name, 'arguments': args})}\n\n"
                        
                        try:
                            if server_name:
                                result = call_mcp_tool(server_name, tool_name, args)
                                yield f"data: {json.dumps({'tool_result': tool_name, 'server': server_name, 'result': result})}\n\n"
                            else:
                                yield f"data: {json.dumps({'error': f'No MCP server found for tool {tool_name}'})}\n\n"
                                break
                            
                            # Send function response
                            from google.generativeai.types import content_types
                            function_response = content_types.FunctionResponse(
                                name=fc.name,
                                response={'result': result}
                            )
                            prompt = content_types.to_content([function_response])
                            break
                        except Exception as e:
                            yield f"data: {json.dumps({'error': f'Tool call failed: {str(e)}'})}\n\n"
                            break
        
        if not has_tool_calls:
            break
    
    yield f"data: {json.dumps({'done': True})}\n\n"

def stream_openrouter(messages, model):
    """Stream responses from OpenRouter with tool calling support"""
    # Get MCP tools from all servers
    all_tools = []
    server_tool_map = {}  # tool_name -> server_name
    
    for server_name in MCP_SERVERS.keys():
        session = get_or_create_mcp_session(server_name)
        if session and session.tools:
            for tool in session.tools:
                server_tool_map[tool['name']] = server_name
            all_tools.extend(session.tools)
    
    tools = None
    payload = {'model': model, 'messages': messages, 'stream': True}
    if all_tools:
        tools = convert_mcp_tools_to_openai(all_tools)
        payload['tools'] = tools
        payload['tool_choice'] = 'auto'
        print(f"[OpenRouter] Publishing {len(tools)} tools to model {model}: {[t['function']['name'] for t in tools]}")
    
    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json'
        },
        json=payload,
        stream=True
    )
    
    tool_calls = []
    current_tool_call = None
    
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                if data_str == '[DONE]':
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    break
                try:
                    data = json.loads(data_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        delta = choice.get('delta', {})
                        
                        # Handle content
                        content = delta.get('content', '')
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                        
                        # Handle tool calls (OpenRouter uses OpenAI format)
                        if 'tool_calls' in delta:
                            for tc_delta in delta['tool_calls']:
                                if tc_delta.get('index') is not None:
                                    idx = tc_delta['index']
                                    if len(tool_calls) <= idx:
                                        tool_calls.append({
                                            'id': tc_delta.get('id', ''),
                                            'name': '',
                                            'arguments': ''
                                        })
                                    current_tool_call = tool_calls[idx]
                                    
                                    if 'function' in tc_delta:
                                        if 'name' in tc_delta['function']:
                                            current_tool_call['name'] = tc_delta['function']['name']
                                        if 'arguments' in tc_delta['function']:
                                            current_tool_call['arguments'] += tc_delta['function']['arguments']
                        
                        # Handle tool call finish
                        if choice.get('finish_reason') == 'tool_calls':
                            for tool_call in tool_calls:
                                try:
                                    args = json.loads(tool_call['arguments'])
                                    tool_name = tool_call['name']
                                    server_name = server_tool_map.get(tool_name, list(MCP_SERVERS.keys())[0] if MCP_SERVERS else None)
                                    
                                    yield f"data: {json.dumps({'tool_call': tool_name, 'server': server_name, 'arguments': args})}\n\n"
                                    
                                    if server_name:
                                        result = call_mcp_tool(server_name, tool_name, args)
                                        yield f"data: {json.dumps({'tool_result': tool_name, 'result': result})}\n\n"
                                except Exception as e:
                                    print(f"Tool call error: {e}")
                except json.JSONDecodeError:
                    pass

def parse_sse_message(line: str) -> Optional[Dict[str, Any]]:
    """Parse SSE message line"""
    if line.startswith('data: '):
        try:
            return json.loads(line[6:])
        except:
            return None
    return None

def send_jsonrpc_request(server_url: str, auth_token: str, session_id: str, method: str, params: Dict[str, Any], request_id: int = None) -> Optional[Dict[str, Any]]:
    """Send a JSON-RPC request to MCP server and get response"""
    if request_id is None:
        request_id = int(time.time() * 1000)
    
    headers = {
        'Accept': 'application/json, text/event-stream',
        'Content-Type': 'application/json'
    }
    if auth_token:
        headers['Authorization'] = f'Bearer {auth_token}'
    if session_id:
        headers['mcp-session-id'] = session_id
    
    request_body = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params
    }
    
    try:
        response = requests.post(
            server_url,
            headers=headers,
            json=request_body,
            stream=True,
            timeout=10
        )
        
        # Read SSE stream for response
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    if data.get('id') == request_id:
                        response.close()
                        return data
                except json.JSONDecodeError:
                    continue
        
        response.close()
        return None
    except Exception as e:
        print(f"JSON-RPC request error [{method}]: {e}")
        return None

def initialize_mcp_session(server_name: str, server_config: Dict[str, Any]) -> Optional[MCPSession]:
    """Initialize a new MCP SSE session with tool discovery"""
    
    server_url = server_config.get('url')
    auth_token = server_config.get('auth_token', '')
    
    if not server_url:
        return None
    
    try:
        headers = {'Accept': 'text/event-stream'}
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
        
        # Establish SSE connection to get session ID
        response = requests.get(
            server_url,
            headers=headers,
            stream=True,
            timeout=10
        )
        
        if response.status_code not in [200, 400]:
            print(f"MCP [{server_name}] connection failed: {response.status_code}")
            return None
        
        # Get session ID from headers
        session_id = response.headers.get('mcp-session-id')
        response.close()
        
        if not session_id:
            print(f"MCP [{server_name}] no session ID received")
            return None
        
        print(f"MCP [{server_name}] session ID: {session_id}")
        
        # Send initialize request
        init_response = send_jsonrpc_request(
            server_url,
            auth_token,
            session_id,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "ai-web-chat",
                    "version": "1.0.0"
                }
            },
            request_id=1
        )
        
        if not init_response or 'error' in init_response:
            print(f"MCP [{server_name}] initialize failed: {init_response}")
            return None
        
        print(f"MCP [{server_name}] initialized successfully")
        
        # Send initialized notification (no response expected for notifications)
        # Note: Notifications in JSON-RPC don't have an id field
        try:
            headers = {
                'Accept': 'application/json, text/event-stream',
                'Content-Type': 'application/json'
            }
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            if session_id:
                headers['mcp-session-id'] = session_id
            
            notification_body = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            
            requests.post(server_url, headers=headers, json=notification_body, timeout=5)
        except Exception as e:
            print(f"MCP [{server_name}] notification error (non-critical): {e}")
        
        # List available tools
        tools_response = send_jsonrpc_request(
            server_url,
            auth_token,
            session_id,
            "tools/list",
            {},
            request_id=3
        )
        
        tools = []
        if tools_response and 'result' in tools_response:
            tools = tools_response['result'].get('tools', [])
            print(f"MCP [{server_name}] discovered {len(tools)} tools: {[t['name'] for t in tools]}")
        else:
            print(f"MCP [{server_name}] no tools discovered")
        
        # Create session
        session = MCPSession(
            server_name=server_name,
            session_id=session_id,
            response=None,  # We don't keep connection open
            tools=tools,
            initialized=True,
            created_at=time.time(),
            server_config=server_config
        )
        
        return session
        
    except Exception as e:
        print(f"MCP [{server_name}] session error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_or_create_mcp_session(server_name: str) -> Optional[MCPSession]:
    """Get active MCP session for a server or create new one"""
    global mcp_active_sessions
    
    if server_name not in MCP_SERVERS:
        print(f"MCP server '{server_name}' not configured")
        return None
    
    with mcp_session_lock:
        # Check if session exists and is recent (< 5 minutes old)
        if server_name in mcp_active_sessions:
            session = mcp_active_sessions[server_name]
            if time.time() - session.created_at < 300:
                return session
            
            # Close old session
            try:
                session.response.close()
            except:
                pass
            del mcp_active_sessions[server_name]
        
        # Create new session
        session = initialize_mcp_session(server_name, MCP_SERVERS[server_name])
        if session:
            mcp_active_sessions[server_name] = session
        return session

def call_mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call an MCP tool and return the result"""
    session = get_or_create_mcp_session(server_name)
    if not session:
        return {'error': f'Failed to establish MCP session for server: {server_name}'}
    
    server_config = session.server_config
    
    try:
        result = send_jsonrpc_request(
            server_config['url'],
            server_config.get('auth_token', ''),
            session.session_id,
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments
            }
        )
        
        if result and 'result' in result:
            return result['result']
        elif result and 'error' in result:
            return {'error': result['error']}
        else:
            return {'error': 'No response from MCP server'}
        
    except Exception as e:
        return {'error': str(e)}

def convert_mcp_tools_to_openai(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MCP tool schema to OpenAI function format"""
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool['name'],
                "description": tool.get('description', ''),
                "parameters": tool.get('inputSchema', {})
            }
        })
    return openai_tools

def convert_mcp_tools_to_anthropic(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MCP tool schema to Anthropic format"""
    anthropic_tools = []
    for tool in mcp_tools:
        anthropic_tools.append({
            "name": tool['name'],
            "description": tool.get('description', ''),
            "input_schema": tool.get('inputSchema', {})
        })
    return anthropic_tools

def convert_mcp_tools_to_google(mcp_tools: List[Dict[str, Any]]) -> List[Any]:
    """Convert MCP tool schema to Google function declarations"""
    from google.generativeai.types import FunctionDeclaration
    
    google_tools = []
    for tool in mcp_tools:
        # Convert JSON Schema to Google format
        params = tool.get('inputSchema', {})
        google_tools.append(FunctionDeclaration(
            name=tool['name'],
            description=tool.get('description', ''),
            parameters=params
        ))
    return google_tools

@app.route('/api/mcp/servers', methods=['GET'])
def mcp_servers():
    """Get list of configured MCP servers"""
    servers = []
    for name, config in MCP_SERVERS.items():
        servers.append({
            'name': name,
            'url': config.get('url'),
            'connected': name in mcp_active_sessions
        })
    return jsonify({'servers': servers})

@app.route('/api/mcp/tools', methods=['GET'])
def mcp_tools():
    """Get available MCP tools from all servers"""
    if not MCP_SERVERS:
        return jsonify({'error': 'No MCP servers configured'}), 400
    
    server_name = request.args.get('server', list(MCP_SERVERS.keys())[0] if MCP_SERVERS else None)
    if not server_name:
        return jsonify({'error': 'No server specified'}), 400
    
    session = get_or_create_mcp_session(server_name)
    if not session:
        return jsonify({'error': f'Failed to establish MCP session for {server_name}', 'tools': []}), 500
    
    return jsonify({'server': server_name, 'tools': session.tools})

@app.route('/api/mcp/call', methods=['POST'])
def mcp_call():
    """Call an MCP tool"""
    if not MCP_SERVER_URL:
        return jsonify({'error': 'MCP server not configured'}), 400
    
    data = request.json
    tool_name = data.get('tool')
    arguments = data.get('arguments', {})
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        }
        if MCP_AUTH_TOKEN:
            headers['Authorization'] = f'Bearer {MCP_AUTH_TOKEN}'
        response = requests.post(
            f'{MCP_SERVER_URL}/call',
            json={'tool': tool_name, 'arguments': arguments},
            headers=headers,
            timeout=30
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
