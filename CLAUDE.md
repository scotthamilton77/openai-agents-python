# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses `uv` for dependency management. Essential commands:

- `make sync` - Install all dependencies and development tools
- `make tests` - Run the full test suite with pytest
- `make mypy` - Run type checking
- `make lint` - Run ruff linting checks
- `make format` - Auto-format code with ruff
- `make coverage` - Run tests with coverage reporting (requires 95% coverage)

For testing specific areas:
- `uv run pytest tests/test_agent_runner.py` - Test a specific file
- `uv run pytest -k test_name` - Run specific test by name
- `uv run pytest --inline-snapshot=fix` - Update inline snapshots

Documentation:
- `make serve-docs` - Serve documentation locally with MkDocs
- `make build-docs` - Build documentation

## Architecture Overview

The OpenAI Agents SDK is a multi-agent workflow framework built around these core concepts:

### Core Components

**Agent (`src/agents/agent.py`)**: The central abstraction representing an LLM with instructions, tools, guardrails, and handoff capabilities. Agents can have structured output types and lifecycle hooks.

**Runner (`src/agents/run.py`)**: Orchestrates the agent execution loop. Handles the iterative process of LLM calls, tool execution, handoffs, and guardrail validation until reaching a final output.

**Tools (`src/agents/tool.py`)**: Extensible system supporting function tools, built-in tools (computer use, file search, code interpreter, web search), and MCP (Model Context Protocol) servers.

**Handoffs (`src/agents/handoffs.py`)**: Mechanism for transferring control between agents, enabling complex multi-agent workflows with input filtering and routing logic.

**Guardrails (`src/agents/guardrail.py`)**: Input and output validation system that can modify, reject, or approve agent inputs/outputs with configurable safety checks.

### Key Architectural Patterns

**Provider-Agnostic Models**: The `models/` directory abstracts LLM providers through the `Model` and `ModelProvider` interfaces. Supports OpenAI Responses API, Chat Completions API, and LiteLLM for 100+ providers.

**Tracing System**: Built-in observability through `tracing/` module with spans, traces, and extensible processors. Automatically tracks agent runs, tool calls, handoffs, and guardrails.

**Voice Support**: The `voice/` module provides TTS/STT capabilities with pipelines, workflows, and real-time audio processing for voice-enabled agents.

**MCP Integration**: `mcp/` module enables connection to Model Context Protocol servers for external tool and resource access.

### Agent Execution Loop

1. LLM generates response (may include tool calls)
2. Process tool calls and collect results  
3. Run guardrails on outputs
4. Check for handoffs to other agents
5. Determine if final output reached, otherwise repeat

The loop continues until structured output is produced (if `output_type` set) or a message without tool calls/handoffs is generated.

## Testing Conventions

- Tests use fake models by default to avoid real API calls
- Use `@pytest.mark.allow_call_model_methods` for integration tests requiring real models
- Tracing is automatically configured with test processors
- Inline snapshots are used for complex response validation
- Coverage requirement is 95%

## Voice Module

Voice functionality requires the `voice` optional dependency (`pip install 'openai-agents[voice]'`). Key components:
- `VoiceAgentMixin` - Adds voice capabilities to agents
- `VoicePipeline` - Handles audio streaming and processing
- `STTModel`/`TTSModel` interfaces for speech services
- OpenAI implementation with websocket-based streaming

## MCP (Model Context Protocol)

MCP servers provide external tools and resources. The SDK includes:
- Automatic discovery and connection to MCP servers
- Tool approval workflows for security
- Caching and connection management
- Examples in `examples/mcp/` and `examples/hosted_mcp/`