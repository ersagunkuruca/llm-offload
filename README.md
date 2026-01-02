# Log Compression Pipeline for Claude Code

A complete log summarization pipeline that reduces token spend by compressing and summarizing logs **before** they enter Claude's context window.

## How Token Savings Work

**Important**: To save tokens, you must run `clog` BEFORE sending logs to Claude:

```bash
# Step 1: Compress logs locally (uses local LLM, not Claude)
clog app.log > summary.txt

# Step 2: Send ONLY the summary to Claude Code
# Copy summary.txt content, paste to Claude
```

This way Claude never sees the raw 15,000 lines - only the ~50 line summary.

## Components

1. **Ollama** - Local LLM runtime (runs on your machine, free)
2. **Qwen2.5 7B Instruct** - Default model for summarization (local)
3. **clog** - CLI tool for log compression and summarization
4. **MCP Server** - Optional native Claude Code tool integration
5. **Reference docs** - Agent behavior guides in `~/.claude/agents/`

## Quick Start

```bash
# Compress and summarize a log file
clog app.log

# Compression only (no LLM)
clog --no-llm app.log

# Focus on a specific question
clog -p "What caused the OOM error?" app.log

# From stdin
docker logs myapp | clog

# JSON output for automation
clog --json app.log
```

## Installation Verification

```bash
# Check Ollama is running
ollama list

# Check clog is available
clog --help

# Check agents are installed
ls ~/.claude/agents/
```

## clog Command Reference

```
Usage: clog [OPTIONS] [FILES...]

Options:
  --no-llm             Skip LLM summary, only compression
  --max-lines N        Process only first N lines
  --around-error K     Keep K lines around errors (default: 5)
  --json               Output as JSON
  --model NAME         Use different Ollama model
  --compression-only   Alias for --no-llm
  -p, --prompt TEXT    Focus LLM analysis on a specific question
```

## Using with Streaming Logs (logcat, Xcode, etc.)

clog processes logs after they complete (batch mode). For streaming tools:

```bash
# Android logcat - capture then analyze
adb logcat -d > logcat.txt && clog logcat.txt
# Or with timeout:
timeout 30 adb logcat > logcat.txt; clog logcat.txt

# Xcode console - copy logs to file, then:
clog ~/Desktop/xcode_console.log

# Tail with limit - capture last N lines then analyze
adb logcat -t 1000 | clog
kubectl logs --tail=500 deployment/api | clog

# Live tail with periodic snapshots (shell loop)
while true; do
  adb logcat -t 500 | clog --no-llm > /tmp/latest_summary.txt
  sleep 30
done
```

For true real-time streaming, capture to a file and analyze periodically, or use `--max-lines` to limit processing.

## What clog Does

### Deterministic Compression Pass
1. **Deduplicates** consecutive identical lines
2. **Normalizes** timestamps, UUIDs, IDs, IPs to placeholders
3. **Templates** similar lines and counts occurrences
4. **Preserves** error context windows (lines around ERROR/WARN/EXCEPTION)
5. **Tracks** coarse timeline from first to last timestamp

### LLM Summary (via Qwen2.5 7B)
Produces structured markdown with:
- What happened (1-3 bullets)
- Top error signatures (table)
- Timeline (coarse)
- Most likely root causes (ranked)
- Suggested debugging steps (ranked)

## Example Output

```markdown
# Log Compression Summary

**Original lines:** 15000
**Unique templates:** 47
**Compression ratio:** 319.1x

## Error Signatures
| Count | First | Last | Template |
|-------|-------|------|----------|
| 127 | L1205 | L14892 | `ERROR: Connection to <IP>:<PORT> failed: <ID>` |

## Error Context Windows
### Window 1 (lines 1200-1210)
...

## LLM Analysis
### What Happened
- Application started normally at 10:30
- Connection failures began at 10:35
- Service degraded to 50% capacity

### Most Likely Root Causes
1. Database connection pool exhaustion (evidence: 127 connection failures)
2. Network partition to database server (evidence: all failures to same IP)

### Suggested Next Debugging Steps
1. Check database server health and connection limits
2. Review connection pool configuration
3. Check for network issues between app and DB
```

## Recommended Workflow

### For Log Analysis (saves the most tokens)

```bash
# 1. Run clog locally FIRST
clog app.log

# 2. Copy the output
# 3. Paste to Claude: "Here's my compressed log summary, what went wrong?"
```

### For Quick Local Analysis (no Claude needed)

```bash
# Get analysis from local model only
clog app.log
# The output includes LLM analysis from Qwen2.5 7B
```

### For General Tasks with Local Model

```bash
# Summarize any text
cat document.txt | ollama run qwen2.5:7b-instruct "Summarize this"

# Generate commit message
git diff --staged | ollama run qwen2.5:7b-instruct "Write a commit message"

# Quick code explanation
cat script.py | ollama run qwen2.5:7b-instruct "What does this do?"
```

## Reference Documentation

The files in `~/.claude/agents/` are reference guides for using this pipeline:
- `log-summarizer.md` - Best practices for log analysis workflow
- `local-agent.md` - When and how to use the local model

These are documentation files, not automated hooks. To save tokens, always run `clog` manually before pasting logs to Claude.

## Troubleshooting

### "clog: command not found"
```bash
# Add to PATH
export PATH="$HOME/bin:$PATH"
# Or source your shell config
source ~/.zshrc
```

### "Ollama not running"
```bash
# Start Ollama service
brew services start ollama
# Or run manually
ollama serve
```

### "Model not found"
```bash
# Pull the model
ollama pull qwen2.5:7b-instruct
```

### Slow first run
Normal - the model needs to load into memory. Subsequent runs are faster.

### Out of memory
Try a smaller model:
```bash
ollama pull qwen2.5:3b-instruct
clog --model qwen2.5:3b-instruct app.log
```

## Advanced Usage

### Using with Docker logs
```bash
docker logs --tail 10000 mycontainer 2>&1 | clog
```

### Using with Kubernetes
```bash
kubectl logs -f deployment/myapp --tail=5000 | clog
```

### Using with journalctl
```bash
journalctl -u myservice --since "1 hour ago" | clog
```

### Processing multiple files
```bash
clog app.log error.log debug.log
```

### JSON output for scripts
```bash
clog --json app.log | jq '.templates | length'
```

## File Locations

- CLI tool: `~/bin/clog` → `~/Library/Python/3.9/bin/clog`
- Source: `~/claude-log-compression/clog/clog.py`
- Agents: `~/.claude/agents/log-summarizer.md`, `~/.claude/agents/local-agent.md`
- Ollama models: `~/.ollama/models/`

## Sharing with Colleagues

### Option 1: Share the install script

```bash
# Copy this folder to a shared location or git repo
cp -r ~/claude-log-compression /path/to/shared/

# Colleagues run:
/path/to/shared/claude-log-compression/install.sh
```

### Option 2: Quick one-liner (if hosted)

If you host the install script (e.g., on GitHub, internal server):
```bash
curl -fsSL https://your-server/install.sh | bash
```

### Option 3: Manual steps

Share these instructions:
```bash
# 1. Install Ollama
brew install ollama  # macOS
# or: curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# 2. Start Ollama
brew services start ollama

# 3. Pull model
ollama pull qwen2.5:7b-instruct

# 4. Install clog (copy clog.py to their machine)
mkdir -p ~/.local/share/clog ~/bin
# Copy clog.py to ~/.local/share/clog/
echo '#!/bin/bash
exec python3 ~/.local/share/clog/clog.py "$@"' > ~/bin/clog
chmod +x ~/bin/clog
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
```

### What colleagues need

- macOS or Linux
- Python 3.8+
- ~5GB disk space for model
- 8GB+ RAM recommended

## Extending

### Add more models
```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull codellama:7b  # For code-heavy logs
```

### Modify compression patterns
Edit `clog/clog.py` and update the `PATTERNS` dict.

### Customize summary format
Edit the `generate_llm_summary()` function in clog.py.

## MCP Server (Native Claude Tool)

You can also expose `clog` and `local_llm` as native Claude Code tools via MCP (Model Context Protocol). This allows Claude to use these tools directly without manual invocation.

### Install MCP Server

```bash
# 1. Create virtual environment (requires Python 3.10+)
cd ~/claude-log-compression/mcp-server
/opt/homebrew/bin/python3 -m venv venv  # macOS with Homebrew
# or: python3.10 -m venv venv  # Linux

# 2. Install MCP
source venv/bin/activate
pip install mcp

# 3. Register with Claude Code
claude mcp add --transport stdio llm-offload --scope user -- \
  ~/claude-log-compression/mcp-server/venv/bin/python \
  ~/claude-log-compression/mcp-server/server.py

# 4. Verify
claude mcp list
```

### Available MCP Tools

Once registered, Claude Code has access to:

| Tool | Description |
|------|-------------|
| `clog` | Compress and summarize log files using local LLM |
| `local_llm` | Run any prompt through Ollama (for token-heavy simple tasks) |
| `clog_file_list` | List log files in a directory with sizes |

### Example Usage (from Claude's perspective)

Claude can now call these tools directly:

```python
# Analyze a log file with a focused question
clog(file_path="/var/log/app.log", prompt="What caused the OOM?")

# Analyze log content directly (e.g., from command output)
clog(log_content="...", prompt="What caused the errors?")

# Offload summarization to local model
local_llm(prompt="Summarize this in one sentence", input_text="...")

# Generate boilerplate locally
local_llm(prompt="Write unit tests for these functions", input_file="api.py")

# Find log files to analyze
clog_file_list(directory="./logs", pattern="*.log")
```

### Verified Working

Both tools have been tested and work correctly:
- `local_llm`: Handles creative prompts, summarization, and text processing
- `clog`: Compresses logs, identifies error patterns, and provides LLM analysis

### Remove MCP Server

```bash
claude mcp remove llm-offload -s user
```
