# LLM Offload for Claude Code

Offload token-heavy tasks to a local LLM to save context window space and reduce costs. Claude reads files and command outputs locally through MCP tools - you never paste large logs or outputs into the chat.

## How It Works

Once installed, just ask Claude to analyze logs or process files. Claude uses local MCP tools that read content on your machine - the raw data never enters the conversation, only the processed summary.

```
You: "Analyze the logs in /var/log/app.log, what caused the crash?"
Claude: [Uses clog tool locally, returns summary]

You: "Check the docker logs for my api container"
Claude: [Uses pipe_to_clog tool, runs docker logs, returns analysis]

You: "Write a changelog from the last 5 commits"
Claude: [Uses pipe_to_llm tool, runs git diff, returns changelog]
```

## Installation

```bash
# Clone and run installer
git clone https://github.com/ersagunkuruca/llm-offload.git
cd llm-offload
./install.sh
```

This installs:
- **Ollama** - Local LLM runtime
- **Qwen2.5 3B** - Default model for analysis
- **clog** - Log compression CLI
- **MCP Server** - Claude Code integration

### Verify Installation

```bash
ollama list                 # Check Ollama is running
claude mcp list             # Check MCP server is registered
```

## What You Can Ask Claude

### Analyze Log Files

```
"Analyze /var/log/nginx/error.log"
"What errors are in ./app.log?"
"Check ~/logs/api.log for OOM errors"
```

Claude uses the `clog` tool which:
- Reads the file locally (you never see raw logs)
- Compresses and deduplicates log lines
- Identifies error patterns and context
- Returns a structured summary with LLM analysis

### Analyze Container/Service Logs

```
"Check the docker logs for myapp container"
"What's in the kubernetes logs for the api deployment?"
"Analyze journalctl logs for nginx from the last hour"
```

Claude uses the `pipe_to_clog` tool which:
- Runs the log command (docker logs, kubectl logs, journalctl, etc.)
- Pipes output through clog compression
- Returns analysis without you seeing raw output

Example commands Claude might run:
- `docker logs myapp --tail 1000`
- `kubectl logs deployment/api`
- `journalctl -u nginx --since "1 hour ago"`

### Process Command Output

```
"Write a changelog from the last 5 commits"
"Summarize the git diff"
"What dependencies are in package.json?"
```

Claude uses the `pipe_to_llm` tool which:
- Runs your command (git diff, cat, etc.)
- Sends output to local Ollama
- Returns the LLM's response

You never see the raw command output - just the processed result.

### Process Files with Local LLM

```
"Summarize the README in that repo"
"Add docstrings to utils.py"
"Convert config.yaml to JSON format"
```

Claude uses the `local_llm` tool which:
- Reads the file locally
- Processes it with Ollama
- Returns the result

Good for token-heavy simple tasks: summarization, format conversion, boilerplate generation.

### Find Log Files

```
"What log files are in ./logs?"
"Find all .log files in /var/log"
```

Claude uses `clog_file_list` to find and list log files with their sizes.

## MCP Tools Reference

| Tool | What It Does | When Claude Uses It |
|------|--------------|---------------------|
| `clog` | Analyze log files | "Analyze this log file" |
| `pipe_to_clog` | Run command, analyze output as logs | "Check docker/kubectl/journalctl logs" |
| `pipe_to_llm` | Run command, process output with LLM | "Summarize git diff", "Write changelog" |
| `local_llm` | Process file with LLM | "Summarize this file", "Add docstrings" |
| `clog_file_list` | List log files in directory | "What log files exist?" |

### Key Design Principle

All tools read files and run commands locally - Claude never sees the raw content, only the processed result. This is what saves tokens.

### Verified Working

All 5 MCP tools have been tested:
- `clog_file_list` - Lists log files with sizes
- `local_llm` - Runs prompts through Ollama
- `pipe_to_llm` - Pipes command output to LLM
- `clog` - Analyzes log files with compression + LLM summary
- `pipe_to_clog` - Pipes command output through clog

## Troubleshooting

### "Ollama not running"
```bash
brew services start ollama   # macOS
# or
ollama serve                 # manual start
```

### "Model not found"
```bash
ollama pull qwen2.5:3b-instruct
```

### Slow first run
Normal - the model loads into memory. Subsequent runs are faster.

### Out of memory
Try a smaller model:
```bash
ollama pull qwen2.5:3b-instruct
```

### MCP server not working
```bash
# Check it's registered
claude mcp list

# Re-register if needed
claude mcp add --transport stdio llm-offload --scope user -- \
  ~/llm-offload/mcp-server/venv/bin/python \
  ~/llm-offload/mcp-server/server.py
```

---

# Manual CLI Usage (Optional)

You can also use these tools directly from the command line, without Claude.

## clog Command

Compress and analyze log files:

```bash
# Basic usage
clog app.log

# With focused question
clog -p "What caused the OOM error?" app.log

# Pipe from commands
docker logs myapp | clog
kubectl logs deployment/api | clog
journalctl -u nginx --since "1 hour ago" | clog

# Compression only (no LLM)
clog --no-llm app.log
```

### clog Options

```
Usage: clog [OPTIONS] [FILES...]

Options:
  --no-llm             Skip LLM summary, only compression
  --max-lines N        Process only first N lines
  --around-error K     Keep K lines around errors (default: 5)
  --json               Output as JSON
  --model NAME         Use different Ollama model
  -p, --prompt TEXT    Focus LLM analysis on a specific question
```

### What clog Does

**Compression Pass:**
1. Deduplicates consecutive identical lines
2. Normalizes timestamps, UUIDs, IDs, IPs to placeholders
3. Templates similar lines and counts occurrences
4. Preserves error context windows (lines around ERROR/WARN/EXCEPTION)
5. Tracks timeline from first to last timestamp

**LLM Summary:**
- What happened (1-3 bullets)
- Top error signatures
- Timeline
- Most likely root causes
- Suggested debugging steps

## Direct Ollama Usage

```bash
# Summarize any text
cat document.txt | ollama run qwen2.5:3b-instruct "Summarize this"

# Generate commit message
git diff --staged | ollama run qwen2.5:3b-instruct "Write a commit message"

# Code explanation
cat script.py | ollama run qwen2.5:3b-instruct "What does this do?"
```

## Streaming Logs (logcat, Xcode, etc.)

clog processes logs in batch mode. For streaming tools:

```bash
# Android logcat - capture then analyze
adb logcat -d > logcat.txt && clog logcat.txt

# With timeout
timeout 30 adb logcat > logcat.txt; clog logcat.txt

# Tail with limit
adb logcat -t 1000 | clog
kubectl logs --tail=500 deployment/api | clog
```

## Example clog Output

```markdown
# Log Compression Summary

**Original lines:** 15000
**Unique templates:** 47
**Compression ratio:** 319.1x

## Error Signatures
| Count | First | Last | Template |
|-------|-------|------|----------|
| 127 | L1205 | L14892 | `ERROR: Connection to <IP>:<PORT> failed` |

## LLM Analysis
### What Happened
- Application started normally at 10:30
- Connection failures began at 10:35
- Service degraded to 50% capacity

### Most Likely Root Causes
1. Database connection pool exhaustion
2. Network partition to database server
```

---

# Advanced

## Add More Models

```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull codellama:7b  # For code-heavy logs
```

## Remove MCP Server

```bash
claude mcp remove llm-offload -s user
```

## File Locations

- MCP Server: `~/llm-offload/mcp-server/server.py`
- clog CLI: `~/bin/clog`
- Ollama models: `~/.ollama/models/`

## Requirements

- macOS or Linux
- Python 3.10+
- ~5GB disk space for model
- 8GB+ RAM recommended
