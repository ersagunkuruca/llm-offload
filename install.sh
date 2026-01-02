#!/bin/bash
#
# Log Compression Pipeline Installer
# Installs: Ollama, Qwen2.5 7B model, clog CLI tool
#
# Usage: curl -fsSL <url>/install.sh | bash
#    or: ./install.sh
#

set -e

echo "==================================="
echo "Log Compression Pipeline Installer"
echo "==================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Darwin) OS_TYPE="macos" ;;
    Linux)  OS_TYPE="linux" ;;
    *)      error "Unsupported OS: $OS" ;;
esac

echo "Detected OS: $OS_TYPE"
echo

# Step 1: Install Ollama
echo "Step 1: Checking Ollama..."
if command -v ollama &> /dev/null; then
    success "Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
    echo "Installing Ollama..."
    if [ "$OS_TYPE" = "macos" ]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            error "Homebrew not found. Please install Ollama manually: https://ollama.ai"
        fi
    else
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
    success "Ollama installed"
fi
echo

# Step 2: Start Ollama service
echo "Step 2: Starting Ollama service..."
if [ "$OS_TYPE" = "macos" ]; then
    if brew services list | grep -q "ollama.*started"; then
        success "Ollama service already running"
    else
        brew services start ollama 2>/dev/null || ollama serve &>/dev/null &
        sleep 3
        success "Ollama service started"
    fi
else
    if pgrep -x ollama > /dev/null; then
        success "Ollama service already running"
    else
        ollama serve &>/dev/null &
        sleep 3
        success "Ollama service started"
    fi
fi
echo

# Step 3: Pull the model
echo "Step 3: Pulling Qwen2.5 7B model (this may take a few minutes)..."
if ollama list 2>/dev/null | grep -q "qwen2.5:7b-instruct"; then
    success "Model already downloaded"
else
    ollama pull qwen2.5:7b-instruct
    success "Model downloaded"
fi
echo

# Step 4: Install clog
echo "Step 4: Installing clog CLI tool..."

# Create the clog directory
INSTALL_DIR="$HOME/.local/share/clog"
mkdir -p "$INSTALL_DIR"

# Download clog.py
cat > "$INSTALL_DIR/clog.py" << 'CLOG_EOF'
#!/usr/bin/env python3
"""
clog - Compress and summarize logs for efficient context usage.
"""

import argparse
import json
import re
import sys
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

PATTERNS = {
    'timestamp_iso': (
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?',
        '<TIMESTAMP>'
    ),
    'timestamp_common': (
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
        '<TIMESTAMP>'
    ),
    'timestamp_unix': (r'\b1[6-9]\d{8,11}\b', '<UNIX_TS>'),
    'uuid': (
        r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
        '<UUID>'
    ),
    'hex_id': (r'\b[0-9a-fA-F]{16,}\b', '<HEX_ID>'),
    'numeric_id': (r'\b\d{4,}\b', '<ID>'),
    'ip_address': (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>'),
    'port': (r':\d{2,5}\b', ':<PORT>'),
    'temp_path': (r'/tmp/[^\s]+', '<TEMP_PATH>'),
    'memory_addr': (r'0x[0-9a-fA-F]+', '<ADDR>'),
    'bracket_id': (r'\[[0-9a-fA-F-]{8,}\]', '[<REQ_ID>]'),
}

ERROR_PATTERNS = [
    r'\bERROR\b', r'\bFAILED?\b', r'\bFATAL\b', r'\bCRITICAL\b',
    r'\bWARN(?:ING)?\b', r'\bEXCEPTION\b', r'\bTraceback\b',
    r'\bpanic\b', r'\bsegfault\b', r'\bSEGV\b', r'\bAborted\b',
    r'\bTimeout\b', r'\bConnection refused\b', r'\bPermission denied\b',
]
ERROR_REGEX = re.compile('|'.join(ERROR_PATTERNS), re.IGNORECASE)

@dataclass
class LogTemplate:
    template: str
    count: int = 0
    first_seen: int = 0
    last_seen: int = 0
    exemplar: str = ""
    is_error: bool = False

@dataclass
class CompressedOutput:
    original_lines: int = 0
    compressed_lines: int = 0
    templates: list = field(default_factory=list)
    error_windows: list = field(default_factory=list)
    timeline: list = field(default_factory=list)

def normalize_line(line: str) -> str:
    result = line
    for name, (pattern, replacement) in PATTERNS.items():
        result = re.sub(pattern, replacement, result)
    return result

def is_error_line(line: str) -> bool:
    return bool(ERROR_REGEX.search(line))

def extract_timestamp(line: str) -> Optional[str]:
    match = re.match(r'^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', line)
    if match:
        return match.group(1)
    match = re.match(r'^((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})', line)
    if match:
        return match.group(1)
    match = re.match(r'^\[([^\]]+)\]', line)
    if match:
        return match.group(1)
    return None

def compress_logs(lines, around_error=5, max_lines=None):
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
    output = CompressedOutput(original_lines=len(lines))
    error_indices = [i for i, line in enumerate(lines) if is_error_line(line)]
    error_window_lines = set()
    for idx in error_indices:
        start = max(0, idx - around_error)
        end = min(len(lines), idx + around_error + 1)
        for i in range(start, end):
            error_window_lines.add(i)
    template_map = {}
    prev_line = None
    for i, line in enumerate(lines):
        line = line.rstrip()
        if not line:
            continue
        if line == prev_line:
            continue
        prev_line = line
        normalized = normalize_line(line)
        is_error = is_error_line(line)
        if normalized not in template_map:
            template_map[normalized] = LogTemplate(
                template=normalized, count=1, first_seen=i + 1,
                last_seen=i + 1, exemplar=line, is_error=is_error
            )
        else:
            template_map[normalized].count += 1
            template_map[normalized].last_seen = i + 1
    error_windows = []
    if error_indices:
        windows = []
        for idx in error_indices:
            start = max(0, idx - around_error)
            end = min(len(lines), idx + around_error + 1)
            if windows and start <= windows[-1][1]:
                windows[-1] = (windows[-1][0], end)
            else:
                windows.append((start, end))
        for start, end in windows:
            window_lines = [lines[i].rstrip() for i in range(start, end) if lines[i].strip()]
            if window_lines:
                error_windows.append({
                    'start_line': start + 1, 'end_line': end, 'lines': window_lines
                })
    timeline = []
    first_ts, last_ts = None, None
    for line in lines:
        ts = extract_timestamp(line)
        if ts:
            if not first_ts:
                first_ts = ts
            last_ts = ts
    if first_ts:
        timeline.append({'start': first_ts, 'end': last_ts or first_ts})
    sorted_templates = sorted(template_map.values(), key=lambda t: (-t.count, t.first_seen))
    output.templates = [asdict(t) for t in sorted_templates]
    output.error_windows = error_windows
    output.timeline = timeline
    output.compressed_lines = len(template_map)
    return output

def format_compressed_output(compressed, as_json=False):
    if as_json:
        return json.dumps(asdict(compressed), indent=2)
    lines = []
    lines.append(f"# Log Compression Summary")
    lines.append(f"")
    lines.append(f"**Original lines:** {compressed.original_lines}")
    lines.append(f"**Unique templates:** {compressed.compressed_lines}")
    lines.append(f"**Compression ratio:** {compressed.original_lines / max(1, compressed.compressed_lines):.1f}x")
    lines.append("")
    if compressed.timeline:
        lines.append("## Timeline")
        for t in compressed.timeline:
            lines.append(f"- Start: {t['start']}")
            lines.append(f"- End: {t['end']}")
        lines.append("")
    error_templates = [t for t in compressed.templates if t['is_error']]
    if error_templates:
        lines.append("## Error Signatures")
        lines.append("")
        lines.append("| Count | First | Last | Template |")
        lines.append("|-------|-------|------|----------|")
        for t in error_templates[:20]:
            template_short = t['template'][:80] + "..." if len(t['template']) > 80 else t['template']
            lines.append(f"| {t['count']} | L{t['first_seen']} | L{t['last_seen']} | `{template_short}` |")
        lines.append("")
    if compressed.error_windows:
        lines.append("## Error Context Windows")
        lines.append("")
        for i, window in enumerate(compressed.error_windows[:5]):
            lines.append(f"### Window {i+1} (lines {window['start_line']}-{window['end_line']})")
            lines.append("```")
            for wl in window['lines'][:15]:
                lines.append(wl)
            if len(window['lines']) > 15:
                lines.append(f"... ({len(window['lines']) - 15} more lines)")
            lines.append("```")
            lines.append("")
    normal_templates = [t for t in compressed.templates if not t['is_error']]
    if normal_templates:
        lines.append("## Top Message Templates")
        lines.append("")
        lines.append("| Count | Template |")
        lines.append("|-------|----------|")
        for t in normal_templates[:15]:
            template_short = t['template'][:80] + "..." if len(t['template']) > 80 else t['template']
            lines.append(f"| {t['count']} | `{template_short}` |")
        lines.append("")
    return "\n".join(lines)

def call_ollama(prompt, model="qwen2.5:7b-instruct"):
    try:
        result = subprocess.run(
            ["ollama", "run", model], input=prompt,
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return f"Error calling Ollama: {result.stderr}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: Ollama request timed out after 120 seconds"
    except FileNotFoundError:
        return "Error: Ollama not found. Please install Ollama first."
    except Exception as e:
        return f"Error calling Ollama: {e}"

def generate_llm_summary(compressed, model="qwen2.5:7b-instruct", user_prompt=None):
    compressed_text = format_compressed_output(compressed, as_json=False)
    if user_prompt:
        prompt = f"""You are a log analysis expert. Analyze the following compressed log data to answer this specific question:

**USER QUESTION:** {user_prompt}

---

Here is the compressed log data:

{compressed_text}

---

Focus your analysis on answering the user's question. Be concise and provide evidence from the logs."""
    else:
        prompt = f"""You are a log analysis expert. Analyze the following compressed log data and provide a summary in EXACTLY this markdown format:

## What Happened
- [1-3 bullet points describing the main events/activities in the logs]

## Top Error Signatures

| Signature | Count | First Seen | Last Seen | Exemplar |
|-----------|-------|------------|-----------|----------|
[Table of error signatures, max 10 rows]

## Timeline
[Coarse timeline of events]

## Most Likely Root Causes
1. [Ranked cause with evidence from logs]

## Suggested Next Debugging Steps
1. [Most important action to take]

---

Here is the compressed log data:

{compressed_text}

Provide your analysis now. Be concise and focus on actionable insights. If there are no errors, say so clearly."""
    return call_ollama(prompt, model)

def main():
    parser = argparse.ArgumentParser(
        description='Compress and summarize logs for efficient context usage.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  clog app.log                              # Compress and summarize app.log
  clog *.log                                # Process multiple log files
  cat app.log | clog                        # Read from stdin
  clog --no-llm app.log                     # Only compression, no LLM summary
  clog --json app.log                       # Output as JSON
  clog --around-error 10 app.log            # Keep 10 lines around errors
  clog --model llama3:8b app.log            # Use a different model
  clog -p "Why did the DB fail?" app.log    # Focus analysis on specific question
"""
    )
    parser.add_argument('files', nargs='*', help='Log files to process')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM summary')
    parser.add_argument('--max-lines', type=int, default=None, help='Maximum lines to process')
    parser.add_argument('--around-error', '-K', type=int, default=5, help='Lines around errors (default: 5)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--model', type=str, default='qwen2.5:7b-instruct', help='Ollama model')
    parser.add_argument('--compression-only', action='store_true', help='Alias for --no-llm')
    parser.add_argument('--prompt', '-p', type=str, default=None, help='Custom prompt/question to focus the LLM analysis')
    args = parser.parse_args()
    lines = []
    if args.files:
        for filepath in args.files:
            try:
                with open(filepath, 'r', errors='replace') as f:
                    lines.extend(f.readlines())
            except Exception as e:
                print(f"Error reading {filepath}: {e}", file=sys.stderr)
                sys.exit(1)
    else:
        if sys.stdin.isatty():
            print("Error: No input. Provide log files or pipe data via stdin.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        lines = sys.stdin.readlines()
    if not lines:
        print("Error: No log lines to process.", file=sys.stderr)
        sys.exit(1)
    compressed = compress_logs(lines, around_error=args.around_error, max_lines=args.max_lines)
    skip_llm = args.no_llm or args.compression_only
    if args.json:
        output = asdict(compressed)
        if not skip_llm:
            output['llm_summary'] = generate_llm_summary(compressed, args.model, args.prompt)
        print(json.dumps(output, indent=2))
    else:
        print(format_compressed_output(compressed))
        if not skip_llm:
            print("\n" + "=" * 60)
            if args.prompt:
                print(f"## LLM Analysis: {args.prompt}")
            else:
                print("## LLM Analysis")
            print("=" * 60 + "\n")
            summary = generate_llm_summary(compressed, args.model, args.prompt)
            print(summary)

if __name__ == '__main__':
    main()
CLOG_EOF

chmod +x "$INSTALL_DIR/clog.py"

# Create wrapper script
mkdir -p "$HOME/bin"
cat > "$HOME/bin/clog" << EOF
#!/bin/bash
exec python3 "$INSTALL_DIR/clog.py" "\$@"
EOF
chmod +x "$HOME/bin/clog"

# Add to PATH if needed
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_RC="$HOME/.bash_profile"
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q 'HOME/bin' "$SHELL_RC" 2>/dev/null; then
        echo 'export PATH="$HOME/bin:$PATH"' >> "$SHELL_RC"
        warn "Added ~/bin to PATH in $SHELL_RC - restart your shell or run: source $SHELL_RC"
    fi
fi

success "clog installed to ~/bin/clog"
echo

# Step 5: Install reference docs (optional)
echo "Step 5: Installing reference documentation..."
mkdir -p "$HOME/.claude/agents"

cat > "$HOME/.claude/agents/log-summarizer.md" << 'AGENT_EOF'
# Log Summarizer Workflow

To save tokens when analyzing logs:

1. Run `clog app.log` locally FIRST
2. Copy the compressed output
3. Paste only the summary to Claude

The clog tool:
- Deduplicates and normalizes log lines
- Groups into templates with counts
- Preserves error context windows
- Generates LLM analysis locally (free, uses Qwen2.5 7B)
- Supports focused questions with -p flag
AGENT_EOF

cat > "$HOME/.claude/agents/local-agent.md" << 'AGENT_EOF'
# Local LLM Agent

Use the local Ollama model for:
- Summarization
- Data transformation
- Simple analysis
- Bulk operations

Commands:
  ollama run qwen2.5:7b-instruct "Your prompt"
  cat file.txt | ollama run qwen2.5:7b-instruct "Summarize this"
  clog app.log                          # Log compression + summary
  clog -p "What caused the crash?" log  # Focused analysis
AGENT_EOF

success "Reference docs installed to ~/.claude/agents/"
echo

# Verification
echo "==================================="
echo "Installation Complete!"
echo "==================================="
echo
echo "Installed components:"
echo "  - Ollama LLM runtime"
echo "  - Qwen2.5 7B Instruct model"
echo "  - clog CLI tool"
echo
echo "Quick test:"
echo "  echo '2024-01-15T10:00:00Z ERROR Connection failed' | clog --no-llm"
echo
echo "Usage:"
echo "  clog app.log                        # Compress + summarize"
echo "  clog --no-llm app.log               # Compression only"
echo "  clog -p 'What caused the OOM?' log  # Focused question"
echo "  docker logs app | clog              # From stdin"
echo
echo "To save Claude tokens:"
echo "  1. Run: clog app.log > summary.txt"
echo "  2. Paste summary.txt content to Claude"
echo
if [ -n "$SHELL_RC" ]; then
    echo "Note: Run 'source $SHELL_RC' or restart your terminal to use clog"
fi
