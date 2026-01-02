#!/usr/bin/env python3
"""
clog - Compress and summarize logs for efficient context usage.

This tool performs:
1. Deterministic compression (dedupe, normalize, template, keep exemplars)
2. LLM-powered summary with structured markdown output
"""

import argparse
import json
import re
import sys
import subprocess
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


# ============================================================================
# Regex patterns for normalization
# ============================================================================

PATTERNS = {
    # ISO timestamps: 2024-01-15T10:30:45.123Z or 2024-01-15 10:30:45
    'timestamp_iso': (
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?',
        '<TIMESTAMP>'
    ),
    # Common log timestamps: Jan 15 10:30:45 or 15/Jan/2024:10:30:45
    'timestamp_common': (
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
        '<TIMESTAMP>'
    ),
    # Unix timestamps (10-13 digits)
    'timestamp_unix': (r'\b1[6-9]\d{8,11}\b', '<UNIX_TS>'),
    # UUIDs
    'uuid': (
        r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
        '<UUID>'
    ),
    # Hex IDs (16+ chars)
    'hex_id': (r'\b[0-9a-fA-F]{16,}\b', '<HEX_ID>'),
    # Numeric IDs (standalone numbers 4+ digits)
    'numeric_id': (r'\b\d{4,}\b', '<ID>'),
    # IP addresses
    'ip_address': (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>'),
    # Ports
    'port': (r':\d{2,5}\b', ':<PORT>'),
    # File paths with variable components
    'temp_path': (r'/tmp/[^\s]+', '<TEMP_PATH>'),
    # Memory addresses
    'memory_addr': (r'0x[0-9a-fA-F]+', '<ADDR>'),
    # Session/request IDs in brackets
    'bracket_id': (r'\[[0-9a-fA-F-]{8,}\]', '[<REQ_ID>]'),
}

# Error indicators to look for
ERROR_PATTERNS = [
    r'\bERROR\b',
    r'\bFAILED?\b',
    r'\bFATAL\b',
    r'\bCRITICAL\b',
    r'\bWARN(?:ING)?\b',
    r'\bEXCEPTION\b',
    r'\bTraceback\b',
    r'\bpanic\b',
    r'\bsegfault\b',
    r'\bSEGV\b',
    r'\bAborted\b',
    r'\bTimeout\b',
    r'\bConnection refused\b',
    r'\bPermission denied\b',
]

ERROR_REGEX = re.compile('|'.join(ERROR_PATTERNS), re.IGNORECASE)


@dataclass
class LogTemplate:
    """A normalized log template with count and exemplars."""
    template: str
    count: int = 0
    first_seen: int = 0
    last_seen: int = 0
    exemplar: str = ""
    is_error: bool = False


@dataclass
class CompressedOutput:
    """Output from the compression pass."""
    original_lines: int = 0
    compressed_lines: int = 0
    templates: list = field(default_factory=list)
    error_windows: list = field(default_factory=list)
    timeline: list = field(default_factory=list)


def normalize_line(line: str) -> str:
    """Normalize a log line by replacing variable parts with placeholders."""
    result = line
    for name, (pattern, replacement) in PATTERNS.items():
        result = re.sub(pattern, replacement, result)
    return result


def is_error_line(line: str) -> bool:
    """Check if a line contains error indicators."""
    return bool(ERROR_REGEX.search(line))


def extract_timestamp(line: str) -> Optional[str]:
    """Try to extract a timestamp from the beginning of a line."""
    # Try ISO format first
    match = re.match(r'^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', line)
    if match:
        return match.group(1)
    # Try common syslog format
    match = re.match(r'^((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})', line)
    if match:
        return match.group(1)
    # Try bracketed timestamp
    match = re.match(r'^\[([^\]]+)\]', line)
    if match:
        return match.group(1)
    return None


def compress_logs(lines: list[str], around_error: int = 5, max_lines: Optional[int] = None) -> CompressedOutput:
    """
    Compress logs using deterministic methods:
    1. Dedupe consecutive identical lines
    2. Normalize and group into templates
    3. Keep error context windows
    4. Track timeline
    """
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]

    output = CompressedOutput(original_lines=len(lines))

    # Step 1: Find all error line indices
    error_indices = [i for i, line in enumerate(lines) if is_error_line(line)]

    # Step 2: Build error windows (lines to always keep verbatim)
    error_window_lines = set()
    for idx in error_indices:
        start = max(0, idx - around_error)
        end = min(len(lines), idx + around_error + 1)
        for i in range(start, end):
            error_window_lines.add(i)

    # Step 3: Group lines into templates
    template_map: dict[str, LogTemplate] = {}
    prev_line = None
    consecutive_count = 0

    for i, line in enumerate(lines):
        line = line.rstrip()
        if not line:
            continue

        # Track consecutive duplicates
        if line == prev_line:
            consecutive_count += 1
            continue
        elif consecutive_count > 0:
            # Record the deduped run
            pass

        prev_line = line
        consecutive_count = 0

        # Normalize and template
        normalized = normalize_line(line)
        is_error = is_error_line(line)

        if normalized not in template_map:
            template_map[normalized] = LogTemplate(
                template=normalized,
                count=1,
                first_seen=i + 1,
                last_seen=i + 1,
                exemplar=line,
                is_error=is_error
            )
        else:
            template_map[normalized].count += 1
            template_map[normalized].last_seen = i + 1

    # Step 4: Collect error windows
    error_windows = []
    if error_indices:
        # Merge overlapping windows
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
                    'start_line': start + 1,
                    'end_line': end,
                    'lines': window_lines
                })

    # Step 5: Build timeline (coarse-grained)
    timeline = []
    first_ts = None
    last_ts = None
    for line in lines:
        ts = extract_timestamp(line)
        if ts:
            if not first_ts:
                first_ts = ts
            last_ts = ts
    if first_ts:
        timeline.append({'start': first_ts, 'end': last_ts or first_ts})

    # Sort templates by count descending
    sorted_templates = sorted(template_map.values(), key=lambda t: (-t.count, t.first_seen))

    output.templates = [asdict(t) for t in sorted_templates]
    output.error_windows = error_windows
    output.timeline = timeline
    output.compressed_lines = len(template_map)

    return output


def format_compressed_output(compressed: CompressedOutput, as_json: bool = False) -> str:
    """Format the compressed output for display or LLM input."""
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

    # Error templates first
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

    # Error windows
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

    # Top non-error templates
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


def call_ollama(prompt: str, model: str = "qwen2.5:7b-instruct") -> str:
    """Call Ollama API to get LLM summary."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120
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


def generate_llm_summary(compressed: CompressedOutput, model: str = "qwen2.5:7b-instruct", user_prompt: str = None) -> str:
    """Generate an LLM summary of the compressed logs."""

    # Build prompt
    compressed_text = format_compressed_output(compressed, as_json=False)

    if user_prompt:
        # User provided a specific question/focus
        prompt = f"""You are a log analysis expert. Analyze the following compressed log data to answer this specific question:

**USER QUESTION:** {user_prompt}

---

Here is the compressed log data:

{compressed_text}

---

Focus your analysis on answering the user's question. Be concise and provide evidence from the logs."""
    else:
        # Default structured analysis
        prompt = f"""You are a log analysis expert. Analyze the following compressed log data and provide a summary in EXACTLY this markdown format:

## What Happened
- [1-3 bullet points describing the main events/activities in the logs]

## Top Error Signatures

| Signature | Count | First Seen | Last Seen | Exemplar |
|-----------|-------|------------|-----------|----------|
[Table of error signatures, max 10 rows]

## Timeline
[Coarse timeline of events, e.g., "10:30-10:35: Normal operation", "10:35-10:40: Error spike"]

## Most Likely Root Causes
1. [Ranked cause with evidence from logs]
2. [Second cause if applicable]
3. [Third cause if applicable]

## Suggested Next Debugging Steps
1. [Most important action to take]
2. [Second action]
3. [Third action]

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
  clog --prompt "What caused the OOM?"      # Custom analysis prompt
"""
    )

    parser.add_argument('files', nargs='*', help='Log files to process (reads stdin if none)')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM summary, only show compression')
    parser.add_argument('--max-lines', type=int, default=None, help='Maximum lines to process')
    parser.add_argument('--around-error', '-K', type=int, default=5, help='Lines to keep around errors (default: 5)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--model', type=str, default='qwen2.5:7b-instruct', help='Ollama model to use (default: qwen2.5:7b-instruct)')
    parser.add_argument('--compression-only', action='store_true', help='Alias for --no-llm')
    parser.add_argument('--prompt', '-p', type=str, default=None, help='Custom prompt/question to focus the LLM analysis')

    args = parser.parse_args()

    # Read input
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

    # Compress
    compressed = compress_logs(lines, around_error=args.around_error, max_lines=args.max_lines)

    # Output
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
