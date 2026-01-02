#!/usr/bin/env python3
"""
MCP Server for llm-offload tools.
Exposes clog and local LLM as native Claude Code tools.
"""

import subprocess
import sys
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("llm-offload")

@mcp.tool()
def clog(
    file_path: str = None,
    log_content: str = None,
    prompt: str = None,
    no_llm: bool = False,
    max_lines: int = None,
    around_error: int = 5
) -> str:
    """Compress and summarize log files using local LLM.

    Reduces log files to structured summaries, saving context window space.
    Uses deterministic compression + local Qwen2.5 7B for analysis.

    Args:
        file_path: Path to log file to analyze
        log_content: Raw log content (alternative to file_path)
        prompt: Specific question to focus the analysis on (e.g., "What caused the OOM?")
        no_llm: If True, only run compression without LLM summary
        max_lines: Maximum lines to process
        around_error: Lines to keep around errors (default: 5)

    Returns:
        Compressed log summary with optional LLM analysis
    """
    # Build command
    cmd = ["clog"]

    if prompt:
        cmd.extend(["-p", prompt])
    if no_llm:
        cmd.append("--no-llm")
    if max_lines:
        cmd.extend(["--max-lines", str(max_lines)])
    if around_error != 5:
        cmd.extend(["--around-error", str(around_error)])

    try:
        if file_path:
            cmd.append(file_path)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PATH": f"{os.environ.get('HOME')}/bin:{os.environ.get('PATH', '')}"}
            )
        elif log_content:
            result = subprocess.run(
                cmd,
                input=log_content,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PATH": f"{os.environ.get('HOME')}/bin:{os.environ.get('PATH', '')}"}
            )
        else:
            return "Error: Either file_path or log_content must be provided"

        if result.returncode != 0:
            return f"Error running clog: {result.stderr}"

        return result.stdout

    except subprocess.TimeoutExpired:
        return "Error: clog timed out after 120 seconds"
    except FileNotFoundError:
        return "Error: clog not found. Run install.sh first."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def local_llm(
    prompt: str,
    input_text: str = None,
    input_file: str = None,
    model: str = "qwen2.5:7b-instruct"
) -> str:
    """Run a prompt through the local Ollama LLM.

    Use this for token-heavy simple tasks: summarization, format conversion,
    boilerplate generation, bulk transformations.

    Args:
        prompt: The instruction/question for the LLM
        input_text: Optional text to process (sent before prompt)
        input_file: Optional file path to read and process
        model: Ollama model to use (default: qwen2.5:7b-instruct)

    Returns:
        LLM response
    """
    try:
        # Build the full prompt
        full_prompt = ""

        if input_file:
            try:
                with open(input_file, 'r', errors='replace') as f:
                    full_prompt = f.read() + "\n\n"
            except Exception as e:
                return f"Error reading file: {e}"
        elif input_text:
            full_prompt = input_text + "\n\n"

        full_prompt += prompt

        result = subprocess.run(
            ["ollama", "run", model],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return f"Error calling Ollama: {result.stderr}"

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        return "Error: Ollama timed out after 120 seconds"
    except FileNotFoundError:
        return "Error: Ollama not found. Is it installed and running?"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def clog_file_list(directory: str = ".", pattern: str = "*.log") -> str:
    """List log files in a directory that can be analyzed with clog.

    Args:
        directory: Directory to search (default: current directory)
        pattern: Glob pattern to match (default: *.log)

    Returns:
        List of log files with their sizes
    """
    import glob
    from pathlib import Path

    try:
        search_path = Path(directory).expanduser()
        files = list(search_path.glob(f"**/{pattern}"))

        if not files:
            return f"No files matching '{pattern}' found in {directory}"

        # Sort by size descending
        file_info = []
        for f in files:
            try:
                size = f.stat().st_size
                lines = sum(1 for _ in open(f, 'rb'))
                file_info.append((f, size, lines))
            except:
                continue

        file_info.sort(key=lambda x: x[1], reverse=True)

        result = f"Found {len(file_info)} log files:\n\n"
        result += "| File | Size | Lines |\n|------|------|-------|\n"
        for f, size, lines in file_info[:20]:
            size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
            result += f"| {f} | {size_str} | {lines:,} |\n"

        if len(file_info) > 20:
            result += f"\n... and {len(file_info) - 20} more files"

        return result

    except Exception as e:
        return f"Error listing files: {e}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
