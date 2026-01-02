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
    file_path: str,
    prompt: str = None,
    no_llm: bool = False,
    max_lines: int = None,
    around_error: int = 5
) -> str:
    """Compress and summarize log files using local LLM.

    The file is read locally by clog, not by Claude - this saves context window space.
    Only the compressed summary is returned.

    Args:
        file_path: Path to log file to analyze
        prompt: Specific question to focus the analysis on (e.g., "What caused the OOM?")
        no_llm: If True, only run compression without LLM summary
        max_lines: Maximum lines to process
        around_error: Lines to keep around errors (default: 5)

    Returns:
        Compressed log summary with optional LLM analysis
    """
    cmd = ["clog"]

    if prompt:
        cmd.extend(["-p", prompt])
    if no_llm:
        cmd.append("--no-llm")
    if max_lines:
        cmd.extend(["--max-lines", str(max_lines)])
    if around_error != 5:
        cmd.extend(["--around-error", str(around_error)])

    cmd.append(file_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PATH": f"{os.environ.get('HOME')}/bin:{os.environ.get('PATH', '')}"}
        )

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
    input_file: str = None,
    model: str = "qwen2.5:7b-instruct"
) -> str:
    """Run a prompt through the local Ollama LLM.

    Use this for token-heavy simple tasks: summarization, format conversion,
    boilerplate generation, bulk transformations.

    The file is read locally, not by Claude - this saves context window space.

    Args:
        prompt: The instruction/question for the LLM
        input_file: Optional file path to read and process (read locally, saves tokens)
        model: Ollama model to use (default: qwen2.5:7b-instruct)

    Returns:
        LLM response
    """
    try:
        full_prompt = ""

        if input_file:
            try:
                with open(input_file, 'r', errors='replace') as f:
                    full_prompt = f.read() + "\n\n"
            except Exception as e:
                return f"Error reading file: {e}"

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
def pipe_to_llm(
    command: str,
    prompt: str,
    model: str = "qwen2.5:7b-instruct"
) -> str:
    """Run a shell command and pipe its output to the local LLM.

    The command output is never seen by Claude - only the LLM's response is returned.
    This saves tokens when processing large command outputs.

    Args:
        command: Shell command to run (e.g., "git diff", "docker logs app")
        prompt: Instruction for the LLM on how to process the output
        model: Ollama model to use (default: qwen2.5:7b-instruct)

    Returns:
        LLM response based on command output
    """
    try:
        # Run the command
        cmd_result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        cmd_output = cmd_result.stdout
        if cmd_result.stderr:
            cmd_output += "\n\nSTDERR:\n" + cmd_result.stderr

        if not cmd_output.strip():
            return "Command produced no output"

        # Pipe to LLM
        full_prompt = cmd_output + "\n\n" + prompt

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
        return "Error: Command or Ollama timed out"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def pipe_to_clog(
    command: str,
    prompt: str = None,
    no_llm: bool = False
) -> str:
    """Run a shell command and pipe its output through clog for log analysis.

    The command output is never seen by Claude - only the compressed summary is returned.
    Use this for log-producing commands like docker logs, kubectl logs, journalctl, etc.

    Args:
        command: Shell command to run (e.g., "docker logs app", "kubectl logs pod")
        prompt: Optional question to focus the analysis (e.g., "What caused the crash?")
        no_llm: If True, only run compression without LLM summary

    Returns:
        Compressed log summary with optional LLM analysis
    """
    try:
        # Run the command
        cmd_result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        cmd_output = cmd_result.stdout
        if cmd_result.stderr:
            cmd_output += "\n" + cmd_result.stderr

        if not cmd_output.strip():
            return "Command produced no output"

        # Pipe to clog
        clog_cmd = ["clog"]
        if prompt:
            clog_cmd.extend(["-p", prompt])
        if no_llm:
            clog_cmd.append("--no-llm")

        result = subprocess.run(
            clog_cmd,
            input=cmd_output,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PATH": f"{os.environ.get('HOME')}/bin:{os.environ.get('PATH', '')}"}
        )

        if result.returncode != 0:
            return f"Error running clog: {result.stderr}"

        return result.stdout

    except subprocess.TimeoutExpired:
        return "Error: Command or clog timed out"
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
