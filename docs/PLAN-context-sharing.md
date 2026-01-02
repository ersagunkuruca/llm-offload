# Plan: Context Sharing with Local LLM

## Problem

When Claude calls local LLM tools (clog, pipe_to_llm, etc.), the local LLM has no context about:
- What project we're working on
- What we've already tried and ruled out
- Conventions and patterns in the codebase
- The user's actual goal

This leads to generic analysis that might suggest things we already checked.

## Idea

Automatically pass conversation context to local LLM tools so they give more relevant answers.

**Example:** Instead of clog suggesting "check if auth token is valid" when we already verified that 5 minutes ago, it would know to look elsewhere.

## Session Data Location

Claude Code stores sessions at:
```
~/.claude/projects/{project-path}/{session-id}.jsonl
```

Each line is a JSON object with conversation messages, tool calls, and metadata.

## Possible Approaches

### Approach 1: Read Session File Directly

MCP tools read the session file and extract recent context.

```python
def get_session_context():
    project_dir = "~/.claude/projects/-Users-..."
    # Find most recent .jsonl file
    # Parse last N user/assistant messages
    # Summarize or extract key facts
    return context_string
```

**Pros:**
- Automatic, no manual updates needed
- Full conversation available

**Cons:**
- Race conditions (file being written)
- Large files need filtering
- Hard to identify "current" session
- Privacy concerns (all conversation exposed to local LLM)

### Approach 2: Context File I Maintain

Claude writes key facts to a `.llm-context` file as we work.

```python
# Claude updates this file during conversation
# MCP tools read it automatically
```

**Pros:**
- Curated, relevant context only
- No parsing needed
- Claude controls what's shared

**Cons:**
- I need to remember to update it
- Could get stale or inconsistent

### Approach 3: Explicit Context Parameter

Add optional `context` parameter to tools. Claude passes relevant info.

```python
clog(
    file_path="app.log",
    context="Android project, Retrofit networking, already ruled out: auth token, network connectivity",
    prompt="What's causing the API failures?"
)
```

**Pros:**
- Simple to implement
- Explicit control
- Works today

**Cons:**
- Manual, might forget to pass context
- Adds verbosity to every call

### Approach 4: Hybrid - Session Summary File

Claude periodically writes a summary to a known location. MCP tools read it.

```
~/.claude/current-context.md

## Current Session
- Project: llm-offload (Android)
- Goal: Debug login failures
- Ruled out: auth token valid, network OK
- Current hypothesis: server-side rate limiting
```

**Pros:**
- Curated but semi-automatic
- Single known location
- Human-readable

**Cons:**
- Still requires Claude to maintain it

## Recommended Path

1. **Start with Approach 3** (explicit context parameter) - simplest, works now
2. **After real usage**, identify what context actually helps
3. **Then consider** session file reading or hybrid approach

## Implementation Notes

### Finding Current Session

```bash
# Most recently modified .jsonl in project folder
ls -t ~/.claude/projects/{project}/*.jsonl | head -1
```

### Extracting User Messages

```bash
grep '"type":"user"' session.jsonl | tail -10
```

### Token Budget

3B model has 128K context but slower with more input. Keep context under 500 tokens for speed.

## Open Questions

- What context actually improves local LLM output? (Need real usage data)
- Should context be per-tool or global?
- How to handle multi-session scenarios?
- Privacy: should all conversation go to local LLM?

## Status

**Status:** Idea documented, awaiting real-world usage experience before implementation.

**Next step:** Use tools on real project, note when context would have helped, then revisit.
