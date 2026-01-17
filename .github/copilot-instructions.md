# AI Coding Assistant Instructions for adk-nietzsche-gen

## Project Overview
This is a Google ADK (Agent Development Kit) application that generates non-fiction books using parallel AI agents. The system orchestrates a complex workflow: user provides a JSON spec → outline planning → parallel chapter generation → manuscript assembly → EPUB conversion via Cloud Run.

## Architecture
- **Root Agent**: `p_book_gen.agent.root_agent` - A SequentialAgent chaining 15+ sub-agents
- **Agent Types**: LlmAgent (Gemini), SequentialAgent, ParallelAgent, custom BaseAgent subclasses
- **Storage**: Google Cloud Storage for all intermediate artifacts (chapters, manuscripts, metadata)
- **Worker**: Flask app (`main.py`) triggered by GCS events via Eventarc, uses pandoc for MD→EPUB conversion

## Key Workflows
### Agent Development
- Agents store outputs in shared `ctx.session.state` using `output_key` parameter
- State keys follow patterns: `chapter_{n}_text`, `book_manuscript`, `front_matter`, `cover_prompts`
- Access state in prompts using templated variables: `{{chapter_1_text}}`
- Framework-first approach: Outline with Nietzsche quotes created first, then parallel chapter generation (up to 30 chapters)
- Book title from JSON (`book_title` field) takes precedence over generated titles

### GCS Integration
- All file operations use `google.cloud.storage.Client()`
- URIs: `gs://{bucket}/{path}`
- Bucket from env: `BOOK_GEN_BUCKET`
- Paths: `books/{safe_topic}-{session_timestamp}/chapters/`, `books/{safe_topic}-{session_timestamp}/manuscripts/`, `books/{safe_topic}-{session_timestamp}/epub/`
- Directory naming: Uses book topic slug + session timestamp for easy location

### EPUB Pipeline
- Job JSON uploaded to GCS triggers Eventarc → Cloud Run worker
- Worker downloads MD manuscript, runs `pandoc` with metadata, uploads EPUB
- Removes Kindle-incompatible `<mbp:pagebreak />` from MD before conversion

## Code Patterns
### Agent Construction
```python
# LlmAgent with output_key for state storage
chapter_agent = LlmAgent(
    name=f"chapter_{n}_agent",
    model=GEMINI_MODEL,  # "gemini-1.5-pro"
    instruction=_chapter_instruction(n),
    output_key=f"chapter_{n}_text"
)
```

### State Management
```python
# Store in state
state["book_manuscript"] = merged_markdown
# Access in prompts
"You can access chapters via {{chapter_1_text}}"
```

### GCS Operations
```python
# Download text
text = storage_client.bucket(bucket).blob(blob).download_as_text()
# Upload file
storage_client.bucket(bucket).blob(blob).upload_from_filename(local_path)
```

### Prose Normalization
- Remove AI artifacts: semicolons → periods, bold lead-ins, lists → paragraphs
- Demote H3+ headings to plain text
- Conservative dehyphenation for compound words

## Dependencies & Environment
- **ADK**: `google-adk` for agent framework
- **AI**: `google-genai` for Gemini models
- **Cloud**: `google-cloud-storage`, `google-cloud-aiplatform`
- **Worker**: `flask`, `gunicorn`, `pandoc` (in Docker)
- **Env Vars**: `BOOK_GEN_BUCKET`, `GOOGLE_CLOUD_PROJECT`

## Conventions
- **Markdown**: Kindle-compatible with `<mbp:pagebreak />` between sections
- **Language**: UK English throughout
- **JSON Spec**: First user message contains book metadata (topic, book_title, book_subtitle, author, audience, etc.)
- **Framework First**: Outline agent creates chapter framework using only Nietzsche quotes, then chapters are built based on this framework
- **Book Title Enforcement**: The book_title from JSON takes precedence over generated titles
- **Nietzsche Focus**: All aphorisms must be genuine Friedrich Nietzsche aphorisms with source book references; content must be crafted based on aphorism meanings
- **Chapter Forms**: Rotate through 16 narrative styles (anecdote-led, idea-first, case study, counterargument, braided narrative, second-person, decision-point reconstruction, post-mortem analysis, doctrine exposed, unsaid version, constraint-first framing, long-horizon consequence, mirror test, split-lens narrative, myth vs mechanism, counterfactual simulation)
- **Word Counts**: Vary by chapter (1300-3000 words)
- **Business Cases**: Each chapter includes one well-known true business case study
- **Chapter Headings**: H1 format (# Chapter N: Title) with Nietzsche aphorism with source book, no subheadings, page break at end
- **Nietzsche Quotes**: Each chapter must begin with a genuine Friedrich Nietzsche quote; content must directly relate to and expand upon the quote
- **Paragraphs**: 3-6 sentences with variation (mix short and longer paragraphs)
- **Punctuation**: No semicolons except in correct English usage

## Testing & Debugging
- Agents run in ADK environment; test via `adk dev` or deployed app
- Worker testable locally with `python main.py` + GCS emulator
- State inspection via ADK session logs
- EPUB validation: check pandoc output, metadata inclusion

## File Organization
- `p_book_gen/agent.py`: Root agent factory
- `p_book_gen/custom_agents.py`: All agent implementations (1900+ lines)
- `p_book_gen/tools.py`: GCS save functions
- `main.py`: Cloud Run EPUB worker
- `app.yaml`: ADK app config
- `dockerfile`: Worker container with pandoc