# p_book_gen/custom_agents.py
"""
Custom agent factory for the parallel book generator (tool-less test version).

Current behaviour (Step 2):

- Expect the user to send ONE JSON object as their first message, with fields like:
  {
    "book_topic": "...",
    "author_name": "...",
    "author_bio": "...",
    "author_voice_style": "...",
    "target_audience": "...",
    "book_purpose": "...",
    "min_chapters": 20
  }

Workflow:

1) Parallel chapter generation
   - Up to N LlmAgents (chapters 1..N) run in parallel.
   - Each writes a complete chapter and stores it in state under:
       chapter_1_text, chapter_2_text, ..., chapter_N_text

2) Merge agent
   - Single LlmAgent reads the original JSON (from conversation) and
     the chapter texts from state.
   - Produces a single Markdown manuscript with:
       - Title page
       - Short introduction
       - Chapters 1..N in order.

"""

#from typing import List

#from google.adk.agents import LlmAgent, SequentialAgent
#from google.adk.agents.parallel_agent import ParallelAgent
import os
import uuid
import subprocess
import tempfile
import json
import re

from pathlib import Path

from pydantic import Field

from datetime import datetime
from typing import Optional,List, AsyncGenerator

from google.cloud import storage
from google.genai import types as genai_types

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.agents.parallel_agent import ParallelAgent

# ----------------------------------------------------------------------
# KDP / Kindle formatting constants
# ----------------------------------------------------------------------

KDP_PAGEBREAK = "<mbp:pagebreak />"        # Kindle proprietary pagebreak
MD_PAGEBREAK = "\n\n---\n\n"               # Markdown fallback for non-Kindle e-readers

class ProseNormaliserAgent(BaseAgent):
    def __init__(self, *, name: str = "prose_normaliser_agent") -> None:
        super().__init__(name=name, description="Deterministically removes AI-signalling formatting.")

    def _normalise(self, md: str) -> str:
        text = md

        # 1) Replace semicolons (strong signal)
        text = text.replace(";", ".")

        # 2) Remove bold lead-in labels at paragraph starts
        # e.g. **Key idea:** blah -> blah
        text = re.sub(r'(?m)^\*\*[^*\n]{2,40}\*\*:\s*', '', text)

        # 3) Convert bullet/numbered list blocks into paragraphs
        # Join consecutive list lines into one paragraph.
        def _join_list_block(match):
            block = match.group(0)
            lines = [re.sub(r'^\s*(?:[-*•]|\d+\.)\s+', '', ln).strip() for ln in block.splitlines()]
            lines = [ln for ln in lines if ln]
            return "\n\n" + " ".join(lines) + "\n\n"

        # Bullet blocks
        text = re.sub(r'(?ms)(?:^\s*(?:[-*•])\s+.+\n)+', _join_list_block, text)
        # Numbered blocks
        text = re.sub(r'(?ms)(?:^\s*\d+\.\s+.+\n)+', _join_list_block, text)

        # 4) Demote accidental ### headings into plain text
        text = re.sub(r'(?m)^#{3,6}\s+', '', text)

        # 5) Clean spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        return text.strip() + "\n"

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        manuscript = state.get("book_manuscript")
        if not isinstance(manuscript, str) or not manuscript.strip():
            yield Event(author=self.name, content=genai_types.Content(
                role="system",
                parts=[genai_types.Part(text="No manuscript found to normalise.")]
            ))
            return

        state["book_manuscript"] = self._normalise(manuscript)

        yield Event(author=self.name, content=genai_types.Content(
            role="system",
            parts=[genai_types.Part(text="Normalised manuscript formatting (lists/bold lead-ins/semicolons).")]
        ))


class QueueEpubJobAgent(BaseAgent):
    """
    Writes an EPUB build job JSON to GCS. This is what triggers Cloud Run via Eventarc.
    Stores state['epub_job_gs_uri'] and state['book_epub_gs_uri'] (expected output location).
    """

    def __init__(self, *, name: str = "queue_epub_job_agent") -> None:
        super().__init__(name=name, description="Queues an EPUB build job (job JSON) in GCS.")
        self._storage_client = None

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        bucket_name = os.environ.get("BOOK_GEN_BUCKET")
        if not bucket_name:
            yield Event(author=self.name, content=genai_types.Content(
                role="system",
                parts=[genai_types.Part(text="BOOK_GEN_BUCKET is not set. Cannot queue EPUB job.")]
            ))
            return

        manuscript_gs_uri = state.get("book_gs_uri")  # produced by SaveManuscriptAgent
        if not isinstance(manuscript_gs_uri, str) or not manuscript_gs_uri.startswith("gs://"):
            yield Event(author=self.name, content=genai_types.Content(
                role="system",
                parts=[genai_types.Part(text="book_gs_uri is missing/invalid. Cannot queue EPUB job.")]
            ))
            return

        # Canonical metadata resolved earlier in your workflow
        title = (state.get("book_title") or "Untitled Book").strip()
        subtitle = (state.get("book_subtitle") or "").strip()
        author = (state.get("book_author") or state.get("author_name") or "Unknown Author").strip()

        session_id = ctx.session.id
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        random_id = uuid.uuid4().hex[:8]

        # Match your GCS layout & the Cloud Run guard: must be under /jobs/ and end with -epub-job.json
        job_object_name = f"books/run-{session_id}/jobs/{timestamp}-{random_id}-epub-job.json"

        # Where the worker should write the resulting EPUB:
        epub_object_name = f"books/run-{session_id}/epub/{timestamp}-{random_id}.epub"
        output_epub_gs_uri = f"gs://{bucket_name}/{epub_object_name}"

        job = {
            "job_id": random_id,
            "session_id": session_id,
            "manuscript_gs_uri": manuscript_gs_uri,
            "output_epub_gs_uri": output_epub_gs_uri,
            "metadata": {
                "title": title,
                "subtitle": subtitle,
                "author": author,
                "lang": "en-GB",
            },
        }

        client = self._get_client()
        bucket = client.bucket(bucket_name)
        bucket.blob(job_object_name).upload_from_string(
            json.dumps(job, ensure_ascii=False, indent=2),
            content_type="application/json; charset=utf-8",
        )

        job_gs_uri = f"gs://{bucket_name}/{job_object_name}"
        state["epub_job_gs_uri"] = job_gs_uri
        state["book_epub_gs_uri"] = output_epub_gs_uri  # expected output location

        yield Event(author=self.name, content=genai_types.Content(
            role="system",
            parts=[genai_types.Part(text=f"Queued EPUB job: {job_gs_uri}\nOutput will be: {output_epub_gs_uri}")]
        ))

class SaveEndMatterAgent(BaseAgent):
    """
    Saves end-matter sections (conclusion, action plan, etc.) from state to GCS.
    Stores URIs in state['end_matter_gs_uris'] as a dict[str, str].
    """

    def __init__(self, *, name: str = "save_end_matter_agent") -> None:
        super().__init__(name=name, description="Saves end-matter sections to GCS.")
        self._storage_client = None

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        bucket_name = os.environ.get("BOOK_GEN_BUCKET")
        if not bucket_name:
            yield Event(author=self.name, content=genai_types.Content(
                role="system",
                parts=[genai_types.Part(text="BOOK_GEN_BUCKET is not set. Cannot save end matter.")]
            ))
            return

        client = self._get_client()
        bucket = client.bucket(bucket_name)
        session_id = ctx.session.id
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        # Add or remove keys here as you add sections.
        section_keys = [
            ("conclusion", "conclusion_text"),
            ("action-plan", "action_plan_text"),
            # ("final-word", "final_word_text"),
            # ("further-reading", "further_reading_text"),
        ]

        uris: dict[str, str] = {}
        saved_lines: list[str] = []

        for slug, state_key in section_keys:
            txt = state.get(state_key)
            if not isinstance(txt, str) or not txt.strip():
                continue

            object_name = f"books/run-{session_id}/end_matter/{timestamp}-{slug}.md"
            normalised = txt.strip() + "\n"

            blob = bucket.blob(object_name)
            blob.upload_from_string(normalised, content_type="text/markdown; charset=utf-8")

            gs_uri = f"gs://{bucket_name}/{object_name}"
            uris[slug] = gs_uri
            saved_lines.append(f"{slug}: {gs_uri}")

        state["end_matter_gs_uris"] = uris

        msg = "Saved end matter to GCS:\n" + ("\n".join(saved_lines) if saved_lines else "(none)")
        yield Event(author=self.name, content=genai_types.Content(
            role="system",
            parts=[genai_types.Part(text=msg)]
        ))


class ParseTitleSynthesisAgent(BaseAgent):
    def __init__(self, *, name: str = "parse_title_synthesis_agent") -> None:
        super().__init__(name=name, description="Parses title/subtitle into canonical state keys.")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        raw = state.get("title_synthesis_raw", "")

        title = ""
        subtitle = ""

        if isinstance(raw, str):
            for line in raw.splitlines():
                line = line.strip()
                if line.upper().startswith("TITLE:"):
                    title = line.split(":", 1)[1].strip()
                if line.upper().startswith("SUBTITLE:"):
                    subtitle = line.split(":", 1)[1].strip()

        # Hard safety fallback only if the LLM output is malformed
        if not title:
            title = (state.get("book_topic") or "Untitled Book").strip() if isinstance(state.get("book_topic"), str) else "Untitled Book"

        state["book_title"] = title
        state["book_subtitle"] = subtitle

        msg = f"Canonical title set:\n- {title}\n- {subtitle}" if subtitle else f"Canonical title set:\n- {title}"
        yield Event(author=self.name, content=genai_types.Content(role="system", parts=[genai_types.Part(text=msg)]))


class BuildEpubAgent(BaseAgent):
    """
    Converts the final Markdown manuscript into an EPUB using pandoc
    and uploads it to GCS.
    """

    def __init__(self, *, name: str = "build_epub_agent") -> None:
        super().__init__(
            name=name,
            description="Builds an EPUB from the final Markdown manuscript.",
        )
        self._storage_client = None

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        manuscript = state.get("book_manuscript")

        if not isinstance(manuscript, str) or not manuscript.strip():
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    role="system",
                    parts=[genai_types.Part(text="No manuscript found to convert to EPUB.")]
                ),
            )
            return

        bucket_name = os.environ.get("BOOK_GEN_BUCKET")
        if not bucket_name:
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    role="system",
                    parts=[genai_types.Part(text="BOOK_GEN_BUCKET is not set.")]
                ),
            )
            return
        # ------------------------------------------------------------------
        # Metadata (set these in MergeFromGcsAgent; fallback if missing)
        # ------------------------------------------------------------------
        title = (state.get("book_title") or "Untitled Book").strip()
        author = (state.get("book_author") or "Unknown Author").strip()
        subtitle = (state.get("book_subtitle") or "").strip()
         
        # ------------------------------------------------------------------
        # EPUB cannot use Kindle mbp:pagebreak. Remove or replace.
        # Simple + reliable: remove and keep spacing.
        # ------------------------------------------------------------------
        epub_md = manuscript.replace("<mbp:pagebreak />", "\n\n")


        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            md_path = tmpdir / "book.md"
            epub_path = tmpdir / "book.epub"

            #md_path.write_text(manuscript, encoding="utf-8")
            md_path.write_text(epub_md, encoding="utf-8")


            # Run pandoc
            args = [
                "pandoc",
                str(md_path),
                "-o",
                str(epub_path),
                "--toc",
                "--metadata", "lang=en-GB",
                "--metadata", f"title={title}",
                "--metadata", f"author={author}",
            ]
            if subtitle:
                args += ["--metadata", f"subtitle={subtitle}"]

            subprocess.run(args, check=True)


            client = self._get_client()
            bucket = client.bucket(bucket_name)

            session_id = ctx.session.id
            object_name = (
                f"books/run-{session_id}/epub/book.epub"
            )

            blob = bucket.blob(object_name)
            blob.upload_from_filename(
                str(epub_path),
                content_type="application/epub+zip",
            )

            gs_uri = f"gs://{bucket_name}/{object_name}"
            state["book_epub_gs_uri"] = gs_uri

            yield Event(
                author=self.name,
                content=genai_types.Content(
                    role="system",
                    parts=[genai_types.Part(text=f"EPUB created at {gs_uri}")]
                ),
            )


def kdp_pagebreak_block() -> str:
    """
    Returns the official Kindle pagebreak block that works
    for KDP conversions reliably.
    """
    return f"\n\n{KDP_PAGEBREAK}\n\n"

def md_pagebreak_block() -> str:
    return MD_PAGEBREAK

# ----------------------------------------------------------------------
# Official heading rules for the manuscript
# ----------------------------------------------------------------------

def make_book_title(title: str) -> str:
    return f"# {title}\n\n"

def make_chapter_heading(idx: int, title: str) -> str:
    return f"## Chapter {idx}: {title}\n\n"


GEMINI_MODEL = "gemini-2.0-flash"  # or any other model you prefer

# ---------------------------------------------------------------------
# Outline / planning agent
# ---------------------------------------------------------------------


def _outline_instruction(num_chapters: int) -> str:
    """
    Instruction for the outline planning agent.

    It reads the JSON spec from the user message and produces a chapter-by-chapter
    outline, including a UNIQUE Nietzsche aphorism for each chapter.
    """
    return f"""
You are a planning agent for a non-fiction Kindle book inspired by Friedrich Nietzsche's philosophy.

The user will provide ONE JSON object as their first message. It will include fields like:
- book_topic
- book_title
- author_name
- author_bio
- author_voice_style
- target_audience
- book_purpose
- min_chapters

Your task:

1. Read and understand the JSON object, particularly the book_title and book_topic.
2. The book title is: {{book_title}} - use this exact title as the foundation for your outline.
3. FIRST, select appropriate Nietzsche aphorisms, THEN create chapter titles and descriptions around them.
4. Create a chapter-by-chapter outline for exactly {{min_chapters}} chapters (or {num_chapters} if min_chapters is not specified).
5. For each chapter, start by selecting one of Nietzsche's most famous aphorisms, then craft:
   - Chapter title that connects the aphorism to the book_title and topic
   - A short one-line description of how the aphorism applies to the chapter's focus
   - The complete aphorism text
   - The source book where the aphorism appears
6. Each Nietzsche aphorism must be:
   - A real, authentic aphorism from Friedrich Nietzsche's works
   - One of Nietzsche's most famous and widely recognized aphorisms
   - Assigned to exactly ONE chapter in this outline
   - NOT reused across chapters
   - Include the specific book/source reference
7. Prioritize Nietzsche's most popular aphorisms such as:
   - "God is dead" from The Gay Science
   - "What does not kill me makes me stronger" from Twilight of the Idols
   - "He who has a why to live can bear almost any how" from various works
   - "The Übermensch" concepts from Thus Spoke Zarathustra
   - "Eternal recurrence" from The Gay Science
   - "Will to power" concepts from various works
   - And other well-known aphorisms from Beyond Good and Evil, Human, All Too Human, etc.
8. The outline should create a cohesive framework where each chapter explores the deeper meaning and implications of its assigned aphorism as applied to the book's topic and purpose.

Output format:

- Use plain text or Markdown.
- Use a numbered list, one entry per chapter, in this pattern:

  Chapter 1: <Chapter title>
  Description: <one line>
  Aphorism: "Exact Nietzsche aphorism text."
  Source: <Book title where aphorism appears>
  Author: Friedrich Nietzsche

  Chapter 2: ...
  ...

- Do NOT include any JSON or code in your output.
- Do NOT mention tools or models.
- Use UK English spelling wherever applicable.
- Every aphorism must be from Friedrich Nietzsche - no other authors allowed.
"""


def build_outline_agent(num_chapters: int) -> LlmAgent:
    """
    Build the planning agent that creates the outline and assigns
    a unique quote to each chapter.

    The outline text is stored in state under 'chapter_outline'.
    """
    return LlmAgent(
        name="outline_agent",
        model=GEMINI_MODEL,
        description="Creates the book outline and assigns a unique quote to each chapter.",
        instruction=_outline_instruction(num_chapters),
        output_key="chapter_outline",
    )

# ---------------------------------------------------------------------
# Front matter agent (Dedication, Foreword, Intro, About the Author)
# ---------------------------------------------------------------------


def _front_matter_instruction() -> str:
    """
    Instruction for the front-matter agent.

    It creates dedication, foreword, introduction, and about-the-author
    pages in UK English, based on the JSON spec, using KDP-friendly Markdown.
    """
    return """
You are a specialist non-fiction book writer responsible for the front matter
of a Kindle-ready book.

The user will provide ONE JSON object as their first message. It will include fields like:
- book_topic
- author_name
- author_bio
- author_voice_style
- target_audience
- book_purpose
- min_chapters

Your task:

1. Read and understand the JSON object.
2. Draft the following front-matter sections, in order:

   a) Dedication
      - 1–3 short lines, personal and sincere.
      - Do not invent specific names unless the JSON explicitly provides them.
      - If you cannot infer specific people, keep the dedication general
        (for example “To everyone who chooses to lead when it would be easier to stand still.”).

   b) Foreword
      - 500–800 words.
      - Written in a third-person voice, as if by a thoughtful peer who
        understands why this book matters for the target audience.
      - Explain the problem space, why the author is credible, and what
        the reader will gain.

   c) Introduction
      - 800–1,200 words.
      - Written in the author’s own voice (first person or close third,
        as feels natural), aligned with author_voice_style.
      - Set out the central promise of the book, how it is structured,
        and how the reader should engage with it.

   d) About the Author
      - 250–400 words.
      - Based on author_name and author_bio in the JSON.
      - You may lightly elaborate on the bio for narrative flow, but do
        not invent specific employers or roles that clearly contradict
        the given bio.

3. Use UK English spelling and a natural human voice throughout.
4. Do not mention models, AI, or tools.
5. Do not include any JSON or code in your output.

KDP / formatting constraints:

- Do NOT use any H1 headings (# ...). H1 is reserved for the overall book title.
- Each front-matter section must start with a single Markdown H2 heading (##), in this order:

  ## Dedication
  ...

  ## Foreword
  ...

  ## Introduction
  ...

  ## About the Author
  ...

- Do NOT insert any explicit page-break markup (such as <mbp:pagebreak /> or ---).
  Page breaks will be added later by another part of the workflow.

- Use normal paragraphs and minimal inline formatting (for example, emphasis) so the text
  renders cleanly on Kindle devices.

Additional constraints:

- The canonical book title and subtitle have been generated earlier in the workflow and are available in shared state as:
  - book_title
  - book_subtitle

- You MUST reference the exact title (book_title) at least once in the Foreword and once in the Introduction.
- Do not invent a different title.



Return only the Markdown content of these sections, in the order listed.
"""



def build_front_matter_agent() -> LlmAgent:
    """
    Build the agent that generates front matter.

    The result is stored in state under 'front_matter'.
    """
    return LlmAgent(
        name="front_matter_agent",
        model=GEMINI_MODEL,
        description="Creates Dedication, Foreword, Introduction, and About the Author.",
        instruction=_front_matter_instruction(),
        output_key="front_matter",
    )

    # ---------------------------------------------------------------------
# Cover prompts agent (title, subtitle, cover concepts, back-cover copy)
# ---------------------------------------------------------------------


def _cover_prompts_instruction() -> str:
    """
    Instruction for the cover-prompts agent.

    It creates:
    - A strong book title and subtitle.
    - A front-cover image prompt.
    - A back-cover image prompt.
    - A back-cover blurb suitable for a Kindle / print edition.
    """
    return """
You are a specialist in non-fiction book positioning and cover design concepts.

The user will provide ONE JSON object as their first message. It will include fields like:
- book_topic
- author_name
- author_bio
- author_voice_style
- target_audience
- book_purpose
- min_chapters

Your task:

1. Read and understand the JSON object and infer:
   - The real problem this book is solving.
   - The emotional tone that will resonate with the target audience.
   - How the author should be presented on the cover (for example, as a guide, a peer, a strategist).

2. Create:
   a) A strong, marketable book title.
   b) A clear, concise subtitle that signals the promise and audience.
   c) A front-cover image prompt that could be given to an image-generation model.
      - Describe the composition, mood, colour palette, typography feel and any key visual metaphors.
   d) A back-cover image prompt (for example, a simplified or complementary visual that still feels on-brand).
   e) A back-cover blurb of 200–300 words that:
      - Hooks the reader in the first sentence.
      - Names the audience and their pain.
      - Explains what the reader will gain.
      - Briefly references the author’s credibility based on the JSON.
      - Ends with a clear, confident invitation to start reading.

3. Use UK English spelling and a natural, human marketing voice.
4. Do not mention models, AI, tools, or prompts in the blurb itself.
5. Do not include any JSON or code in your output.

Output format (Markdown):

# Book Title
<your title>

# Subtitle
<your subtitle>

# Front Cover Image Prompt
<your detailed description>

# Back Cover Image Prompt
<your detailed description>

# Back Cover Blurb
<your 200–300 word blurb>
"""


def build_cover_prompts_agent() -> LlmAgent:
    """
    Build the agent that generates title, subtitle, cover image prompts,
    and back-cover blurb.

    The result is stored in state under 'cover_prompts'.
    """
    return LlmAgent(
        name="cover_prompts_agent",
        model=GEMINI_MODEL,
        description="Creates book title/subtitle, cover image prompts, and back-cover blurb.",
        instruction=_cover_prompts_instruction(),
        output_key="cover_prompts",
    )


def _chapter_form(chapter_number: int) -> str:
    forms = [
        "Anecdote-led reflection",
        "Idea-first deep dive",
        "Case study and consequences",
        "Counterargument and reversal",
        "Braided narrative (two threads)",
        "Second-person confrontation",
        "Decision-Point Reconstruction",
        "Post-Mortem Analysis",
        "Doctrine Exposed",
        "The Unsaid Version",
        "Constraint-First Framing",
        "Long-Horizon Consequence",
        "The Mirror Test",
        "Split-Lens Narrative",
        "Myth vs Mechanism",
        "Counterfactual Simulation",
    ]
    return forms[(chapter_number - 1) % len(forms)]


def _chapter_length_range(chapter_number: int) -> tuple[int, int]:
    ranges = [
        (1400, 1800),
        (1800, 2400),
        (2200, 2800),
        (1600, 2100),
        (2000, 2600),
        (1500, 1900),
        (1700, 2200),
        (1900, 2500),
        (2100, 2700),
        (1300, 1700),
        (2300, 2900),
        (1800, 2300),
        (1600, 2000),
        (2000, 2600),
        (1400, 1900),
        (2400, 3000),
    ]
    return ranges[(chapter_number - 1) % len(ranges)]


def _chapter_instruction(chapter_number: int) -> str:
    """
    Create the system instruction for a given chapter agent.

    The agent will:
    - Read the JSON spec from the user's first message.
    - Write one complete chapter with heading, subheading, and body text.
    - Use UK English and a natural human voice.
    - Follow Kindle/KDP-friendly Markdown conventions.
    """
    return f"""
You are a specialist non-fiction book writer for chapter {chapter_number} of a Kindle book inspired by Friedrich Nietzsche's philosophy.
Do not write in “workshop facilitator” tone. Write like an author: opinionated, specific, occasionally anecdotal, and willing to linger on a point for a full paragraph before moving on.

The user will provide ONE JSON object as their first message. It will include fields like:
- book_topic
- book_title
- author_name
- author_bio
- author_voice_style
- target_audience
- book_purpose
- min_chapters

The book title is: {{book_title}} - ensure your chapter contributes to this overall theme and fits within the book's narrative arc.
Book subtitle (if any): use the "book_subtitle" field from the input JSON if present.


Before your turn, an outline planning agent has already created a full outline
for all chapters, including a unique Friedrich Nietzsche aphorism for each chapter. That outline
appears earlier in the conversation and is also stored in shared state.

Your task for chapter {chapter_number}:

0. Carefully read and parse the JSON object from the user message.
   - Look for the field "min_chapters".

   - If "min_chapters" is not present, assume that all configured chapters should be written.

1. If you are not skipping this chapter, read the outline that was generated earlier and
   identify the entry for chapter {chapter_number}.
   - Use the chapter title and short description from that outline as the basis for this chapter.
   - Use the EXACT Friedrich Nietzsche aphorism assigned to chapter {chapter_number} in the outline.
   - The chapter content MUST be crafted based on the deep meaning and implications of this aphorism.
   - Include the source book reference as provided in the outline.

2. Use UK English and a natural, human narrative voice. Avoid any meta-commentary about being an AI.

3. Draft a chapter of about {_chapter_length_range(chapter_number)[0]}–{_chapter_length_range(chapter_number)[1]} words with the following STRICT formatting rules:

   Heading and quote:
   - Begin the chapter with a SINGLE Markdown H1 heading in EXACTLY this format:

       # Chapter {chapter_number}: <Title from outline>

   - Directly below the heading, include the aphorism from the outline in this exact format:

       “Exact quote text.”
       — Friedrich Nietzsche, <Source Book>

   - Then write the body text that aligns with the book topic, purpose, target audience,
     and the outline description directly after the quote.

   Additional formatting constraints:
   - Do NOT use any additional H1 headings. H1 is reserved for chapter titles.
   - Do NOT use any H2 headings. If you need internal structure, prefer paragraphs
     or, at most, modest emphasis, but avoid extra heading levels.
   - Insert page-break markup at the END of each chapter in EXACTLY this format:

       <mbp:pagebreak />

4. The aphorism you use MUST:
   - Be a genuine Friedrich Nietzsche aphorism that fits the chapter theme.
   - If the outline provides a different aphorism, you must replace it with an appropriate Nietzsche aphorism.
   - Not be reused from any other chapter in the outline.
   - The chapter content must be crafted based on the aphorism's deep meaning and implications.

5. The body should be written as conventional non-fiction prose, with paragraphs and no bullet points.

6. Do NOT include any tool calls, code, or JSON in your response.

7. Do NOT mention models, tools, Gemini, Vertex AI, or artificial intelligence.

8. Do NOT apologise or describe the writing process. Just write the chapter.

Style and tone:
- UK English spelling.
- Clear, confident, practical, and aimed at the specified target audience.
- Ensure the chapter clearly connects back to the aphorism and shows why it matters for the reader.
Hard bans (must not appear anywhere in the chapter):
- No bullet lists, numbered lists, or checklist formatting of any kind.
  (No lines starting with '-', '*', '•', '1.', '2.', etc.)
- No bold lead-in labels such as:
  **Key idea:**, **In practice:**, **Step 1:**, **Tip:**, **Summary:**
- No colon-led “label: sentence” structures at paragraph starts.
- Do not use semicolons unless they are correct English punctuation (e.g., in complex lists). Prefer full stops or commas.
- Do not write in “framework voice” (e.g., “First… Second… Third…”).
- Do not use subheadings beyond the single required H1 chapter heading.

To avoid identical pacing across chapters, you MUST follow this chapter form:
Chapter form for this chapter: {_chapter_form(chapter_number)}

CRITICAL: You must actively implement this specific narrative form in your writing. Do not just write a generic chapter - structure your content according to this exact form's requirements. Each form demands a different approach to pacing, structure, and reader engagement.
- Anecdote-led reflection: open with a concrete scene; let the idea emerge from it; return briefly to the scene later.
- Idea-first deep dive: open with a claim or tension; expand through reasoning; introduce a vignette late as proof.
- Case study and consequences: centre on one situation; explore what happened, why it mattered, what it revealed; include an uncomfortable trade-off.
- Counterargument and reversal: start with a common belief; dismantle it slowly; let your own position be tested.
- Braided narrative (two threads): interleave two threads that illuminate each other; merge them only in the final paragraphs.
- Second-person confrontation: occasional "you" to place the reader inside the tension; grounded recognition, not coaching.
- Decision-Point Reconstruction: focus on key choice moments; explore the options available and consequences of each path.
- Post-Mortem Analysis: examine events with hindsight; analyze what went wrong/right and the lessons that emerged.
- Doctrine Exposed: reveal underlying principles through their practical application and real-world testing.
- The Unsaid Version: explore what remained unspoken or undone; examine the power of silence and restraint.
- Constraint-First Framing: begin with limitations and boundaries; show how they define and enable possibilities.
- Long-Horizon Consequence: trace effects far into the future; consider generational or systemic impacts.
- The Mirror Test: use self-reflection as the central device; hold ideas up to personal examination.
- Split-Lens Narrative: present situations through multiple simultaneous perspectives; show how viewpoints shape reality.
- Myth vs Mechanism: contrast idealized stories with practical realities; expose the machinery behind the myth.
- Counterfactual Simulation: explore "what if" scenarios; examine alternative outcomes and their implications.

Paragraph structure (STRICT):
- Keep paragraphs to standard length: 3-6 sentences per paragraph.
- Create variation: mix short (2-3 sentences) and longer (5-7 sentences) paragraphs throughout the chapter.
- Avoid very long paragraphs (more than 8 sentences).
- Use paragraph breaks to create natural pauses and emphasis.
- Consecutive paragraphs should vary in length to maintain reader engagement.

Narrative style requirements:
- Write as continuous, human prose with paragraphs that flow.
- Use varied sentence length and occasional rhetorical questions.
- Use concrete examples and small situational vignettes (1–3 sentences) to ground concepts.
- Use transitions between paragraphs.
- If structure is needed, do it implicitly via paragraphing, not formatting.


"""



def build_chapter_agent(chapter_number: int) -> LlmAgent:
    """
    Build one chapter-writing LlmAgent, for the given chapter number.

    The agent's output is stored in the shared state at:
    state[f"chapter_{chapter_number}_text"]
    via the output_key parameter.
    """
    return LlmAgent(
        name=f"chapter_{chapter_number}_agent",
        model=GEMINI_MODEL,
        description=f"Writes chapter {chapter_number} content.",
        instruction=_chapter_instruction(chapter_number),
        output_key=f"chapter_{chapter_number}_text",

        # No tools in this first test version.
        # We let the agent reply with the full chapter directly.
    )


def build_parallel_book_agent(max_chapters: int) -> ParallelAgent:

    """
    Build the root ParallelAgent that runs up to `max_chapters` chapter agents
    in parallel.
    """
    num_chapters = max(1, max_chapters)

    chapter_agents: List[LlmAgent] = [
        build_chapter_agent(i) for i in range(1, num_chapters + 1)
    ]

    parallel_agent = ParallelAgent(
        name="p_book_gen_parallel",
        description="Runs chapter-writing agents in parallel.",
        sub_agents=chapter_agents,
    )
    return parallel_agent

# ---------------------------------------------------------------------
# Merge agent (runs after parallel step)
# ---------------------------------------------------------------------


def _merge_instruction(num_chapters: int) -> str:
    """
    Instruction for the merge agent.

    It reads chapter texts from state keys:
      chapter_1_text, chapter_2_text, ..., chapter_N_text
    and assembles a single Kindle-style manuscript in Markdown.

    The original JSON spec is available in the conversation history,
    so it can respect book_topic, author_name, author_voice_style,
    target_audience, and book_purpose.
    """
    # Build a small description of available keys, for clarity in the prompt.
    chapter_keys_desc = ", ".join(
        f"chapter_{i}_text" for i in range(1, num_chapters + 1)
    )

    return f"""
You are a book assembler agent.

Context:
- The user provided ONE JSON object as their first message with fields such as:
  book_topic, book_title, author_name, author_bio, author_voice_style, target_audience, book_purpose, min_chapters.
- An outline planning agent has already created an outline for all chapters using only Friedrich Nietzsche quotes.
- A front-matter agent has already created Dedication, Foreword, Introduction, and About the Author
  and stored that text in shared state under 'front_matter'.
- Several chapter-writing agents have already run and stored their outputs in shared state.
- A cover-prompts agent has already created a proposed book title, subtitle, and cover copy,
  and stored that text in shared state under 'cover_prompts'.

You can access the chapter texts via the following state keys:
- {chapter_keys_desc}

You can also access the front matter via:
- front_matter

You can access the cover prompts via:
- cover_prompts

Some chapter slots may have been logically disabled if the requested min_chapters
in the JSON is smaller than the maximum configured chapters. Any chapter text that
is exactly the single line "SKIP_CHAPTER" must be treated as non-existent and
must not appear in the final manuscript.

Use templated variables to access chapter content, e.g.:
- {{chapter_1_text}}
- {{chapter_2_text}}
- ...
- {{chapter_{num_chapters}_text}}

Your task:

1. Read the original JSON spec from the conversation history AND the cover prompts
   from state['cover_prompts'] if they exist.

   Title and subtitle resolution:
   - FIRST, check state for "book_title" (set by ParseSpecAgent from JSON). If present, use this as the canonical book title.
   - If "book_title" is not in state, then check the original JSON spec for a "book_title" field.
   - If still no title, check cover_prompts for generated titles.
   - For subtitle, check state for "book_subtitle", then JSON, then cover_prompts, then derive from book_purpose/target_audience.

   Author name:
   - Use author_name exactly as provided in the JSON.

2. Read the front matter from state['front_matter'] if it exists.
   - It should already use H2 headings such as "## Dedication", "## Foreword", etc.

3. Read all available chapter texts from the state keys above.
   - Ignore any chapter whose text is exactly "SKIP_CHAPTER".
   - Preserve each chapter's existing "## Chapter X: Title" heading and content.

4. Assemble a single, coherent manuscript in Markdown with this structure:

   a) Title page
      - A SINGLE H1 heading with the final book title:
          # <Final Book Title>
      - On the next line, an italicised subtitle (if appropriate), for example:
          _Subtitle text here_
      - On a separate line, the author line, for example:
          By <Author Name>

      - After the title page content, insert a Kindle-compatible pagebreak:
          <mbp:pagebreak />

   b) Front matter
      - Insert the front matter content (Dedication, Foreword, Introduction, About the Author)
        exactly as provided in state['front_matter'], preserving its H2 headings.
      - You may make very light edits for continuity but do not substantially rewrite or reorder it.
      - After the entire front matter block, insert a pagebreak:
          <mbp:pagebreak />

   c) Chapters
      - For each chapter i from 1 to {num_chapters}:
        * If chapter_i_text is present and not equal to "SKIP_CHAPTER", include it in order.
        * Do not alter the chapter's H1 heading ("# Chapter i: Title") or epigraph aphorism formatting,
          except for very light edits to fix obvious typos.
        * Maintain the chapter body as written; do not radically rewrite it.
        * After each included chapter, insert a pagebreak:
            <mbp:pagebreak />

5. Flow and continuity:
   - You may add very short transition sentences between major sections if strictly necessary
     (for example between front matter and the first chapter), but keep these minimal.
   - Do not add new headings beyond those already described.
   - Do not insert any additional structural sections (such as a Table of Contents) unless explicitly
     present in the available content.

Output format rules:

- Output MUST be a single Markdown document, ready for Kindle ingestion.
- Use exactly one H1 heading in the entire document: the book title on the title page.
- All other major sections (front matter and chapters) must use H2 headings ("##").
- Every pagebreak must be written exactly as:
    <mbp:pagebreak />
  on its own line, surrounded by blank lines.

- Do NOT include any JSON, code, or commentary.
- Do NOT mention tools, models, Gemini, Vertex AI, or artificial intelligence.
- Use UK English spelling consistently.
- Avoid anything that suggests the manuscript was generated by a machine.
"""


def build_merge_agent(num_chapters: int) -> LlmAgent:
    """
    Build the LlmAgent that merges the chapters into a single manuscript.

    The final manuscript is stored in state under 'book_manuscript'.
    """
    return LlmAgent(
        name="merge_book_agent",
        model=GEMINI_MODEL,
        description="Merges chapter texts into a single Kindle-style manuscript.",
        instruction=_merge_instruction(num_chapters),
        output_key="book_manuscript",
    )
# ---------------------------------------------------------------------
# Save chapters agent (runs after parallel chapters)
# ---------------------------------------------------------------------


class SaveChaptersAgent(BaseAgent):
    """
    Saves each realised chapter in state['chapter_X_text'] to GCS and stores
    state['chapter_gs_uris'] = {chapter_index: gs://...}
    """

    #num_chapters: int = Field(default=0)
    num_chapters: int = 0
    _storage_client: Optional[storage.Client] = None

    def __init__(self, num_chapters: int, *, name: str = "save_chapters_agent") -> None:
        super().__init__(
            name=name,
            description="Saves each chapter to GCS as a separate Markdown file.",
            #num_chapters=int(num_chapters),  # <-- set as pydantic field

        )
        self.num_chapters = int(num_chapters)

        self._storage_client: storage.Client | None = None

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        bucket_name = os.environ.get("BOOK_GEN_BUCKET")
        if not bucket_name:
            yield Event(
                author=self.name,
                content=genai_types.Content(
                    role="system",
                    parts=[genai_types.Part(
                        text="BOOK_GEN_BUCKET environment variable is not set. Cannot save chapters."
                    )],
                ),
            )
            return

        client = self._get_client()
        bucket = client.bucket(bucket_name)

        session_id = ctx.session.id
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        chapter_uris: dict[int, str] = {}
        saved_lines: list[str] = []

        for i in range(1, self.num_chapters + 1):
            key = f"chapter_{i}_text"
            chapter_text = state.get(key)

            if not isinstance(chapter_text, str) or not chapter_text.strip():
                continue
            if chapter_text.strip() == "SKIP_CHAPTER":
                continue

            object_name = f"books/run-{session_id}/chapters/{timestamp}-chapter-{i:02d}.md"
            normalised = chapter_text.strip() + "\n"

            blob = bucket.blob(object_name)
            blob.upload_from_string(normalised, content_type="text/markdown; charset=utf-8")

            gs_uri = f"gs://{bucket_name}/{object_name}"
            chapter_uris[i] = gs_uri
            saved_lines.append(f"Chapter {i}: {gs_uri}")

        state["chapter_gs_uris"] = chapter_uris

        msg = (
            "No chapter texts were found in state; no chapter files were saved."
            if not saved_lines
            else "Saved chapter files to GCS:\n" + "\n".join(saved_lines)
        )

        yield Event(
            author=self.name,
            content=genai_types.Content(role="system", parts=[genai_types.Part(text=msg)]),
        )


class SaveFrontMatterAgent(BaseAgent):
    """
    Saves the front matter in state['front_matter'] to Google Cloud Storage
    and stores the resulting GCS URI in state['front_matter_gs_uri'].
    """

    def __init__(self, *, name: str = "save_front_matter_agent") -> None:
        super().__init__(
            name=name,
            description="Saves the front matter to GCS.",
        )
        self._storage_client = None  # lazy init

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        front_matter = state.get("front_matter")
        if not isinstance(front_matter, str) or not front_matter.strip():
            content = genai_types.Content(
                role="system",
                parts=[genai_types.Part(text="No front matter found in state['front_matter']. Nothing to save.")],
            )
            yield Event(author=self.name, content=content)
            return

        bucket_name = os.environ.get("BOOK_GEN_BUCKET")
        if not bucket_name:
            content = genai_types.Content(
                role="system",
                parts=[
                    genai_types.Part(
                        text=(
                            "BOOK_GEN_BUCKET environment variable is not set. "
                            "Cannot save front matter to Google Cloud Storage."
                        )
                    )
                ],
            )
            yield Event(author=self.name, content=content)
            return

        client = self._get_client()
        bucket = client.bucket(bucket_name)

        session_id = ctx.session.id
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        random_id = uuid.uuid4().hex[:8]

        object_name = (
            f"books/run-{session_id}/front_matter/"
            f"{timestamp}-{random_id}-front-matter.md"
        )

        normalised = front_matter.strip() + "\n"

        blob = bucket.blob(object_name)
        blob.upload_from_string(
            normalised,
            content_type="text/markdown; charset=utf-8",
        )

        gs_uri = f"gs://{bucket_name}/{object_name}"
        state["front_matter_gs_uri"] = gs_uri

        text = (
            f"Saved front matter to {gs_uri}\n\n"
            "Front matter content was normalised (trimmed) before saving."
        )

        content = genai_types.Content(
            role="system",
            parts=[genai_types.Part(text=text)],
        )
        yield Event(author=self.name, content=content)


# ---------------------------------------------------------------------
# Save agent (runs after merge, writes manuscript to GCS)
# ---------------------------------------------------------------------


class SaveManuscriptAgent(BaseAgent):
    """
    Deterministic agent that saves the final manuscript in state['book_manuscript']
    to Google Cloud Storage using the BOOK_GEN_BUCKET environment variable.

    It also sets state['book_gs_uri'] and returns a message with the GCS URI
    plus the full manuscript, so you still see the content in ADK Web.
    """

    def __init__(self, *, name: str = "save_manuscript_agent") -> None:
        super().__init__(
            name=name,
            description="Saves the final manuscript to GCS.",
        )
        self._storage_client = None  # lazy init

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        manuscript = state.get("book_manuscript")
        if not manuscript:
            content = genai_types.Content(
                role="system",
                parts=[
                    genai_types.Part(
                        text="No manuscript found in state['book_manuscript']. Nothing to save."
                    )
                ],
            )
            yield Event(author=self.name, content=content)
            return

        bucket_name = os.environ.get("BOOK_GEN_BUCKET")
        if not bucket_name:
            content = genai_types.Content(
                role="system",
                parts=[
                    genai_types.Part(
                        text=("BOOK_GEN_BUCKET environment variable is not set. "
                        "Cannot save manuscript to Google Cloud Storage."
                    )
                    )
                ],
            )
            yield Event(author=self.name, content=content)
            return

        client = self._get_client()
        bucket = client.bucket(bucket_name)

        session_id = ctx.session.id
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        random_id = uuid.uuid4().hex[:8]

        object_name = (
            f"books/run-{session_id}/manuscripts/"
            f"{timestamp}-{random_id}.md"
        )

        blob = bucket.blob(object_name)
        blob.upload_from_string(
            manuscript,
            content_type="text/markdown; charset=utf-8",
        )

        gs_uri = f"gs://{bucket_name}/{object_name}"
        state["book_gs_uri"] = gs_uri

        # Return a message that includes the URI and also the full manuscript,
        # so you can still read it directly in ADK Web.
        text = (
            f"Saved manuscript to {gs_uri}\n\n"
            "Below is the manuscript that was saved:\n\n"
            f"{manuscript}"
        )

        content = genai_types.Content(
            role="system",
            parts=[genai_types.Part(text=text)],
        )
        yield Event(author=self.name, content=content)

# ---------------------------------------------------------------------
# Save cover prompts agent (runs after cover prompts generation)
# ---------------------------------------------------------------------


class SaveCoverPromptsAgent(BaseAgent):
    """
    Deterministic agent that saves the cover prompts in state['cover_prompts']
    to Google Cloud Storage using the BOOK_GEN_BUCKET environment variable.

    It sets state['cover_prompts_gs_uri'] and returns a message with the GCS URI
    plus the cover prompt content.
    """

    def __init__(self, *, name: str = "save_cover_prompts_agent") -> None:
        super().__init__(
            name=name,
            description="Saves the cover prompts (title, subtitle, image prompts, blurb) to GCS.",
        )
        self._storage_client = None  # lazy init

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        cover_prompts = state.get("cover_prompts")
        if not cover_prompts:
            content = genai_types.Content(
                role="system",
                parts=[
                    genai_types.Part(
                        text="No cover prompts found in state['cover_prompts']. Nothing to save."
                    )
                ],
            )
            yield Event(author=self.name, content=content)
            return

        bucket_name = os.environ.get("BOOK_GEN_BUCKET")
        if not bucket_name:
            content = genai_types.Content(
                role="system",
                parts=[
                    genai_types.Part(
                        text=(
                            "BOOK_GEN_BUCKET environment variable is not set. "
                            "Cannot save cover prompts to Google Cloud Storage."
                        )
                    )
                ],
            )
            yield Event(author=self.name, content=content)
            return

        client = self._get_client()
        bucket = client.bucket(bucket_name)

        session_id = ctx.session.id
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        random_id = uuid.uuid4().hex[:8]

        object_name = (
            f"books/run-{session_id}/cover/"
            f"{timestamp}-{random_id}-cover-prompts.md"
        )

        blob = bucket.blob(object_name)
        blob.upload_from_string(
            cover_prompts,
            content_type="text/markdown; charset=utf-8",
        )

        gs_uri = f"gs://{bucket_name}/{object_name}"
        state["cover_prompts_gs_uri"] = gs_uri

        text = (
            f"Saved cover prompts to {gs_uri}\n\n"
            "Below are the cover prompts that were saved:\n\n"
            f"{cover_prompts}"
        )

        content = genai_types.Content(
            role="system",
            parts=[genai_types.Part(text=text)],
        )
        yield Event(author=self.name, content=content)


class MergeFromGcsAgent(BaseAgent):
    """
    Deterministic agent that merges title page, front matter, and chapters
    by reading their Markdown content directly from GCS.

    - Uses state['front_matter_gs_uri'] for front matter (if present)
    - Uses state['chapter_gs_uris'] (dict[int, str]) for chapters
    - Writes the final manuscript to state['book_manuscript']

    Notes:
    - Resolves title/subtitle preferring state['cover_prompts'] ("# Book Title", "# Subtitle")
      when present, otherwise falls back to spec JSON (book_topic, book_purpose, target_audience).
    - Parses the user's first message as JSON text (robust for ADK Web pasted JSON).
    - Stores resolved metadata in state: book_title, book_subtitle, book_author
    """

    def __init__(self, *, name: str = "merge_from_gcs_agent") -> None:
        super().__init__(
            name=name,
            description="Merges front matter and chapters from GCS into a KDP-ready manuscript.",
        )
        self._storage_client = None  # lazy init

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    def _download_gs_text(self, client: storage.Client, gs_uri: str) -> str:
        # gs_uri format: gs://bucket/path/to/object
        if not gs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gs_uri}")

        path = gs_uri[len("gs://") :]
        bucket_name, _, blob_name = path.partition("/")
        if not bucket_name or not blob_name:
            raise ValueError(f"Invalid GCS URI: {gs_uri}")

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text(encoding="utf-8")

    def _extract_cover_section(self, md: str, heading: str) -> str:
        """
        Extracts the content under a Markdown H1 heading like:
          # Book Title
          <content>
        stopping at the next '# ' heading (or end of string).
        """
        if not isinstance(md, str) or not md.strip():
            return ""
        pattern = rf"(?ms)^\#\s+{re.escape(heading)}\s*\n(.*?)(?=^\#\s+|\Z)"
        m = re.search(pattern, md)
        return (m.group(1).strip() if m else "")

    def _extract_json_spec_from_first_message(self, ctx: InvocationContext) -> dict:
        """
        Robust JSON extraction for ADK Web.

        ADK Web does NOT expose ctx.session.messages.
        User input appears as early session EVENTS with Content.parts[].text.

        This function:
        - Scans session.events in order
        - Finds the first text payload that looks like JSON
        - Attempts json.loads()
        """
        try:
            events = getattr(ctx.session, "events", None) or []
            for ev in events:
                content = getattr(ev, "content", None)
                if not content:
                    continue

                parts = getattr(content, "parts", None) or []
                for part in parts:
                    text = getattr(part, "text", None)
                    if not isinstance(text, str):
                        continue

                    candidate = text.strip()
                    if candidate.startswith("{") and candidate.endswith("}"):
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
        except Exception:
            pass

        return {}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        client = self._get_client()

        # --------------------------------------------------------------
        # 1) Resolve title/subtitle/author (canonical state first)
        # --------------------------------------------------------------
        spec = self._extract_json_spec_from_first_message(ctx)
        
        # Author still comes from JSON (you said title/subtitle should NOT be passed in JSON)
        author_name = (state.get("book_author") or state.get("author_name") or "").strip()

        if not author_name:
            # very last resort
            spec = state.get("spec") or {}
            if isinstance(spec, dict):
                author_name = (state.get("book_author") or spec.get("author_name") or "Unknown Author").strip()

        
        
        # Canonical title/subtitle should already be generated earlier and stored in state
        title = (state.get("book_title") or "").strip()
        subtitle = (state.get("book_subtitle") or "").strip()
        
        # Optional fallback: if canonical values are missing, try cover_prompts parsing
        cover_prompts = state.get("cover_prompts")
        if (not title) and isinstance(cover_prompts, str):
            title = self._extract_cover_section(cover_prompts, "Book Title").strip()
        if (not subtitle) and isinstance(cover_prompts, str):
            subtitle = self._extract_cover_section(cover_prompts, "Subtitle").strip()
        
        # Last resort fallback ONLY (avoid if possible)
        if not title:
            # do NOT treat book_topic as canonical title; this is just a safe fallback
            book_topic = (spec.get("book_topic") or "").strip()
            title = book_topic or "Untitled Book"
        
        if not author_name:
            author_name = "Unknown Author"
        
        # Store for downstream steps (EPUB metadata, etc.)
        state["book_title"] = title
        state["book_subtitle"] = subtitle
        state["book_author"] = author_name

        lines: list[str] = []

        # --------------------------------------------------------------
        # 2. Title page (single H1 + subtitle + author) + page break
        # --------------------------------------------------------------
        lines.append(f"# {title}")
        if subtitle:
            lines.append(f"_{subtitle}_")
        lines.append(f"By {author_name}")
        lines.append("")
        lines.append("<mbp:pagebreak />")
        lines.append("")

        # --------------------------------------------------------------
        # 3. Front matter from GCS (if any) + page break
        # --------------------------------------------------------------
        front_matter_gs_uri = state.get("front_matter_gs_uri")
        if isinstance(front_matter_gs_uri, str) and front_matter_gs_uri.strip():
            try:
                fm_text = self._download_gs_text(client, front_matter_gs_uri)
                if fm_text.strip():
                    lines.append(fm_text.strip())
                    lines.append("")
                    lines.append("<mbp:pagebreak />")
                    lines.append("")
            except Exception as e:
                # Fail soft: continue without front matter
                lines.append(f"<!-- Failed to load front matter from {front_matter_gs_uri}: {e} -->")
                lines.append("")

        # --------------------------------------------------------------
        # 4. Chapters from GCS (ordered by chapter index) + page breaks
        # --------------------------------------------------------------
        chapter_uris = state.get("chapter_gs_uris") or {}

        def _key_int(k):
            try:
                return int(k)
            except Exception:
                return k

        for idx in sorted(chapter_uris.keys(), key=_key_int):
            gs_uri = chapter_uris[idx]
            try:
                chapter_text = self._download_gs_text(client, gs_uri)
                if chapter_text.strip():
                    lines.append(chapter_text.strip())
                    lines.append("")
                    lines.append("<mbp:pagebreak />")
                    lines.append("")
            except Exception as e:
                lines.append(f"<!-- Failed to load chapter {idx} from {gs_uri}: {e} -->")
                lines.append("")

        # --------------------------------------------------------------
        # 5. End matter from GCS (in a fixed order) + page breaks
        # --------------------------------------------------------------
        end_matter = state.get("end_matter_gs_uris") or {}

        # Order matters; slugs must match what SaveEndMatterAgent writes.
        end_order = ["conclusion", "action-plan"]

        for slug in end_order:
            gs_uri = end_matter.get(slug)
            if not isinstance(gs_uri, str) or not gs_uri.strip():
                continue
            try:
                t = self._download_gs_text(client, gs_uri)
                if t.strip():
                    lines.append(t.strip())
                    lines.append("")
                    lines.append("<mbp:pagebreak />")
                    lines.append("")
            except Exception as e:
                lines.append(
                    f"<!-- Failed to load end matter {slug} from {gs_uri}: {e} -->"
                )
                lines.append("")

        manuscript = "\n".join(lines).strip()
        state["book_manuscript"] = manuscript

        content = genai_types.Content(
            role="system",
            parts=[genai_types.Part(text=manuscript)],
        )
        yield Event(author=self.name, content=content)

def build_conclusion_agent() -> LlmAgent:
    return LlmAgent(
        name="conclusion_agent",
        model=GEMINI_MODEL,
        description="Writes the book conclusion section.",
        instruction=_conclusion_instruction(),
        output_key="conclusion_text",
    )


def _action_plan_instruction() -> str:
    return """
You are a specialist non-fiction book writer creating a practical action plan to help the reader apply the book.

Write a section that:
- Write it as a narrative plan in three phases (first month, next two months, following three months), but expressed as flowing paragraphs with minimal signposting. No lists, no labelled steps, no bold lead-ins.
- Uses short paragraphs and clear structure, but do NOT use bullet points.
- Gives concrete, realistic actions aligned to the target audience.

Formatting rules:
- Output Markdown only.
- Start with a single H2 heading:
  ## From Insight to Action
- No H1 headings.
- No pagebreak markup.
- Use UK English.
- 600–1,000 words.
"""

def build_action_plan_agent() -> LlmAgent:
    return LlmAgent(
        name="action_plan_agent",
        model=GEMINI_MODEL,
        description="Writes a practical action plan section.",
        instruction=_action_plan_instruction(),
        output_key="action_plan_text",
    )



def _title_synthesis_instruction() -> str:
    return """
You are a non-fiction book editor and positioning specialist.

The user provided ONE JSON object with fields like:
- book_topic
- author_name
- author_bio
- author_voice_style
- target_audience
- book_purpose
- min_chapters

Your task:

1) Create a marketable, credible non-fiction title and subtitle for this book.
2) The title and subtitle must:
   - Be driven by book_topic, target_audience, and book_purpose.
   - Be specific and non-generic.
   - Avoid hypey language.
   - Use UK English spelling.
3) Output MUST be exactly two lines in this format:

TITLE: <Final Book Title>
SUBTITLE: <Final Subtitle>

No extra text. No headings. No JSON. No bullets.
"""

# ---------------------------------------------------------------------
# Full workflow: SequentialAgent(root) = Parallel chapters → Merge
# ---------------------------------------------------------------------

def build_full_workflow_agent(max_chapters: int = 30) -> SequentialAgent:

    """
    Build the root SequentialAgent for this step.

    - Uses max_chapters as N, the maximum number of chapter agents.
    - Outline agent plans N chapters and assigns unique quotes.
    - Front-matter agent generates Dedication, Foreword, Introduction, About the Author.
    - Cover-prompts agent generates title, subtitle, cover image prompts, and back-cover blurb.
    - Parallel chapter agents write up to N chapters (respecting min_chapters via SKIP_CHAPTER).
    - SaveChaptersAgent saves each realised chapter to GCS.
    - Merge agent assembles a single KDP-friendly Markdown manuscript with:
        * One H1 book title page
        * H2 front-matter sections
        * H2 chapter headings
        * <mbp:pagebreak /> between major sections and after each chapter.
    - SaveManuscriptAgent saves the full manuscript to GCS.
    - SaveCoverPromptsAgent saves the cover prompts to GCS.
    """

    num_chapters = max(1, max_chapters)
    outline_agent = build_outline_agent(num_chapters)
    front_matter_agent = build_front_matter_agent()
    save_front_matter_agent = SaveFrontMatterAgent()
    cover_prompts_agent = build_cover_prompts_agent()

    parallel_agent = build_parallel_book_agent(num_chapters)
    save_chapters_agent = SaveChaptersAgent(num_chapters=num_chapters)
    #merge_agent = MergeFromGcsAgent()          # <- use the deterministic GCS-based merge
    save_manuscript_agent = SaveManuscriptAgent()
    save_cover_prompts_agent = SaveCoverPromptsAgent()
    #build_epub_agent = BuildEpubAgent()
    queue_epub_job_agent = QueueEpubJobAgent()
    title_synthesis_agent = build_title_synthesis_agent()
    parse_title_agent = ParseTitleSynthesisAgent()
    conclusion_agent = build_conclusion_agent()
    action_plan_agent = build_action_plan_agent()
    save_end_matter_agent = SaveEndMatterAgent()
    merge_from_gcs_agent = MergeFromGcsAgent()  # use this, not the LLM merge agent
    parse_spec_agent = ParseSpecAgent()
    prose_normaliser_agent = ProseNormaliserAgent()



    workflow = SequentialAgent(
        name="p_book_gen_workflow",
        description=(
            "Deterministic workflow: spec parsing, outline planning, front-matter generation, "
            "front-matter save, cover prompts, parallel chapter generation, "
            "save chapters to GCS, merge from GCS into a single manuscript, "
            "then save the full book and cover prompts to GCS."
        ),
        sub_agents=[
            parse_spec_agent,
            outline_agent,
            front_matter_agent,
            save_front_matter_agent,
            cover_prompts_agent,
            parallel_agent,
            save_chapters_agent,
            conclusion_agent,
            action_plan_agent,
            save_end_matter_agent,
            merge_from_gcs_agent ,
            #merge_agent,
            #prose_normaliser_agent,
            save_manuscript_agent,
            queue_epub_job_agent,    # queues job JSON to trigger Cloud Run pandoc worker
            save_cover_prompts_agent,
                    ],
    )
    return workflow

def _conclusion_instruction() -> str:
    return """
You are a specialist non-fiction book writer creating the closing conclusion of the book.

Inputs available earlier in the conversation and/or shared state:
- The original JSON spec (book_topic, author_name, author_voice_style, target_audience, book_purpose)
- The complete chapter outline in state['chapter_outline']
- The full set of chapters already written (you may refer to their themes, but do not quote long passages)

Write a concluding section that:
- Synthesises the book’s core thesis and the most important recurring ideas.
- Feels earned and grounded (no hype).
- Ends with momentum and clarity for the reader.

Formatting rules:
- Output Markdown only.
- Start with a single H2 heading:
  ## Conclusion
- No H1 headings.
- No pagebreak markup.
- Use UK English.
- No bullets (paragraph prose only).
- 800–1,200 words.
"""


def build_title_synthesis_agent() -> LlmAgent:
    return LlmAgent(
        name="title_synthesis_agent",
        model=GEMINI_MODEL,
        description="Generates canonical book title and subtitle from the spec.",
        instruction=_title_synthesis_instruction(),
        output_key="title_synthesis_raw",
    )

class ParseSpecAgent(BaseAgent):
    """
    Robustly finds the first user-provided JSON spec anywhere in the session messages,
    parses it, and stores key fields in state so downstream agents never depend on
    ctx.session.messages[0].
    """

    def __init__(self, *, name: str = "parse_spec_agent") -> None:
        super().__init__(name=name, description="Parses the user JSON spec into state.")

    def _find_first_json_text(self, ctx: InvocationContext) -> str | None:
        """
        ADK Web-compatible JSON spec locator.
    
        - ADK Web stores conversation history on ctx.session.events (not messages).
        - We first prefer events that appear to be from the user (when role is available).
        - Then we fallback to scanning all events.
        """
        events = getattr(ctx.session, "events", None) or []
    
        def _extract_json_from_event(ev) -> str | None:
            content = getattr(ev, "content", None)
            if not content:
                return None
            parts = getattr(content, "parts", None) or []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str):
                    s = t.strip()
                    if s.startswith("{") and s.endswith("}"):
                        return s
            return None
    
        # Pass 1: prefer "user" when role metadata exists
        for ev in events:
            role = (
                getattr(ev, "role", None)
                or getattr(getattr(ev, "content", None), "role", None)
            )
            if role and str(role).lower() != "user":
                continue
    
            found = _extract_json_from_event(ev)
            if found:
                return found

        # Pass 2: fallback scan across all events
        for ev in events:
            found = _extract_json_from_event(ev)
            if found:
                return found

        return None


    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        raw = self._find_first_json_text(ctx)
        if not raw:
            yield Event(author=self.name, content=genai_types.Content(
                role="system",
                parts=[genai_types.Part(text="Could not find JSON spec text in session messages.")]
            ))
            return

        try:
            spec = json.loads(raw)
            if not isinstance(spec, dict):
                raise ValueError("Spec JSON is not an object.")
        except Exception as e:
            yield Event(author=self.name, content=genai_types.Content(
                role="system",
                parts=[genai_types.Part(text=f"Failed to parse JSON spec: {e}")]
            ))
            return

        # Persist canonical fields for downstream agents
        state["spec"] = spec
        for k in ["book_topic", "book_title", "book_subtitle", "author_name", "author_bio", "author_voice_style", "target_audience", "book_purpose", "min_chapters"]:
            if k in spec:
                state[k] = spec.get(k)

        author = (str(spec.get("author_name") or "").strip()) or "Unknown Author"
        state["book_author"] = author  # canonical author key used by merge + epub

        yield Event(author=self.name, content=genai_types.Content(
            role="system",
            parts=[genai_types.Part(text=f"Parsed spec. Author set to: {author}")]
        ))