# p_book_gen/agent.py
"""
Root agent for the parallel book generator.

Current design (Step 2):
- Use a SequentialAgent to orchestrate:
  1) Parallel chapter generation (up to N chapters in parallel).
  2) Merge agent that assembles a single Kindle-style manuscript.

Later steps will add:
- Front matter (dedication, foreword, detailed introduction),
- Quotes per chapter,
- GCS save of final manuscript and metadata,
- Cover image prompts.
"""

from .custom_agents import build_full_workflow_agent

# For now, default to a maximum of 20 chapters in the workflow.
#root_agent = build_full_workflow_agent(min_chapters_for_now=20)
root_agent = build_full_workflow_agent()

