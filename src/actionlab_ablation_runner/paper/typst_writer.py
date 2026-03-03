from __future__ import annotations

from pathlib import Path

from actionlab_ablation_runner.schemas import GeminiSection


def write_typst(section: GeminiSection, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    citations = "\n".join(f"- {url}" for url in section.citations)
    content = f"""= ActionLab Ablation Report

== Abstract
{section.abstract}

== Hypothesis Framing
{section.hypothesis_framing}

== Methods
{section.methods}

== Statistical Testing
{section.statistical_testing}

== Interpretation
{section.interpretation}

== Threats to Validity
{section.threats_to_validity}

== Ethical Considerations
{section.ethical_considerations}

== References
{citations}
"""
    out_file.write_text(content)
