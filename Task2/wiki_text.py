"""Transform MediaWiki markup into searchable Aetherfall text."""

from __future__ import annotations

import html
import re
from typing import Final

import mwparserfromhell

from .knowledge_models import Term

MALFORMED_PLATFORM_PREFIX: Final = re.compile(
    r"^(?:On|In|Unless on) (?:the )?(?:,|when\b|will\b)",
    flags=re.IGNORECASE,
)


def apply_terms(text: str, terms: tuple[Term, ...]) -> str:
    ordered = sorted(terms, key=lambda term: len(term.source), reverse=True)
    choices = "|".join(re.escape(term.source) for term in ordered)
    pattern = re.compile(rf"(?<!\w)({choices})(?!\w)", flags=re.IGNORECASE)
    replacements = {term.source.casefold(): term.replacement for term in terms}

    def replace(match: re.Match[str]) -> str:
        return replacements[match.group(0).casefold()]

    return pattern.sub(replace, text)


def clean_wikitext(raw: str, section_names: frozenset[str]) -> str:
    parsed = mwparserfromhell.parse(raw)
    sections = parsed.get_sections(include_lead=True, levels=[2])
    fragments: list[str] = []

    for index, section in enumerate(sections):
        headings = section.filter_headings()
        heading = "" if index == 0 else headings[0].title.strip_code().strip()
        if index > 0 and heading not in section_names:
            continue
        section_body = mwparserfromhell.parse(str(section))
        section_headings = section_body.filter_headings()
        if section_headings:
            section_body.remove(section_headings[0])
        for link in section_body.filter_wikilinks():
            namespace = str(link.title).partition(":")[0].casefold()
            if namespace in {"file", "image"}:
                section_body.remove(link)
        for tag in section_body.filter_tags():
            if str(tag.tag).casefold() == "ref":
                section_body.remove(tag)
        plain = html.unescape(section_body.strip_code(normalize=True, collapse=True))
        plain = _normalize_text(plain)
        if plain:
            fragments.append(plain if index == 0 else f"## {heading}\n\n{plain}")

    return "\n\n".join(fragments)


def _normalize_text(text: str) -> str:
    lines = (re.sub(r"[ \t]+", " ", line).strip(" *#:;") for line in text.splitlines())
    cleaned = "\n".join(line for line in lines if line and not _is_noisy_line(line))
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _is_noisy_line(line: str) -> bool:
    return bool(
        MALFORMED_PLATFORM_PREFIX.search(line)
        or " / / " in line
        or line.casefold().startswith("thumb|")
        or '""' in line
    )
