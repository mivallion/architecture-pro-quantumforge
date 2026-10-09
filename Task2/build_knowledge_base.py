# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "mwparserfromhell==0.7.2",
#     "pydantic==2.12.5",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv or pip install needed):
#      uv run --with mwparserfromhell==0.7.2 --with pydantic==2.12.5 python -m Task2.build_knowledge_base
# 3. Or use the repository environment:
#      venv/bin/python -m Task2.build_knowledge_base
# ──────────────────

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final, assert_never

from pydantic import ConfigDict, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from .knowledge_models import (
    Document,
    EntityKind,
    EntitySpec,
    FictionalName,
    MissingEntityTermsError,
    MissingPagesError,
    SourceTitle,
    Term,
    TermsMapError,
    ValidationError,
    WikiPage,
)
from .wiki_text import (
    apply_terms as apply_terms,  # noqa: PLC0414 - preserve public import path.
)
from .wiki_text import (
    clean_wikitext as clean_wikitext,  # noqa: PLC0414 - preserve public import path.
)

MEDIAWIKI_NAMESPACE: Final = "{http://www.mediawiki.org/xml/export-0.11/}"
TERM_MAP_ADAPTER: Final = TypeAdapter(
    dict[str, str],
    config=ConfigDict(strict=True),
)
ENTITY_CATALOG: Final[tuple[tuple[SourceTitle, EntityKind], ...]] = (
    (SourceTitle("King Slime"), EntityKind.SOVEREIGN),
    (SourceTitle("Eye of Cthulhu"), EntityKind.SOVEREIGN),
    (SourceTitle("Eater of Worlds"), EntityKind.SOVEREIGN),
    (SourceTitle("Brain of Cthulhu"), EntityKind.SOVEREIGN),
    (SourceTitle("Queen Bee"), EntityKind.SOVEREIGN),
    (SourceTitle("Skeletron"), EntityKind.SOVEREIGN),
    (SourceTitle("Deerclops"), EntityKind.SOVEREIGN),
    (SourceTitle("Wall of Flesh"), EntityKind.SOVEREIGN),
    (SourceTitle("Queen Slime"), EntityKind.SOVEREIGN),
    (SourceTitle("The Twins"), EntityKind.SOVEREIGN),
    (SourceTitle("The Destroyer"), EntityKind.SOVEREIGN),
    (SourceTitle("Skeletron Prime"), EntityKind.SOVEREIGN),
    (SourceTitle("Plantera"), EntityKind.SOVEREIGN),
    (SourceTitle("Golem"), EntityKind.SOVEREIGN),
    (SourceTitle("Duke Fishron"), EntityKind.SOVEREIGN),
    (SourceTitle("Empress of Light"), EntityKind.SOVEREIGN),
    (SourceTitle("Lunatic Cultist"), EntityKind.SOVEREIGN),
    (SourceTitle("Moon Lord"), EntityKind.SOVEREIGN),
    (SourceTitle("Guide"), EntityKind.SETTLER),
    (SourceTitle("Merchant"), EntityKind.SETTLER),
    (SourceTitle("Nurse"), EntityKind.SETTLER),
    (SourceTitle("Demolitionist"), EntityKind.SETTLER),
    (SourceTitle("Dye Trader"), EntityKind.SETTLER),
    (SourceTitle("Angler"), EntityKind.SETTLER),
    (SourceTitle("Zoologist"), EntityKind.SETTLER),
    (SourceTitle("Dryad"), EntityKind.SETTLER),
    (SourceTitle("Painter"), EntityKind.SETTLER),
    (SourceTitle("Golfer"), EntityKind.SETTLER),
    (SourceTitle("Arms Dealer"), EntityKind.SETTLER),
    (SourceTitle("Tavernkeep"), EntityKind.SETTLER),
    (SourceTitle("Stylist"), EntityKind.SETTLER),
    (SourceTitle("Goblin Tinkerer"), EntityKind.SETTLER),
    (SourceTitle("Witch Doctor"), EntityKind.SETTLER),
    (SourceTitle("Mechanic"), EntityKind.SETTLER),
    (SourceTitle("Party Girl"), EntityKind.SETTLER),
    (SourceTitle("Wizard"), EntityKind.SETTLER),
)


def load_terms(path: Path) -> tuple[Term, ...]:
    try:
        term_map = TERM_MAP_ADAPTER.validate_json(path.read_bytes())
    except (OSError, PydanticValidationError) as error:
        raise TermsMapError(path=path, reason=str(error)) from error

    empty_terms = tuple(
        source
        for source, replacement in term_map.items()
        if not source.strip() or not replacement.strip()
    )
    if empty_terms:
        raise TermsMapError(
            path=path, reason="terms and replacements must not be empty"
        )
    return tuple(
        Term(SourceTitle(source), FictionalName(replacement))
        for source, replacement in term_map.items()
    )


def create_entity_specs(terms: tuple[Term, ...]) -> tuple[EntitySpec, ...]:
    replacements = {term.source: term.replacement for term in terms}
    missing = tuple(
        source for source, _ in ENTITY_CATALOG if source not in replacements
    )
    if missing:
        raise MissingEntityTermsError(missing)
    return tuple(
        EntitySpec(
            source_title=SourceTitle(source),
            fictional_name=replacements[SourceTitle(source)],
            kind=EntityKind(kind),
        )
        for source, kind in ENTITY_CATALOG
    )


def extract_pages(
    xml_path: Path,
    wanted_titles: frozenset[SourceTitle],
) -> tuple[WikiPage, ...]:
    pages: list[WikiPage] = []
    for _, element in ET.iterparse(xml_path):
        if element.tag != f"{MEDIAWIKI_NAMESPACE}page":
            continue
        title = SourceTitle(element.findtext(f"{MEDIAWIKI_NAMESPACE}title", default=""))
        namespace = element.findtext(f"{MEDIAWIKI_NAMESPACE}ns", default="")
        if namespace == "0" and title in wanted_titles:
            raw = element.findtext(f".//{MEDIAWIKI_NAMESPACE}text", default="")
            pages.append(WikiPage(title=title, raw=raw))
        element.clear()

    found = frozenset(page.title for page in pages)
    missing = tuple(sorted(wanted_titles - found))
    if missing:
        raise MissingPagesError(titles=missing)
    return tuple(sorted(pages, key=lambda page: page.title))


def create_document(
    spec: EntitySpec,
    raw: str,
    terms: tuple[Term, ...],
) -> Document:
    match spec.kind:
        case EntityKind.SOVEREIGN:
            sections = frozenset({"Spawn", "Behavior", "Aftermath"})
        case EntityKind.SETTLER:
            sections = frozenset({"Spawning", "Living preferences"})
        case unreachable:
            assert_never(unreachable)
    body = apply_terms(clean_wikitext(raw, sections), terms)
    return Document(name=spec.fictional_name, kind=spec.kind, body=body)


def validate_documents(
    documents: tuple[Document, ...],
    terms: tuple[Term, ...],
    *,
    minimum_documents: int = 30,
) -> None:
    issues: list[str] = []
    if len(documents) < minimum_documents:
        issues.append(
            f"expected at least {minimum_documents} documents, got {len(documents)}"
        )

    names = [document.name.casefold() for document in documents]
    if len(names) != len(set(names)):
        issues.append("fictional entity names must be unique")

    wiki_markers = ("{{", "}}", "[[", "]]", "<ref", "</ref")
    for document in documents:
        if len(document.body) < 160:
            issues.append(
                f"'{document.name}' is too short ({len(document.body)} characters)"
            )
        for marker in wiki_markers:
            if marker.casefold() in document.body.casefold():
                issues.append(f"'{document.name}' contains wiki marker '{marker}'")
        for term in terms:
            pattern = re.compile(
                rf"(?<!\w){re.escape(term.source)}(?!\w)",
                flags=re.IGNORECASE,
            )
            if pattern.search(document.body):
                issues.append(f"'{document.name}' contains source term '{term.source}'")

    if issues:
        raise ValidationError(tuple(issues))


def render_document(document: Document) -> str:
    return (
        "---\n"
        f"entity: {document.name}\n"
        f"category: {document.kind}\n"
        "world: Aetherfall\n"
        "---\n\n"
        f"# {document.name}\n\n"
        f"{document.body}\n"
    )


def _slugify(name: FictionalName) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    if not slug:
        msg = f"cannot create filename for '{name}'"
        raise ValueError(msg)
    return slug


def write_outputs(
    output_dir: Path,
    documents: tuple[Document, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_document in output_dir.glob("*.md"):
        stale_document.unlink()
    for index, document in enumerate(documents, start=1):
        filename = f"{index:02d}-{_slugify(document.name)}.md"
        _ = (output_dir / filename).write_text(
            render_document(document),
            encoding="utf-8",
        )


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    task_dir = Path(__file__).resolve().parent
    xml_path = repository_root / "terraria_gamepedia_pages_current.xml"
    terms = load_terms(task_dir / "terms_map.json")
    entity_specs = create_entity_specs(terms)
    pages = extract_pages(
        xml_path,
        frozenset(spec.source_title for spec in entity_specs),
    )
    raw_by_title = {page.title: page.raw for page in pages}
    documents = tuple(
        create_document(spec, raw_by_title[spec.source_title], terms)
        for spec in entity_specs
    )
    validate_documents(documents, terms)
    write_outputs(task_dir / "knowledge_base", documents)
    print(f"Generated {len(documents)} documents in {task_dir / 'knowledge_base'}")


if __name__ == "__main__":
    main()
