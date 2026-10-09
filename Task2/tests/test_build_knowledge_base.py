from __future__ import annotations

from pathlib import Path

import pytest

from Task2.build_knowledge_base import (
    apply_terms,
    clean_wikitext,
    create_document,
    extract_pages,
    load_terms,
    render_document,
    validate_documents,
    write_outputs,
)
from Task2.knowledge_models import (
    Document,
    EntityKind,
    EntitySpec,
    FictionalName,
    MissingPagesError,
    SourceTitle,
    Term,
    TermsMapError,
    ValidationError,
)


def test_load_terms_reads_replacements_from_json(tmp_path: Path) -> None:
    # Given
    terms_path = tmp_path / "terms_map.json"
    _ = terms_path.write_text(
        '{"Guide": "Pathkeeper", "Night": "Dark Cycle"}',
        encoding="utf-8",
    )

    # When
    result = load_terms(terms_path)

    # Then
    assert result == (
        Term(SourceTitle("Guide"), FictionalName("Pathkeeper")),
        Term(SourceTitle("Night"), FictionalName("Dark Cycle")),
    )


def test_load_terms_rejects_non_string_replacement(tmp_path: Path) -> None:
    # Given
    terms_path = tmp_path / "terms_map.json"
    _ = terms_path.write_text('{"Guide": 42}', encoding="utf-8")

    # When / Then
    with pytest.raises(TermsMapError, match="terms_map.json"):
        _ = load_terms(terms_path)


def test_apply_terms_replaces_complete_name_once() -> None:
    # Given
    terms = (
        Term(SourceTitle("Eye of Cthulhu"), FictionalName("Watcher of the Rift")),
        Term(SourceTitle("Cthulhu"), FictionalName("Nythrax")),
    )

    # When
    result = apply_terms("The Eye of Cthulhu serves Cthulhu.", terms)

    # Then
    assert result == "The Watcher of the Rift serves Nythrax."


def test_clean_wikitext_keeps_lead_and_selected_section() -> None:
    # Given
    raw = """{{npc infobox|type=Boss}}[[File:Battle.png|thumb|A known battle.]]
The [[Guide]] arrives before the first battle.<ref>internal note</ref>
On the {{desktop version}}, a platform-only sound will play.

== Spawning ==
It appears at [[Night|night]].

== History ==
Version-specific trivia.
"""

    # When
    result = clean_wikitext(raw, frozenset({"Spawning"}))

    # Then
    assert result == (
        "The Guide arrives before the first battle.\n\n"
        "## Spawning\n\nIt appears at night."
    )


def test_extract_pages_reports_missing_requested_title(tmp_path: Path) -> None:
    # Given
    xml_path = tmp_path / "wiki.xml"
    _ = xml_path.write_text(
        """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
<page><title>Guide</title><ns>0</ns><revision><text>Guide text.</text></revision></page>
<page><title>Template:Guide</title><ns>10</ns><revision><text>Ignored.</text></revision></page>
</mediawiki>""",
        encoding="utf-8",
    )

    # When / Then
    with pytest.raises(MissingPagesError, match="Dryad"):
        _ = extract_pages(
            xml_path,
            frozenset({SourceTitle("Guide"), SourceTitle("Dryad")}),
        )


def test_create_document_has_fictional_identity_and_clean_body() -> None:
    # Given
    spec = EntitySpec(
        source_title=SourceTitle("Guide"),
        fictional_name=FictionalName("Pathkeeper"),
        kind=EntityKind.SETTLER,
    )
    terms = (Term(spec.source_title, spec.fictional_name),)

    # When
    result = create_document(
        spec,
        "The [[Guide]] helps every player.<ref>note</ref>",
        terms,
    )

    # Then
    assert result == Document(
        name=FictionalName("Pathkeeper"),
        kind=EntityKind.SETTLER,
        body="The Pathkeeper helps every player.",
    )


def test_validate_documents_rejects_source_terms_and_wiki_markup() -> None:
    # Given
    documents = (
        Document(
            name=FictionalName("Pathkeeper"),
            kind=EntityKind.SETTLER,
            body="The Guide carries an unfinished [[link]].",
        ),
    )
    terms = (Term(SourceTitle("Guide"), FictionalName("Pathkeeper")),)

    # When / Then
    with pytest.raises(ValidationError) as error:
        validate_documents(documents, terms, minimum_documents=1)

    assert "source term 'Guide'" in str(error.value)
    assert "wiki marker '[[" in str(error.value)


def test_render_document_adds_searchable_metadata() -> None:
    # Given
    document = Document(
        name=FictionalName("Watcher of the Rift"),
        kind=EntityKind.SOVEREIGN,
        body="A floating sovereign of the Dark Cycle.",
    )

    # When
    result = render_document(document)

    # Then
    assert result == (
        "---\n"
        "entity: Watcher of the Rift\n"
        "category: sovereign\n"
        "world: Aetherfall\n"
        "---\n\n"
        "# Watcher of the Rift\n\n"
        "A floating sovereign of the Dark Cycle.\n"
    )


def test_write_outputs_removes_stale_generated_document(tmp_path: Path) -> None:
    # Given
    output_dir = tmp_path / "knowledge_base"
    output_dir.mkdir()
    stale_path = output_dir / "obsolete.md"
    _ = stale_path.write_text("stale", encoding="utf-8")
    documents = (
        Document(
            name=FictionalName("Pathkeeper"),
            kind=EntityKind.SETTLER,
            body="A sufficiently long body for an output-only unit test.",
        ),
    )
    # When
    write_outputs(output_dir, documents)

    # Then
    assert not stale_path.exists()
    assert (output_dir / "01-pathkeeper.md").is_file()
