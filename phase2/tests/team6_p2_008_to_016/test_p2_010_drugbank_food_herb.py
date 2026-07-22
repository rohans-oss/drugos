"""Regression tests for P2-010: drugbank_parser.py food/herb interactions.

P2-010 ROOT FIX: the previous parser only parsed ``<drug-interaction>``
elements, silently dropping ``<food-interaction>`` and
``<herb-interaction>`` elements. This missed safety-critical signals
(e.g. grapefruit juice + statins = rhabdomyolysis). The fix adds:

  * ``_parse_food_interactions`` -- parses ``<food-interaction>`` elements.
  * ``_parse_herb_interactions`` -- parses ``<herb-interaction>`` elements.
  * ``_classify_food_herb_severity`` -- infers severity from description.
  * ``drugbank_to_food_herb_edges`` -- emits Drug-Food and Drug-Herb
    ``causes_adverse_event`` edges.
  * ``DrugRecord.food_interactions`` / ``DrugRecord.herb_interactions``
    fields.

The tests use atorvastatin's grapefruit-juice interaction as the
canonical example per the P2-010 spec.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "phase2"))

from drugos_graph.drugbank_parser import (  # noqa: E402
    DrugRecord,
    _classify_food_herb_severity,
    _parse_food_interactions,
    _parse_herb_interactions,
    drugbank_to_food_herb_edges,
    parse_drug,
)


DB_NS = {"db": "http://www.drugbank.ca"}


def _make_drug_xml(
    food_interactions: list[str] | None = None,
    herb_interactions: list[tuple[str, str]] | None = None,
) -> ET.Element:
    """Build a minimal <drug> XML element with food/herb interactions."""
    food_interactions = food_interactions or []
    herb_interactions = herb_interactions or []
    food_xml = "".join(
        f"<food-interaction><description>{desc}</description></food-interaction>"
        for desc in food_interactions
    )
    herb_xml = "".join(
        f"<herb-interaction><name>{name}</name>"
        f"<description>{desc}</description></herb-interaction>"
        for name, desc in herb_interactions
    )
    xml_str = f"""\
<drug type="small molecule" xmlns="http://www.drugbank.ca">
  <drugbank-id primary="true">DB01076</drugbank-id>
  <name>Atorvastatin</name>
  <cas-number>134523-00-5</cas-number>
  <inchikey>XUKUURHRXDUEBC-UHFFFAOYSA-N</inchikey>
  <food-interactions>
    {food_xml}
  </food-interactions>
  <herb-interactions>
    {herb_xml}
  </herb-interactions>
</drug>"""
    return ET.fromstring(xml_str)


class TestP2010ParseFoodInteractions:
    """Tests for ``_parse_food_interactions``."""

    def test_parse_grapefruit_atorvastatin(self) -> None:
        """P2-010 spec: atorvastatin's grapefruit-juice interaction must
        be parsed and classified as 'severe' (rhabdomyolysis risk)."""
        elem = _make_drug_xml(food_interactions=[
            "Grapefruit juice increases serum concentration of "
            "atorvastatin, increasing the risk of rhabdomyolysis.",
        ])
        result = _parse_food_interactions(elem, DB_NS, "DB01076")
        assert len(result) == 1
        assert result[0]["kind"] == "food"
        assert result[0]["severity"] == "severe"
        assert "rhabdomyolysis" in result[0]["description"].lower()

    def test_parse_multiple_food_interactions(self) -> None:
        elem = _make_drug_xml(food_interactions=[
            "Take with food. Food increases bioavailability.",
            "Avoid grapefruit juice -- risk of hypertensive crisis.",
        ])
        result = _parse_food_interactions(elem, DB_NS, "DB01076")
        assert len(result) == 2

    def test_empty_food_interactions(self) -> None:
        elem = _make_drug_xml()
        result = _parse_food_interactions(elem, DB_NS, "DB01076")
        assert result == []

    def test_skips_empty_food_interaction_elements(self) -> None:
        xml_str = """\
<drug xmlns="http://www.drugbank.ca">
  <food-interactions>
    <food-interaction><description></description></food-interaction>
    <food-interaction><description>Real interaction</description></food-interaction>
  </food-interactions>
</drug>"""
        elem = ET.fromstring(xml_str)
        result = _parse_food_interactions(elem, DB_NS, "DB01076")
        assert len(result) == 1
        assert result[0]["description"] == "Real interaction"


class TestP2010ParseHerbInteractions:
    """Tests for ``_parse_herb_interactions``."""

    def test_parse_st_johns_wort_warfarin(self) -> None:
        elem = _make_drug_xml(herb_interactions=[
            ("St. John's Wort",
             "The therapeutic efficacy of Warfarin can be decreased "
             "when used in combination with St. John's Wort."),
        ])
        result = _parse_herb_interactions(elem, DB_NS, "DB00682")
        assert len(result) == 1
        assert result[0]["kind"] == "herb"
        assert result[0]["name"] == "St. John's Wort"
        assert "decreased" in result[0]["description"].lower()

    def test_parse_multiple_herb_interactions(self) -> None:
        elem = _make_drug_xml(herb_interactions=[
            ("St. John's Wort", "May decrease effect of drug."),
            ("Garlic", "May increase bleeding risk."),
        ])
        result = _parse_herb_interactions(elem, DB_NS, "DB00682")
        assert len(result) == 2

    def test_empty_herb_interactions(self) -> None:
        elem = _make_drug_xml()
        result = _parse_herb_interactions(elem, DB_NS, "DB01076")
        assert result == []


class TestP2010ClassifyFoodHerbSeverity:
    """Tests for ``_classify_food_herb_severity``."""

    @pytest.mark.parametrize(
        "description,expected_severity",
        [
            ("Risk of rhabdomyolysis", "severe"),
            ("May cause serotonin syndrome", "severe"),
            ("Hypertensive crisis possible", "severe"),
            ("Contraindicated with MAOIs", "severe"),
            ("Avoid grapefruit juice", "moderate"),
            ("Monitor patient closely", "moderate"),
            ("May decrease effect", "moderate"),
            ("Take with food", "mild"),
            ("Food increases bioavailability", "mild"),
            ("Random text with no keywords", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_severity_classification(
        self, description: str, expected_severity: str,
    ) -> None:
        assert _classify_food_herb_severity(description) == expected_severity


class TestP2010DrugRecordFields:
    """Tests for the new ``food_interactions`` / ``herb_interactions`` fields."""

    def test_drug_record_has_food_herb_fields(self) -> None:
        record = DrugRecord(drugbank_id="DB01076", name="Atorvastatin")
        assert hasattr(record, "food_interactions")
        assert hasattr(record, "herb_interactions")
        assert record.food_interactions == []
        assert record.herb_interactions == []

    def test_parse_drug_populates_food_herb_fields(self) -> None:
        """The full ``parse_drug`` pipeline must populate the new fields."""
        elem = _make_drug_xml(
            food_interactions=["Take with food."],
            herb_interactions=[("Garlic", "May increase bleeding risk.")],
        )
        drug = parse_drug(elem)
        assert drug.drugbank_id == "DB01076"
        assert len(drug.food_interactions) == 1
        assert drug.food_interactions[0]["kind"] == "food"
        assert len(drug.herb_interactions) == 1
        assert drug.herb_interactions[0]["name"] == "Garlic"
        assert drug.herb_interactions[0]["kind"] == "herb"


class TestP2010FoodHerbEdgeEmission:
    """Tests for ``drugbank_to_food_herb_edges``."""

    def test_emits_food_edge_with_causes_adverse_event_relation(self) -> None:
        drug = DrugRecord(
            drugbank_id="DB01076",
            name="Atorvastatin",
            inchikey="XUKUURHRXDUEBC-UHFFFAOYSA-N",
            food_interactions=[{
                "drugbank_id": "",
                "name": "Grapefruit juice increases serum concentration...",
                "description": "Grapefruit juice increases serum concentration "
                               "of atorvastatin -> rhabdomyolysis.",
                "severity": "severe",
                "kind": "food",
                "orphan_interaction": False,
            }],
        )
        edges = drugbank_to_food_herb_edges([drug])
        assert len(edges) == 1
        assert edges[0]["rel_type"] == "causes_adverse_event"
        assert edges[0]["src_type"] == "Compound"
        assert edges[0]["dst_type"] == "Food"
        assert edges[0]["severity"] == "severe"
        assert edges[0]["kind"] == "food"
        assert edges[0]["dst_id"].startswith("Food:")

    def test_emits_herb_edge_with_causes_adverse_event_relation(self) -> None:
        drug = DrugRecord(
            drugbank_id="DB00682",
            name="Warfarin",
            inchikey="PJVWKTKQMONHTF-UHFFFAOYSA-N",
            herb_interactions=[{
                "drugbank_id": "",
                "name": "St. John's Wort",
                "description": "St. John's Wort decreases warfarin efficacy.",
                "severity": "moderate",
                "kind": "herb",
                "orphan_interaction": False,
            }],
        )
        edges = drugbank_to_food_herb_edges([drug])
        assert len(edges) == 1
        assert edges[0]["rel_type"] == "causes_adverse_event"
        assert edges[0]["src_type"] == "Compound"
        assert edges[0]["dst_type"] == "Herb"
        assert edges[0]["severity"] == "moderate"
        assert edges[0]["kind"] == "herb"
        assert edges[0]["dst_id"].startswith("Herb:")
        assert edges[0]["partner_name"] == "St. John's Wort"

    def test_same_food_across_drugs_merges_to_same_dst_id(self) -> None:
        """Two drugs with the same food interaction partner MUST merge
        to the same Food node -- this is critical for KG consistency."""
        drug1 = DrugRecord(
            drugbank_id="DB01076", name="Atorvastatin",
            inchikey="XUKUURHRXDUEBC-UHFFFAOYSA-N",
            food_interactions=[{
                "drugbank_id": "", "name": "Grapefruit juice",
                "description": "Risk of rhabdomyolysis.",
                "severity": "severe", "kind": "food",
                "orphan_interaction": False,
            }],
        )
        drug2 = DrugRecord(
            drugbank_id="DB01095", name="Simvastatin",
            inchikey="RYMZZMVNJRMZDD-UHFFFAOYSA-N",
            food_interactions=[{
                "drugbank_id": "", "name": "Grapefruit juice",
                "description": "Risk of rhabdomyolysis.",
                "severity": "severe", "kind": "food",
                "orphan_interaction": False,
            }],
        )
        edges = drugbank_to_food_herb_edges([drug1, drug2])
        assert len(edges) == 2
        # Both edges must point to the same Food node.
        assert edges[0]["dst_id"] == edges[1]["dst_id"]
        assert edges[0]["dst_id"].startswith("Food:")

    def test_no_food_or_herb_interactions_returns_empty(self) -> None:
        drug = DrugRecord(drugbank_id="DB00001", name="Test")
        edges = drugbank_to_food_herb_edges([drug])
        assert edges == []

    def test_skips_drugs_without_drugbank_id(self) -> None:
        drug = DrugRecord(drugbank_id="", name="Test",
                          food_interactions=[{
                              "drugbank_id": "", "name": "food",
                              "description": "desc", "severity": "mild",
                              "kind": "food", "orphan_interaction": False,
                          }])
        edges = drugbank_to_food_herb_edges([drug])
        assert edges == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
