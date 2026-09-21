"""
Golden evaluation set — 30 questions with expected source documents.

Mix:
  - 8  runbook questions
  - 8  design-doc questions
  - 7  ticket / change questions
  - 2  config questions
  - 5  unanswerable questions (no docs in corpus cover them)

Phrasing is intentionally natural / operational rather than by document title.
``expected_sources`` lists the source *file stems* that must appear in top-k
results (at least one must match for a hit to count).
``unanswerable`` questions should produce a refusal (no relevant chunk).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldenQuestion:
    qid: str
    question: str
    expected_sources: list[str]   # file stems (without path / extension) that should rank
    doc_types: list[str]          # expected doc_type(s) for the winning chunks
    unanswerable: bool = False
    notes: str = ""


GOLDEN_SET: list[GoldenQuestion] = [
    # -----------------------------------------------------------------------
    # Runbook questions (8)
    # -----------------------------------------------------------------------
    GoldenQuestion(
        qid="Q01",
        question="What is the runbook procedure when a BGP session flaps?",
        expected_sources=["RB-001_bgp_flap_diagnosis"],
        doc_types=["runbook"],
        notes="Phrasing variant that was previously refused despite confidence=2.72",
    ),
    GoldenQuestion(
        qid="Q02",
        question="How do I diagnose a BGP flap?",
        expected_sources=["RB-001_bgp_flap_diagnosis"],
        doc_types=["runbook"],
    ),
    GoldenQuestion(
        qid="Q03",
        question="Which commands should I run first when a BGP neighbour goes down?",
        expected_sources=["RB-001_bgp_flap_diagnosis"],
        doc_types=["runbook"],
    ),
    GoldenQuestion(
        qid="Q04",
        question="What steps does the runbook recommend for an OMP route that is missing?",
        expected_sources=["RB-002_omp_route_issues"],
        doc_types=["runbook"],
    ),
    GoldenQuestion(
        qid="Q05",
        question="How do I recover a tunnel that has gone down between vEdge sites?",
        expected_sources=["RB-003_tunnel_down_recovery"],
        doc_types=["runbook"],
    ),
    GoldenQuestion(
        qid="Q06",
        question="What is the procedure for adding a new site using zero-touch provisioning?",
        expected_sources=["RB-005_ztp_new_site_deployment"],
        doc_types=["runbook"],
    ),
    GoldenQuestion(
        qid="Q07",
        question="How do I renew the vManage certificate?",
        expected_sources=["RB-006_vmanage_cert_renewal"],
        doc_types=["runbook"],
    ),
    GoldenQuestion(
        qid="Q08",
        question="What should I do if I suspect a BGP route leak is occurring?",
        expected_sources=["RB-007_bgp_route_leak_response"],
        doc_types=["runbook"],
    ),

    # -----------------------------------------------------------------------
    # Design-doc questions (8)
    # -----------------------------------------------------------------------
    GoldenQuestion(
        qid="Q09",
        question="What BGP ASN does Contoso use for its private AS?",
        expected_sources=["DD-003_bgp_policy_design"],
        doc_types=["design_doc"],
    ),
    GoldenQuestion(
        qid="Q10",
        question="How does the SD-WAN overlay connect hub and branch sites?",
        expected_sources=["DD-001_sdwan_overlay_design"],
        doc_types=["design_doc"],
    ),
    GoldenQuestion(
        qid="Q11",
        question="What are the differences between hub-spoke and full-mesh topologies in the design?",
        expected_sources=["DD-002_hub_spoke_vs_full_mesh"],
        doc_types=["design_doc"],
    ),
    GoldenQuestion(
        qid="Q12",
        question="How is OSPF area design structured across the Contoso WAN?",
        expected_sources=["DD-004_ospf_area_design"],
        doc_types=["design_doc"],
    ),
    GoldenQuestion(
        qid="Q13",
        question="How is internet traffic forwarded via Zscaler ZIA from branch sites?",
        expected_sources=["DD-005_zscaler_zia_integration"],
        doc_types=["design_doc"],
    ),
    GoldenQuestion(
        qid="Q14",
        question="What BGP communities are used to tag routes from MPLS providers?",
        expected_sources=["DD-003_bgp_policy_design"],
        doc_types=["design_doc"],
    ),
    GoldenQuestion(
        qid="Q15",
        question="How does hybrid cloud connectivity work between on-prem and AWS?",
        expected_sources=["DD-006_hybrid_cloud_connectivity"],
        doc_types=["design_doc"],
    ),
    GoldenQuestion(
        qid="Q16",
        question="What is the iBGP design between regional data centres?",
        expected_sources=["DD-003_bgp_policy_design"],
        doc_types=["design_doc"],
    ),

    # -----------------------------------------------------------------------
    # Ticket / change questions (7)
    # -----------------------------------------------------------------------
    GoldenQuestion(
        qid="Q17",
        question="What was the rollback plan for the BGP route-map cleanup on LON-DC01?",
        expected_sources=["CRQ-2024-1015"],
        doc_types=["ticket"],
        notes="CRQ-2024-1015 rollback plan was outranking runbooks for BGP flap queries",
    ),
    GoldenQuestion(
        qid="Q18",
        question="Which change ticket covered the BGP route leak incident cleanup?",
        expected_sources=["CRQ-2024-1015"],
        doc_types=["ticket"],
    ),
    GoldenQuestion(
        qid="Q19",
        question="Was there a certificate renewal change performed on vManage?",
        expected_sources=["CRQ-2024-1101", "CRQ-2024-1110", "CRQ-2024-1120"],
        doc_types=["ticket"],
    ),
    GoldenQuestion(
        qid="Q20",
        question="What change was made to BGP timers in August 2024?",
        expected_sources=["CRQ-2024-0801", "CRQ-2024-0823"],
        doc_types=["ticket"],
    ),
    GoldenQuestion(
        qid="Q21",
        question="What was the outcome of the failover test in September 2024?",
        expected_sources=["CRQ-2024-0905", "CRQ-2024-0918"],
        doc_types=["ticket"],
    ),
    GoldenQuestion(
        qid="Q22",
        question="What new site was deployed in January 2025?",
        expected_sources=["CRQ-2025-0108", "CRQ-2025-0115", "CRQ-2025-0122"],
        doc_types=["ticket"],
    ),
    GoldenQuestion(
        qid="Q23",
        question="Which tickets reference the LON-DC01-RTR01 router?",
        expected_sources=["CRQ-2024-1015", "CRQ-2024-1002"],
        doc_types=["ticket"],
    ),

    # -----------------------------------------------------------------------
    # Config questions (2)
    # -----------------------------------------------------------------------
    GoldenQuestion(
        qid="Q24",
        question="What BGP configuration is on the LON-DC01 router?",
        expected_sources=["LON-DC01-RTR01"],
        doc_types=["config"],
    ),
    GoldenQuestion(
        qid="Q25",
        question="What interfaces are configured on the NYC data centre router?",
        expected_sources=["NYC-DC01-RTR01"],
        doc_types=["config"],
    ),

    # -----------------------------------------------------------------------
    # Unanswerable questions (5) — corpus does not contain answers
    # -----------------------------------------------------------------------
    GoldenQuestion(
        qid="Q26",
        question="What is the configuration of the Paris office router?",
        expected_sources=[],
        doc_types=[],
        unanswerable=True,
        notes="No Paris office in corpus",
    ),
    GoldenQuestion(
        qid="Q27",
        question="What is the SNMP community string used on Contoso devices?",
        expected_sources=[],
        doc_types=[],
        unanswerable=True,
        notes="SNMP not covered in corpus",
    ),
    GoldenQuestion(
        qid="Q28",
        question="How do I configure Cisco ACI fabric for the London data centre?",
        expected_sources=[],
        doc_types=[],
        unanswerable=True,
        notes="ACI not in scope",
    ),
    GoldenQuestion(
        qid="Q29",
        question="What is the WiFi password for the Contoso guest network?",
        expected_sources=[],
        doc_types=[],
        unanswerable=True,
        notes="WiFi not in corpus",
    ),
    GoldenQuestion(
        qid="Q30",
        question="What is the SLA for Contoso's SD-WAN vendor support contract?",
        expected_sources=[],
        doc_types=[],
        unanswerable=True,
        notes="Vendor contracts not in corpus",
    ),
]
