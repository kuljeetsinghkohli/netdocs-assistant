# DD-002: Hub-and-Spoke vs Full Mesh Topology Selection

**Document Type:** Design Document  
**Author:** Sandra Lin, Network Engineering Lead  
**Reviewed By:** Alex Mercer, Principal Network Architect  
**Version:** 1.4  
**Date:** 2024-06-20  
**Classification:** Internal — Restricted

---

## 1. Purpose

This document evaluates hub-and-spoke and full-mesh SD-WAN topologies for the Contoso Global WAN and documents the rationale for the selected hybrid approach.

---

## 2. Topology Options Evaluated

### 2.1 Hub-and-Spoke

In a hub-and-spoke topology, spoke sites communicate exclusively through a designated hub site. All inter-spoke traffic is hair-pinned through the hub.

**Advantages:**
- Centralised policy enforcement at hub.
- Simpler vSmart configuration (fewer TLOC entries).
- Scales well to large spoke counts (N spokes = N hub tunnels, not N² spoke tunnels).
- Supports centralised firewall insertion without complex routing.

**Disadvantages:**
- Hub is a single point of congestion for spoke-to-spoke traffic.
- Additional latency for geographically distant spoke-to-spoke paths.
- Hub capacity must be sized for aggregate spoke bandwidth.

**Recommended for:** Contoso EMEA region (hub at LON-DC01), APAC region (hub at SGP-DC01).

### 2.2 Full Mesh

In a full-mesh topology, every vEdge establishes direct IPSec tunnels to every other vEdge. Inter-site traffic takes the direct path.

**Advantages:**
- Optimal latency for spoke-to-spoke communication.
- No hub congestion bottleneck.
- Resilient: no single hub failure disrupts inter-spoke connectivity.

**Disadvantages:**
- Scales as O(N²) tunnels — 50 spokes = 1,225 tunnels.
- BFD sessions multiply proportionally — risk of control plane overload.
- Complex policy enforcement: must push policy to all edges.
- vEdge platform tunnel limits may be reached at large scale.

**Recommended for:** Contoso Americas DC-to-DC interconnect (NYC-DC01 ↔ LON-DC01 ↔ SGP-DC01).

### 2.3 Partial Mesh / Hybrid (Selected)

The selected design uses a hybrid topology:

- **Branch-to-DC:** Hub-and-spoke. Branches connect to their regional hub.
- **DC-to-DC:** Full mesh between the 4 regional data centres.
- **Cloud On-Ramp:** Cloud vEdge connects in full mesh to all 4 DCs; branches access cloud via regional hub.

This hybrid provides optimal DC-to-DC latency while keeping branch tunnel counts manageable.

---

## 3. Tunnel Count Analysis

| Scenario | Sites | Tunnels (per transport) |
|---|---|---|
| Pure hub-and-spoke (all 47 branches) | 47 spokes + 4 hubs | 94 (dual-homed) |
| Pure full mesh (all 47 branches) | 51 nodes | 1,275 |
| **Hybrid (selected)** | **47 spokes + 4 DCs + 3 cloud** | **~110** |

With dual transport (MPLS + biz-internet), total tunnel count is approximately 220, well within the Cisco vEdge 1000 tunnel limit per platform.

---

## 4. Latency Impact

Round-trip latency measurements (baseline, pre-SD-WAN):

| Path | Legacy WAN (ms) | Hub-and-Spoke (ms) | Direct Mesh (ms) |
|---|---|---|---|
| LON-BR01 → LON-DC01 | 8 | 8 | — |
| LON-BR01 → NYC-DC01 | 72 | 72 | 70 |
| LON-BR01 → LON-BR05 | 14 | 18 (via hub) | 14 |
| NYC-DC01 → SGP-DC01 | 210 | 210 | 208 |

For branch-to-branch traffic (LON-BR01 → LON-BR05), hub-and-spoke adds ~4 ms. This is acceptable for the branch-heavy Contoso workload profile, where 85% of traffic is branch-to-DC.

---

## 5. Failover Behaviour by Topology

### 5.1 Hub-and-Spoke Failover

If LON-DC01 hub becomes unreachable:
1. BFD detects hub tunnel failure within 600 ms.
2. Spoke routes to LON-DC01 are withdrawn from OMP.
3. Traffic reroutes via backup hub (NYC-DC01 via DC full-mesh).
4. Spoke latency increases by DC-to-DC RTT (~72 ms) during outage.

### 5.2 Full Mesh DC Failover

If NYC-DC01 becomes unreachable:
1. All direct mesh tunnels to NYC-DC01 fail.
2. Traffic to NYC-DC01-connected prefixes reroutes via alternative DC hubs.
3. Cloud on-ramp for AWS us-east-1 remains available via AZ-WEU-COR or LON-DC01 path.

---

## 6. Decision and Rationale

**Decision:** Hybrid partial mesh (§2.3) is selected.

**Rationale:**
- Branch tunnel count remains bounded and predictable.
- DC interconnect is optimal-path (full mesh).
- Failover paths exist for both hub and DC failures.
- Policy enforcement remains centralised at vSmart with per-region segmentation.

---

## 7. References

- DD-001: Contoso Global SD-WAN Overlay Design
- DD-005: Zscaler ZIA Integration Design
- Cisco SD-WAN Scalability Guide (Cisco CCO)
