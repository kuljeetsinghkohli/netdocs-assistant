# DD-004: OSPF Area Design — Contoso Data Centre Sites

**Document Type:** Design Document  
**Author:** Tom Bradley, DC Network Engineer  
**Reviewed By:** Priya Nair, BGP/Routing SME  
**Version:** 2.0  
**Date:** 2024-07-18  
**Classification:** Internal — Restricted  
**OSPF Process ID:** 1 (all DC sites)

---

## 1. Purpose

This document defines the OSPF area design used within Contoso Global data centre sites. OSPF is the intra-DC routing protocol; it does not extend beyond the DC boundary. Inter-DC and WAN routing is handled by BGP (DD-003) and OMP (DD-001).

---

## 2. Area Assignments

Each data centre is an OSPF domain boundary. Areas are allocated per DC:

| Data Centre | OSPF Area | Area Type | Notes |
|---|---|---|---|
| LON-DC01 | Area 0 (backbone) | Normal | Backbone; all DC areas connect here |
| NYC-DC01 | Area 1 | Normal | |
| SGP-DC01 | Area 2 | Normal | |
| DXB-DC01 | Area 3 | Stub | No external LSAs; reduces LSDB size |

### 2.1 Area 0 — LON-DC01 Backbone

All inter-area routing transits Area 0. LON-DC01 is the OSPF backbone because it is the primary peering point for MPLS and the SD-WAN hub root.

Routers in Area 0:
- LON-DC01-RTR01 (ABR — connects to NYC Area 1 via virtual link if needed)
- LON-DC01-RTR02 (backup ABR)
- LON-DC01-SW01 (Layer 3 distribution — participates in Area 0)

Area 0 networks: 10.10.0.0/20 (server VLAN), 10.200.1.1/32 (loopback), 10.200.0.0/24 (management)

### 2.2 Virtual Link

A virtual link is configured between LON-DC01-RTR01 and NYC-DC01-RTR01 to provide a backup backbone path in the unlikely event that the direct Area 0 adjacency over the DC interconnect is lost:

```
router ospf 1
 area 0 virtual-link 10.200.1.2
!
```

This virtual link is pre-configured but not normally active (the physical path is preferred).

---

## 3. Router Roles

| Router | Role | Areas | LSA Types Originated |
|---|---|---|---|
| LON-DC01-RTR01 | ABR + ASBR | 0, 1 | Type 1, 3, 5 |
| LON-DC01-RTR02 | ABR | 0 | Type 1, 3 |
| NYC-DC01-RTR01 | ABR | 0, 1 | Type 1, 3 |
| SGP-DC01-RTR01 | ABR | 0, 2 | Type 1, 3 |
| DXB-DC01-RTR01 | ABR | 0, 3 | Type 1, 3 (stub — no Type 5) |

ASBRs (LON-DC01-RTR01) redistribute BGP routes into OSPF as Type 5 External LSAs with metric-type E2.

---

## 4. OSPF Timers

### 4.1 Interface Timers

All DC point-to-point links use aggressive timers for fast convergence:

```
interface GigabitEthernet0/0/0
 ip ospf hello-interval 1
 ip ospf dead-interval 3
 ip ospf network point-to-point
!
```

### 4.2 SPF Timers

```
router ospf 1
 timers throttle spf 50 200 5000
 timers throttle lsa 50 200 5000
!
```

SPF initial delay: 50 ms, minimum hold: 200 ms, maximum hold: 5000 ms.

---

## 5. Route Summarisation

ABRs summarise Type 3 inter-area LSAs to reduce LSDB size:

```
router ospf 1
 area 1 range 10.11.0.0 255.255.0.0
 area 2 range 10.12.0.0 255.255.0.0
 area 3 range 10.13.0.0 255.255.0.0
!
```

External routes are summarised before redistribution into BGP:

```
router ospf 1
 summary-address 10.10.0.0 255.255.240.0
!
```

---

## 6. Authentication

All OSPF adjacencies use MD5 authentication:

```
router ospf 1
 area 0 authentication message-digest
!
interface GigabitEthernet0/0/0
 ip ospf message-digest-key 1 md5 <redacted>
!
```

Key rotation is performed quarterly via the Contoso change management process.

---

## 7. Passive Interfaces

Server-facing and management interfaces are configured as passive to prevent unauthorised OSPF neighbours:

```
router ospf 1
 passive-interface default
 no passive-interface GigabitEthernet0/0/0
 no passive-interface GigabitEthernet0/0/1
!
```

---

## 8. OSPF-to-BGP Redistribution (ASBR)

At LON-DC01-RTR01 (ASBR), OSPF routes are redistributed into BGP 65001 for inter-DC and WAN propagation:

```
router bgp 65001
 redistribute ospf 1 route-map RM-OSPF-TO-BGP
!
route-map RM-OSPF-TO-BGP permit 10
 match ip address prefix-list PL-DC-PREFIXES
 set community 65001:200
!
```

Only DC-local prefixes (tagged `65001:200`) are redistributed. Provider-learned and internet prefixes are excluded via route-map.

---

## 9. References

- DD-003: BGP Policy Design
- DD-001: SD-WAN Overlay Design
- RFC 2328 — OSPF Version 2
- RFC 3101 — OSPF Not-So-Stubby Area (NSSA)
