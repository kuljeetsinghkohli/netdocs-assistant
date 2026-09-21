# DD-003: BGP Policy Design — Contoso Global WAN

**Document Type:** Design Document  
**Author:** Priya Nair, BGP/Routing SME  
**Reviewed By:** Alex Mercer, Principal Network Architect  
**Version:** 3.1  
**Date:** 2024-10-05  
**Classification:** Internal — Restricted  
**BGP ASN:** 65001 (Contoso private ASN)

---

## 1. Scope

This document describes the BGP routing policy design for:
- MPLS provider peering (eBGP) at each hub site.
- DC-to-DC iBGP between regional data centres.
- BGP-to-OMP redistribution at SD-WAN hubs.
- Internet edge BGP at LON-DC01 (transit AS: 65100 — Contoso Internet Edge).

---

## 2. AS Structure

```
                        ┌──────────────────────┐
                        │   MPLS Provider      │
                        │   AS 12345 (BT)      │
                        │   AS 45678 (Lumen)   │
                        └──────┬───────────────┘
                               │ eBGP
                 ┌─────────────┴──────────────┐
                 │    AS 65001 (Contoso)       │
           LON-DC01 ── iBGP ── NYC-DC01        │
                │                │             │
           SGP-DC01 ── iBGP ── DXB-DC01        │
                └─────────────────────────────┘
```

### 2.1 iBGP Design

All four hub routers participate in a full-mesh iBGP (no route reflectors — site count is small enough). Peer group `IBGP-HUBS` is configured on each hub.

iBGP sessions use loopback addresses:

| Device | Loopback0 | iBGP Neighbor |
|---|---|---|
| LON-DC01-RTR01 | 10.200.1.1/32 | NYC, SGP, DXB loopbacks |
| NYC-DC01-RTR01 | 10.200.1.2/32 | LON, SGP, DXB loopbacks |
| SGP-DC01-RTR01 | 10.200.1.3/32 | LON, NYC, DXB loopbacks |
| DXB-DC01-RTR01 | 10.200.1.4/32 | LON, NYC, SGP loopbacks |

`next-hop-self` is configured on all iBGP sessions.

### 2.2 eBGP Peering

eBGP sessions to MPLS providers:

| Hub | Provider | Provider ASN | Provider PE IP | Contoso CE IP |
|---|---|---|---|---|
| LON-DC01 | BT MPLS | 12345 | 172.16.0.1 | 172.16.0.2 |
| LON-DC01 | Lumen backup | 45678 | 172.16.1.1 | 172.16.1.2 |
| NYC-DC01 | Lumen | 45678 | 172.17.0.1 | 172.17.0.2 |
| SGP-DC01 | BT MPLS | 12345 | 172.18.0.1 | 172.18.0.2 |
| DXB-DC01 | Etisalat | 99999 | 172.19.0.1 | 172.19.0.2 |

---

## 3. Route Policy Design

### 3.1 Inbound Policy (from MPLS Providers)

**Objective:** Accept only specific prefixes from providers; reject default routes.

```
route-map RM-FROM-BT-IN permit 10
  match ip address prefix-list PL-BT-ACCEPTABLE
  set local-preference 200
  set community 65001:100
!
route-map RM-FROM-BT-IN deny 9999
!
ip prefix-list PL-BT-ACCEPTABLE seq 10 permit 10.0.0.0/8 le 24
ip prefix-list PL-BT-ACCEPTABLE seq 20 permit 172.16.0.0/12 le 30
ip prefix-list PL-BT-ACCEPTABLE seq 9999 deny 0.0.0.0/0
```

**Local Preference Values:**

| Source | Local-Pref | Meaning |
|---|---|---|
| BT MPLS (primary) | 200 | Preferred path |
| Lumen MPLS (backup) | 150 | Secondary path |
| Internet edge | 100 | Last resort |

### 3.2 Outbound Policy (to MPLS Providers)

Contoso advertises a summary prefix 10.0.0.0/8 to providers. More-specific routes are filtered.

```
route-map RM-TO-BT-OUT permit 10
  match ip address prefix-list PL-CONTOSO-SUMMARY
  set as-path prepend 65001 65001
!
ip prefix-list PL-CONTOSO-SUMMARY seq 10 permit 10.0.0.0/8
ip prefix-list PL-CONTOSO-SUMMARY seq 9999 deny 0.0.0.0/0 le 32
```

AS-path prepend (2× 65001) is applied to the backup Lumen path to make BT the preferred inbound path.

### 3.3 MED Policy

MED is used to influence inbound traffic from providers where BT and Lumen both advertise the same prefix range:

```
route-map RM-SET-MED permit 10
  set metric 100
!
```

MED 100 is set on primary (BT); Lumen receives MED 200 by default (no explicit set — relies on provider default). **Note:** MED comparison is only meaningful between paths from the same AS; cross-provider MED manipulation is not relied upon.

### 3.4 BGP Community Design

| Community | Meaning | Action |
|---|---|---|
| 65001:100 | Received from BT | Prefer over Lumen paths |
| 65001:150 | Received from Lumen | Secondary path |
| 65001:200 | Internal — DC originated | Do not advertise to providers |
| 65001:300 | Internet edge | Redistribute to OMP |
| 65001:999 | Blackhole | Drop traffic (RTBH) |

---

## 4. BGP-to-OMP Redistribution

At each hub, BGP routes are redistributed into OMP so that SD-WAN branches can reach provider-learned prefixes.

```
sdwan
 omp
  redistribute bgp
  redistribute static
 !
```

**Filter:** Only routes with community `65001:300` (internet edge) and `65001:100/150` (MPLS learned) are redistributed into OMP. Routes tagged `65001:200` (DC internal) are excluded — DC prefixes are originated directly in OMP via network statements.

---

## 5. Route Dampening

BGP route dampening is enabled on eBGP sessions to providers to suppress flapping prefixes:

```
router bgp 65001
 bgp dampening 15 750 2000 60
!
```

Parameters: half-life 15 min, reuse threshold 750, suppress threshold 2000, max-suppress 60 min.

---

## 6. BFD for BGP Fast Failover

BFD is enabled on all eBGP and iBGP sessions:

```
router bgp 65001
 neighbor 172.16.0.1 fall-over bfd
 neighbor 10.200.1.2 fall-over bfd
!
```

BFD interval: 300 ms / multiplier 3 (900 ms detection). This allows sub-second BGP convergence on link failure.

---

## 7. References

- DD-001: SD-WAN Overlay Design (OMP/BGP redistribution boundary)
- DD-004: OSPF Area Design (DC intra-site routing)
- Contoso IP Address Management (IPAM) database
- RFC 4271 — BGP-4
- RFC 7911 — BGP Additional Paths
