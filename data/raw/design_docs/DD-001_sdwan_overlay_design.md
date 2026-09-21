# DD-001: Contoso Global SD-WAN Overlay Design

**Document Type:** Design Document  
**Author:** Alex Mercer, Principal Network Architect  
**Reviewed By:** Sandra Lin, Network Engineering Lead  
**Version:** 2.3  
**Date:** 2024-09-12  
**Classification:** Internal — Restricted  
**Site IDs in Scope:** All (Global)

---

## 1. Executive Summary

This document describes the Contoso Global SD-WAN overlay architecture based on Cisco Viptela (SD-WAN) technology. The design provides a secure, resilient, and policy-driven WAN fabric connecting 47 branch sites, 4 regional data centres, and 3 cloud on-ramp points (AWS us-east-1, Azure West Europe, Azure UAE North).

The architecture leverages an OMP (Overlay Management Protocol) control plane hosted on vSmart controllers, with a Cisco vManage NMS/orchestration platform deployed in high-availability mode in the LON-DC01 data centre.

---

## 2. Design Goals

- Provide sub-second failover between transport circuits (MPLS, broadband, LTE).
- Enforce per-application traffic steering policies centrally via vManage.
- Segment network traffic into VPNs (Corporate, Guest, OT/IoT, Management) with no cross-VPN leakage.
- Support zero-touch provisioning (ZTP) for branch deployments.
- Achieve BFD-driven failover within 500 ms for critical application traffic.

---

## 3. Topology Overview

### 3.1 Control Plane

The control plane consists of:

- **2 × Cisco vSmart Controllers** (Active/Active) deployed in LON-DC01 at 10.200.0.10 and 10.200.0.11.
- **1 × Cisco vManage** (HA pair) at 10.200.0.20/10.200.0.21 — primary management and policy push point.
- **2 × Cisco vBond Orchestrators** at 10.200.0.30 and 10.200.0.31 — responsible for NAT traversal and initial device authentication.

All control plane components reside in VPN 512 (Management VPN) and communicate over TLS/DTLS using certificates issued by the Contoso PKI (CA root: `contoso-root-ca.crt`).

### 3.2 Data Plane

Each branch site runs one or two Cisco vEdge or Catalyst SD-WAN (IOS-XE-based) edge routers. The data plane is formed by IPSec tunnels (IKEv2) automatically established between vEdge devices based on OMP-advertised TLOC routes.

Transport colours in use:

| Colour | Transport Type | Priority |
|---|---|---|
| `mpls` | Dedicated MPLS circuit | Primary |
| `biz-internet` | Broadband / business internet | Secondary |
| `lte` | 4G/LTE backup | Tertiary (failover only) |

### 3.3 Site Roles

| Role | Description | Example Sites |
|---|---|---|
| **Hub** | Regional aggregation; terminates spoke tunnels; connects to DC/cloud | LON-DC01, NYC-DC01, SGP-DC01, DXB-DC01 |
| **Spoke** | Branch site; connects to two hubs for redundancy | LON-BR01 through LON-BR12, NYC-BR01 through NYC-BR08 |
| **Cloud On-Ramp** | vEdge Cloud on AWS/Azure; provides optimised cloud access | AWS-USE1-COR, AZ-WEU-COR, AZ-UAE-COR |

---

## 4. OMP Design

### 4.1 OMP Route Distribution

OMP is the SD-WAN control protocol analogous to BGP in traditional WAN. All vEdge devices establish OMP sessions to both vSmart controllers. The vSmart selects best paths and distributes:

- **OMP Routes** — prefixes reachable via the overlay.
- **TLOC Routes** — transport location identifiers (IP, colour, encap).
- **Service Routes** — service chaining entries (e.g., firewall, IDS).

Route policy is applied on the vSmart using `centralized-policy` blocks pushed from vManage.

### 4.2 OMP Timers

```
omp
 timers
  hello-interval        1
  hold-time             3
  advertisement-interval 1
  graceful-restart-timer 43200
 !
```

### 4.3 OMP Route Filtering

Traffic steering is implemented using data policies and centralized policies. Example: all traffic destined for 10.0.0.0/8 (corporate) prefers `mpls` colour; all traffic to 0.0.0.0/0 (internet) is steered via `biz-internet` or local internet breakout at the branch.

---

## 5. VPN Segmentation

| VPN ID | Name | Description |
|---|---|---|
| 0 | Transport VPN | Underlay — WAN interface IP addresses |
| 1 | Corporate | Enterprise user traffic |
| 2 | Guest | Visitor / BYOD — local internet breakout only |
| 3 | OT/IoT | Operational technology; isolated; no internet |
| 10 | Voice | QoS-prioritised voice/video |
| 512 | Management | Out-of-band management; NMS access |

Segmentation is enforced at the vEdge by assigning interfaces to VPN IDs. Inter-VPN routing is explicitly denied except where a service-route chain through the LON-DC01 firewall cluster is permitted.

---

## 6. High Availability

### 6.1 Dual-Homing

Every spoke site is dual-homed to two regional hubs. OMP redistributes TLOC routes for both hubs. BFD runs on all tunnels with interval 200 ms / multiplier 3 (600 ms detection).

### 6.2 Controller Redundancy

vSmart and vManage operate in Active/Active mode. If one vSmart fails, OMP sessions reconverge to the surviving controller within 3–5 seconds. Policy remains active on vEdge devices during this window (cached).

### 6.3 Transport Failover

Failover sequence:
1. BFD detects tunnel failure on `mpls` transport.
2. vEdge marks TLOC as down; traffic immediately shifts to `biz-internet`.
3. If `biz-internet` also fails, traffic shifts to `lte`.
4. When primary recovers, traffic preempts back within 30 s (preemption timer).

---

## 7. Security

- All overlay tunnels use IPSec (AES-256-GCM) with IKEv2.
- Certificates for device identity are signed by the Contoso PKI.
- vBond validates device serial numbers against the Cisco PnP portal allow-list.
- vManage RBAC restricts policy push to users in the `network-admin` role.
- Configuration audit logs are shipped to the Contoso SIEM (10.200.5.100) via syslog TLS.

---

## 8. IP Addressing Summary

| Segment | Range | Notes |
|---|---|---|
| vSmart/vBond/vManage | 10.200.0.0/24 | Management VPN 512 |
| Hub loopbacks | 10.200.1.0/28 | OMP router-IDs |
| Branch loopbacks | 10.200.2.0/22 | /32 per device |
| Corporate VPN1 | 10.10.0.0/16 | Summary; subnets per region |
| Guest VPN2 | 172.20.0.0/16 | NAT to biz-internet |
| OT/IoT VPN3 | 192.168.100.0/22 | Isolated |
| Voice VPN10 | 10.50.0.0/20 | QoS marked DSCP EF |

---

## 9. References

- Cisco SD-WAN Design Guide v20.x
- Contoso PKI Certificate Policy v1.2
- DD-002: Hub-and-Spoke vs Full Mesh Topology Selection
- DD-004: OSPF Area Design for DC Sites
