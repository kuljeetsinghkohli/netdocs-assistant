# DD-005: Zscaler ZIA Integration Design

**Document Type:** Design Document  
**Author:** Meera Patel, Security Architect  
**Reviewed By:** Alex Mercer, Principal Network Architect  
**Version:** 1.5  
**Date:** 2024-08-30  
**Classification:** Internal — Restricted  
**Zscaler Tenant:** contoso.zslogin.net

---

## 1. Overview

This document describes the integration of Zscaler Internet Access (ZIA) with the Contoso Global SD-WAN fabric. ZIA provides cloud-delivered secure web gateway (SWG), DNS security, and CASB capabilities. The integration routes branch internet-bound traffic directly to Zscaler PoPs, bypassing the hub data centres for internet access.

---

## 2. Integration Architecture

### 2.1 Internet Traffic Flow (Before ZIA)

Pre-ZIA, all internet traffic from branches was backhauled to LON-DC01, traversing the enterprise firewall cluster, then egressing via the LON-DC01 internet uplink. This created:
- Single egress point for all 47 branches.
- High latency for branches distant from LON (e.g., DXB branches: +150 ms).
- Bandwidth bottleneck at LON-DC01 internet uplink (2 × 1 Gbps).

### 2.2 Internet Traffic Flow (With ZIA — Local Internet Breakout)

With ZIA, branches perform **local internet breakout** via their `biz-internet` transport. IPSec tunnels (GRE over IPSec) are established from each vEdge to the nearest Zscaler PoP.

```
Branch vEdge ──(biz-internet transport)──► Zscaler PoP ──► Internet
```

Zscaler PoPs used by region:

| Region | Primary ZIA PoP | Secondary ZIA PoP |
|---|---|---|
| EMEA | Frankfurt, DE | London, UK |
| Americas | Ashburn, VA | Chicago, IL |
| APAC | Singapore | Tokyo, JP |
| Middle East | Dubai, UAE | Riyadh, SA |

### 2.3 Corporate Traffic Flow

Traffic destined for internal Contoso IP space (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) continues to traverse the SD-WAN overlay via OMP and does not touch Zscaler.

---

## 3. SD-WAN Data Policy for ZIA

Traffic steering is enforced via SD-WAN data policies pushed from vManage:

```yaml
# vManage Data Policy — ZIA Steering (simplified representation)
data-policy:
  name: "ZIA-Internet-Breakout"
  vpn: 1
  sequence:
    - seq: 10
      match:
        destination-ip: 10.0.0.0/8
      action: accept                  # Corporate — keep in overlay
    - seq: 20
      match:
        destination-ip: 172.16.0.0/12
      action: accept
    - seq: 30
      match:
        destination-ip: 0.0.0.0/0    # Internet default
      action:
        nat: outside                  # Local breakout
        zscaler: primary              # ZIA tunnel
```

### 3.1 Application-Aware Routing

Microsoft 365 and Zoom traffic is identified using SAAS-specific data policies and steered directly to the internet via ZIA, bypassing any deep inspection (per Zscaler and Microsoft guidance):

- Office 365 categories: `Microsoft-Exchange-Online`, `Microsoft-SharePoint`, `Microsoft-Teams`.
- Policy action: direct NAT (no ZIA inspection) for these categories — uses SD-WAN SaaS optimisation tunnel directly to Microsoft peering IPs.

---

## 4. Zscaler Tunnel Configuration

Each vEdge establishes two GRE/IPSec tunnels to ZIA:

```
# vEdge Zscaler tunnel configuration (template)
interface tunnel1
 description "ZIA-Primary-GRE"
 ip address 10.255.1.1/30
 tunnel source GigabitEthernet0/0   # biz-internet interface
 tunnel destination <Zscaler-PoP-Primary-IP>
 tunnel mode gre ip
 ip mtu 1400
!
interface tunnel2
 description "ZIA-Secondary-GRE"
 ip address 10.255.2.1/30
 tunnel source GigabitEthernet0/0
 tunnel destination <Zscaler-PoP-Secondary-IP>
 tunnel mode gre ip
 ip mtu 1400
!
```

Authentication: HMAC-SHA256 shared secret per site, stored in vManage template variables.

---

## 5. DNS Security

All branch DNS queries are redirected to Zscaler DNS (163.27.0.0/16 — Zscaler DNS range) via the data policy. Internal DNS (Active Directory DNS at 10.200.5.50) is used for `.contoso.internal` domains via split DNS.

```
ip name-server 10.200.5.50           # Internal — .contoso.internal
ip name-server 185.46.212.88         # Zscaler DNS — all other queries
```

---

## 6. Authentication and SAML

Branch users are authenticated to ZIA using SAML 2.0 with Contoso Azure AD as the IdP:

- SAML IDP: `login.microsoftonline.com/contoso-tenant-id`
- Zscaler SP: `contoso.zslogin.net`
- Attribute mapping: `UPN` → Zscaler `user`, `Department` → Zscaler group for policy.

Machine-authenticated tunnels (for non-browser traffic) use Zscaler Client Connector (ZCC) deployed via Microsoft Intune.

---

## 7. Bypass Lists

The following destinations bypass ZIA inspection (direct NAT only):

- Contoso AWS VPC CIDRs: 10.100.0.0/16 (us-east-1), 10.101.0.0/16 (eu-west-1)
- Contoso Azure VNet CIDRs: 10.102.0.0/16 (West Europe), 10.103.0.0/16 (UAE North)
- Microsoft 365 optimise category (per Microsoft documentation)

---

## 8. Rollback Plan

If ZIA integration causes traffic disruption:
1. Disable ZIA data policy on affected sites via vManage (takes effect within 30 s).
2. Traffic reverts to hub-backhaul path via `mpls` transport.
3. Incident to be raised as P1 (CRQ-XXXX rollback procedure).

---

## 9. References

- DD-001: SD-WAN Overlay Design
- Zscaler ZIA Deployment Guide for Cisco SD-WAN
- Microsoft 365 Network Connectivity Principles
- Contoso Azure AD SSO Configuration Guide
