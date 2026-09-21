# DD-006: Hybrid Cloud Connectivity — AWS and Azure

**Document Type:** Design Document  
**Author:** Alex Mercer, Principal Network Architect  
**Reviewed By:** Meera Patel, Security Architect; Tom Bradley, DC Network Engineer  
**Version:** 2.1  
**Date:** 2024-11-01  
**Classification:** Internal — Restricted  
**AWS Account:** 123456789012 (Contoso Production)  
**Azure Subscription:** contoso-prod-subscription-01

---

## 1. Overview

This document describes hybrid cloud connectivity between Contoso Global on-premises infrastructure and AWS (us-east-1, eu-west-1) and Azure (West Europe, UAE North). The design provides dedicated, private connectivity supplemented by SD-WAN cloud on-ramp for resilience.

---

## 2. Connectivity Methods

### 2.1 AWS — AWS Direct Connect

Contoso uses AWS Direct Connect (DX) for primary connectivity to AWS:

| Parameter | Value |
|---|---|
| Connection type | Hosted Connection (via Equinix) |
| Bandwidth | 1 Gbps per connection |
| Locations | London (Equinix LD4), New York (Equinix NY4) |
| Virtual Interfaces | Private VIF (VPC access) + Transit VIF (Transit Gateway) |
| BGP ASN (AWS) | 64512 |
| BGP ASN (Contoso CE) | 65001 |
| Peering IPs | 169.254.100.1/30 (AWS), 169.254.100.2/30 (Contoso) |

Direct Connect terminates at LON-DC01-RTR01 (primary) and NYC-DC01-RTR01 (secondary via DX cross-connect).

#### 2.1.1 AWS Transit Gateway

A Transit Gateway (TGW) in each AWS region aggregates connectivity:

- VPCs attach to the TGW.
- Contoso on-premises prefixes are propagated via DX BGP into the TGW route table.
- TGW route table: `10.0.0.0/8` → on-prem DX attachment.

AWS VPC CIDRs:

| VPC | CIDR | Region | Purpose |
|---|---|---|---|
| prod-vpc-use1 | 10.100.0.0/16 | us-east-1 | Production workloads |
| dev-vpc-use1 | 10.100.128.0/17 | us-east-1 | Dev/test |
| prod-vpc-euw1 | 10.101.0.0/16 | eu-west-1 | EMEA production |

### 2.2 AWS — SD-WAN Cloud On-Ramp (Backup)

A Cisco Catalyst SD-WAN (CSR 1000V) is deployed as a cloud on-ramp in AWS us-east-1:

- Instance: `AWS-USE1-COR` (CSR 1000V — c5.xlarge)
- OMP participant: connects to LON-DC01 and NYC-DC01 vSmart
- IPSec tunnels to all 4 regional DC hubs
- Provides failover path if DX is unavailable

Traffic flow via cloud on-ramp:
```
Branch → OMP overlay → AWS-USE1-COR → AWS TGW → VPC
```

### 2.3 Azure — ExpressRoute

Azure connectivity uses ExpressRoute (ER) via the same Equinix LD4 facility:

| Parameter | Value |
|---|---|
| Circuit bandwidth | 500 Mbps |
| Peering type | Private Peering |
| Provider | Equinix Fabric (co-location) |
| BGP ASN (Azure) | 12076 |
| BGP ASN (Contoso) | 65001 |
| Primary peering | 172.16.10.0/30 |
| Secondary peering | 172.16.10.4/30 |

ExpressRoute terminates at LON-DC01-RTR01 via a dedicated VLAN (VLAN 200).

Azure VNet CIDRs:

| VNet | CIDR | Region | Purpose |
|---|---|---|---|
| prod-vnet-weu | 10.102.0.0/16 | West Europe | Production |
| dev-vnet-weu | 10.102.128.0/17 | West Europe | Dev/test |
| prod-vnet-uaen | 10.103.0.0/16 | UAE North | Middle East production |

### 2.4 Azure — SD-WAN Cloud On-Ramp (Backup)

Two cloud on-ramp CSR instances in Azure:

- `AZ-WEU-COR` — West Europe (Standard_D4s_v3)
- `AZ-UAE-COR` — UAE North (Standard_D4s_v3)

---

## 3. BGP Design for Cloud Connectivity

### 3.1 AWS Direct Connect BGP Policy

Inbound from AWS (DX BGP):
```
route-map RM-FROM-AWS-DX-IN permit 10
  match ip address prefix-list PL-AWS-VPCS
  set local-preference 180
  set community 65001:400
!
ip prefix-list PL-AWS-VPCS seq 10 permit 10.100.0.0/15 le 24
ip prefix-list PL-AWS-VPCS seq 20 permit 10.101.0.0/16 le 24
```

Outbound to AWS (DX BGP):
```
route-map RM-TO-AWS-DX-OUT permit 10
  match ip address prefix-list PL-CONTOSO-TO-AWS
  set community 65001:200
!
ip prefix-list PL-CONTOSO-TO-AWS seq 10 permit 10.10.0.0/16
ip prefix-list PL-CONTOSO-TO-AWS seq 20 permit 10.11.0.0/16
```

### 3.2 Azure ExpressRoute BGP Policy

ExpressRoute uses Microsoft BGP community values for route categorisation. Contoso accepts all Azure private peering routes from community `12076:20003` (West Europe) and `12076:20023` (UAE North).

---

## 4. QoS and Traffic Priority

Cloud-bound traffic QoS marking:

| Traffic Type | DSCP Marking | Queue |
|---|---|---|
| SAP HANA DB replication | EF (46) | Priority |
| Azure Active Directory | AF31 (26) | Assured |
| General cloud app | AF21 (18) | Best effort |
| Backup / bulk transfer | CS1 (8) | Scavenger |

---

## 5. Resiliency and Failover

Failover hierarchy for cloud access:

```
Primary:  Direct Connect / ExpressRoute (private, dedicated)
Secondary: SD-WAN Cloud On-Ramp over biz-internet (IPSec overlay)
Tertiary:  Zscaler ZIA → cloud provider peering (internet only, limited to SaaS)
```

Switchover is automatic via BGP preference (local-pref):
- DX/ER routes: local-pref 180
- Cloud on-ramp OMP routes: local-pref 120
- Zscaler path: not in BGP (policy-based routing only)

---

## 6. Security Zones

Cloud resources are placed in dedicated security zones in the Contoso Palo Alto firewall (LON-DC01-FW01):

| Zone | Description | Policy |
|---|---|---|
| CLOUD-PROD | AWS/Azure production VPCs | Permit from CORP; deny from GUEST, OT |
| CLOUD-DEV | Dev/test cloud VPCs | Permit from DEV VLAN only |
| CLOUD-MGMT | Cloud management plane | Permit from MGMT only |

---

## 7. References

- DD-001: SD-WAN Overlay Design (cloud on-ramp sections)
- DD-003: BGP Policy Design
- DD-005: Zscaler ZIA Integration
- AWS Direct Connect User Guide
- Azure ExpressRoute Documentation
- Contoso AWS Landing Zone Design v3.0
