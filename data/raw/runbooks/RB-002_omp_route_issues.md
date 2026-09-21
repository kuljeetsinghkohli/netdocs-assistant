# RB-002: OMP Route Issues — SD-WAN Overlay Troubleshooting

**Runbook Type:** Operational  
**Author:** Alex Mercer, Principal Network Architect  
**Version:** 1.8  
**Last Updated:** 2024-09-20  
**Applies To:** All vEdge and Catalyst SD-WAN devices; vSmart controllers  
**Severity Trigger:** Branch site unreachable via SD-WAN overlay; OMP route missing alert

---

## Prerequisites

- Access to vManage (https://10.200.0.20) — `netops` RBAC role minimum
- SSH access to affected vEdge (via vManage SSH proxy or jump host 10.200.5.1)
- vSmart CLI access (10.200.0.10, 10.200.0.11)

---

## Step 1 — Check OMP Session Status on vEdge

Log in to the affected vEdge device:

```bash
ssh admin@<vedge-management-ip>
```

```
show omp summary
show omp peers
```

Expected output: OMP sessions to both vSmart controllers in `Up` state. If sessions are down, proceed to Step 2. If sessions are up but routes are missing, proceed to Step 3.

---

## Step 2 — OMP Sessions Down: Diagnose Connectivity

Check if vEdge can reach vSmart controllers:

```
ping vpn 512 10.200.0.10
ping vpn 512 10.200.0.11
traceroute vpn 512 10.200.0.10
```

If connectivity fails from VPN 512 (Management): check that the `tunnel-interface` configuration on the vEdge transport interface is correct and that the vBond has been reached:

```
show control connections
show control connection-history
```

If vBond is unreachable: verify DNS resolution and vBond IP in device config:

```
show run | section system
show system status
```

---

## Step 3 — OMP Routes Missing: Check vSmart Policy

On vManage, navigate to **Monitor > Network > Select Device > OMP Routes**.

Alternatively, from vSmart CLI:

```
show omp routes vpn 1
show omp routes vpn 1 | filter prefix 10.10.0.0/16
```

If routes exist on vSmart but are not reaching the vEdge:

```
# On vEdge:
show omp received-routes
show omp tlocs
```

Check if a centralized policy is filtering routes on the vSmart:

```
# On vSmart:
show policy from-vsmart
show omp routes detail | include rejected
```

---

## Step 4 — Check TLOC Reachability

OMP routes depend on TLOC (Transport Location) reachability. If a TLOC is down, associated routes are withdrawn:

```
# On vEdge:
show omp tlocs
show bfd sessions
show bfd sessions detail
```

If BFD sessions are down: the IPSec tunnels to the remote TLOC are failing. Check tunnel status:

```
show tunnel statistics
show interface tunnel-interface
```

Verify that the transport IP (WAN IP) is correct and that the interface is up/up:

```
show interface GigabitEthernet0
```

---

## Step 5 — Re-advertise OMP Routes (Temporary Fix)

If OMP routes need to be refreshed without a hard reset:

```
# On vEdge:
request omp rediscover
```

Monitor for route recovery:

```
watch show omp received-routes
```

---

## Step 6 — Check Control Plane Certificate Issues

OMP sessions require valid device certificates. If the certificate is expired or revoked:

```
show certificate installed
show certificate validity
show certificate status
```

If expired: contact NOC to initiate certificate renewal via vManage PKI workflow.

---

## Step 7 — Escalation Path

If the above steps do not resolve the issue:

1. Capture full diagnostics: `request technical-support` (uploads to vManage automatically).
2. Escalate to Cisco TAC with case number referencing the tech-support bundle.
3. Open P1 incident in ITSM if more than 1 site is affected.

---

## Step 8 — Post-Recovery Validation

```
show omp summary
show omp routes vpn 1
show bfd sessions
ping vpn 1 <remote-branch-ip>
```

Confirm route table reflects expected prefixes and no unexpected policy rejections remain.

---

## References

- DD-001: SD-WAN Overlay Design
- RB-001: BGP Flap Diagnosis (if MPLS underlay is implicated)
- Cisco vManage/vSmart documentation: https://developer.cisco.com/sdwan
- NOC escalation: noc@contoso.internal / +44 20 1234 5678
