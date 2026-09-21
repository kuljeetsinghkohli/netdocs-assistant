# RB-004: SD-WAN Failover Test Procedure

**Runbook Type:** Operational (Planned Maintenance)  
**Author:** Sandra Lin, Network Engineering Lead  
**Version:** 1.3  
**Last Updated:** 2024-07-05  
**Applies To:** All branch sites with dual transport  
**Pre-requisite:** Change Request approved (see §1)

---

## 1. Change Control Requirement

This runbook describes a **planned** transport failover test. A change request MUST be approved before execution. Template CRQ-TEMPLATE-FAILOVER should be raised in ITSM with:
- Site(s) affected
- Test window (minimum 30 minutes)
- Approver: Network Engineering Lead

---

## 2. Pre-Test Checks

Before starting the test, verify baseline state:

```
# On affected vEdge:
show omp summary
show bfd sessions
show tunnel statistics
show sdwan app-route statistics
```

Note down:
- Current active transport for each VPN
- Prefix count via each transport
- BFD session states (all should be Up)

Notify NOC: "Failover test starting at <HH:MM> UTC on site <site-id>. Expected duration 30 min."

---

## 3. Simulate Primary Transport Failure

Shut down the primary (MPLS) transport interface on the vEdge:

```
# On vEdge at test site:
interface GigabitEthernet0
 shutdown
!
```

This triggers:
1. BFD session on MPLS TLOC drops within 600 ms.
2. OMP withdraws MPLS TLOC from affected site.
3. Traffic reroutes to `biz-internet` transport.

---

## 4. Validate Failover

Wait 5 seconds after interface shutdown, then validate:

```
show bfd sessions
show omp tlocs
show tunnel statistics
show sdwan app-route statistics
```

Expected:
- BFD sessions on `biz-internet` remain Up.
- OMP routes for VPN 1, 10 still present (via biz-internet TLOC).
- App-route statistics show traffic flowing on biz-internet.

From a test workstation at the site, confirm application reachability:

```bash
ping -c 100 10.10.0.10   # LON-DC01 server
curl -I https://sharepoint.contoso.internal
```

Record:
- Failover time (seconds from interface shutdown to first successful ping).
- Any application errors observed during failover window.

---

## 5. Validate LTE Fallback (If Applicable)

For sites with LTE as tertiary:

```
interface GigabitEthernet1   # biz-internet
 shutdown
!
```

Validate BFD and OMP for LTE TLOC. Confirm application reachability.

Allow LTE failover to stabilise for 2 minutes before restoring.

---

## 6. Restore Primary Transport

Restore the primary transport interface:

```
interface GigabitEthernet0
 no shutdown
!
```

Monitor preemption: within 30 seconds, traffic should re-prefer the MPLS transport:

```
watch show omp tlocs
watch show sdwan app-route statistics
```

Verify traffic has moved back to MPLS by checking the `colour` field in app-route statistics.

---

## 7. Post-Test Documentation

Record all findings in the change ticket:
- Measured failover time (BFD detection + OMP reconvergence).
- Application impact during failover window.
- Any unexpected behaviour.
- Pass/Fail criteria: failover < 2 s is PASS; > 5 s requires investigation.

Update vManage Site Health dashboard to confirm all tunnels green.

---

## References

- DD-001: SD-WAN Overlay Design (§6 High Availability)
- DD-002: Hub-and-Spoke vs Full Mesh (§5 Failover)
- RB-003: Tunnel Down Recovery (for unplanned failures)
