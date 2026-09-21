# RB-001: BGP Peer Flapping — Diagnosis and Recovery

**Runbook Type:** Operational  
**Author:** Priya Nair, BGP/Routing SME  
**Version:** 2.1  
**Last Updated:** 2024-08-15  
**Applies To:** All Contoso hub routers (LON-DC01-RTR01/02, NYC-DC01-RTR01, SGP-DC01-RTR01, DXB-DC01-RTR01)  
**Severity Trigger:** BGP peer down alert from SIEM / NOC monitoring

---

## Prerequisites

- SSH access to the affected router (MGMT VPN 512, jump host: 10.200.5.1)
- Read-only access to vManage (https://10.200.0.20) for OMP state
- ITSM access to raise/update incident ticket

---

## Step 1 — Confirm the Flap and Identify the Peer

Log in to the affected router and check BGP summary:

```bash
ssh netops@<router-management-ip>
```

```
show bgp summary
show bgp neighbors <peer-ip> | include BGP state|flap|reset reason
show ip bgp neighbors <peer-ip>
```

Expected output if peer is flapping:
```
Neighbor        V    AS   MsgRcvd  MsgSent  TblVer  InQ  OutQ  Up/Down  State/PfxRcd
172.16.0.1      4  12345   10231    10198     4521    0     0  00:02:11  Established
```

Check flap count and last reset reason:
```
show ip bgp neighbors 172.16.0.1 | include resets|flaps|reset reason
```

---

## Step 2 — Check Underlying Connectivity

Test Layer 3 reachability to the BGP peer IP:

```
ping 172.16.0.1 repeat 100 size 1500 df-bit
traceroute 172.16.0.1
```

If pings fail: escalate to provider (BT/Lumen NOC contact in §8 References).

If pings succeed but BGP is down: proceed to Step 3.

---

## Step 3 — Review BGP Hold Timer and Keepalives

Check if the session is timing out due to keepalive loss:

```
show ip bgp neighbors 172.16.0.1 | include Hold time|Keepalive|Last read|Last write
```

Compare configured vs negotiated timers:
```
show run | section router bgp
```

If hold time mismatch: align timers with provider. Contoso standard: hold-time 90 s, keepalive 30 s.

```
router bgp 65001
 neighbor 172.16.0.1 timers 30 90
```

---

## Step 4 — Check for Route Policy Issues

A route policy rejecting all prefixes can cause BGP to appear flapping if the provider resets on 0-prefix receipt:

```
show ip bgp neighbors 172.16.0.1 received-routes | head 20
show ip bgp neighbors 172.16.0.1 advertised-routes | head 20
```

Verify route-map is not blocking all:
```
show route-map RM-FROM-BT-IN
show ip bgp regexp ^12345
```

---

## Step 5 — Check Interface and Physical Layer

Verify the WAN interface is up/up and error-free:

```
show interfaces GigabitEthernet0/0/1
show interfaces GigabitEthernet0/0/1 counters errors
```

Look for: input errors, CRC, giants, resets. If non-zero errors: raise with provider as physical layer issue.

---

## Step 6 — Check BFD Status

If BFD is triggering BGP resets:

```
show bfd neighbors
show bfd neighbors details
```

If BFD is flapping but the peer is reachable: adjust BFD timers:

```
router bgp 65001
 neighbor 172.16.0.1 fall-over bfd
!
interface GigabitEthernet0/0/1
 bfd interval 300 min_rx 300 multiplier 3
```

Note: BFD timer changes take effect immediately and do not require BGP reset.

---

## Step 7 — Controlled BGP Reset (Last Resort)

If all above checks pass and the session remains unstable, perform a soft reset:

```
clear ip bgp 172.16.0.1 soft
```

If soft reset fails, perform hard reset (causes brief traffic loss — confirm with NOC first):

```
clear ip bgp 172.16.0.1
```

After reset: monitor `show bgp summary` for 5 minutes to confirm stability.

---

## Step 8 — Post-Recovery Validation

After stabilisation:
1. Verify prefix count is within expected range (see IPAM for expected counts).
2. Confirm OMP route redistribution is intact: `show sdwan omp routes`.
3. Update incident ticket with root cause and resolution.
4. If recurrence is likely, raise change request to adjust dampening or BFD timers.

---

## References

- DD-003: BGP Policy Design
- BT NOC: +44 800 800 152 (ref: Contoso circuit ID BT-WAN-LON-001)
- Lumen NOC: +1 800 359 5353 (ref: circuit ID LU-WAN-NYC-002)
- ITSM: https://itsm.contoso.internal
