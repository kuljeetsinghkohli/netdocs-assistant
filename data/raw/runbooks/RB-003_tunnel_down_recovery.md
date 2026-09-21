# RB-003: SD-WAN Tunnel Down — Recovery Procedure

**Runbook Type:** Operational  
**Author:** Sandra Lin, Network Engineering Lead  
**Version:** 2.0  
**Last Updated:** 2024-10-10  
**Applies To:** All vEdge IPSec tunnels (site-to-site and cloud on-ramp)  
**Severity Trigger:** Tunnel-down alert in vManage / SIEM

---

## Prerequisites

- vManage access (https://10.200.0.20)
- vEdge SSH access via jump host 10.200.5.1
- Carrier NOC contacts (see References)

---

## Step 1 — Identify the Affected Tunnel

In vManage, navigate to **Monitor > Network** and search for the affected device. Under **Tunnel Statistics**, identify the down tunnel(s).

Note the following:
- **Local TLOC**: IP + colour (e.g., 203.0.113.5 / mpls)
- **Remote TLOC**: IP + colour (e.g., 198.51.100.10 / mpls)
- **VPN**: Affected VPN segment
- **Time down**: Duration of outage

---

## Step 2 — Verify Transport Interface Status

SSH to the affected vEdge:

```bash
ssh admin@<vedge-mgmt-ip>
```

```
show interface
show interface GigabitEthernet0   # Transport 0 (usually MPLS)
show interface GigabitEthernet1   # Transport 1 (usually biz-internet)
```

Check for:
- Interface up/down state
- Input/output errors (CRC, drops)
- IP address assigned (DHCP or static)

If the transport interface is down: escalate to provider immediately (Step 7).

---

## Step 3 — Check IPSec / DTLS Tunnel Status

```
show tunnel statistics
show ipsec inbound-connections
show ipsec outbound-connections
```

Verify IKEv2 SA status:

```
show crypto ikev2 sa
show crypto ipsec sa
```

If IKE SA is in INIT or MM_WAIT state: NAT traversal may be failing. Check:

```
show nat translations
show sdwan nat-table
```

---

## Step 4 — BFD Session Validation

BFD failure triggers tunnel down in OMP. Check BFD:

```
show bfd sessions
show bfd sessions detail
```

If BFD is detecting loss:
- Check for packet loss on the transport: `ping <remote-transport-ip> repeat 1000`
- Evaluate if BFD timers are too aggressive for the circuit quality (consult DD-001 §6.1)

Temporary workaround — increase BFD multiplier (reduces sensitivity, increases failover time):

```
interface tunnel-interface
 bfd interval 1000 min_rx 1000 multiplier 5
!
```

Apply the same on the remote vEdge. Restore to standard values after investigation.

---

## Step 5 — Force Tunnel Re-establishment

Clear the IPSec SA to force renegotiation:

```
clear crypto ipsec sa peer <remote-transport-ip>
clear crypto ikev2 sa remote <remote-transport-ip>
```

Monitor:

```
debug platform hardware ipsec
show tunnel statistics
```

Wait 60 seconds. If tunnel does not come up: proceed to Step 6.

---

## Step 6 — Reload Tunnel Interface

As a non-service-impacting step (OMP will reroute to backup transport immediately):

```
interface GigabitEthernet0
 shutdown
 no shutdown
!
```

Monitor BFD and OMP:

```
show bfd sessions
show omp tlocs
```

---

## Step 7 — Escalation

If the transport interface is down or there is a persistent L1/L2 issue:

| Provider | Contact | Ref |
|---|---|---|
| BT MPLS | +44 800 800 152 | Circuit: BT-WAN-LON-001 |
| Lumen | +1 800 359 5353 | Circuit: LU-WAN-NYC-002 |
| Etisalat | +971 800 5800 | Circuit: ET-WAN-DXB-001 |
| ISP (biz-internet) | +44 330 123 4567 | Account: CONT-BBand-001 |

When calling: provide site name, circuit ID, and time the outage started. Request a line test and confirmation of physical status.

---

## Step 8 — Post-Recovery Validation

After tunnel recovery:

```
show bfd sessions
show omp tlocs
show omp routes vpn 1 | include <remote-prefix>
ping vpn 1 <remote-host-ip>
```

Confirm traffic is flowing via the expected transport (check with `show policy service-path vpn 1 interface <app-traffic>`).

Update the incident ticket with root cause (transport failure / IKE issue / BFD false-positive / config error).

---

## References

- DD-001: SD-WAN Overlay Design (§6 HA, §3.2 Transport Colours)
- RB-002: OMP Route Issues
- ITSM: https://itsm.contoso.internal
