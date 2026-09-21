# RB-008: OSPF Neighbour Down — Diagnosis and Recovery

**Runbook Type:** Operational  
**Author:** Tom Bradley, DC Network Engineer  
**Version:** 1.5  
**Last Updated:** 2024-09-01  
**Applies To:** All DC routers participating in OSPF Process 1  
**Severity Trigger:** OSPF neighbour down alert from SIEM / Zabbix

---

## Prerequisites

- SSH access to affected DC router
- Topology diagram (DD-004 OSPF Area Design)
- ITSM access to raise incident

---

## Step 1 — Identify the Failed Adjacency

Log in to affected router:

```bash
ssh netops@<router-mgmt-ip>
```

```
show ip ospf neighbor
show ip ospf neighbor detail
show ip ospf interface brief
```

Identify the neighbor in EXSTART, EXCHANGE, or DOWN state. Note:
- Neighbor IP
- Interface the adjacency is on
- Area number
- State and duration in that state

---

## Step 2 — Check Physical and Layer 2

```
show interfaces <interface>
show interfaces <interface> counters errors
show cdp neighbors <interface> detail
```

Verify:
- Interface is up/up (not administratively down)
- No CRC errors, input drops, or resets
- CDP confirms the expected remote device on this interface (use as a sanity check)

If interface is down: check cable, SFP, and switch port. Escalate to DC hands team if physical access needed (dcops@contoso.internal).

---

## Step 3 — Verify OSPF Configuration Consistency

OSPF adjacencies fail if these parameters mismatch between neighbors:

| Parameter | Command to Check |
|---|---|
| Hello/Dead timers | `show ip ospf interface <int>` |
| Area ID | `show ip ospf interface <int>` |
| Network type | `show ip ospf interface <int>` |
| Authentication | `show ip ospf interface <int>` |
| MTU | `show interfaces <int>` + `ip ospf mtu-ignore` |

Most common cause of EXSTART/EXCHANGE stuck state: **MTU mismatch**. Verify both sides have matching MTU or apply `ip ospf mtu-ignore` as a temporary measure:

```
interface GigabitEthernet0/0/1
 ip ospf mtu-ignore
!
```

---

## Step 4 — Check OSPF Authentication

All OSPF adjacencies use MD5 authentication (DD-004 §6). Mismatched keys are a common cause:

```
show ip ospf interface <interface> | include Auth
debug ip ospf adj  (caution: generates volume on busy router)
```

If authentication mismatch suspected:
1. Verify key ID and MD5 key value on both ends.
2. Keys are stored in Contoso HashiCorp Vault: `vault kv get network/ospf/area0-key`.
3. Reconfigure if mismatch confirmed:

```
interface GigabitEthernet0/0/1
 ip ospf message-digest-key 1 md5 <correct-key>
!
```

Note: Authentication key changes may cause a brief OSPF adjacency reset (few seconds).

---

## Step 5 — Review OSPF SPF and LSA Activity

If adjacency appears to form but then drops repeatedly:

```
show ip ospf statistics
show ip ospf database | include LSA|Age
debug ip ospf events  (limit to 30 seconds: undebug all after)
```

Check for LSA storms (Type 4/5 from ASBR) or database overflow that could be destabilising the process.

---

## Step 6 — Controlled OSPF Process Reset

If above steps do not resolve the adjacency:

```
clear ip ospf process   (confirm with 'yes')
```

This resets all OSPF adjacencies on the router. Ensure traffic has a redundant path (verify with NOC) before performing this action.

---

## Step 7 — Post-Recovery Validation

```
show ip ospf neighbor
show ip route ospf
show ip ospf database summary
```

Verify:
- All expected neighbours are in FULL state.
- OSPF route table matches pre-incident baseline.
- No Type 5 LSA flooding anomalies.

Inform NOC that the issue is resolved. Update incident ticket.

---

## References

- DD-004: OSPF Area Design
- DC Hands Team: dcops@contoso.internal / +44 20 1234 5679
- Contoso HashiCorp Vault: https://vault.contoso.internal
