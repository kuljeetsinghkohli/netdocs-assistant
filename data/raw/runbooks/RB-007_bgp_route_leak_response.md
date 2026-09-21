# RB-007: BGP Route Leak Detection and Containment

**Runbook Type:** Operational (Incident Response)  
**Author:** Priya Nair, BGP/Routing SME  
**Version:** 1.2  
**Last Updated:** 2024-05-22  
**Applies To:** All hub routers with eBGP peering  
**Severity Trigger:** Unexpected prefix advertisement detected by SIEM or provider notification

---

## Step 1 — Identify the Leaked Prefix

Receive notification from provider (BT/Lumen) or SIEM alert: unexpected prefix being advertised by Contoso to provider.

Confirm leaked prefix on affected router:

```
show ip bgp neighbors <provider-peer-ip> advertised-routes | include <leaked-prefix>
show ip bgp <leaked-prefix>
```

Determine the origin of the prefix:

```
show ip bgp <leaked-prefix> | include Origin|AS_PATH|Local Pref|Next Hop
```

---

## Step 2 — Immediate Containment — Null Route

Apply a null route to the leaked prefix to prevent forwarding while diagnosis continues:

```
ip route <leaked-prefix> Null0 254
```

This adds a high-metric static route that blackholes traffic to the leaked prefix without withdrawing BGP advertisements immediately (which could cause a route flap visible to the provider).

---

## Step 3 — Block Advertisement to Provider

Apply a route-map to block the leaked prefix from being advertised:

```
ip prefix-list PL-EMERGENCY-DENY seq 5 deny <leaked-prefix>
ip prefix-list PL-EMERGENCY-DENY seq 9999 permit 0.0.0.0/0 le 32
!
route-map RM-TO-BT-OUT permit 5
 match ip address prefix-list PL-EMERGENCY-DENY
!
clear ip bgp <provider-peer-ip> soft out
```

Verify the prefix is no longer advertised:

```
show ip bgp neighbors <provider-peer-ip> advertised-routes | include <leaked-prefix>
```

---

## Step 4 — Root Cause Investigation

Common causes of route leaks:
1. **Incorrect redistribution**: OSPF-to-BGP or static-to-BGP redistribute without adequate prefix-list filtering.
2. **Policy misconfiguration**: A route-map `permit` clause was too broad (e.g., `permit 0.0.0.0/0 le 32`).
3. **ZTP template error**: A new device deployed with incorrect outbound policy.

Check redistribution config:

```
show run | section router bgp
show run | section route-map RM-TO-BT
show run | include redistribute
```

---

## Step 5 — Permanent Fix

After identifying the cause:
1. Update the affected prefix-list or route-map.
2. Remove the emergency null route: `no ip route <leaked-prefix> Null0 254`.
3. Remove the emergency prefix-list deny entry.
4. Perform soft outbound reset: `clear ip bgp <provider-peer> soft out`.
5. Confirm only expected prefixes are advertised.

---

## Step 6 — Post-Incident Review

1. Document in the incident ticket: timeline, leaked prefix, root cause, fix.
2. Raise a change request to codify the fix in the standard config template.
3. Schedule a BGP policy audit within 2 weeks (use RB-008 if available).
4. Notify provider NOC that the issue is resolved.

---

## References

- DD-003: BGP Policy Design (§3.2 Outbound Policy)
- BT NOC: +44 800 800 152
- Lumen NOC: +1 800 359 5353
- ITSM: https://itsm.contoso.internal
