# RB-005: vEdge Zero-Touch Provisioning (ZTP) — New Site Deployment

**Runbook Type:** Operational (Deployment)  
**Author:** Alex Mercer, Principal Network Architect  
**Version:** 2.5  
**Last Updated:** 2024-11-15  
**Applies To:** New branch site deployments using Cisco vEdge or Catalyst SD-WAN  
**Pre-requisite:** Site survey complete; circuits provisioned; device shipped

---

## Step 1 — Pre-Provisioning (Before Device Ships)

### 1.1 Add Device to vManage

1. Log in to vManage (https://10.200.0.20) as `network-admin`.
2. Navigate to **Configuration > Devices > WAN Edge List**.
3. Click **Upload WAN Edge List** and upload the device serial number CSV.
   - CSV format: `Chassis Serial,UUID,Model,Description`
   - Example: `FTX2345G001,xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx,vedge-1000,LON-BR13`
4. Verify the device appears in the **Invalid** bucket (normal before claiming).

### 1.2 Create or Assign Device Template

1. Navigate to **Configuration > Templates > Device Template**.
2. For standard branch: clone template `BRANCH-VEDGE-STANDARD-v3`.
3. Modify site-specific variables:
   - `system_site_id`: new site ID (e.g., 1013 for LON-BR13)
   - `system_host_name`: `LON-BR13-VE01`
   - `vpn0_interface_ip`: WAN IP assigned by ISP
   - `vpn512_management_ip`: 10.200.2.x/32 (from IPAM)
4. Attach template to the device serial number.

### 1.3 Pre-Stage Device Certificate

If using manual certificate (non-Cisco PnP):
1. Generate CSR on vManage: **Configuration > Certificates > WAN Edge > Generate CSR**.
2. Submit CSR to Contoso PKI CA.
3. Upload signed certificate to vManage.

---

## Step 2 — Physical Installation at Site

At the branch site, the local hands technician:
1. Rack and cable the vEdge (power + WAN circuits + LAN uplink to access switch).
2. Verify WAN circuit LEDs are active.
3. Do NOT configure the device — ZTP handles all configuration.

---

## Step 3 — ZTP Process (Automatic)

Once powered on and WAN connected, the vEdge:
1. Sends DHCP request on VPN 0 (transport VPN).
2. If DHCP Option 43 is present, uses vBond IP from DHCP. Otherwise, uses factory default vBond: contacts Cisco PnP portal.
3. Contoso vBond (10.200.0.30 / 10.200.0.31) authenticates the serial number.
4. vBond redirects the device to vManage for certificate exchange.
5. vManage pushes the device template (pre-attached in Step 1.2).
6. Device completes ZTP; OMP session to vSmart comes up automatically.

Typical ZTP duration: 5–10 minutes.

---

## Step 4 — Post-ZTP Validation

From vManage, verify:
1. **Monitor > Network**: New device appears with green status.
2. **Control Status**: OMP sessions to both vSmarts are Up.
3. **BFD Status**: BFD sessions to hub(s) are Up.

From the new vEdge CLI:

```
show system status
show control connections
show omp summary
show bfd sessions
show tunnel statistics
```

From a workstation at the site, test corporate connectivity:

```bash
ping 10.10.0.10        # LON-DC01 server
ping 10.10.0.20        # NYC-DC01 server
nslookup contoso.internal 10.200.5.50
```

---

## Step 5 — Update IPAM and Documentation

1. Update IPAM with new site IP assignments.
2. Add site to the site inventory spreadsheet (SharePoint: Network > Sites > WAN-Sites.xlsx).
3. Raise change ticket CRQ to record the deployment.
4. Notify NOC to add site to monitoring (Zabbix template: BRANCH-VEDGE).

---

## References

- DD-001: SD-WAN Overlay Design (§3.3 Site Roles, §7 Security)
- Cisco ZTP Configuration Guide
- Contoso IPAM: https://ipam.contoso.internal
