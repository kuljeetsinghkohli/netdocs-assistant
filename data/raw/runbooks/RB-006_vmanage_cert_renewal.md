# RB-006: vManage Certificate Renewal

**Runbook Type:** Operational (Scheduled Maintenance)  
**Author:** Sandra Lin, Network Engineering Lead  
**Version:** 1.1  
**Last Updated:** 2024-04-10  
**Applies To:** vManage, vSmart, vBond, and all WAN Edge device certificates  
**Schedule:** Annual (certificates expire after 2 years; begin process 60 days before expiry)

---

## Step 1 — Check Certificate Expiry Dates

On vManage:

```
Navigate: Administration > Certificate Management > WAN Edge Certificate
Filter: Expiry < 60 days
```

For controller certificates:

```bash
# On vManage CLI:
show certificate installed
show certificate validity
```

Export the expiry report to CSV for tracking.

---

## Step 2 — Renew vManage / Controller Certificates

1. On vManage, navigate to **Administration > Certificate Management > Controllers**.
2. Select the certificate to renew (vManage, vSmart-1, vSmart-2, vBond-1, vBond-2).
3. Click **Generate CSR** for each controller.
4. Submit CSRs to Contoso PKI CA (email: pki@contoso.internal or via CMP API at https://pki.contoso.internal/cmp).
5. Download signed certificates (PEM format).
6. Upload to vManage under **Certificate Management > Install Certificate**.

**Important:** Install vManage certificate first, then vSmart, then vBond. Out-of-order installation can cause OMP disruption.

---

## Step 3 — Renew WAN Edge Device Certificates

For devices using the Contoso PKI (not Cisco PnP cloud):

1. In vManage: **Configuration > Certificates > WAN Edge**.
2. Select all devices expiring within 60 days.
3. Click **Renew Certificate** — vManage will auto-generate CSR and submit to the configured CA if automated enrolment is enabled.

If automated enrolment is not enabled:
1. Export CSRs for each device.
2. Submit to PKI CA in batch.
3. Import signed certificates back to vManage.
4. Push certificates to devices: **Send to Controllers**.

---

## Step 4 — Validate Certificate Installation

On each vEdge:

```
show certificate installed
show certificate validity
show certificate status
show control connections
```

Expected: certificate valid for 2 years from today; control connections status: Valid.

On vSmart:

```
show certificate installed
show omp summary
```

Confirm OMP sessions did not drop during certificate push (certificate update is non-disruptive if performed correctly).

---

## Step 5 — Post-Renewal Documentation

1. Record new expiry dates in the Certificate Register (SharePoint: Security > PKI > Certificate-Register.xlsx).
2. Set a calendar reminder for the next renewal cycle (next renewal: current expiry date minus 60 days).
3. Close the maintenance change ticket.

---

## References

- Contoso PKI Certificate Policy v1.2
- DD-001: SD-WAN Overlay Design (§7 Security)
- Cisco vManage Certificate Management Guide
