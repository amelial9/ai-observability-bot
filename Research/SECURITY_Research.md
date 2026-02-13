# Network Security Architecture Research
## OTEL Collector → Splunk Integration

**Date**: 2026-02-12
**Status**: Pending Security Rule Configuration

---

## Executive Summary

This document presents security architecture options for connecting our OTEL Collector to Splunk Enterprise while maintaining secure remote dashboard access.

**Key Requirements**:
- Secure internal VM communication (OTEL → Splunk HEC)
- Remote dashboard access from authorized locations
- Minimal attack surface exposure
- Zero trust security principles

**Current Environment**:
- **OTEL Collector VM**: `10.0.0.126` (private subnet)
- **Splunk VM**: `10.0.0.249` (private) / `132.226.90.42` (public)
- **Subnet**: `10.0.0.0/24` (OCI VCN)

**Critical Design Principle**:
- Internal VM communication uses **private IPs only** (10.0.0.x)
- Public IP used exclusively for external dashboard access
- Data ingestion port (8088) never exposed to public internet

---

## Proposed Security Options

### Option 1: IP Whitelisting (RECOMMENDED)

**Overview**: Restrict public access using source IP whitelisting while maintaining internal communication via private network.

**Security Rules** (Splunk VM: 10.0.0.249):

| Rule | Source CIDR | Protocol | Port | Purpose | Exposure |
|------|-------------|----------|------|---------|----------|
| 1 | `10.0.0.0/24` | TCP | 8088 | HEC Data Ingestion | Internal Only |
| 2 | `10.0.0.0/24` | TCP | 8089 | Management API | Internal Only |
| 3 | `<Your_IP>/32` | TCP | 8000 | Dashboard Access | Single IP |

**Implementation**:
```
OCI Console → Networking → VCN → Security Lists → Add Ingress Rules

Rule 1 - HEC (Internal):
  Source CIDR: 10.0.0.0/24
  Protocol: TCP
  Port: 8088
  Description: Splunk HEC - Internal Only

Rule 2 - API (Internal):
  Source CIDR: 10.0.0.0/24
  Protocol: TCP
  Port: 8089
  Description: Splunk API - Internal Only

Rule 3 - Web UI (Restricted):
  Source CIDR: <Obtain via: curl ifconfig.me>/32
  Protocol: TCP
  Port: 8000
  Description: Dashboard - Authorized IP Only
```

**Advantages**:
- ✅ HEC port completely isolated from public internet
- ✅ Dashboard accessible from authorized locations
- ✅ Simple to implement and maintain
- ✅ No additional infrastructure cost
- ✅ Suitable for individual users or small teams

**Disadvantages**:
- ⚠️ Requires update if public IP changes (e.g., home ISP DHCP)

**Effort**: Low | **Cost**: $0 | **Security**: High | **Customer Experience**: High

---

### Option 2: SSH Tunneling (Zero Public Exposure)

**Overview**: Access dashboard via encrypted SSH tunnel, eliminating all public port exposure.

**Security Rules** (Splunk VM):
```
Single Rule Required:
  Source CIDR: 10.0.0.0/24
  Protocol: TCP
  Port: 8088
  Description: HEC Internal Only
```

**Client-Side Setup**:
```bash
# Linux/Mac/Windows
ssh -L 8000:10.0.0.249:8000 ubuntu@132.226.90.42 -N

# Access via browser: http://localhost:8000
```

**Advantages**:
- ✅ Zero public exposure of dashboard port
- ✅ End-to-end encryption via SSH
- ✅ Maximum security posture
- ✅ No firewall rule changes needed after initial setup

**Disadvantages**:
- ⚠️ Requires active SSH connection for access
- ⚠️ Additional step before dashboard access
- ⚠️ User training required

**Effort**: Low | **Cost**: $0 | **Security**: Very High | **Customer Experience**: Low

---

### Option 3: OCI Bastion Service (Enterprise-Grade)

**Overview**: Utilize OCI native Bastion service for secure, audited access with zero public exposure.

**Implementation Steps**:
1. Provision OCI Bastion service in VCN
2. Create port forwarding session to Splunk VM
3. Establish local tunnel through Bastion

**Command Example**:
```bash
# Create port forwarding session
oci bastion session create-port-forwarding \
  --bastion-id <bastion-ocid> \
  --target-resource-port 8000 \
  --target-private-ip 10.0.0.249 \
  --session-ttl-in-seconds 10800

# Client connection
ssh -i <key.pem> -N -L 8000:10.0.0.249:8000 <session-ocid>@host
```

**Advantages**:
- ✅ Cloud-native solution with full audit trails
- ✅ Zero public port exposure
- ✅ Compliance-ready (SOC2, ISO27001)
- ✅ Centralized access management
- ✅ Session time limits enforced

**Disadvantages**:
- ⚠️ Additional OCI service cost (~$0.03/hour)
- ⚠️ Higher implementation complexity

**Effort**: Medium | **Cost**: ~$22/month | **Security**: Very High | | **Customer Experience**: High

---

### Option 4: Multi-IP Whitelist (Team Access)

**Overview**: Extend Option 1 to support multiple authorized users/locations.

**Configuration Example**:
```
Rule 1 - Internal HEC:
  Source: 10.0.0.0/24, TCP, Port: 8088

Rule 2 - User 1 (Home):
  Source: <IP_1>/32, TCP, Port: 8000

Rule 3 - User 2 (Office):
  Source: <IP_2>/32, TCP, Port: 8000

Rule 4 - User 3:
  Source: <IP_3>/32, TCP, Port: 8000
```

**Advantages**:
- ✅ Supports distributed team access
- ✅ Granular IP-based access control
- ✅ Simple rule management

**Disadvantages**:
- ⚠️ Rule management overhead as team scales
- ⚠️ Dynamic IP addresses require updates

**Effort**: Low-Medium | **Cost**: $0 | **Security**: High


---

## Recommendation

**Primary Recommendation: Option 1 (IP Whitelisting)**

**Rationale**:
1. **Security**: Adequate protection for current use case
   - HEC port isolated to internal network
   - Dashboard restricted to specific IP
   - Meets industry security standards

2. **Operational Efficiency**:
   - Zero infrastructure overhead
   - No cost implications
   - Simple troubleshooting

3. **Scalability**:
   - Easy migration to Option 4 if team expands
   - Can upgrade to Bastion if compliance requirements change

**Fallback**: Option 2 (SSH Tunnel) if public IP address frequently changes

---

## Future Multi-Tenancy Readiness

**Why Option 1 is optimal for upcoming multi-tenant customer access**:

As we transition to a multi-tenant architecture where customers will require direct dashboard access, Option 1 provides the ideal foundation:

1. **Seamless Scaling**: Option 1 naturally evolves into Option 4 (Multi-IP Whitelist) by simply adding security rules. Each customer gets their own dedicated ingress rule with their specific IP address.

2. **Zero Refactoring**: Current implementation requires no architectural changes when onboarding customers - only incremental security rule additions (free operation).

3. **Tenant Isolation**: IP-based access control provides network-layer tenant isolation, meeting B2B SaaS security requirements without additional infrastructure.

4. **Compliance Ready**: Individual IP rules per customer create clear audit trails for SOC2, ISO27001, and customer security questionnaires.

5. **Cost-Effective Growth**: Scales linearly without additional infrastructure costs (unlike Bastion service which adds ~$22/month per tenant).

---

