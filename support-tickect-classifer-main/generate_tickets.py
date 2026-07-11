import csv
import random
import os

random.seed(42)

# ─────────────────────────────────────────────
# SHARED VOCABULARY POOLS
# ─────────────────────────────────────────────

GREETINGS = [
    "Hi IT Support,", "Dear Helpdesk Team,", "Hello Support,",
    "Good morning IT team,", "Hi all,", "Dear IT,",
    "To the IT Support Team,", "Hi there,", "Good afternoon,",
    "Dear Support Team,", "Hello,", "Hi Support,",
]

CLOSINGS = [
    "Kind regards,", "Best regards,", "Many thanks,",
    "Thanks in advance,", "Regards,", "Thank you,",
    "Appreciate your help,", "Thanks and regards,",
    "With thanks,", "Looking forward to your response,",
]

FOLLOW_UP_PREFIXES = [
    "Following up on the below. ",
    "Chasing the above as this is still unresolved. ",
    "RE: RE: Still awaiting a resolution on this. ",
    "This issue has now been ongoing for several days. ",
    "",  # No prefix — fresh ticket
    "",
    "",
    "",
]

STEPS_TRIED = [
    "I have already restarted my machine and the issue persists.",
    "I have tried clearing browser cache and cookies with no success.",
    "I attempted to resolve this myself by reinstalling the application, but the problem remains.",
    "I have reproduced this on two different devices, so it does not appear to be machine-specific.",
    "My line manager has also confirmed this is not a permissions issue on their end.",
    "I have already submitted a request via the self-service portal with no response.",
    "I tried logging in from a different browser and the same error occurs.",
    "I contacted a colleague who is experiencing the same issue.",
    "I have run the built-in diagnostic tool, which shows no local errors.",
    "I have already raised this with my team lead who advised me to escalate to IT.",
]

URGENCY_PHRASES = {
    3: [
        "This is critically urgent as it is blocking all work for my team.",
        "Please treat this as P1 — production is down and customers are affected.",
        "This is a complete work stoppage and I have a client delivery in two hours.",
        "Entire floor cannot work. Please escalate immediately.",
        "This is blocking a live client demo scheduled for this afternoon.",
        "Production environment is fully down — all transactions are failing.",
        "This requires immediate escalation as our SLA breach window opens in one hour.",
    ],
    2: [
        "This is affecting my ability to complete today's deliverables.",
        "Could you please prioritise this as it is impacting a project deadline.",
        "The issue is causing significant disruption to the team's workflow.",
        "We have a sprint review tomorrow and this is blocking our progress.",
        "Please action as soon as possible — this is time-sensitive.",
        "Our department is partially operational but the situation is deteriorating.",
    ],
    1: [
        "Not urgent, but please action when convenient.",
        "Low priority — happy to wait for the next available slot.",
        "Please address this at your earliest convenience.",
        "No immediate business impact, but would appreciate resolution by end of week.",
        "When you have capacity, please look into the following.",
        "This is non-critical but would improve day-to-day productivity.",
    ],
}

IMPACT_PHRASES = {
    3: [
        "This is affecting the entire organisation — all users are impacted.",
        "All staff in the London and Warsaw offices are affected.",
        "The issue extends across multiple departments and service lines.",
        "Every user who attempts to access the system is encountering the same error.",
        "The entire project team of 40+ people is blocked.",
    ],
    2: [
        "This is affecting our entire department of approximately 15 people.",
        "The issue is impacting a team of 8 developers and is blocking the sprint.",
        "Multiple colleagues on the same floor are experiencing the same problem.",
        "The finance team has confirmed they are all affected by this issue.",
        "Around 20 users in the same Active Directory group are impacted.",
    ],
    1: [
        "As far as I am aware, this is only affecting my account.",
        "The issue appears to be isolated to my machine.",
        "I have checked with colleagues and they are not experiencing this.",
        "This seems to be specific to my user profile only.",
        "No colleagues are impacted — this appears to be a single-user issue.",
    ],
}

DEPARTMENTS = [
    "Finance", "HR", "Engineering", "DevOps", "Marketing",
    "Sales", "Legal", "Operations", "Procurement", "IT",
    "Business Intelligence", "Product Management", "Customer Success",
    "Infrastructure", "Data Analytics", "Project Delivery",
]

SYSTEMS = [
    "SAP", "Salesforce", "Jira", "Confluence", "SharePoint",
    "Azure AD", "Okta", "ServiceNow", "Oracle", "Dynamics 365",
    "Exchange Online", "Teams", "Outlook", "OneDrive", "Power BI",
    "Citrix", "Active Directory", "GitHub Enterprise", "Workday",
    "Tableau", "Slack", "Zoom", "Ariba", "DocuSign",
]

ERROR_CODES = [
    "0x80070002", "0x8009030C", "403 Forbidden", "500 Internal Server Error",
    "LDAP Error 49", "Event ID 4625", "NTE_BAD_KEYSET", "0xC000006D",
    "SSL_ERROR_RX_RECORD_TOO_LONG", "HRESULT 0x80004005", "error code -2147023570",
    "0x800704CF", "RPC Server Unavailable", "Kerberos Error KDC_ERR_C_PRINCIPAL_UNKNOWN",
    "HTTP 502 Bad Gateway", "HTTP 504 Gateway Timeout", "NTLM Auth Failed",
]


# ─────────────────────────────────────────────
# CATEGORY: TECHNICAL SUPPORT
# ─────────────────────────────────────────────

TS_SCENARIOS = [
    # VPN
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"{random.choice(FOLLOW_UP_PREFIXES)}"
        f"I am unable to establish a VPN connection from my home network since {random.choice(['this morning', 'yesterday afternoon', 'the patch deployed last night', 'the infrastructure maintenance window'])}. "
        f"The client returns the error: \"{random.choice(ERROR_CODES)}\". "
        f"I have confirmed my credentials are correct and MFA is functioning. "
        f"{random.choice(STEPS_TRIED)} "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Server error
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"{random.choice(FOLLOW_UP_PREFIXES)}"
        f"We are encountering a persistent server error on the {random.choice(SYSTEMS)} environment. "
        f"The application returns HTTP {random.choice(['500', '502', '503', '504'])} at irregular intervals, "
        f"making the service unstable for end users. "
        f"The error began appearing after the {random.choice(['latest deployment', 'configuration update', 'database migration', 'infrastructure change'])} on {random.choice(['Monday', 'last Friday', 'the weekend'])}. "
        f"{random.choice(STEPS_TRIED)} "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Configuration
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am raising a ticket regarding a configuration issue affecting {random.choice(SYSTEMS)}. "
        f"Following the recent {random.choice(['group policy update', 'system configuration change', 'registry modification', 'environment variable update'])}, "
        f"the service is no longer behaving as expected. "
        f"Specifically, {random.choice(['SSL certificate validation is failing', 'the proxy settings are not being applied', 'OAuth redirect URIs are misconfigured', 'DNS resolution is returning incorrect records', 'the LDAP binding configuration is invalid'])}. "
        f"{random.choice(STEPS_TRIED)} "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Timeout
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"{random.choice(FOLLOW_UP_PREFIXES)}"
        f"We are experiencing consistent timeout errors when accessing {random.choice(SYSTEMS)}. "
        f"Requests are timing out after approximately {random.choice(['30 seconds', '2 minutes', '5 minutes', '45 seconds'])} "
        f"without returning a response. "
        f"This appears to be related to {random.choice(['database query performance', 'network latency between the application and database tiers', 'an unresponsive dependency service', 'connection pool exhaustion'])}. "
        f"{random.choice(STEPS_TRIED)} "
        f"We have reviewed application logs and identified repeated connection timeout entries. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Crash
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am reporting a recurring application crash affecting {random.choice(SYSTEMS)}. "
        f"The application terminates unexpectedly {random.choice(['on startup', 'when opening large files', 'during report generation', 'when switching between modules', 'after approximately 10 minutes of use'])}. "
        f"The Windows Event Viewer shows {random.choice(ERROR_CODES)} at the time of each crash. "
        f"{random.choice(STEPS_TRIED)} "
        f"The crash has been reproduced consistently across {random.choice(['three different machines', 'two user accounts', 'multiple operating system versions'])}. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Deployment
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"We have encountered a deployment failure in the {random.choice(['staging', 'UAT', 'production', 'pre-production'])} environment "
        f"for the {random.choice(SYSTEMS)} application. "
        f"The deployment pipeline {random.choice(['failed at the container build stage', 'is failing during the database migration step', 'is throwing a dependency resolution error', 'failed post-deployment health checks'])}. "
        f"The CI/CD logs show: \"{random.choice(ERROR_CODES)}\". "
        f"We have reviewed the configuration but cannot identify the root cause. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # API failure
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"We are experiencing API failure between {random.choice(SYSTEMS)} and {random.choice(SYSTEMS)}. "
        f"The integration endpoint is returning {random.choice(['HTTP 401 Unauthorized', 'HTTP 403 Forbidden', 'HTTP 500', 'connection refused', 'malformed JSON responses', 'empty response bodies'])} "
        f"for all requests since {random.choice(['this morning', '14:30 yesterday', 'the last deployment', 'the API key rotation'])}. "
        f"The authentication token has been regenerated but the issue persists. "
        f"Downstream processes that depend on this API are now failing silently. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Network/connectivity
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am unable to reach {random.choice(['the internal file server', 'corporate intranet resources', 'the shared network drive', 'internal DNS-resolved services'])} "
        f"from my workstation. "
        f"External internet access is functioning normally, suggesting this is an internal routing or firewall configuration issue. "
        f"Running traceroute shows the path {random.choice(['dropping packets at the third hop', 'timing out at the gateway', 'resolving to an incorrect IP address'])}. "
        f"{random.choice(STEPS_TRIED)} "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Software crash / BSOD
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"My workstation is experiencing {random.choice(['a Blue Screen of Death (BSOD)', 'repeated kernel panics', 'unexpected reboots'])} "
        f"approximately {random.choice(['once per hour', 'twice daily', 'every time I open the application', 'once per session'])}. "
        f"The stop code displayed is {random.choice(ERROR_CODES)}. "
        f"This started after {random.choice(['the Windows Update applied last Tuesday', 'the driver update pushed via SCCM', 'the hardware replacement last week'])}. "
        f"{random.choice(STEPS_TRIED)} "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Print / peripheral
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"The {random.choice(['network printer on floor 3', 'shared departmental printer', 'label printer in the warehouse', 'MFD on level 2'])} "
        f"is {random.choice(['offline and not accepting print jobs', 'returning a paper jam error with no jam present', 'printing blank pages', 'not discoverable on the network'])} since {random.choice(['this morning', 'the weekend', 'the office relocation'])}. "
        f"The printer's IP address is reachable via ping but the print spooler is not connecting. "
        f"{random.choice(STEPS_TRIED)} "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,3)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
]

# ─────────────────────────────────────────────
# CATEGORY: ACCOUNT ACCESS
# ─────────────────────────────────────────────

AA_SCENARIOS = [
    # Password reset
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am unable to log in to {random.choice(SYSTEMS)} as my password has expired. "
        f"I attempted to use the self-service password reset portal, but "
        f"{random.choice(['the reset email is not arriving in my inbox or junk folder', 'the reset link has already expired by the time I click it', 'the portal returns an error after entering my username', 'my recovery phone number is no longer active'])}. "
        f"Could you please initiate a manual password reset for my account "
        f"({random.choice(['j.smith', 'sarah.jones', 'm.patel', 'd.brown', 'a.kowalski'])}@company.com)? "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[1])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Account locked
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"{random.choice(FOLLOW_UP_PREFIXES)}"
        f"My account has been locked following {random.choice(['several failed login attempts this morning', 'an incorrect password entry while my Caps Lock was on', 'what appears to be an automated process using cached credentials'])}. "
        f"I am locked out of {random.choice(SYSTEMS)} and cannot access any corporate resources. "
        f"Could you please unlock my Active Directory account and advise whether any suspicious activity was detected? "
        f"{random.choice(URGENCY_PHRASES[random.randint(2,3)])} "
        f"{random.choice(IMPACT_PHRASES[1])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # MFA issue
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am experiencing an MFA issue that is preventing me from completing login to {random.choice(SYSTEMS)}. "
        f"Specifically, {random.choice(['the authenticator app is generating codes that are being rejected', 'I have replaced my phone and the authenticator app is not migrated', 'the SMS verification code is not being delivered to my registered number', 'I am being prompted for MFA even on a trusted corporate device', 'the push notification is not appearing in the Microsoft Authenticator app'])}. "
        f"{random.choice(STEPS_TRIED)} "
        f"Please could you either reset my MFA registration or provision an alternative verification method? "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[1])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Login failure
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am encountering a login failure when attempting to access {random.choice(SYSTEMS)}. "
        f"The system returns the error \"{random.choice(['Invalid username or password', 'Your account is not authorised to access this application', 'LDAP authentication failed', 'User not found in directory', 'Session token invalid — please re-authenticate'])}\" "
        f"despite my credentials being correct. "
        f"{random.choice(STEPS_TRIED)} "
        f"My colleague with the same role and permissions can log in without issue. "
        f"Could you please investigate whether there is an issue with my user object in {random.choice(['Active Directory', 'Azure AD', 'Okta', 'the application user database'])}? "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[1])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # New joiner provisioning
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am writing on behalf of a new joiner, {random.choice(['Alex Chen', 'Priya Sharma', 'Michael Torres', 'Emma Wilson', 'Daniel Okonkwo'])}, "
        f"who is starting in the {random.choice(DEPARTMENTS)} department on {random.choice(['Monday', 'next Wednesday', '15th of this month', 'the 1st'])}. "
        f"Please could you provision the following access prior to their start date:\n"
        f"- Corporate email account\n"
        f"- VPN access\n"
        f"- {random.choice(SYSTEMS)} read/write access\n"
        f"- {random.choice(SYSTEMS)} view-only access\n"
        f"- Access to the shared {random.choice(DEPARTMENTS)} team drive\n\n"
        f"Line manager approval has been provided — {random.choice(['CC-d on this email', 'approved via the HR system', 'confirmed by phone with IT'])}. "
        f"{random.choice(URGENCY_PHRASES[2])} "
        f"{random.choice(IMPACT_PHRASES[1])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Leaver / offboarding
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am writing to notify IT that {random.choice(['James Robertson', 'Fatima Al-Rashid', 'Chris Pearce', 'Nadia Volkov'])} "
        f"has left the organisation as of {random.choice(['today', 'yesterday', 'end of last week', 'this Friday'])}. "
        f"Please could you action the following:\n"
        f"- Disable their Active Directory account immediately\n"
        f"- Revoke all application access including {random.choice(SYSTEMS)}\n"
        f"- Redirect their emails to their line manager for {random.choice(['30 days', '60 days', 'the duration of the handover period'])}\n"
        f"- Recover and re-assign their corporate hardware\n\n"
        f"Please confirm once actioned. "
        f"{random.choice(URGENCY_PHRASES[2])} "
        f"{random.choice(IMPACT_PHRASES[1])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Permission denied / elevated access
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am receiving an 'Access Denied' error when attempting to "
        f"{random.choice(['open a shared folder on the network drive', 'edit documents in the project SharePoint site', 'run a report in ' + random.choice(SYSTEMS), 'access the admin panel in ' + random.choice(SYSTEMS), 'approve purchase orders above my current limit'])}. "
        f"My role requires this access in order to {random.choice(['complete the month-end reporting process', 'review project documentation ahead of client sign-off', 'perform the system configuration changes approved by management', 'process the vendor invoices due today'])}. "
        f"Could you please grant the necessary permissions? Line manager authorisation is attached. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # SSO failure
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"Single sign-on is failing when I attempt to authenticate to {random.choice(SYSTEMS)} via our corporate identity provider. "
        f"The SSO redirect completes but I am {random.choice(['redirected to a blank page', 'shown a \"SAML assertion invalid\" error', 'prompted to log in again immediately after authenticating', 'shown a 403 error despite successful IdP authentication'])}. "
        f"{random.choice(STEPS_TRIED)} "
        f"Other applications using the same SSO provider are functioning correctly, suggesting the issue may be in the {random.choice(SYSTEMS)} service provider configuration. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # VPN account / cert
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"My VPN client certificate has {random.choice(['expired', 'been revoked following the recent PKI rotation', 'become invalid after my machine was re-imaged'])}. "
        f"I am unable to connect to the corporate network remotely and therefore cannot access any internal systems. "
        f"Could you please issue a replacement certificate or re-enrol my device in the VPN provisioning system? "
        f"{random.choice(STEPS_TRIED)} "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[1])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Two-factor bypass / emergency
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am currently travelling internationally and my registered MFA device has been {random.choice(['lost', 'stolen', 'broken during transit'])}. "
        f"I am unable to receive verification codes and cannot access any corporate systems. "
        f"I understand this requires security verification on your side — please advise on the identity verification process required to temporarily grant access. "
        f"I have my employee ID ({random.choice(['EMP-' + str(random.randint(10000,99999)) for _ in range(1)])}) and can confirm my start date and line manager. "
        f"{random.choice(URGENCY_PHRASES[3])} "
        f"{random.choice(IMPACT_PHRASES[1])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
]

# ─────────────────────────────────────────────
# CATEGORY: BILLING
# ─────────────────────────────────────────────

BILLING_SYSTEMS = [
    "SAP Financials", "Oracle ERP", "Dynamics 365 Finance",
    "the billing portal", "the invoice management system",
    "Ariba", "Coupa", "the accounts payable system",
]

AMOUNTS = [
    "£2,450.00", "$3,200.00", "€1,875.50", "£12,000.00",
    "$450.00", "€6,340.00", "£890.00", "$1,250.00",
]

VENDORS = [
    "Adobe", "Microsoft", "Salesforce", "AWS", "our primary cloud provider",
    "the software vendor", "the managed service provider",
    "the external contractor", "the SaaS platform provider",
]

BILLING_SCENARIOS = [
    # Duplicate charge
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am writing to flag a duplicate charge that has appeared on our account. "
        f"We have been billed {random.choice(AMOUNTS)} twice for the same service period "
        f"({random.choice(['Q1 2025', 'March 2025', 'the period 01/02/2025 – 28/02/2025'])}) "
        f"by {random.choice(VENDORS)}. "
        f"The duplicate entry is visible in {random.choice(BILLING_SYSTEMS)} under transaction references "
        f"{random.choice(['INV-20250301-A', 'INV-88421'])} and {random.choice(['INV-20250301-B', 'INV-88562'])}. "
        f"Please could you arrange a credit note or refund for the duplicate amount at the earliest opportunity? "
        f"I have attached both invoices for reference. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Invoice discrepancy
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"We have received invoice {random.choice(['INV-' + str(random.randint(10000,99999)) for _ in range(1)])} "
        f"from {random.choice(VENDORS)} for {random.choice(AMOUNTS)}, "
        f"however this does not match the agreed contract value of {random.choice(AMOUNTS)}. "
        f"The discrepancy appears to relate to {random.choice(['an incorrect user count', 'a service tier that was not approved', 'a prorated calculation error', 'an addition of out-of-scope services'])}. "
        f"Could you please liaise with the vendor to issue a corrected invoice? "
        f"I have attached the original purchase order and the signed contract for reference. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,2)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Payment failed
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"We have received a payment failure notification from {random.choice(VENDORS)} "
        f"for invoice {random.choice(['INV-' + str(random.randint(10000,99999)) for _ in range(1)])} "
        f"totalling {random.choice(AMOUNTS)}. "
        f"The payment was submitted via {random.choice(['BACS', 'SWIFT transfer', 'the automated payment system', 'our corporate credit card'])} "
        f"on {random.choice(['01/04/2025', '15/03/2025', '28/02/2025'])} but has not been confirmed as received. "
        f"Could you please investigate whether the payment was processed correctly and, if not, resubmit? "
        f"Failure to resolve this may result in {random.choice(['service suspension', 'late payment penalties', 'loss of the early payment discount'])}. "
        f"{random.choice(URGENCY_PHRASES[random.randint(2,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Refund request
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am requesting a refund of {random.choice(AMOUNTS)} relating to "
        f"{random.choice(['a cancelled subscription that was charged after the cancellation date', 'a service that was not delivered as agreed', 'an overpayment made in error', 'a return of unused licence fees following a contract renegotiation'])}. "
        f"The original payment reference is {random.choice(['PAY-' + str(random.randint(100000,999999)) for _ in range(1)])} "
        f"and was processed on {random.choice(['12/01/2025', '28/02/2025', '05/03/2025'])}. "
        f"Please confirm the expected timeline for the refund to be credited. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,2)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Invoice not received
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"We have not yet received the invoice for services provided by {random.choice(VENDORS)} "
        f"for the period {random.choice(['January 2025', 'Q4 2024', 'February–March 2025'])}. "
        f"The payment deadline is approaching and our accounts payable team requires the invoice "
        f"in order to process payment within the agreed net {random.choice(['30', '45', '60'])} day terms. "
        f"Could you please request the invoice from the vendor and forward to finance? "
        f"{random.choice(URGENCY_PHRASES[random.randint(2,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # ERP billing module error
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"The finance team is experiencing an error in {random.choice(BILLING_SYSTEMS)} "
        f"when attempting to process outgoing invoices for this month's billing cycle. "
        f"The system returns \"{random.choice(['Posting period not open', 'GL account not found', 'Tax code configuration error', 'Document number range exhausted'])}\" "
        f"for all invoice posting attempts since {random.choice(['Monday morning', 'the system update last week', 'the start of the new financial year'])}. "
        f"Month-end close cannot proceed until this is resolved. "
        f"{random.choice(URGENCY_PHRASES[3])} "
        f"{random.choice(IMPACT_PHRASES[2])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # License billing
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"We have been invoiced for {random.choice(['250', '500', '75', '1,000'])} user licences by {random.choice(VENDORS)}, "
        f"however our current active user count is only {random.choice(['180', '420', '60', '850'])}. "
        f"The discrepancy of approximately {random.choice(['70', '80', '15', '150'])} licences is being billed incorrectly. "
        f"Could you please work with the vendor to reconcile the licence count and issue a credit note for the unused licences? "
        f"I have attached the current Active Directory user export as supporting evidence. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,2)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Subscription renewal issue
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"Our subscription to {random.choice(VENDORS)} was due for renewal on "
        f"{random.choice(['01/04/2025', '01/01/2025', '01/03/2025'])}, "
        f"however the renewal has {random.choice(['not been processed despite the purchase order being submitted on time', 'been processed at the incorrect tier price', 'resulted in a service interruption due to a billing system error'])}. "
        f"The team is currently {random.choice(['unable to access the service', 'operating under a grace period that expires today', 'seeing reduced functionality due to the lapsed subscription'])}. "
        f"Please could you expedite the renewal process or contact the vendor on our behalf? "
        f"{random.choice(URGENCY_PHRASES[random.randint(2,3)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
]

# ─────────────────────────────────────────────
# CATEGORY: FEATURE REQUEST
# ─────────────────────────────────────────────

FR_SCENARIOS = [
    # Dashboard enhancement
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I would like to submit a feature request for a dashboard enhancement in {random.choice(SYSTEMS)}. "
        f"Currently, the {random.choice(['executive summary dashboard', 'operational metrics view', 'team performance dashboard', 'project status board'])} "
        f"does not support {random.choice(['custom date range filtering', 'drill-down by department', 'real-time data refresh', 'comparison against prior period', 'KPI threshold alerting'])}. "
        f"This enhancement would benefit the {random.choice(DEPARTMENTS)} team by allowing "
        f"{random.choice(['more granular reporting without manual exports', 'faster identification of performance anomalies', 'self-service insight generation reducing reliance on the BI team', 'alignment with our OKR tracking process'])}. "
        f"Happy to provide further business justification or arrange a call to discuss requirements. "
        f"{random.choice(URGENCY_PHRASES[1])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Export functionality
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I would like to request the addition of export functionality to {random.choice(SYSTEMS)}. "
        f"At present, data from the {random.choice(['reporting module', 'audit log view', 'user management screen', 'transaction history', 'approval queue'])} "
        f"cannot be exported and must be manually copied, which is time-consuming and error-prone. "
        f"We would like the ability to export to {random.choice(['CSV', 'Excel (.xlsx)', 'PDF', 'both CSV and Excel'])} "
        f"with {random.choice(['all currently visible columns', 'a configurable column selection', 'the applied filters preserved in the export'])}. "
        f"This would significantly reduce the manual effort currently required by the {random.choice(DEPARTMENTS)} team during {random.choice(['month-end reporting', 'audit preparation', 'stakeholder reporting cycles'])}. "
        f"{random.choice(URGENCY_PHRASES[1])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Dark mode
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I would like to raise a request for a dark mode option to be introduced in {random.choice(SYSTEMS)}. "
        f"A number of team members work extended hours and find the current high-contrast white interface causes eye strain, "
        f"particularly during evening and early morning shifts. "
        f"Dark mode is now a standard accessibility feature in most enterprise applications and has been formally requested by "
        f"{random.choice(['multiple users in our team', 'the accessibility working group', 'the wellbeing committee', 'staff survey feedback'])}. "
        f"We would appreciate this being added to the product roadmap for consideration. "
        f"{random.choice(URGENCY_PHRASES[1])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # API integration
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"We would like to request an API integration between {random.choice(SYSTEMS)} and {random.choice(SYSTEMS)}. "
        f"Currently, data must be manually transferred between the two systems, which is "
        f"{random.choice(['creating a significant administrative overhead', 'introducing data inconsistency due to manual re-entry errors', 'delaying our reporting cycle by several hours', 'requiring a full-time resource to manage'])}. "
        f"A REST API integration would allow us to {random.choice(['automate the data sync on a scheduled basis', 'trigger real-time updates when records change', 'eliminate the manual export/import process entirely', 'ensure data consistency across both platforms'])}. "
        f"We can provide detailed technical requirements and are willing to work with the development team through the specification process. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,2)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Notification / alerting
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I would like to request the addition of configurable notifications in {random.choice(SYSTEMS)}. "
        f"The system currently does not send alerts when "
        f"{random.choice(['a task assigned to me is approaching its due date', 'an approval request has been pending for more than 24 hours', 'a monitored threshold is breached', 'a document I am reviewing has been updated by another user'])}. "
        f"We would like notifications to be delivered via "
        f"{random.choice(['email', 'Microsoft Teams channel', 'both email and Teams', 'the in-application notification centre'])}. "
        f"This would reduce the number of missed actions and improve overall process compliance. "
        f"{random.choice(URGENCY_PHRASES[1])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Bulk operations
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am submitting a feature request for bulk operation support in {random.choice(SYSTEMS)}. "
        f"Currently, actions such as {random.choice(['archiving old records', 'updating the status of multiple items', 'reassigning tasks to a different owner', 'applying a tag or category to a group of entries'])} "
        f"must be performed one record at a time. "
        f"Our team regularly processes {random.choice(['hundreds of records during month-end', 'large data volumes during onboarding cycles', 'batch imports from third-party systems'])} "
        f"and the lack of bulk operations creates a significant time cost. "
        f"We estimate this enhancement would save approximately {random.choice(['3 hours', '5 hours', 'a full working day'])} per {random.choice(['week', 'month', 'reporting cycle'])}. "
        f"{random.choice(URGENCY_PHRASES[1])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Role / permission model enhancement
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I would like to request an enhancement to the role-based access control model in {random.choice(SYSTEMS)}. "
        f"Currently the system only supports {random.choice(['two permission levels (Admin and Viewer)', 'a flat permission model with no granularity', 'role assignment at the application level only'])}. "
        f"Our organisation requires {random.choice(['department-level permission scoping', 'the ability to create custom roles with specific capability combinations', 'row-level security to restrict data visibility by business unit', 'time-limited role elevation for temporary project access'])}. "
        f"This has been identified as a gap during our recent {random.choice(['ISO 27001 audit', 'internal security review', 'compliance assessment', 'SOC 2 readiness exercise'])}. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,2)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Mobile / responsive UI
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"We would like to request improved mobile responsiveness for {random.choice(SYSTEMS)}. "
        f"A significant portion of our workforce accesses the system from mobile devices, "
        f"and the current interface {random.choice(['is not optimised for smaller screen sizes', 'requires horizontal scrolling on mobile browsers', 'has buttons and form fields that are too small to interact with accurately', 'does not support touch-based navigation effectively'])}. "
        f"An improved mobile experience would particularly benefit our "
        f"{random.choice(['field engineering team', 'sales team who work predominantly on the road', 'warehouse staff using handheld devices', 'executive team who prefer mobile access'])}. "
        f"{random.choice(URGENCY_PHRASES[1])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Reporting enhancement
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I would like to request an enhancement to the reporting capabilities within {random.choice(SYSTEMS)}. "
        f"The existing reports do not cover {random.choice(['trend analysis over custom time windows', 'cross-system aggregated metrics', 'variance against budget or target', 'individual-level performance breakdowns', 'geographic segmentation of results'])}. "
        f"These additional report types are required to support {random.choice(['the quarterly board pack', 'our regulatory reporting obligations', 'the upcoming external audit', 'the strategic planning process for FY2026'])}. "
        f"We would be happy to share detailed requirements documentation prepared by our {random.choice(DEPARTMENTS)} team. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,2)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
    # Workflow automation
    lambda: (
        f"{random.choice(GREETINGS)}\n\n"
        f"I am raising a request to introduce workflow automation in {random.choice(SYSTEMS)} for the "
        f"{random.choice(['purchase order approval process', 'expense claim review cycle', 'employee onboarding task sequence', 'contract review and sign-off workflow', 'IT change request approval chain'])}. "
        f"At present this process is handled manually via email, leading to "
        f"{random.choice(['delays in approval turnaround times', 'lost requests when approvers are out of office', 'no audit trail for compliance purposes', 'inconsistent SLA adherence across teams'])}. "
        f"Automating this workflow within {random.choice(SYSTEMS)} would ensure consistency, auditability, and reduced administrative overhead. "
        f"{random.choice(URGENCY_PHRASES[random.randint(1,2)])} "
        f"{random.choice(IMPACT_PHRASES[random.randint(1,2)])}\n\n"
        f"{random.choice(CLOSINGS)}"
    ),
]


# ─────────────────────────────────────────────
# GENERATION ENGINE
# ─────────────────────────────────────────────

CATEGORY_CONFIG = {
    "Technical Support":  {"scenarios": TS_SCENARIOS,      "count": 200, "urgency_weights": [0.15, 0.45, 0.40], "impact_weights": [0.35, 0.40, 0.25]},
    "Account Access":     {"scenarios": AA_SCENARIOS,      "count": 200, "urgency_weights": [0.20, 0.45, 0.35], "impact_weights": [0.65, 0.25, 0.10]},
    "Billing":            {"scenarios": BILLING_SCENARIOS, "count": 200, "urgency_weights": [0.20, 0.50, 0.30], "impact_weights": [0.45, 0.40, 0.15]},
    "Feature Request":    {"scenarios": FR_SCENARIOS,      "count": 200, "urgency_weights": [0.70, 0.25, 0.05], "impact_weights": [0.55, 0.35, 0.10]},
}

def pick_weighted(values, weights):
    r = random.random()
    cumulative = 0
    for v, w in zip(values, weights):
        cumulative += w
        if r < cumulative:
            return v
    return values[-1]

rows = []

for category, cfg in CATEGORY_CONFIG.items():
    scenarios = cfg["scenarios"]
    for _ in range(cfg["count"]):
        scenario_fn = random.choice(scenarios)
        body = scenario_fn()
        urgency = pick_weighted([1, 2, 3], cfg["urgency_weights"])
        impact  = pick_weighted([1, 2, 3], cfg["impact_weights"])
        rows.append({"body": body, "category": category, "urgency": urgency, "impact": impact})

random.shuffle(rows)

# ─────────────────────────────────────────────
# WRITE CSV
# ─────────────────────────────────────────────

output_path = os.path.join(os.path.dirname(__file__), "data", "support_tickets.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["body", "category", "urgency", "impact"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows -> {output_path}")

# Quick distribution check
from collections import Counter
cats    = Counter(r["category"] for r in rows)
urgency = Counter(r["urgency"]  for r in rows)
impact  = Counter(r["impact"]   for r in rows)
print("\nCategory distribution:")
for k, v in sorted(cats.items()):    print(f"  {k}: {v}")
print("\nUrgency distribution:")
for k, v in sorted(urgency.items()): print(f"  {k}: {v}")
print("\nImpact distribution:")
for k, v in sorted(impact.items()):  print(f"  {k}: {v}")
