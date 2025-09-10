# Adverse Action Explanation Template (Canada — PIPEDA-aligned)

**Case ID:** {{case_id}}  
**Date/Time (UTC):** {{timestamp}}  
**Model Version:** {{model_version}}  
**Probability of Fraud:** {{probability}}  
**Operational Decision:** {{decision}} (e.g., "Held for manual review")  
**Reviewer:** {{reviewer}} (if applicable)

## Main Factors (Top 3)
1. {{reason_1}}  
2. {{reason_2}}  
3. {{reason_3}}

*Examples:*  
- High transaction amount relative to typical behavior.  
- Cross-border transaction with unfamiliar merchant category.  
- Unusual time (night) combined with online channel.

## Customer Notice
This assessment used an AI model to assist analysis. A human reviewer can reassess upon request.  
For inquiries or to contest, contact: **{{contact_email}}**.

## Data Used
Only necessary data attributes were processed for this assessment. No sensitive PII was required.

## Audit Log
- Input features hash: {{features_hash}}  
- SHAP top features: {{shap_top}}  
- Data sources: {{data_sources}}  
- Actions taken: {{actions_taken}}
