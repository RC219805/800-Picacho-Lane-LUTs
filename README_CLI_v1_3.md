
## v1.3 additions
### Auto-measure eye-line & gutters
```bash
python presence_security_v1_2/presence_cli_v1_3.py measure --image In-Command_In-Conversation_2400x3000.jpg --aspect 4:5
# -> JSON report: eye_line_pct, gutters, confidence
```

### Verify-manifest (consent gating; optional signature)
```bash
python presence_security_v1_2/presence_cli_v1_3.py verify-manifest   --manifest presence_manifest_example.json   --hero In-Command_In-Conversation_2400x3000.jpg   --web  In-Command_In-Conversation_1065x1330.jpg   --public ed25519_public.key --signature signature.b64 --require-signature
```
