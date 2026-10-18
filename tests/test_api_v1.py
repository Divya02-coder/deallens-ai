from fastapi.testclient import TestClient
from backend.main import app
from backend.services.redaction import redact_text
from backend.services.evidence import fuse_evidence

client=TestClient(app)

def test_health():
    assert client.get("/health").json()["status"]=="ok"

def test_redaction_masks_email_pan_and_phone():
    result=redact_text("Contact jane@example.com, +1 212 555 0198, PAN ABCDE1234F")
    assert "jane@example.com" not in result["text"]
    assert "ABCDE1234F" not in result["text"]
    assert result["redactions"] >= 2

def test_evidence_fusion_keeps_source_and_marks_not_verified():
    findings=[{"title":"Supplier concentration exposure","category":"Dependency","detail":"supplier risk"}]
    chunks=[{"source":"suppliers.csv","page":1,"location":"table","row_start":1,"row_end":3,
             "text":"supplier, spend\\nAcme, 90%"}]
    result=fuse_evidence(findings,chunks)
    assert result[0]["document_evidence"][0]["source"]=="suppliers.csv"
    assert result[0]["document_evidence"][0]["verification_status"]=="retrieval_match_not_independent_verification"

def test_upload_rejects_unsupported_file():
    response=client.post("/api/v1/documents",files={"file":("payload.exe",b"nope","application/octet-stream")})
    assert response.status_code==415
