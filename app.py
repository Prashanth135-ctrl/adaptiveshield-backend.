import os, re, time, pickle, zipfile, shutil, urllib.request
from urllib.parse import urlparse
from datetime import datetime
from typing import Optional, List

import numpy as np
import Levenshtein
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)

# ── Setup ──────────────────────────────────────────────
app = FastAPI(title="AdaptiveShield API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DEVICE       = torch.device("cpu")
MAX_LEN      = 128
MAX_URL_LEN  = 200
NUM_FEATURES = 30

TOP_DOMAINS = ["google.com","youtube.com","facebook.com","amazon.com",
               "wikipedia.org","twitter.com","instagram.com","linkedin.com",
               "microsoft.com","apple.com","netflix.com","paypal.com",
               "ebay.com","reddit.com","github.com","stackoverflow.com",
               "dropbox.com","spotify.com","adobe.com","yahoo.com"]

SUSPICIOUS_TLDS = [".xyz",".tk",".ml",".ga",".cf",".pw",".top",
                   ".ru",".cn",".info",".biz",".click",".link"]

BRAND_KEYWORDS = ["paypal","amazon","google","microsoft","apple","facebook",
                  "netflix","bank","secure","login","verify","account",
                  "update","confirm","password","credit","debit","wallet"]

URL_CHARS   = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_~:/?#[]@!$&()*+,;=%"
char_to_idx = {c: i+2 for i, c in enumerate(URL_CHARS)}
char_to_idx["<PAD>"] = 0
char_to_idx["<UNK>"] = 1
VOCAB_SIZE   = len(char_to_idx)

feedback_store = []
scan_history   = []

# ── CNN Model ──────────────────────────────────────────
class PhishingCNN(nn.Module):
    def __init__(self, vocab_size=None, embed_dim=128, num_filters=128,
                 filter_sizes=[2,3,4,5], num_classes=2, dropout=0.5):
        super().__init__()
        vs = vocab_size or VOCAB_SIZE
        self.embedding  = nn.Embedding(vs, embed_dim, padding_idx=0)
        self.convs      = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embed_dim, num_filters, fs),
                          nn.BatchNorm1d(num_filters), nn.ReLU())
            for fs in filter_sizes
        ])
        total = num_filters * len(filter_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(total, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(dropout*0.6), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        emb    = self.embedding(x).permute(0, 2, 1)
        pooled = [F.max_pool1d(c(emb), c(emb).size(2)).squeeze(2) for c in self.convs]
        return self.classifier(torch.cat(pooled, dim=1))

# ── GNN Model ─────────────────────────────────────────
GNN_AVAILABLE = False
try:
    from torch_geometric.nn import SAGEConv, BatchNorm as GNNBatchNorm
    class PhishingGNN(nn.Module):
        def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
            super().__init__()
            self.conv1 = SAGEConv(num_features, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim*2)
            self.conv3 = SAGEConv(hidden_dim*2, hidden_dim)
            self.bn1   = GNNBatchNorm(hidden_dim)
            self.bn2   = GNNBatchNorm(hidden_dim*2)
            self.bn3   = GNNBatchNorm(hidden_dim)
            self.cls   = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, num_classes)
            )
            self.drop = dropout
        def forward(self, x, ei):
            x = F.dropout(F.relu(self.bn1(self.conv1(x,ei))), p=self.drop, training=self.training)
            x = F.dropout(F.relu(self.bn2(self.conv2(x,ei))), p=self.drop, training=self.training)
            x = F.dropout(F.relu(self.bn3(self.conv3(x,ei))), p=self.drop, training=self.training)
            return self.cls(x)
    GNN_AVAILABLE = True
except Exception as e:
    print(f"GNN not available: {e}")

# ── Feature Functions ──────────────────────────────────
def compute_entropy(text):
    if not text: return 0.0
    freq = [text.count(c)/len(text) for c in set(text)]
    return -sum(p*np.log2(p+1e-10) for p in freq)

def min_typo_distance(domain):
    if not domain: return 10
    clean = domain.replace("www.", "")
    return min(Levenshtein.distance(clean, d) for d in TOP_DOMAINS)

def is_ip(domain):
    return bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain))

def count_encoded(url):
    return len(re.findall(r"%[0-9a-fA-F]{2}", url))

def extract_domain_name(url):
    try:
        parsed = urlparse(url if url.startswith("http") else "http://"+url)
        parts  = parsed.netloc.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else parsed.netloc
    except:
        return url

def extract_features(url):
    url = str(url)
    try:
        parsed = urlparse(url if url.startswith("http") else "http://"+url)
        domain, path, query = parsed.netloc, parsed.path, parsed.query
    except:
        domain, path, query = url, "", ""
    td = min_typo_distance(domain)
    return np.array([
        len(url), len(domain), len(path), len(query),
        url.count("."), url.count("-"), url.count("/"),
        url.count("@"), url.count("?"), url.count("="),
        url.count("%"), sum(c.isdigit() for c in url),
        len(domain.split("."))-1 if domain else 0,
        1 if url.startswith("https") else 0,
        1 if is_ip(domain) else 0,
        1 if any(domain.endswith(t) for t in SUSPICIOUS_TLDS) else 0,
        1 if any(b in url.lower() for b in BRAND_KEYWORDS) else 0,
        compute_entropy(url),
        sum(c.isdigit() for c in url)/max(len(url), 1),
        len([p for p in path.split("/") if p]),
        1 if td==1 else 0, 1 if td==2 else 0, td,
        len(re.findall(r"[0-9]", domain)),
        1 if "xn--" in domain else 0,
        url.count("_"), count_encoded(url),
        1 if re.search(r"\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}", domain) else 0,
        len(domain.split(".")[-1]) if domain else 0,
        sum(c.isupper() for c in url)/max(len(url), 1)
    ], dtype=np.float32)

def get_risk_level(prob):
    if prob >= 0.70: return "HIGH"
    elif prob >= 0.40: return "MEDIUM"
    return "LOW"

def analyze_extra(url):
    domain  = extract_domain_name(url)
    td      = min_typo_distance(domain)
    dists   = {d: Levenshtein.distance(domain.replace("www.",""), d) for d in TOP_DOMAINS}
    closest = min(dists, key=dists.get)
    return {
        "typosquatting_detected" : td <= 2,
        "typo_distance"          : int(td),
        "closest_legitimate"     : closest,
        "homograph_detected"     : "xn--" in domain,
        "ip_as_domain"           : is_ip(domain),
        "suspicious_tld"         : any(domain.endswith(t) for t in SUSPICIOUS_TLDS),
        "brand_impersonation"    : any(b in url.lower() for b in BRAND_KEYWORDS),
        "url_entropy"            : round(compute_entropy(url), 4),
        "uses_https"             : url.startswith("https"),
        "url_encoded_chars"      : count_encoded(url),
        "domain"                 : domain
    }

# ── Model Setup ────────────────────────────────────────
models = {}

def download_from_drive(file_id, dest_path):
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 1024 * 1024:
            print(f"Already exists: {dest_path} ({size/1024/1024:.1f} MB)")
            return True
        else:
            os.remove(dest_path)

    import requests
    print(f"Downloading {dest_path}...")

    session = requests.Session()

    # First request to get confirmation token for large files
    url     = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp    = session.get(url, stream=True)

    # Check if Google is asking for confirmation
    token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    # If confirmation needed get the actual file
    if token:
        url  = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
        resp = session.get(url, stream=True)

    # Also try the new Google Drive download format
    if resp.status_code != 200 or len(resp.content) < 1024 * 100:
        url  = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        resp = session.get(url, stream=True)

    # Write file in chunks
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    size = os.path.getsize(dest_path)
    print(f"Downloaded: {dest_path} ({size/1024/1024:.1f} MB)")

    if size < 1024 * 100:
        print(f"WARNING: File too small ({size} bytes). May be a Google Drive error page.")
        os.remove(dest_path)
        return False

    return True

def extract_transformer(zip_path, target_path):
    if os.path.exists(f"{target_path}/config.json"):
        print(f"Already extracted: {target_path}")
        return
    tmp = f"/tmp/ext_{os.path.basename(target_path)}"
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp)
    for root, dirs, files in os.walk(tmp):
        if "config.json" in files and "model.safetensors" in files:
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(root, target_path)
            print(f"Extracted: {target_path}")
            return

def extract_pt(zip_path, pt_path):
    if os.path.exists(pt_path):
        print(f"Already extracted: {pt_path}")
        return
    tmp = f"/tmp/ext_{os.path.basename(pt_path)}"
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp)
    pt_name = os.path.basename(pt_path)
    for root, dirs, files in os.walk(tmp):
        if pt_name in files:
            shutil.copy(f"{root}/{pt_name}", pt_path)
            print(f"Extracted: {pt_path}")
            return

def setup_models():
    os.makedirs("./models/bert",    exist_ok=True)
    os.makedirs("./models/roberta", exist_ok=True)

    ids = {
        "bert_model.zip"    : os.getenv("BERT_FILE_ID",    ""),
        "roberta_model.zip" : os.getenv("ROBERTA_FILE_ID", ""),
        "cnn_model.zip"     : os.getenv("CNN_FILE_ID",     ""),
        "gnn_model.zip"     : os.getenv("GNN_FILE_ID",     ""),
    }

    for fname, fid in ids.items():
        if fid:
            download_from_drive(fid, f"./models/{fname}")

    if os.path.exists("./models/bert_model.zip"):
        extract_transformer("./models/bert_model.zip",    "./models/bert")
    if os.path.exists("./models/roberta_model.zip"):
        extract_transformer("./models/roberta_model.zip", "./models/roberta")
    if os.path.exists("./models/cnn_model.zip"):
        extract_pt("./models/cnn_model.zip", "./models/cnn_best.pt")
    if os.path.exists("./models/gnn_model.zip"):
        extract_pt("./models/gnn_model.zip", "./models/gnn_best.pt")

    print("Model setup complete.")

setup_models()

# ── Load Models ────────────────────────────────────────
print(f"Loading models on {DEVICE}...")

try:
    models["bert_tokenizer"] = BertTokenizer.from_pretrained("./models/bert")
    models["bert"]           = BertForSequenceClassification.from_pretrained("./models/bert").to(DEVICE).eval()
    print("BERT loaded.")
except Exception as e: print(f"BERT failed: {e}")

try:
    models["roberta_tokenizer"] = RobertaTokenizer.from_pretrained("./models/roberta")
    models["roberta"]           = RobertaForSequenceClassification.from_pretrained("./models/roberta").to(DEVICE).eval()
    print("RoBERTa loaded.")
except Exception as e: print(f"RoBERTa failed: {e}")

try:
    ckpt = torch.load("./models/cnn_best.pt", map_location=DEVICE, weights_only=False)
    cnn  = PhishingCNN(vocab_size=ckpt.get("vocab_size", VOCAB_SIZE))
    cnn.load_state_dict(ckpt["model_state"])
    models["cnn"]         = cnn.to(DEVICE).eval()
    models["char_to_idx"] = ckpt.get("char_to_idx", char_to_idx)
    print("CNN loaded.")
except Exception as e: print(f"CNN failed: {e}")

try:
    if GNN_AVAILABLE:
        ckpt = torch.load("./models/gnn_best.pt", map_location=DEVICE, weights_only=False)
        gnn  = PhishingGNN(ckpt.get("num_features", NUM_FEATURES),
                           ckpt.get("hidden_dim", 128),
                           ckpt.get("num_classes", 2),
                           ckpt.get("dropout", 0.3))
        gnn.load_state_dict(ckpt["model_state"])
        models["gnn"]    = gnn.to(DEVICE).eval()
        models["scaler"] = ckpt["scaler"]
        print("GNN loaded.")
except Exception as e: print(f"GNN failed: {e}")

try:
    if "scaler" not in models:
        with open("./models/scaler.pkl", "rb") as f:
            models["scaler"] = pickle.load(f)
except: pass

try:
    with open("./models/fusion_model.pkl", "rb") as f:
        models["fusion"] = pickle.load(f)
    print("Fusion loaded.")
except Exception as e: print(f"Fusion failed: {e}")

loaded = [k for k in models if not k.endswith("tokenizer") and not k.endswith("_to_idx")]
print(f"Models ready: {loaded}")

# ── Prediction Functions ───────────────────────────────
def pb(url):
    if "bert" not in models: return 0.5
    try:
        enc = models["bert_tokenizer"](url, add_special_tokens=True, max_length=MAX_LEN,
                                       padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            return torch.softmax(models["bert"](
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE)
            ).logits, dim=1)[0][1].item()
    except: return 0.5

def pr(url):
    if "roberta" not in models: return 0.5
    try:
        enc = models["roberta_tokenizer"](url, add_special_tokens=True, max_length=MAX_LEN,
                                          padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            return torch.softmax(models["roberta"](
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE)
            ).logits, dim=1)[0][1].item()
    except: return 0.5

def pc(url):
    if "cnn" not in models: return 0.5
    try:
        cidx = models.get("char_to_idx", char_to_idx)
        enc  = [cidx.get(c, 1) for c in str(url)[:MAX_URL_LEN]]
        enc  = enc + [0] * (MAX_URL_LEN - len(enc))
        with torch.no_grad():
            return torch.softmax(models["cnn"](
                torch.tensor([enc], dtype=torch.long).to(DEVICE)
            ), dim=1)[0][1].item()
    except: return 0.5

def pg(url):
    if "gnn" not in models or "scaler" not in models: return 0.5
    try:
        f  = models["scaler"].transform(extract_features(url).reshape(1, -1))
        x  = torch.tensor(f, dtype=torch.float).to(DEVICE)
        ei = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(models["gnn"](x, ei), dim=1)[0][1].item()
    except: return 0.5

def pf(b, r, c, g):
    if "fusion" not in models: return float(np.mean([b, r, c, g]))
    try: return float(models["fusion"].predict_proba(np.array([[b, r, c, g]]))[0][1])
    except: return float(np.mean([b, r, c, g]))

# ── Request Models ─────────────────────────────────────
class ScanRequest(BaseModel):
    url: str

class FeedbackRequest(BaseModel):
    url: str
    is_phishing: bool
    user_comment: Optional[str] = ""

class BulkScanRequest(BaseModel):
    urls: List[str]

# ── Endpoints ──────────────────────────────────────────
@app.get("/")
def root():
    loaded = [k for k in models if not k.endswith("tokenizer") and not k.endswith("_to_idx")]
    return {"message": "AdaptiveShield API", "status": "running",
            "models": loaded, "device": str(DEVICE)}

@app.get("/health")
def health():
    loaded = [k for k in models if not k.endswith("tokenizer") and not k.endswith("_to_idx")]
    return {"status": "healthy", "models_loaded": loaded,
            "timestamp": datetime.now().isoformat()}

@app.post("/scan")
def scan_url(request: ScanRequest):
    url = request.url.strip()
    if not url: raise HTTPException(status_code=400, detail="URL cannot be empty.")
    start    = time.time()
    b,r,c,g  = pb(url), pr(url), pc(url), pg(url)
    fp       = pf(b, r, c, g)
    extra    = analyze_extra(url)
    boost    = 0.0
    if extra["typosquatting_detected"] and extra["typo_distance"] == 1: boost += 0.10
    if extra["ip_as_domain"]:   boost += 0.15
    if extra["homograph_detected"]: boost += 0.10
    if extra["suspicious_tld"] and extra["brand_impersonation"]: boost += 0.08
    final  = min(1.0, fp + boost)
    result = {
        "url"                 : url,
        "label"               : "PHISHING" if final >= 0.5 else "LEGITIMATE",
        "phishing_probability": round(final * 100, 2),
        "risk_level"          : get_risk_level(final),
        "model_scores"        : {
            "bert": round(b*100,2), "roberta": round(r*100,2),
            "cnn" : round(c*100,2), "gnn"    : round(g*100,2),
            "fusion": round(fp*100,2), "final": round(final*100,2)
        },
        "extra_analysis"      : extra,
        "scan_time_ms"        : round((time.time()-start)*1000, 2),
        "timestamp"           : datetime.now().isoformat()
    }
    scan_history.append(result)
    return result

@app.post("/scan/bulk")
def scan_bulk(request: BulkScanRequest):
    if len(request.urls) > 50:
        raise HTTPException(status_code=400, detail="Max 50 URLs.")
    results = []; ph = 0
    for url in request.urls:
        try:
            res = scan_url(ScanRequest(url=url))
            results.append(res)
            ph += 1 if res.get("label") == "PHISHING" else 0
        except Exception as e:
            results.append({"url": url, "error": str(e)})
    return {"total_scanned": len(results), "phishing_found": ph,
            "legitimate_found": len(results)-ph, "results": results}

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    feedback_store.append({"url": request.url, "is_phishing": request.is_phishing,
                            "comment": request.user_comment,
                            "timestamp": datetime.now().isoformat()})
    return {"message": "Feedback received.", "total_feedback": len(feedback_store)}

@app.get("/history")
def history(limit: int = 20):
    return {"total_scans": len(scan_history), "results": scan_history[-limit:]}

@app.get("/stats")
def stats():
    if not scan_history: return {"message": "No scans yet."}
    total = len(scan_history)
    ph    = sum(1 for s in scan_history if s.get("label") == "PHISHING")
    return {"total_scans": total, "phishing_detected": ph,
            "legitimate_detected": total-ph,
            "phishing_rate_percent": round(ph/total*100, 2),
            "average_scan_time_ms": round(np.mean([s.get("scan_time_ms",0) for s in scan_history]), 2)}
