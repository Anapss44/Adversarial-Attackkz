"""
Adversarial Attack Lab — Python Backend
Supports: FGSM, PGD, BIM, DeepFool, C&W
+ Ensemble Detector (statistical + CNN-based)

Usage:
  python adversarial.py attack  <image_path> <attack_type> <params_json> <output_path>
  python adversarial.py detect  <image_path>

Outputs JSON to stdout.
"""

import sys, os, json, base64, io, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# ── ImageNet label map (top subset) ─────────────────────────────────────────
LABELS = {
    0:"tench",1:"goldfish",2:"great white shark",7:"rooster",8:"hen",
    9:"ostrich",88:"Chihuahua",89:"Japanese Chin",90:"Maltese",99:"Beagle",
    207:"golden retriever",208:"Irish setter",281:"tabby cat",282:"tiger cat",
    283:"Persian cat",291:"lion",292:"tiger",340:"zebra",
    385:"Indian elephant",386:"African elephant",
    954:"banana",971:"basketball",980:"volcano",985:"daisy",
    487:"cellular phone",497:"church",508:"computer keyboard",
    562:"fountain",569:"frying pan",574:"garbage truck",
    610:"jersey",652:"matchstick",670:"motor scooter",
    717:"pickup truck",737:"plow",751:"racket",779:"school bus",
    805:"soccer ball",817:"sports car",829:"stopwatch",
    849:"table lamp",860:"toaster",876:"umbrella",
    895:"vending machine",897:"volcano",904:"whiskey jug",
    999:"toilet paper",998:"ear",997:"bolete",996:"hen-of-the-woods mushroom"
}
def get_label(idx): return LABELS.get(idx, f"class_{idx}")

# ── Model & transforms ───────────────────────────────────────────────────────
MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
STD  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def load_model(device):
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m = m.to(device).eval()
    return m

def preprocess(path):
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(path).convert('RGB')
    t = tf(img).unsqueeze(0)
    return t  # raw [0,1]

def normalize(t):   return (t - MEAN) / STD
def denormalize(t): return torch.clamp(t * STD + MEAN, 0, 1)

def to_b64(tensor_01):
    arr = tensor_01.squeeze().detach().cpu().numpy()
    arr = np.transpose(arr,(1,2,0))
    arr = np.clip(arr*255,0,255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def top_preds(output, k=3):
    probs = F.softmax(output, dim=1)
    top_p, top_i = torch.topk(probs, k, dim=1)
    return [{"label":get_label(top_i[0][j].item()),
             "confidence":round(top_p[0][j].item()*100,2),
             "class_id":top_i[0][j].item()} for j in range(k)]

def save_img(tensor_01, path):
    arr = tensor_01.squeeze().detach().cpu().numpy()
    arr = np.transpose(arr,(1,2,0))
    arr = np.clip(arr*255,0,255).astype(np.uint8)
    Image.fromarray(arr).save(path)

# ── Attacks ──────────────────────────────────────────────────────────────────

def attack_fgsm(model, x_raw, device, epsilon=0.1, **kw):
    """Fast Gradient Sign Method — single-step gradient attack."""
    x = normalize(x_raw).to(device).requires_grad_(True)
    out = model(x)
    label = out.max(1)[1]
    loss = F.cross_entropy(out, label)
    model.zero_grad(); loss.backward()
    pert = epsilon * x.grad.sign()
    x_adv_norm = x + pert
    x_adv_raw = torch.clamp(denormalize(x_adv_norm.detach()), 0, 1)
    return x_adv_raw, {"epsilon": epsilon, "steps": 1}


def attack_pgd(model, x_raw, device, epsilon=0.1, alpha=0.01, steps=40, **kw):
    """Projected Gradient Descent — iterative white-box attack."""
    x_raw = x_raw.to(device)
    x_adv = x_raw.clone().detach()
    # random start
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    label = model(normalize(x_raw)).max(1)[1]

    for _ in range(steps):
        x_adv.requires_grad_(True)
        out = model(normalize(x_adv))
        loss = F.cross_entropy(out, label)
        model.zero_grad(); loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x_raw, -epsilon, epsilon)
            x_adv = torch.clamp(x_raw + delta, 0, 1)
    return x_adv.detach(), {"epsilon": epsilon, "alpha": alpha, "steps": steps}


def attack_bim(model, x_raw, device, epsilon=0.1, alpha=None, steps=10, **kw):
    """Basic Iterative Method (I-FGSM) — multi-step FGSM."""
    if alpha is None:
        alpha = epsilon / steps
    x_raw = x_raw.to(device)
    x_adv = x_raw.clone().detach()
    label = model(normalize(x_raw)).max(1)[1]

    for _ in range(steps):
        x_adv.requires_grad_(True)
        out = model(normalize(x_adv))
        loss = F.cross_entropy(out, label)
        model.zero_grad(); loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_raw + epsilon), x_raw - epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach(), {"epsilon": epsilon, "alpha": round(alpha,4), "steps": steps}


def attack_deepfool(model, x_raw, device, steps=50, overshoot=0.02, **kw):
    """DeepFool — minimal perturbation to cross decision boundary."""
    x_raw = x_raw.to(device)
    x_adv = x_raw.clone().detach()
    num_classes = 10  # use top-10 for speed

    r_total = torch.zeros_like(x_raw)

    for _ in range(steps):
        x_adv_var = x_adv.requires_grad_(True)
        out = model(normalize(x_adv_var))
        probs = F.softmax(out, dim=1)
        sorted_idx = probs[0].argsort(descending=True)
        orig_class = sorted_idx[0].item()

        if x_adv_var.grad is not None:
            x_adv_var.grad.zero_()

        # gradient for original class
        out[0, orig_class].backward(retain_graph=True)
        grad_orig = x_adv_var.grad.clone()

        min_pert = float('inf')
        w_hat, f_hat = None, None

        for k in sorted_idx[1:num_classes]:
            if x_adv_var.grad is not None:
                x_adv_var.grad.zero_()
            out[0, k.item()].backward(retain_graph=True)
            grad_k = x_adv_var.grad.clone()

            w_k = grad_k - grad_orig
            f_k = (out[0, k.item()] - out[0, orig_class]).item()
            pert_k = abs(f_k) / (w_k.norm().item() + 1e-8)

            if pert_k < min_pert:
                min_pert = pert_k
                w_hat = w_k
                f_hat = f_k

        if w_hat is None: break

        r_i = (abs(f_hat) / (w_hat.norm()**2 + 1e-8)) * w_hat
        r_total = r_total + r_i.detach()
        x_adv = torch.clamp(x_raw + (1+overshoot)*r_total, 0, 1).detach()

        # check if fooled
        with torch.no_grad():
            new_pred = model(normalize(x_adv)).max(1)[1].item()
            if new_pred != orig_class:
                break

    pert_norm = r_total.abs().mean().item()
    return x_adv, {"steps": steps, "overshoot": overshoot, "perturbation_norm": round(pert_norm,6)}


def attack_cw(model, x_raw, device, c=1.0, kappa=0.0, steps=100, lr=0.01, **kw):
    """Carlini & Wagner L2 attack — optimization-based white-box attack."""
    x_raw = x_raw.to(device)
    label = model(normalize(x_raw)).max(1)[1]

    # tanh-space reparameterization
    w = torch.zeros_like(x_raw, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = x_raw.clone()
    best_dist = float('inf')

    for step in range(steps):
        # map w -> x_adv in [0,1]
        x_adv = 0.5 * (torch.tanh(w) + 1)

        # L2 distance
        dist = ((x_adv - x_raw)**2).sum().sqrt()

        # C&W f6 loss
        out = model(normalize(x_adv))
        target_logit = out[0, label.item()]
        other_logit  = torch.cat([out[0,:label.item()], out[0,label.item()+1:]]).max()
        f_loss = torch.clamp(target_logit - other_logit + kappa, min=0)

        loss = dist + c * f_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if dist.item() < best_dist and f_loss.item() > 0:
            best_dist = dist.item()
            best_adv = x_adv.detach().clone()

    return best_adv, {"c": c, "kappa": kappa, "steps": steps, "lr": lr}


ATTACKS = {
    "fgsm": attack_fgsm,
    "pgd":  attack_pgd,
    "bim":  attack_bim,
    "deepfool": attack_deepfool,
    "cw":   attack_cw,
}

# ── Detector ─────────────────────────────────────────────────────────────────

def statistical_features(img_tensor):
    """
    Extract statistical features for adversarial detection.
    Adversarial images tend to have:
      - Higher high-frequency content
      - Unusual pixel gradient distributions
      - Specific noise patterns in local neighborhoods
    Returns a dict of feature scores and a combined suspicion score [0,1].
    """
    x = img_tensor.squeeze().cpu().numpy()  # (3,H,W)

    feats = {}

    # 1. High-frequency energy (Laplacian-like gradient magnitude)
    from scipy.ndimage import laplace
    hf_scores = []
    for c in range(3):
        lap = laplace(x[c])
        hf_scores.append(np.abs(lap).mean())
    feats["hf_energy"] = float(np.mean(hf_scores))

    # 2. Local pixel variance (small patches)
    patch_vars = []
    h, w = x.shape[1], x.shape[2]
    step = 16
    for i in range(0, h-step, step):
        for j in range(0, w-step, step):
            patch = x[:, i:i+step, j:j+step]
            patch_vars.append(patch.var())
    feats["local_variance"] = float(np.mean(patch_vars))
    feats["local_variance_std"] = float(np.std(patch_vars))

    # 3. Pixel value distribution kurtosis (adversarial perturbations create non-Gaussian tails)
    from scipy.stats import kurtosis
    kurt_scores = [kurtosis(x[c].flatten()) for c in range(3)]
    feats["kurtosis"] = float(np.mean(kurt_scores))

    # 4. Gradient magnitude distribution
    grad_mags = []
    for c in range(3):
        gx = np.abs(np.diff(x[c], axis=1)).mean()
        gy = np.abs(np.diff(x[c], axis=0)).mean()
        grad_mags.append((gx + gy) / 2)
    feats["gradient_magnitude"] = float(np.mean(grad_mags))

    # 5. Cross-channel correlation (adversarial perturbations often misalign channels)
    cc_01 = float(np.corrcoef(x[0].flatten(), x[1].flatten())[0,1])
    cc_02 = float(np.corrcoef(x[0].flatten(), x[2].flatten())[0,1])
    cc_12 = float(np.corrcoef(x[1].flatten(), x[2].flatten())[0,1])
    feats["cross_channel_corr"] = float(np.mean([cc_01, cc_02, cc_12]))

    # ── Scoring heuristic ────────────────────────────────────────────────────
    # Thresholds calibrated on ImageNet ResNet18 (approximate)
    score = 0.0
    weights = []

    # High-frequency energy: adversarial tends > 0.006
    hf_norm = min(feats["hf_energy"] / 0.015, 1.0)
    score += hf_norm * 0.30; weights.append(0.30)

    # Gradient magnitude: adversarial tends > 0.015
    gm_norm = min(feats["gradient_magnitude"] / 0.03, 1.0)
    score += gm_norm * 0.25; weights.append(0.25)

    # Local variance std: adversarial has uneven patch variance
    lv_norm = min(feats["local_variance_std"] / 0.005, 1.0)
    score += lv_norm * 0.20; weights.append(0.20)

    # Kurtosis: high absolute kurtosis → suspicious
    kurt_norm = min(abs(feats["kurtosis"]) / 5.0, 1.0)
    score += kurt_norm * 0.15; weights.append(0.15)

    # Cross-channel correlation: adversarial slightly lower correlation
    corr_score = max(0, 1.0 - feats["cross_channel_corr"])
    score += corr_score * 0.10; weights.append(0.10)

    feats["stat_suspicion"] = round(float(score), 4)
    return feats


def model_based_detector(model, img_tensor, device):
    """
    Use the classifier's own internal signals to detect adversarial inputs.
    Techniques:
      1. Prediction confidence — adversarial tends to have unusual top-1 confidence
      2. Input squeezing — compare predictions before/after median filtering
      3. Feature squeezing via JPEG compression
    """
    from PIL import ImageFilter
    import scipy.ndimage

    results = {}
    x = img_tensor.to(device)

    # --- Original prediction ---
    with torch.no_grad():
        out_orig = model(normalize(x))
        prob_orig = F.softmax(out_orig, dim=1)
        conf_orig = prob_orig.max().item()
        pred_orig = prob_orig.argmax().item()
    results["original_confidence"] = round(conf_orig, 4)
    results["original_pred"] = pred_orig

    # --- Squeeze 1: Median filter (spatial smoothing) ---
    x_np = x.squeeze().cpu().numpy()
    x_squeezed = np.stack([scipy.ndimage.median_filter(x_np[c], size=2) for c in range(3)])
    x_sq_t = torch.from_numpy(x_squeezed).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out_sq = model(normalize(x_sq_t))
        prob_sq = F.softmax(out_sq, dim=1)
        pred_sq = prob_sq.argmax().item()
        conf_sq = prob_sq.max().item()

    results["squeezed_pred"] = pred_sq
    results["pred_changed_after_squeeze"] = (pred_orig != pred_sq)
    results["confidence_drop"] = round(conf_orig - conf_sq, 4)

    # --- Squeeze 2: JPEG compression ---
    arr = (x_np.transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG', quality=75)
    buf.seek(0)
    arr_jpeg = np.array(Image.open(buf)).astype(np.float32) / 255.0
    x_jpeg_t = torch.from_numpy(arr_jpeg.transpose(2,0,1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out_jpeg = model(normalize(x_jpeg_t))
        prob_jpeg = F.softmax(out_jpeg, dim=1)
        pred_jpeg = prob_jpeg.argmax().item()
        conf_jpeg = prob_jpeg.max().item()

    results["jpeg_pred"] = pred_jpeg
    results["pred_changed_after_jpeg"] = (pred_orig != pred_jpeg)
    results["confidence_drop_jpeg"] = round(conf_orig - conf_jpeg, 4)

    # --- Score ---
    score = 0.0
    # Low confidence → suspicious
    conf_score = max(0, 1.0 - conf_orig)
    score += conf_score * 0.30

    # Prediction flips after squeeze → suspicious
    if results["pred_changed_after_squeeze"]:
        score += 0.35
    else:
        score += max(0, results["confidence_drop"]) * 2.0 * 0.20

    # Prediction flips after JPEG → suspicious
    if results["pred_changed_after_jpeg"]:
        score += 0.35
    else:
        score += max(0, results["confidence_drop_jpeg"]) * 2.0 * 0.15

    results["model_suspicion"] = round(min(float(score), 1.0), 4)
    return results


def run_detector(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    img_tensor = preprocess(image_path)

    stat_feats = statistical_features(img_tensor)
    model_results = model_based_detector(model, img_tensor, device)

    # Ensemble: weighted combination
    w_stat, w_model = 0.45, 0.55
    ensemble_score = (
        w_stat  * stat_feats["stat_suspicion"] +
        w_model * model_results["model_suspicion"]
    )
    ensemble_score = round(min(float(ensemble_score), 1.0), 4)

    # Verdict
    if ensemble_score >= 0.60:
        verdict = "ADVERSARIAL"
        confidence = ensemble_score
    elif ensemble_score >= 0.40:
        verdict = "SUSPICIOUS"
        confidence = ensemble_score
    else:
        verdict = "CLEAN"
        confidence = 1.0 - ensemble_score

    # Original prediction
    with torch.no_grad():
        out = model(normalize(img_tensor.to(device)))
        preds = top_preds(out)

    print(json.dumps({
        "success": True,
        "verdict": verdict,
        "ensemble_score": ensemble_score,
        "confidence": round(confidence * 100, 1),
        "stat_suspicion": round(stat_feats["stat_suspicion"] * 100, 1),
        "model_suspicion": round(model_results["model_suspicion"] * 100, 1),
        "features": {
            "hf_energy":            round(stat_feats["hf_energy"], 6),
            "gradient_magnitude":   round(stat_feats["gradient_magnitude"], 6),
            "local_variance_std":   round(stat_feats["local_variance_std"], 6),
            "kurtosis":             round(stat_feats["kurtosis"], 4),
            "cross_channel_corr":   round(stat_feats["cross_channel_corr"], 4),
            "original_confidence":  round(model_results["original_confidence"] * 100, 1),
            "confidence_drop_median": round(model_results["confidence_drop"] * 100, 1),
            "confidence_drop_jpeg": round(model_results["confidence_drop_jpeg"] * 100, 1),
            "pred_changed_squeeze": model_results["pred_changed_after_squeeze"],
            "pred_changed_jpeg":    model_results["pred_changed_after_jpeg"],
        },
        "top_predictions": preds,
        "image_b64": to_b64(img_tensor),
        "device": str(device)
    }))


def run_attack(image_path, attack_type, params, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    x_raw = preprocess(image_path).to(device)

    # original prediction
    with torch.no_grad():
        out_orig = model(normalize(x_raw))
        orig_preds = top_preds(out_orig)

    fn = ATTACKS.get(attack_type)
    if fn is None:
        print(json.dumps({"success": False, "error": f"Unknown attack: {attack_type}"}))
        sys.exit(1)

    x_adv, attack_params = fn(model, x_raw, device, **params)

    # adversarial prediction
    with torch.no_grad():
        out_adv = model(normalize(x_adv.to(device)))
        adv_preds = top_preds(out_adv)

    orig_label = orig_preds[0]["label"]
    adv_label  = adv_preds[0]["label"]

    # perturbation stats
    pert = (x_adv - x_raw).detach().cpu()
    pert_l2   = pert.norm().item()
    pert_linf = pert.abs().max().item()
    pert_mean = pert.abs().mean().item()

    save_img(x_adv.cpu(), output_path)

    print(json.dumps({
        "success": True,
        "attack_type": attack_type,
        "attack_params": attack_params,
        "original": {
            "label": orig_label,
            "confidence": orig_preds[0]["confidence"],
            "top_predictions": orig_preds,
            "image_b64": to_b64(x_raw.cpu())
        },
        "adversarial": {
            "label": adv_label,
            "confidence": adv_preds[0]["confidence"],
            "top_predictions": adv_preds,
            "image_b64": to_b64(x_adv.cpu()),
            "output_path": output_path
        },
        "attack_successful": orig_label != adv_label,
        "perturbation": {
            "l2":   round(pert_l2, 6),
            "linf": round(pert_linf, 6),
            "mean": round(pert_mean, 6),
        },
        "device": str(device)
    }))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode == "attack":
        if len(sys.argv) < 6:
            print(json.dumps({"success": False, "error": "attack requires: image_path attack_type params_json output_path"}))
            sys.exit(1)
        image_path   = sys.argv[2]
        attack_type  = sys.argv[3].lower()
        params       = json.loads(sys.argv[4])
        output_path  = sys.argv[5]
        if not os.path.exists(image_path):
            print(json.dumps({"success": False, "error": f"Image not found: {image_path}"}))
            sys.exit(1)
        try:
            run_attack(image_path, attack_type, params, output_path)
        except Exception as e:
            import traceback
            print(json.dumps({"success": False, "error": str(e), "trace": traceback.format_exc()}))
            sys.exit(1)

    elif mode == "detect":
        if len(sys.argv) < 3:
            print(json.dumps({"success": False, "error": "detect requires: image_path"}))
            sys.exit(1)
        image_path = sys.argv[2]
        if not os.path.exists(image_path):
            print(json.dumps({"success": False, "error": f"Image not found: {image_path}"}))
            sys.exit(1)
        try:
            run_detector(image_path)
        except Exception as e:
            import traceback
            print(json.dumps({"success": False, "error": str(e), "trace": traceback.format_exc()}))
            sys.exit(1)

    else:
        print(json.dumps({"success": False, "error": f"Unknown mode '{mode}'. Use: attack | detect"}))
        sys.exit(1)
