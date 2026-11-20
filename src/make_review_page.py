import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from audit import knn
from common import get_device
from embed import load_raw

BANDS = [(0.99, 1.01), (0.985, 0.99), (0.98, 0.985), (0.975, 0.98)]


def png(img, size):
    buf = io.BytesIO()
    Image.fromarray(img).resize((size, size), Image.BICUBIC).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CIFAR-100 Audit Review</title>
<style>
:root{--bg:#fafaf8;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e3e3e0;--bad:#c2410c;--ok:#15803d;--amb:#a16207;--sel:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1e1e21;--ink:#ececef;--mute:#9a9aa2;--line:#323238;--sel:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:2;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
h1{font-size:18px;margin:0;flex:1 1 auto}h2{font-size:17px;margin:28px 0 6px}
main{max-width:1100px;margin:0 auto;padding:8px 16px 64px}
.hint{color:var(--mute);margin:0 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card.done{border-color:var(--sel)}
.row{display:flex;gap:10px}.big{width:128px;height:128px;border-radius:6px;image-rendering:auto}
.meta{font-size:13px}.meta b{font-weight:600}.mute{color:var(--mute)}
.nbs{display:flex;gap:4px;margin-top:8px;flex-wrap:wrap}.nb{text-align:center;font-size:10px;color:var(--mute);width:52px;overflow:hidden}
.nb img{width:52px;height:52px;border-radius:4px;display:block}
.btns{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}
button{font:inherit;font-size:13px;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer}
button.on[data-v=wrong],button.on[data-v=dup]{background:var(--bad);border-color:var(--bad);color:#fff}
button.on[data-v=correct],button.on[data-v=not_dup]{background:var(--ok);border-color:var(--ok);color:#fff}
button.on[data-v=ambiguous]{background:var(--amb);border-color:var(--amb);color:#fff}
#export{background:var(--sel);border-color:var(--sel);color:#fff}
.pair img{width:128px;height:128px;border-radius:6px}
#progress{color:var(--mute);font-size:14px}
</style></head><body>
<header><h1>Ручная проверка аудита CIFAR-100</h1><span id="progress"></span><button id="export">Скачать ответы (CSV)</button></header>
<main>
<h2>1. Подозрительные метки</h2>
<p class="hint">Большая картинка — пример из train с официальной меткой. Справа: что говорят 10 ближайших соседей и CLIP. Внизу 5 ближайших соседей с их метками.
Отметь: <b>метка неверная</b> (на картинке явно другой класс из 100), <b>метка верная</b>, или <b>неоднозначно</b> (несколько объектов, не разобрать, класс спорный).</p>
<div class="grid" id="labels"></div>
<h2>2. Почти-дубликаты train/test</h2>
<p class="hint">Слева тестовая картинка, справа ближайшая обучающая. По 12 случайных пар из четырёх диапазонов сходства CLIP.
<b>Дубликат</b> — это та же фотография (возможно, сдвинутая, обрезанная, другой цвет/яркость). <b>Не дубликат</b> — просто похожие разные снимки.</p>
<div class="grid" id="dups"></div>
</main>
<script>
const LABELS = __LABELS__;
const DUPS = __DUPS__;
const KEY = "cifar100-audit-review-v1";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} progress(); }
function progress() {
  const n = LABELS.length + DUPS.length, d = Object.keys(state).length;
  document.getElementById("progress").textContent = `отмечено ${d} из ${n}`;
}
function buttons(id, opts) {
  const box = document.createElement("div"); box.className = "btns";
  opts.forEach(([v, t]) => {
    const b = document.createElement("button"); b.dataset.v = v; b.textContent = t;
    if (state[id] === v) b.classList.add("on");
    b.onclick = () => {
      state[id] = v; save();
      box.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
      box.parentElement.classList.add("done");
    };
    box.appendChild(b);
  });
  return box;
}
const L = document.getElementById("labels");
LABELS.forEach(r => {
  const id = "label:" + r.train_idx;
  const c = document.createElement("div"); c.className = "card" + (state[id] ? " done" : "");
  c.innerHTML = `<div class="row"><img class="big" src="${r.img}" alt="train ${r.train_idx}">
    <div class="meta"><div class="mute">train #${r.train_idx} · ранг ${r.rank}</div>
    <div>метка: <b>${r.given}</b></div><div>соседи: <b>${r.knn}</b> <span class="mute">(согласны ${r.agree})</span></div>
    <div>CLIP: <b>${r.clip}</b></div></div></div>
    <div class="nbs">${r.nbs.map(n => `<div class="nb"><img src="${n.img}" alt="">${n.label}</div>`).join("")}</div>`;
  c.appendChild(buttons(id, [["wrong", "Метка неверная"], ["correct", "Метка верная"], ["ambiguous", "Неоднозначно"]]));
  L.appendChild(c);
});
const D = document.getElementById("dups");
DUPS.forEach(r => {
  const id = "dup:" + r.test_idx;
  const c = document.createElement("div"); c.className = "card pair" + (state[id] ? " done" : "");
  c.innerHTML = `<div class="row"><img src="${r.test_img}" alt="test"><img src="${r.train_img}" alt="train"></div>
    <div class="meta mute">test #${r.test_idx} (${r.test_label}) · train #${r.train_idx} (${r.train_label})<br>сходство ${r.sim} · диапазон ${r.band}</div>`;
  c.appendChild(buttons(id, [["dup", "Дубликат"], ["not_dup", "Не дубликат"]]));
  D.appendChild(c);
});
document.getElementById("export").onclick = () => {
  const rows = [["kind", "id", "verdict"]];
  Object.entries(state).forEach(([k, v]) => { const [kind, id] = k.split(":"); rows.push([kind, id, v]); });
  const blob = new Blob([rows.map(r => r.join(",")).join("\\n")], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "stage6_review_verdicts.csv"; a.click();
};
progress();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="results/embeddings/cifar100_clip_b16.npz")
    p.add_argument("--review", default="results/stage6_cifar100_review.csv")
    p.add_argument("--duplicates", default="results/stage6_cifar100_duplicates.csv")
    p.add_argument("--n-labels", type=int, default=50)
    p.add_argument("--n-per-band", type=int, default=12)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results/stage6_review.html")
    args = p.parse_args()

    e = np.load(args.emb)
    classes = list(e["classes"])
    x_train, y_train, x_test, _, _ = load_raw("cifar100", args.data_dir)
    review = pd.read_csv(args.review).head(args.n_labels)
    idx = review.train_idx.to_numpy()
    nb_idx, _ = knn(e["train_emb"][idx], e["train_emb"], 6, False, get_device())

    labels = []
    for rank, (r, nbs) in enumerate(zip(review.itertuples(), nb_idx), start=1):
        nbs = [j for j in nbs if j != r.train_idx][:5]
        labels.append({"train_idx": int(r.train_idx), "rank": rank, "img": png(x_train[r.train_idx], 128),
                       "given": r.given_name, "knn": r.knn_name, "clip": r.clip_name,
                       "agree": f"{r.knn_agree:.0%}",
                       "nbs": [{"img": png(x_train[j], 52), "label": classes[y_train[j]]} for j in nbs]})

    pairs = pd.read_csv(args.duplicates)
    dups = []
    for lo, hi in BANDS:
        band = pairs[(pairs.cos_sim >= lo) & (pairs.cos_sim < hi)]
        for r in band.sample(min(args.n_per_band, len(band)), random_state=1).itertuples():
            dups.append({"test_idx": int(r.test_idx), "train_idx": int(r.train_idx),
                         "test_img": png(x_test[r.test_idx], 128), "train_img": png(x_train[r.train_idx], 128),
                         "test_label": r.test_label, "train_label": r.train_label,
                         "sim": f"{r.cos_sim:.3f}", "band": f"{lo}-{min(hi, 1.0)}"})

    html = PAGE.replace("__LABELS__", json.dumps(labels)).replace("__DUPS__", json.dumps(dups))
    Path(args.out).write_text(html, encoding="utf-8")
    print("saved", args.out, f"({len(labels)} labels, {len(dups)} pairs)")


if __name__ == "__main__":
    main()
