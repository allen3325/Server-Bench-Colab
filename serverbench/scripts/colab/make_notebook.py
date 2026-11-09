#!/usr/bin/env python3
"""Generate ServerBench_Colab.ipynb (edit here, then regenerate):

    python3 serverbench/scripts/colab/make_notebook.py serverbench/scripts/colab/ServerBench_Colab.ipynb
"""
import json
import sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(True)})


md("""
# ServerBench on Colab: Claude vs Codex

這本 notebook 在 **一張 Colab GPU**（預設 L4，也可以換 A100、H100 等，見 `COLAB.zh-TW.md` 的「換 GPU」）上跑 ServerBench 的 serving-system evolve，用來比較 Claude Code 和 Codex。

Colab 不能跑 Docker，所以這裡用的是 repo 裡新增的 **local backend**：
- agent 以一般使用者 `agent` 身分，跑在只有 loopback 的 network namespace 裡。唯一的對外連線是 egress broker，只放行模型供應商的 API，每次連線都會記錄。
- 私有資料（答案池、驗證結果、其他 run、已發佈的 lineage）都設成 root-only。agent 結束、所有程序確認清除之後，才在新的沙箱裡跑最終驗證。
- 版本發佈、`./full-test`、journal、trajectory、計分，用的都是和 Docker 版相同的程式。

**順序**：
1. 設定憑證和 runtime（每個新 session 都要重跑）
2. **校準**：在這張 GPU 上量 SGLang 的參考點和 p90 bar。每種 GPU 只需要做一次（L4 約 5–6 小時），可以分好幾個 session 完成
3. 跑 Claude（3 小時 + 約 45 分鐘驗證）
4. 跑 Codex（3 小時 + 約 45 分鐘驗證）
5. 比較

**斷線與存檔**：結果用 rclone 存到你的 Google Drive 或 GCS，不掛載 Drive。agent 跑的時候，root 端的存檔程式每 10 分鐘、以及每發佈一個新版本後存一次（量測進行中會延後），所以 VM 被回收最多只丟最近一段工作。
- 只是瀏覽器斷線：run 還在跑，回來後執行「重新接上」那格即可。
- VM 被回收：重跑 0–5 格還原，再用第 8 節補跑最後驗證。

> Colab Pro 沒有背景執行，分頁關掉一段時間後 runtime 會被回收；分頁請保持開著。
""")

md("## 0. 參數")
code("""
TASK = None                     # None: the task for this runtime's GPU (e.g. qwen3_5_2b_standard_1xA100)
GPU_RATE = 0.80                 # USD per GPU hour for a NEW GPU's task (does not change an agent comparison)
HOURS = 3                       # each agent's wall-clock budget
CLAUDE_MODEL, CLAUDE_EFFORT = "claude-opus-5", "xhigh"
CODEX_MODEL, CODEX_EFFORT = "gpt-5.6-sol", "max"
REPO_URL = "https://github.com/allen3325/Server-Bench-Colab.git"
BRANCH = "colab"
REPO = "/content/Server-Bench-Colab"
SB_REMOTE = "sb:serverbench-colab"   # <rclone remote>:<folder>
CALIBRATION = "/content/sb-calibration"
import os
os.environ.update(CALIBRATION=CALIBRATION, HF_HOME="/opt/serverbench/hf",
                  SB_REMOTE=SB_REMOTE,
                  SERVERBENCH_TASK_ROOT=REPO + "/serving_tasks",
                  PATH="/opt/serverbench/bin:" + os.environ["PATH"])
""")

md("## 1. 確認 GPU，取得 repo")
code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import os, subprocess
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=False)
%cd {REPO}
# Which task this runtime runs: one task per GPU type, because a task's bars and
# dollars are its own card's (serverbench/scripts/colab/hardware.py).
!python3 serverbench/scripts/colab/hardware.py detect
TASK = TASK or subprocess.run(["python3", "serverbench/scripts/colab/hardware.py", "task"],
                              capture_output=True, text=True, check=True).stdout.strip()
os.environ["TASK"] = TASK
print("TASK =", TASK)
""")

md("""
## 2. 憑證

在 Colab 左側的 🔑 **Secrets** 設定以下項目，並打開 notebook access。所有憑證只會寫進 root 才能讀的 `serverbench/artifacts/secrets/` 或 `/root`，agent 讀不到，也不會被存到雲端。

**Claude**（擇一）

| Secret | 說明 |
|---|---|
| `CLAUDE_CODE_OAUTH_TOKEN` | **訂閱方案**（Pro / Max）。在自己電腦執行 `claude setup-token`，用瀏覽器登入後會印出一組長效 token |
| `ANTHROPIC_API_KEY` | API 金鑰計費 |

**Codex**（擇一）

| 方式 | 說明 |
|---|---|
| 第 4 節的 device code 登入 | **訂閱方案**（ChatGPT Plus / Pro 等），建議用這個。在 Colab 裡執行 `codex login --device-auth`，到瀏覽器輸入代碼即可 |
| Secret `CODEX_AUTH_JSON` | **訂閱方案**的另一種做法：把自己電腦上 `~/.codex/auth.json` 的內容整份貼上。注意：同一份登入同時在兩台機器使用，token 刷新時可能讓另一邊失效 |
| Secret `OPENAI_API_KEY` | API 金鑰計費 |

**存檔**（建議設定，沒有的話 VM 被回收就什麼都不剩）

| Secret | 說明 |
|---|---|
| `RCLONE_CONF` | 一個名為 `sb` 的 rclone remote 設定。Google Drive：在自己電腦執行 `rclone config create sb drive scope=drive.file`（瀏覽器授權），再用 `rclone config show sb` 印出整段貼上。`drive.file` 權限只能存取 rclone 自己建立的檔案，碰不到你 Drive 裡其他東西。也可以改用 GCS bucket 的 remote |
""")
code("""
from google.colab import userdata
import pathlib, subprocess, os

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

secrets_dir = pathlib.Path(REPO, "serverbench/artifacts/secrets")
secrets_dir.mkdir(parents=True, exist_ok=True)
os.chmod(secrets_dir, 0o700)

def write_secret(path, text):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.chmod(path, 0o600)

claude_env = secrets_dir / "claude.env"
if secret("CLAUDE_CODE_OAUTH_TOKEN"):
    write_secret(claude_env, f"CLAUDE_CODE_OAUTH_TOKEN={secret('CLAUDE_CODE_OAUTH_TOKEN')}\\n")
    print("Claude: subscription (setup-token)")
elif secret("ANTHROPIC_API_KEY"):
    write_secret(claude_env, "ANTHROPIC_BASE_URL=https://api.anthropic.com\\n"
                             f"ANTHROPIC_API_KEY={secret('ANTHROPIC_API_KEY')}\\n")
    print("Claude: API key")
else:
    print("Claude: no secret set")

if secret("RCLONE_CONF"):
    write_secret(secrets_dir / "rclone.conf", secret("RCLONE_CONF").strip() + "\\n")
    print("Checkpoints: rclone remote ->", SB_REMOTE)
else:
    print("Checkpoints: NO RCLONE_CONF -- results live only on this VM")
""")

md("## 3. 設定 runtime（每個 session 都要跑）")
code("""
!bash serverbench/scripts/colab/setup.sh 2>&1 | tail -60
""")

md("""
## 4. Codex 登入

依序檢查 `CODEX_AUTH_JSON`、`OPENAI_API_KEY`；都沒有的話，執行下一格用 device code 登入訂閱方案。
""")
code("""
codex_home = pathlib.Path("/root/.codex")
codex_home.mkdir(exist_ok=True)
os.chmod(codex_home, 0o700)
if secret("CODEX_AUTH_JSON"):
    write_secret(codex_home / "auth.json", secret("CODEX_AUTH_JSON").strip() + "\\n")
    print("Codex: subscription (auth.json from your machine)")
elif secret("OPENAI_API_KEY"):
    subprocess.run(["codex", "login", "--with-api-key"], input=secret("OPENAI_API_KEY"),
                   text=True, check=True)
    print("Codex: API key")
elif (codex_home / "auth.json").exists():
    print("Codex: already logged in")
else:
    print("Codex: not logged in yet -- run the next cell")
!codex login status
""")
code("""
# 訂閱方案：印出網址和代碼後，到瀏覽器登入 ChatGPT 並輸入代碼。
# 若提示需要開啟 device code 授權，依照提示到 ChatGPT 的設定開啟後再重跑。
!codex login --device-auth
!codex login status
""")

md("""
## 5. 還原先前存的結果

從 rclone remote 還原校準結果、已校準的任務檔、各 run 與驗證結果。第一次使用時沒有東西可還原。
""")
code("""
!bash serverbench/scripts/colab/persist.sh status
!bash serverbench/scripts/colab/persist.sh restore
""")

md("""
## 5b. 這張 GPU 的任務

每種 GPU 有自己的任務（`qwen3_5_2b_standard_1x<GPU>`）。L4 的任務 repo 內建；其他 GPU 第一次使用時，會從 L4 任務複製一份（模型、token 價格、框架參數都相同），只換硬體和每小時價格 `GPU_RATE`。新任務先沿用 L4 的暫定 bar，**一定要跑第 6 節的校準**，才能拿來比較。建立後會存到雲端，下次 session 會自動還原。
""")
code("""
if not pathlib.Path(REPO, "serving_tasks", TASK).is_dir():
    !serverbench/.venv/bin/python serverbench/scripts/colab/hardware.py new-task --usd-per-gpu-hour {GPU_RATE}
    !bash serverbench/scripts/colab/persist.sh save
!grep -E "^(title|hardware|usd_per_gpu_hour):" serving_tasks/{TASK}/contract.yaml
!grep -A3 service_levels serving_tasks/{TASK}/contract.yaml
""")

md("""
## 6. 校準：在這張 GPU 上量 SGLang 的參考點和 p90 bar（每種 GPU 只做一次）

方法和 repo 原本的一樣（`docs/harness/bar-derivation.md`）：
1. **search**：offered concurrency 1, 2, 4, 8, 16 各跑一個 1,800 秒窗口，取「gross $/$ ÷ p90 端到端延遲」最高的點
2. **seeds**：在選定的點再跑 5 個不同 seed，bar = 5 次中最差的 p90
3. **record**：寫回任務的 `reference.yaml` 和 `contract.yaml`

每個點量完都可以續跑：已經量過的點會直接讀回來。每跑完一步就存檔。
""")
code("""
MODEL_DIR = subprocess.run(["bash", "-c",
    "ls -d /opt/serverbench/hf/hub/models--Qwen--Qwen3.5-2B/snapshots/*/ | head -1"],
    capture_output=True, text=True).stdout.strip()
print(MODEL_DIR)
!serverbench/.venv/bin/python serverbench/scripts/colab/calibrate.py search --task {TASK} --model-dir {MODEL_DIR} --out-dir {CALIBRATION}
!bash serverbench/scripts/colab/persist.sh save
""")
code("""
!serverbench/.venv/bin/python serverbench/scripts/colab/calibrate.py seeds --task {TASK} --model-dir {MODEL_DIR} --out-dir {CALIBRATION}
!bash serverbench/scripts/colab/persist.sh save
""")
code("""
!serverbench/.venv/bin/python serverbench/scripts/colab/calibrate.py record --task {TASK} --out-dir {CALIBRATION}
!bash serverbench/scripts/colab/persist.sh save
!grep -A3 service_levels serving_tasks/{TASK}/contract.yaml
""")

md("""
## 7. 跑 agent

run 在背景執行（`setsid nohup`），瀏覽器斷線不影響。下面的監看 cell 每 5 分鐘印一次進度和最近一次存檔。時間到之後 driver 會凍結最後版本，再跑最終驗證（約 45 分鐘），結束時再存一次檔。
""")
code("""
import subprocess, time, pathlib, json, os

PIDFILE = pathlib.Path("/run/serverbench-local-driver.pid")

def start_run(cli, model, effort):
    if os.path.ismount("/content/drive"):
        from google.colab import drive
        drive.flush_and_unmount()
    if PIDFILE.exists() and pathlib.Path(f"/proc/{PIDFILE.read_text().strip()}").exists():
        raise RuntimeError("a run is already going; use reattach()")
    env = dict(os.environ, AGENT_CLI=cli, AGENT_MODEL=model, AGENT_EFFORT=effort)
    if cli == "claude" and (pathlib.Path(REPO) / "serverbench/artifacts/secrets/claude.env").exists():
        env["AGENT_ENV"] = str(pathlib.Path(REPO) / "serverbench/artifacts/secrets/claude.env")
    log = pathlib.Path(f"/content/run-{cli}.log")
    proc = subprocess.Popen(
        ["setsid", "nohup", "bash", "serverbench/scripts/server_evolve/local-evolve.sh",
         TASK, "--hours", str(HOURS)],
        cwd=REPO, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL)
    print(f"started {cli} ({model}); log {log}")
    return proc, log

def latest_run():
    runs = sorted(pathlib.Path(REPO, "serverbench/evolve_space/evolve", TASK.lower()).glob("*/journal.jsonl"),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1].parent if runs else None

def _running(proc, pid):
    if proc is not None:
        return proc.poll() is None
    try:
        return open(f"/proc/{pid}/stat").read().split(") ", 1)[1][0] != "Z"
    except OSError:
        return False

def watch(proc, log, every=300, pid=None):
    log = pathlib.Path(log)
    while _running(proc, pid):
        run = latest_run()
        events = []
        if run is not None:
            for line in (run / "journal.jsonl").read_text().splitlines()[-400:]:
                try:
                    events.append(json.loads(line).get("event"))
                except ValueError:
                    pass
        saved = [l for l in log.read_text(errors="replace").splitlines() if " checkpoint: " in l]
        print(time.strftime("%H:%M:%S"), "| last event:", events[-1] if events else "-",
              "| versions:", events.count("head_advanced"), "| run:", run.name if run else "-",
              "| last save:", saved[-1].split(" checkpoint: ", 1)[1][:60] if saved else "-", flush=True)
        time.sleep(every)
    print("finished", f"with exit code {proc.returncode}" if proc is not None else "")
    print(log.read_text(errors="replace")[-3000:])

def reattach(cli):
    \"\"\"After a browser disconnect or a kernel restart: follow the run that is still going.\"\"\"
    if not PIDFILE.exists():
        print("no run is going"); return
    watch(None, f"/content/run-{cli}.log", pid=int(PIDFILE.read_text().strip()))

def save_results():
    subprocess.run(["bash", "serverbench/scripts/colab/persist.sh", "save"], cwd=REPO)
""")

md("### 7a. Claude")
code("""
proc, log = start_run("claude", CLAUDE_MODEL, CLAUDE_EFFORT)
watch(proc, log)
save_results()
""")

md("### 7b. Codex")
code("""
proc, log = start_run("codex", CODEX_MODEL, CODEX_EFFORT)
watch(proc, log)
save_results()
""")

md("### 重新接上（瀏覽器斷線或 kernel 重啟後，run 還在跑時用）")
code("""
reattach("claude")   # or "codex"
""")

md("""
## 8. 補跑最後驗證（VM 在驗證期間被回收時）

先跑 0–5 格還原，再對指定的 run 只跑最後驗證：不需要 agent、不花 API 額度，只花 GPU 時間（約 45 分鐘）。已經有驗證結果的 run 會拒絕重跑，除非加上 `--force`。
""")
code("""
!ls -t serverbench/evolve_space/evolve/{TASK.lower()}/
# 預設是最新的 run；也可以指定：verdict-only.sh serverbench/evolve_space/evolve/<task>/<run-id>
!bash serverbench/scripts/colab/verdict-only.sh
""")

md("""
## 9. 比較

讀兩個 lineage 的最終驗證結果（`FINAL.json`）與 `evaluation_progress/<task>.csv`。
""")
code("""
import glob, json, pathlib
rows = []
for final in sorted(glob.glob(f"{REPO}/server_output/*/FINAL.json")):
    body = json.loads(pathlib.Path(final).read_text())
    if TASK not in json.dumps(body):
        continue
    rows.append((pathlib.Path(final).parent.name, body))
for lineage, body in rows:
    print("==", lineage)
    print(json.dumps({k: body.get(k) for k in ("status", "head", "verdict", "dollars_per_dollar",
                                                "gross_dollars_per_dollar", "because") if k in body},
                     indent=1)[:1500])
csv = pathlib.Path(REPO, "evaluation_progress", f"{TASK}.csv")
print(csv.read_text() if csv.exists() else "(no ledger rows yet)")
""")

nb = {"cells": cells, "metadata": {
    "accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
open(sys.argv[1], "w").write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
