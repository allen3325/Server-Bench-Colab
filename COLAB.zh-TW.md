# 在 Google Colab（一張 GPU）上跑 ServerBench

[English](COLAB.md)

這個 fork 在 Harbor/docker 和 Modal 之外，新增第三種環境 backend：**local**。它不需要任何容器 runtime，直接在 driver 所在的機器上跑完整的 evolve。Colab runtime 正好是這種環境：有 root、一張 GPU、有 `CAP_SYS_ADMIN`，但沒有 docker，也沒有 `CAP_NET_ADMIN`（不能用 iptables）。

用法是用 GPU runtime（預設 **L4**，其他 GPU 見下方「換 GPU」）開啟 [`serverbench/scripts/colab/ServerBench_Colab.ipynb`](serverbench/scripts/colab/ServerBench_Colab.ipynb)，由上往下執行。

## 流程

| 步驟 | 做什麼 | 頻率 |
|---|---|---|
| 憑證 | 在 Colab Secrets 設定登入方式與存檔用的 rclone 設定 | 一次 |
| `setup.sh` | 安裝 harness、agent 用的 Python、CUDA 12.8 與 Nsight、Claude 和 Codex CLI、SGLang 0.5.19、模型、replay 語料、rclone | 每個 session（Colab VM 用完即丟） |
| `persist.sh restore` | 從雲端還原先前的結果 | 每個 session |
| `hardware.py new-task` | 為這張 GPU 建立任務（L4 已內建） | 每種 GPU 一次 |
| `calibrate.py search / seeds / record` | 在這張 GPU 上量 SGLang 的參考點和 p90 bar（方法見 `docs/harness/bar-derivation.md`） | 每種 GPU 一次（L4 約 5–6 小時），可分多個 session 完成 |
| `local-evolve.sh <task> --hours 3` | 跑一個 agent（`AGENT_CLI=claude` 或 `codex`），結束後自動跑最後驗證（約 45 分鐘） | 每個 agent 一次 |
| 比較 | 讀兩個 lineage 的 `FINAL.json` | |

L4 任務 `qwen3_5_2b_standard_1xL4` 內附的 bar（p90 TTFT 4100 ms、TPOT 35 ms）是暫定值，`calibrate.py record` 會換成實測值並設定 reference concurrency。

## 換 GPU

可以換。前提是：**一種 GPU 一個任務，而且要先在那張卡上校準**。

### 為什麼不能直接拿 L4 的任務去別張卡跑
任務的 p90 bar 是 SGLang 在那張卡上量出來的，`dollars_per_dollar` 也是用那張卡的每小時價格算的。在 A100 上跑 L4 任務，會拿 A100 的速度去跟 L4 的 bar 比較，分數沒有意義。原本的 harness 不會檢查實際的 GPU 和任務宣告的硬體是否一致，所以這個 fork 加了 `serverbench/scripts/colab/hardware.py`：
- `local-evolve.sh`、`verdict-only.sh`、`calibrate.py search/seeds` 開始前都會檢查。卡不符就拒絕執行（`SB_ALLOW_HARDWARE_MISMATCH=1` 可以強制，只建議除錯時用）。
- 同一張卡上比較的兩個 agent，必須用同一個任務；不同卡的結果不能互相比較。

### 支援狀況

| Colab GPU | 任務字尾 | 狀態 |
|---|---|---|
| L4（24 GB） | `L4` | **支援**：這個 fork 的設計目標，已用同架構（sm_89）的卡完整測試 |
| A100 40 GB | `A100` | 預期可用：Ampere、支援 bf16，SGLang 和工具鏈都支援，但沒有實際跑過完整流程 |
| A100 80 GB | `A10080GB` | 同上（記憶體不同，速度和 SGLang 預設值都不同，所以是另一個任務） |
| H100 80 GB | `H100` | 預期可用：Hopper，同上 |
| RTX PRO 6000（G4） | `RTXPRO6000` | 實驗性：Blackwell（sm_120），CUDA 12.8 支援，但 SGLang 0.5.19 的 kernel 沒驗證過 |
| T4（16 GB） | `T4` | **不支援**：Turing 沒有 bf16，而模型和任務的 SGLang 參數都用 bf16（`SB_ALLOW_UNSUPPORTED_GPU=1` 可以硬試） |

Colab 提供哪些 GPU，取決於你的方案和當下的供應量。

### 步驟（notebook 會自動處理）
1. Colab 選單「執行階段 → 變更執行階段類型」，選想要的 GPU。
2. 第 0 格的 `TASK = None` 保持不動：第 1 格會用 `hardware.py task` 依照偵測到的 GPU 決定任務，例如 `qwen3_5_2b_standard_1xA100`。需要的話，在第 0 格設定這張卡的每小時價格 `GPU_RATE`。
3. 照常跑憑證、`setup.sh`、還原。
4. 第 5b 格：如果這張卡還沒有任務，就從 L4 任務複製一份。模型、token 價格、框架參數、正確性設定都相同，只換硬體和每小時價格。建立後會存到雲端，下次 session 會自動還原。
5. **第 6 節校準一定要跑**。新任務一開始沿用 L4 的暫定 bar，校準後才是這張卡的實測值。耗時取決於卡的速度（L4 約 5–6 小時）。
6. 照常跑 Claude 和 Codex。

不用 notebook 的話，手動步驟如下：

```
python3 serverbench/scripts/colab/hardware.py detect          # 這張卡的字尾和支援狀況
serverbench/.venv/bin/python serverbench/scripts/colab/hardware.py new-task --usd-per-gpu-hour 1.50
export TASK=$(python3 serverbench/scripts/colab/hardware.py task)
serverbench/.venv/bin/python serverbench/scripts/colab/calibrate.py search --task $TASK --model-dir ... --out-dir ...
# seeds、record 同理，再跑 local-evolve.sh $TASK --hours 3
```

### 注意事項
- **每小時價格**（`GPU_RATE` / `--usd-per-gpu-hour`）由你自己設定，預設 0.80 美元。同一張卡上的兩個 agent 會被乘上同一個比例，所以不影響誰贏誰輸；但 `dollars_per_dollar` 的絕對值會跟著改變。
- **校準結果存在哪裡**：新任務的檔案（`serving_tasks/<task>/` 和 `serverbench/harness/final_validate/tasks/<task>.py`）不在 git 裡，靠 `persist.sh` 存到雲端。想把它們 commit 進 repo 的話，還要在 `serverbench/tests/contracts/test_the_p90_bars.py` 的 `P90_ONLY` 補上這個任務的 bar，否則測試會失敗。
- **記憶體不同，SGLang 預設值也不同**：例如 chunked prefill，35 GB 以下是 2048，60 GB 以下是 4096，更大的卡更大。這是每種卡要有自己的任務、自己校準的另一個原因。
- Colab 一個 runtime 只有一張 GPU，所以只能跑單卡（`1x`）的任務。

## 憑證：兩家都支援訂閱方案

在 Colab 左側的 🔑 Secrets 設定，並打開 notebook access。

| Agent | 訂閱方案 | API 金鑰 |
|---|---|---|
| Claude Code | `CLAUDE_CODE_OAUTH_TOKEN`：在自己電腦執行 `claude setup-token` 取得的長效 token | `ANTHROPIC_API_KEY` |
| Codex | notebook 裡執行 `codex login --device-auth`（device code 登入，建議用這個）；或把自己電腦的 `~/.codex/auth.json` 整份貼進 `CODEX_AUTH_JSON` | `OPENAI_API_KEY` |

憑證只會寫進 root 才讀得到的 `serverbench/artifacts/secrets/` 或 `/root`，再由 harness 交給該次 run 的沙箱。存檔時，任何 `secrets/` 目錄都會被排除，不會上傳到雲端。

`CODEX_AUTH_JSON` 的注意事項：ChatGPT 登入的 refresh token 會輪替。同一份登入同時在兩台機器用，一邊刷新後，另一邊可能失效。用 device code 在 Colab 另開一個 session 就沒有這個問題。

## 斷線與存檔

| 情況 | run 還在嗎 | 怎麼辦 |
|---|---|---|
| 瀏覽器斷線、網路斷、筆電睡眠 | 還在（run 用 `setsid nohup` 在背景跑） | 回來後執行 notebook 的「重新接上」那格（`reattach()`） |
| Notebook kernel 重啟 | 還在 | 同上 |
| VM 被回收（Colab Pro 關分頁一段時間後，或碰到 24 小時上限） | 消失 | 重跑 notebook 的 0–5 格還原，再用第 8 節補跑最後驗證 |

**自動存檔**（`serverbench/scripts/colab/checkpoint.py`）：只要設定了 `RCLONE_CONF`，`local-evolve.sh` 會自動帶起存檔程式。

- 每 10 分鐘存一次，每發佈一個新版本後也會盡快存一次。所以 VM 被回收，最多只丟最近一段工作。
- 量測進行中不存檔（`./full-test` 的 replay，以及最後驗證）。上傳會和受測的 server 搶 CPU、磁碟和網路，所以等量測結束再存。
- driver 結束後（包括最後驗證跑完）再存一次。
- 會把 agent 的 Claude Code 對話紀錄複製到 `<run>/claude-projects`，那是上游 make-up run（`run.py --resume-run`）讀取對話的位置。Codex 的對話本來就在 `<run>/codex-sessions`。

**存到哪裡**：`RCLONE_CONF` 是一個名為 `sb` 的 rclone remote。用 Google Drive 的話，在自己電腦執行：

```
rclone config create sb drive scope=drive.file
rclone config show sb      # 把印出來的整段貼進 Colab Secret RCLONE_CONF
```

`drive.file` 權限只能存取 rclone 自己建立的檔案，碰不到 Drive 裡其他東西。也可以改用 GCS bucket 的 remote。預設的目的地是 `sb:serverbench-colab`（可以用 `SB_REMOTE` 改）。

**補跑最後驗證**（`serverbench/scripts/colab/verdict-only.sh [RUN_DIR] [--force]`）：如果 agent 已經跑完，VM 卻在最後驗證期間被回收，還原後對該 run 只跑最後驗證就好。不需要 agent，不花 API 額度，只花 GPU 時間。已經有驗證結果的 run 會拒絕重跑，除非加上 `--force`。補跑這件事會記在 run 的 journal（`final_validation_rerun`）。

### 為什麼雲端存檔不會破壞隔離

- 上傳程式以 root 身分，在 host 的網路上執行。agent 的 network namespace 裡沒有多出任何路由，也沒有多出任何掛載。
- rclone 和它的 token 都只有 root 能用：rclone 在 `/root/bin`（`/root` 是 700），設定檔在 700 的 `secrets/`。agent 讀不到、也執行不了。
- 上傳的檔案裡有 agent 寫的東西，所以 **symlink 一律跳過、不跟隨**。agent 埋一個指向 root 檔案的連結，也不能讓 root 程式替它讀出來。任何 `secrets/` 目錄都排除在外。模擬測試時故意埋了這兩種陷阱，remote 上都沒有出現。
- 量測期間暫停上傳，避免影響計時。

## local backend 用什麼取代 docker

| docker（Harbor） | local |
|---|---|
| 容器加上 iptables 白名單（`restrict-network.sh`） | agent 跑在只有 `lo` 的 **network namespace**（`serving_evolve/local_sandbox.py`），沒有路由，沒有 DNS |
| docker bridge 上的 egress broker | 同一個 broker 改聽 **unix socket**，再轉到 namespace 內的 `127.0.0.1:3128`。只允許 CONNECT 到供應商的主機，每次連線都記錄 |
| image 裡的 `USER agent` | 系統的 `agent` 使用者（uid ≠ 0），用 `nsenter --setuid` 進入 |
| bind mount | repo 內的路徑：主機允許 `mount` 就用 bind mount，否則用 symlink。`/opt/serverbench` 底下的模型和 HF cache 一律用 symlink |
| 容器看不到主機的檔案 | 沒有 mount namespace，所以私有目錄（驗證證據、答案池、憑證、其他 run、已發佈的 lineage）都設成 root-only；能用 bind mount 時連 repo 根目錄也是 |
| 在全新容器裡跑驗證 | 在兩個全新沙箱裡跑驗證（`final_validate/local_verdict.py`），完全沒有對外連線。開始前會殺掉 agent 使用者的所有程序，並確認都已結束 |
| `lmsysorg/sglang:v0.5.19` image | SGLang 0.5.19 裝在自己的 interpreter（`/opt/serverbench/sglang-venv`） |
| agent image 的 Python、CUDA 12.8、Nsight | `/opt/serverbench/agent-python`（image 的版本釘選，從凍結的 wheelhouse 安裝），以及 image 用的 CUDA 12.8 和 Nsight `.deb`，含 checksum 驗證 |

`serverbench/scripts/colab/selfcheck.py` 會在當下這台機器上證明隔離成立：不是 root、不能直接上網、透過 broker 只連得到供應商、私有目錄讀不到、loopback 和 GPU 正常、結束後不留下任何程序。

不變的部分：任務說明、工作區內容、`./version`、`./full-test`、`./measurements`、journal、trajectory、AgentX replay、私有正確性檢查、計分。最後驗證直接重用 Modal backend 的沙箱內 worker（`final_validate/modal_worker.py`），沒有改動。

## 和論文設定的差異

- 硬體是 Colab 的 GPU（預設 L4：sm_89、24 GB、約 300 GB/s），不是論文用的 L40。數字只能和同一個任務（同一種卡）的 run 比較。
- `contract.yaml` 裡的每小時價格（預設 0.80 美元）是暫定值。它會把每個 run 的 `dollars_per_dollar` 乘上同一個比例，所以不影響 agent 之間的比較。
- agent 能看到主機的檔案系統（以一般使用者的權限）。docker 只會讓它看到容器內的檔案。
- `ncu` 可能需要 Colab VM 不開放給非 root 使用者的 profiling 權限。
- 校準完成後，記得更新 `serverbench/tests/contracts/test_the_p90_bars.py` 裡該任務的 bar（`P90_ONLY`）。

## 修正紀錄

**plain reference 的 chunked prefill**（`final_validate/private_correctness.py`，上游的問題）。驗證用的 plain reference 開了 `--enable-deterministic-inference`，在 Triton backend 下，SGLang 只會在 4096 token 的倍數切 prefill chunk。但 35 GB 以下的卡（L4、A10），SGLang 預設的 chunk 是 2048，比對齊單位還小。結果任何超過一個 chunk 的 prompt 都排不進去，scheduler 會一直卡住。EOS 稽核需要重放很長的 trace 請求，所以會讓整個驗證無限期卡住。L40 的預設 chunk 是 4096，上游因此沒遇到。現在固定為 4096；chunk 邊界本來就對齊，所以不影響計分用的 logprob。

**host 端的任務路徑**（local backend 本身的修正）。harness 在 host 上找任務定義時，若沒有指定 `SERVERBENCH_TASK_ROOT`，會去猜 `/workspace/task_context`。docker 版的 host 上沒有 `/workspace`，但 local backend 會在 host 上放一個 `/workspace` symlink 指向 agent 的工作區，而且 run 結束後還留著。結果 host 端的工具，包括 driver 和最後驗證，可能讀到 **agent 改得到的任務定義副本**。現在兩邊都修了：
- 所有 host 端的進入點（`local-evolve.sh`、`verdict-only.sh`、`setup.sh`、`calibrate.py`、`hardware.py`、notebook）都明確指定 `SERVERBENCH_TASK_ROOT=<repo>/serving_tasks`。
- 沙箱收掉時（`local_sandbox.terminate`），會一併移除 `local_env` 在 host 上放的 symlink，並卸載 bind mount。

## 檔案

| 檔案 | 用途 |
|---|---|
| `serverbench/scripts/colab/ServerBench_Colab.ipynb` | notebook（由 `make_notebook.py` 產生，請改產生器再重新產生） |
| `serverbench/scripts/colab/setup.sh` | 每個 session 的環境設定，可重複執行 |
| `serverbench/scripts/colab/selfcheck.py` | 隔離自我檢查 |
| `serverbench/scripts/colab/hardware.py` | 偵測 GPU、檢查 GPU 是否符合任務、為新 GPU 建立任務 |
| `serverbench/scripts/colab/calibrate.py` | L4 校準 |
| `serverbench/scripts/colab/persist.sh` | 用 rclone 存檔、還原、查看最近一次存檔（`save` / `restore` / `status`） |
| `serverbench/scripts/colab/checkpoint.py` | run 期間的自動存檔程式 |
| `serverbench/scripts/colab/verdict-only.sh` | 補跑最後驗證 |
| `serverbench/scripts/server_evolve/local-evolve.sh` | 用 local backend 跑一次 evolve |
| `serverbench/harness/serving_evolve/local_sandbox.py` | namespace 沙箱 |
| `serverbench/scripts/agent/local_env.py` | 立起 agent 的環境（對應 docker 版的 `harbor_env.py`） |
| `serverbench/harness/final_validate/local_verdict.py` | local backend 的最後驗證 |
