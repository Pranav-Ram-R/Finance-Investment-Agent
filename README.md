# FinPlan — Goal-Based Investment Planner Agent

[![CI](https://github.com/Pranav-Ram-R/Finance-Investment-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranav-Ram-R/Finance-Investment-Agent/actions/workflows/ci.yml)

An AI agent that turns a financial **goal** into a personalized, stress-tested
**investment plan** — and tracks it over time. Built with **LangChain**, real
market data (**yfinance**), and a swappable, free-tier LLM backend. Served as an
**async FastAPI** service and packaged with **Docker**.

> *"I have ₹2,00,000 now, can add ₹15,000/month, and want ₹50,00,000 in 12
> years at moderate risk."*
> → risk profile → asset allocation → growth projection + Monte-Carlo range
> → feasibility gap → concrete adjustments.

## What you can ask

You talk to the agent in plain language — it extracts the numbers, normalizes
units, and picks the tools. Inputs fall into four kinds:

**1. Start a plan.** State your goal; the agent pulls out five inputs:

| Input | Examples it understands |
|---|---|
| Initial lump sum | `₹2 lakh`, `5,00,000`, *nothing upfront* (0) |
| Monthly SIP | `₹15,000/month`, `20k monthly`, *can't invest monthly* (0) |
| Horizon | `12 years` |
| Goal amount | `₹50 lakh`, `1.5 crore`, `₹2,00,000` |
| Risk tolerance | `low`/`medium`/`high` — or `conservative`/`moderate`/`aggressive` |

Amounts work as lakh/crore words, ₹ symbols, or grouped digits — you never
convert anything. If a detail is missing, the agent asks before proceeding.

**2. What-if follow-ups** (same conversation, via the granular tools):
*"bump the SIP to ₹20k"*, *"extend to 15 years"*, *"what return would I need?"*,
*"show a more aggressive allocation"*, *"adjust my target for inflation"*,
*"what's my corpus after LTCG tax?"*, *"what's the current market mood for Nifty?"*.

**Multiple goals at once** — *"I want ₹50L for retirement in 20 years and ₹15L for
a car in 5 years"* → each goal is planned independently and the totals (combined
monthly SIP, combined post-tax corpus) are summed.

**3. Messy / partial input** — vague risk (*"I'm cautious"*), missing fields, or
a goal in today's money (handled via the inflation-adjusted target).

**4. Save & track across sessions** — *"save this plan"*, *"what was my plan?"*,
*"I've invested ₹1.8L, portfolio's at ₹2.1L — am I on track?"*

## Why the reasoning is trustworthy

The LLM **never does arithmetic**. Every number — returns, risk, projections,
feasibility — is computed by deterministic Python tools. The model decides
*which* tools to call, interprets the results, personalizes, and explains.
That separation is what makes the financial reasoning accurate and auditable.

## Architecture

```
                    ┌──────────────────────────────┐
   Streamlit UI ──▶ │   LangChain tool-calling      │
   (chat + charts)  │   agent  + conversation memory │
                    └───────────────┬───────────────┘
                                    │ picks tool(s)
        ┌──────────────┬───────────┼───────────┬────────────────┐
        ▼              ▼           ▼           ▼                ▼
   risk_profiler  get_asset_   project_    monte_carlo     check_
   + allocation   data (yf)    growth      _simulation     feasibility
   (rules)        (REAL data)  (FV math)   (uncertainty)   (solver)
        └──────────────┴───────────┴───────────┴────────────────┘
                                    │
                       persistent memory store
                  (profile · goal · plan · progress)
```

Two-model setup (both swappable): a fast **orchestrator** drives the tool loop;
an optional heavier **advisor** writes the final recommendation.

## Swap models from one place

All models are created via `finplan.config.get_chat_model(role)`. Change the
model for a role by editing one line in `.env` — no code changes:

```env
ORCHESTRATOR_MODEL=google_genai:gemini-2.5-flash    # tool-calling loop
ADVISOR_MODEL=groq:deepseek-r1-distill-llama-70b    # final write-up
```

Use a full `provider:model_id` or a preset alias:
`gemini-flash`, `gemini-pro`, `llama-70b`, `deepseek-r1`, `qwen`, `mistral`.
Free keys: [Google AI Studio](https://aistudio.google.com/app/apikey) ·
[Groq Console](https://console.groq.com/keys).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env          # then add your API key(s)
```

## Run the tests

```bash
pytest -q
```

## See the engine in action (no API key needed)

```bash
python -m scripts.demo_engine
```

Chains every tool on live market data: risk profile → allocation → real
return/risk → blended stats → projection → Monte-Carlo range → feasibility.

## Run the app

Add a free API key to `.env` (`GOOGLE_API_KEY` or `GROQ_API_KEY`), then:

```bash
streamlit run app.py        # full chat UI with charts + tool-call trace
python -m scripts.chat      # or a plain terminal chat
```

## REST API

The engine and agent are also exposed as an **async FastAPI** service.
`/plan` and `/plan/multi` are **fully deterministic — they need no LLM key**
(the Python engine computes every figure), so a public demo can't burn free-tier
quota. `/chat` drives the conversational agent and requires a provider key.

```bash
uvicorn finplan.api.main:app --reload    # http://localhost:8000/docs (OpenAPI UI)
```

| Method | Path         | Auth        | Description                                            |
|--------|--------------|-------------|--------------------------------------------------------|
| GET    | `/healthz`   | none        | Liveness + resolved model config                       |
| POST   | `/plan`      | none        | Full goal-based plan (deterministic)                   |
| POST   | `/plan/multi`| none        | Several goals planned independently + combined totals  |
| POST   | `/chat`      | LLM key     | One conversational agent turn (reply + tool trace)     |

```bash
curl -X POST http://localhost:8000/plan -H "Content-Type: application/json" \
  -d '{"initial":"2 lakh","monthly":"15000","years":12,"goal":"50 lakh","risk_tolerance":"medium"}'
```

Money fields accept lakh/crore words, ₹ symbols, or plain numbers — the unit
conversion happens in Python (`parse_amount`), never in an LLM.

## Docker

```bash
docker build -t finplan .
docker run -p 8000:8000 --env-file .env finplan   # /plan works even with an empty .env
# or:  docker compose up
```

Multi-stage build, slim runtime, non-root user, and a `/healthz` container
healthcheck.

## Deploy a live demo

- **API → Render** (free): New → **Blueprint** → point at this repo. The included
  [`render.yaml`](render.yaml) builds from the `Dockerfile`; set `GROQ_API_KEY` in
  the dashboard. `/plan` works even before you add a key.
- **UI → Streamlit Community Cloud** (free, zero-infra): connect the repo, pick
  `app.py`, and add `GROQ_API_KEY` in the app's **Secrets**. One-click public link.

## Observability & evaluation

- **Per-turn metrics** — the `/chat` response carries a `metrics` block (token
  usage, latency, and LLM/tool call counts) from a LangChain callback
  ([`finplan/observability.py`](finplan/observability.py)).
- **Agent evaluation** — a golden set + three deterministic checks (tool
  selection, input extraction, and *numeric grounding* — that the reply quotes the
  engine's figures verbatim), with an optional LLM-as-judge:

  ```bash
  python -m finplan.eval           # pass-rate report (needs an LLM key)
  python -m finplan.eval --judge   # also rate explanation quality 1-5
  ```

## Project layout

```
finplan/
  config.py            # swappable multi-model LLM config
  tools/
    market_data.py     # yfinance: real return/risk stats (corrupt-tick safe)
    projection.py      # exact lump-sum + SIP future value
    risk.py            # risk profile + allocation + portfolio blend
    simulation.py      # Monte-Carlo outcome range + goal probability
    feasibility.py     # solvers: required SIP / years / return; inflation
    tax.py             # post-tax corpus via a deterministic LTCG estimate
    news.py            # "market mood" from yfinance headlines + a lexicon
  agent/
    tools.py           # 15 LangChain tools wrapping the engine
    prompts.py         # workflow-enforcing system prompt
    planner_agent.py   # create_agent + memory + run_turn
  api/
    schemas.py         # Pydantic request/response models (money as text -> ₹)
    main.py            # FastAPI app: /healthz /plan /plan/multi /chat
  eval/                # agent eval: golden set + scoring + `python -m finplan.eval`
  observability.py     # token/latency/tool metrics callback + structured logging
  memory/
    store.py           # SQLite: persistent plans + progress
scripts/
  demo_engine.py       # end-to-end engine demo (no API key needed)
  chat.py              # terminal chat with the agent
tests/                 # unit tests (engine, memory, API, eval, obs) + opt-in live
app.py                 # Streamlit UI (chat + charts + tool-call trace)
Dockerfile             # multi-stage image serving the API
docker-compose.yml     # local orchestration (api [+ optional ui])
render.yaml            # one-click Render (Docker) deploy blueprint
pyproject.toml         # ruff / mypy / pytest config
.github/workflows/ci.yml  # CI: ruff + mypy + pytest (no secrets needed)
```

## Roadmap

- [x] Swappable multi-model config (Gemini / Groq / DeepSeek / Mistral)
- [x] Deterministic engine: market-data, projection, risk/allocation,
      Monte-Carlo, feasibility solvers (corrupt-tick handling)
- [x] LangChain agent (`create_agent`, 15 tools) + conversation & SQLite memory
- [x] Streamlit UI with allocation, projection + Monte-Carlo charts, tool trace
- [x] Runs on LangGraph under the hood (the agent graph is already inspectable)
- [x] Tax-aware: post-tax corpus via a deterministic LTCG estimate (12.5% over ₹1.25L)
- [x] Multi-goal planning: plan several goals at once with combined totals
- [x] News-sentiment "market mood" tool (yfinance headlines + a finance lexicon)
- [x] Async **FastAPI** service (OpenAPI docs; deterministic `/plan` needs no key)
- [x] **Docker** image + compose; **GitHub Actions** CI (ruff + mypy + pytest)
- [x] **Observability**: per-turn token/latency/tool metrics on `/chat`
- [x] **Agent evaluation** harness (golden set, 3 checks + optional LLM-as-judge)
- [x] One-click **deploy** blueprint (`render.yaml`) + Streamlit Cloud steps

## Disclaimer

Educational project. Outputs are **not** financial advice.
