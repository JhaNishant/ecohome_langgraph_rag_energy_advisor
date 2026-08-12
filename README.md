# EcoHome LangGraph RAG Energy Advisor

EcoHome is a Berlin focused smart home energy advisor. It combines live weather data, household energy history, solar production, time of use pricing, and practical energy tips to help people choose better times for EV charging, heating, and flexible appliances.

## What it does

The advisor can:

* Check live weather and solar radiation through Open Meteo.
* Compare Berlin time of use electricity prices in EUR per kWh.
* Review thirty days of EV, HVAC, appliance, and solar history.
* Search a small energy saving knowledge base and cite the source file it used.
* Estimate savings from reducing consumption or moving a task to a cheaper hour.
* Build a personalized tomorrow plan from EV deadlines, comfort, quiet hour, battery, and solar preferences.
* Estimate avoided grid energy and CO₂e using a clear, editable assumption.

The system gives recommendations only. It does not control household devices.

## Project layout

```text
repository root/
├── models/                 SQLite models, preferences, and demo data helper
├── data/documents/         Seven energy saving articles
├── data/energy_data.db     Generated sample database
├── data/user_preferences.json  Editable sample household profile
├── data/reports/           Saved energy insight charts
├── data/vectorstore/       Generated Chroma knowledge base
├── tests/                  Focused checks for tools and data
├── agent.py                LangGraph agent
├── tools.py                Weather, pricing, data, RAG, and savings tools
├── 01_db_setup.ipynb       Database setup and checks
├── 02_rag_setup.ipynb      Knowledge base setup and checks
├── 03_run_and_evaluate.ipynb  Agent tests and evaluation
└── 04_energy_insights.ipynb   Personalized plan and visual reporting
```

## Setup

This project is built and tested with Python 3.14.

```bash
/opt/homebrew/bin/python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Verified local environment: Python 3.14.6, `langgraph==1.2.11`, `langchain-openai==1.4.3`, `langchain-chroma==1.1.0`, `chromadb==1.5.9`, `openai==2.54.0`, `pandas==2.3.3`, `matplotlib==3.11.1`, `SQLAlchemy==2.0.52`, and `requests==2.34.2`. The compatible version ranges are listed in `requirements.txt`.

Add your Vocareum key to `.env` before running the RAG and agent notebooks. The default advisor model is `gpt-5.6-luna`, with `medium` reasoning effort. The knowledge base uses `text-embedding-3-large`. The weather tool uses Open Meteo and does not need a weather API key.

Run the notebooks in order:

1. `01_db_setup.ipynb`
2. `02_rag_setup.ipynb`
3. `03_run_and_evaluate.ipynb`
4. `04_energy_insights.ipynb`

## How the advisor decides

The LangGraph workflow starts with the advisor, calls a tool whenever data is needed, then returns to the advisor to turn that data into a clear recommendation. For a Berlin EV charging question, it checks the forecast and price periods, then recommends hours that balance low prices and useful solar radiation.

Weather data comes from [Open Meteo](https://open-meteo.com/en/docs). If the service is temporarily unavailable, the tool returns clearly marked local fallback data so the rest of the project remains demonstrable.

## Personalization and insight reporting

Edit `data/user_preferences.json` to set an EV departure time and charge target, a comfort band, quiet hours, a battery reserve, and energy priorities. The plan always respects the departure deadline: it uses solar before that deadline when possible, otherwise the cheapest earlier grid period.

`04_energy_insights.ipynb` saves four charts under `data/reports/`. Each chart explains how to read it and what decision it supports. The knowledge search also combines semantic similarity with direct keyword matching before it ranks energy tips. Carbon figures are labelled estimates and use a configurable 350 g CO₂e per kWh grid factor; they are not a live emissions feed.

## Running checks

```bash
.venv/bin/python -m pytest tests -q
```

The final notebook records each scenario, its tool calls, response scores, tool use scores, and a plain language evaluation report.

## Limits

The electricity prices are a reproducible Berlin focused model, not a live tariff. Solar recommendations use weather radiation as an indicator and do not model the exact size, tilt, shading, or battery configuration of a real home.
