├── core/                   # The "World" (remains unchanged)
│   ├── market_physics.py   
│   ├── sandbox_env.py      
│   ├── volatility_scheduler.py
│   └── world_clock.py
├── agents/                 # The OODA MAS Components
│   ├── analyst.py          # Observation/Anomaly Detection
│   ├── strategist.py       # Policy selection & Shadow Pricing (e^t/24)
│   ├── executor.py         # API interface to Sandbox
│   └── human_supervisor.py # The 3-7 day "Strategic Guidance" agent
├── baseline/               # The Industry Baseline Logic
│   ├── rule_engine.py      # Proportional rules (Bid +/- 10%)
│   └── legacy_human.py     # 12-hour/24-hour intervention logic
├── logic/                  # Shared Mathematical Engines
│   ├── optimization.py     # Total Optimization Function (U - P)
│   └── pricing_models.py   # Lagrangian / Shadow Price formulas
├── data/                   # Simulation outputs
│   ├── ib_results.csv      # Industry Baseline results
│   └── mas_results.csv     # OODA MAS results
├── industry_baseline_sim.py        # Entry point to run Industry Baseline
└── ooda_sim.py        # Entry point to run Phase 2 OODA MAS