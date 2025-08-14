# Financial Agents

![Home Page Preview](images/home.png)

A collection of AI-powered financial analysis agents inspired by legendary investors and trading strategies. This project provides a framework for conducting comprehensive financial analysis using specialized agents, each implementing the investment philosophy of a different renowned investor or trading approach.

## Installation and Running

### Prerequisites

- Python >= 3.10
- uv (for dependency management)

### Installation

```bash
# Clone the repository
git clone https://github.com/aostock/financial-agents
cd financial-agents

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Running the Application

```bash
# Start the development server
langgraph dev --allow-blocking --debug-port 5678
```

Alternatively, you can run the FastAPI server directly:

```bash
# Run the FastAPI server
python main.py
```

The server will start on port 2024 by default (configurable via PORT environment variable).

## Frontend UI

The frontend UI for this project is available at: [agent-chat-ui](https://github.com/aostock/agent-chat-ui)

## Data Sources

Financial data for this project is sourced from: [financial-data](https://github.com/aostock/financial-data)

## Agent Overview

This project includes multiple specialized financial analysis agents, each implementing the investment philosophy and methodology of a different renowned investor or trading approach.

### Available Agents

#### 1. Warren Buffett Agent

![Warren Buffett Agent Flow](images/agents/warren_buffett.png)

Implements Warren Buffett's value investing philosophy:

- Focus on businesses within his "circle of competence"
- Evaluate economic moats and competitive advantages
- Assess management quality and capital allocation skills
- Analyze financial strength and consistency
- Calculate intrinsic value and margin of safety

#### 2. Charlie Munger Agent

![Charlie Munger Agent Flow](images/agents/charlie_munger.png)

Applies Charlie Munger's multidisciplinary approach:

- Evaluate management quality and integrity
- Assess moat strength and durability
- Analyze business predictability
- Determine fair valuation

#### 3. Benjamin Graham Agent

![Benjamin Graham Agent Flow](images/agents/ben_graham.png)

Follows Benjamin Graham's value investing principles:

- Analyze earnings stability
- Evaluate financial strength
- Determine intrinsic valuation

#### 4. Peter Lynch Agent

![Peter Lynch Agent Flow](images/agents/peter_lynch.png)

Implements Peter Lynch's growth investing approach:

- Understand the business and its story
- Evaluate earnings quality
- Conduct fundamental analysis
- Analyze growth prospects
- Determine intrinsic value

#### 5. Phil Fisher Agent

![Phil Fisher Agent Flow](images/agents/phil_fisher.png)

Applies Phil Fisher's growth stock approach:

- Evaluate growth quality
- Analyze insider activity
- Determine intrinsic value
- Assess management efficiency
- Analyze margins stability
- Conduct sentiment analysis

#### 6. Aswath Damodaran Agent

![Aswath Damodaran Agent Flow](images/agents/aswath_damodaran.png)

Implements Aswath Damodaran's valuation framework:

- Conduct growth analysis
- Determine intrinsic value
- Perform relative valuation
- Analyze risk factors
- Develop company narrative

#### 7. Michael Burry Agent

![Michael Burry Agent Flow](images/agents/michael_burry.png)

Applies Michael Burry's contrarian investing approach:

- Identify market inefficiencies
- Conduct deep value analysis
- Analyze financial statements
- Assess contrarian opportunities
- Evaluate risk factors

#### 8. Stanley Druckenmiller Agent

![Stanley Druckenmiller Agent Flow](images/agents/stanley_druckenmiller.png)

Implements Stanley Druckenmiller's macro approach:

- Analyze global markets
- Conduct macro analysis
- Evaluate adaptive strategies
- Assess flexibility
- Analyze risk factors

#### 9. Bill Ackman Agent

![Bill Ackman Agent Flow](images/agents/bill_ackman.png)

Applies Bill Ackman's activist investing approach:

- Analyze balance sheets
- Evaluate business quality
- Assess activism potential
- Determine fair valuation

#### 10. Cathie Wood Agent

![Cathie Wood Agent Flow](images/agents/cathie_wood.png)

Implements Cathie Wood's disruptive innovation approach:

- Evaluate disruptive potential
- Analyze innovation growth
- Determine valuation

#### 11. Rakesh Jhunjhunwala Agent

![Rakesh Jhunjhunwala Agent Flow](images/agents/rakesh_jhunjhunwala.png)

Applies Rakesh Jhunjhunwala's value investing approach:

- Analyze balance sheets
- Evaluate cash flows
- Conduct fundamental analysis
- Assess growth prospects
- Determine intrinsic value
- Analyze management quality

#### 12. Fundamentals Agent

![Fundamentals Agent Flow](images/agents/fundamentals.png)

Comprehensive fundamental analysis:

- Evaluate business consistency
- Conduct fundamental analysis
- Analyze growth metrics
- Assess quality factors
- Determine valuation

#### 13. Information Query Agent

![Information Query Agent Flow](images/agents/information_query.png)

Provides detailed company information and key financial metrics analysis.

#### 14. Portfolio Manager Agent

![Portfolio Manager Agent Flow](images/agents/portfolio_manager.png)

Manages portfolio analysis and optimization.

#### 15. Risk Manager Agent

![Risk Manager Agent Flow](images/agents/risk_manager.png)

Analyzes and manages investment risks.

#### 16. Sentiment Agent

![Sentiment Agent Flow](images/agents/sentiment.png)

Analyzes market sentiment from multiple sources:

- News sentiment
- Social media sentiment
- Insider activity sentiment
- Technical analysis sentiment
- Composite sentiment analysis

#### 17. Technicals Agent

![Technicals Agent Flow](images/agents/technicals.png)

Applies technical analysis approaches:

- Trend analysis
- Momentum analysis
- Mean reversion analysis
- Volatility analysis
- Statistical arbitrage analysis

#### 18. Trading Agent

![Trading Agent Flow](images/agents/trading.png)

Implements trading strategies based on technical analysis.

#### 19. Valuation Agent

![Valuation Agent Flow](images/agents/valuation.png)

Specialized valuation methodologies:

- Discounted Cash Flow (DCF) analysis
- EV/EBITDA analysis
- Owner earnings analysis
- Residual income analysis

## Agent Architecture

Each agent follows a modular architecture with specialized analysis components. The agents are built using LangGraph and integrate with various financial data sources through MCP (Model Coordination Protocol) adapters.

## Inspiration and References

This project draws inspiration from:

- https://github.com/virattt/ai-hedge-fund
- https://github.com/TauricResearch/TradingAgents

These projects provided valuable insights into implementing AI-powered financial analysis systems and trading agent frameworks.
