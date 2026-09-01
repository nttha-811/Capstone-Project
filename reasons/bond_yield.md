# 2-Year Government of Canada Bond Yield Predictor Rationale

## Series Information
- **Series Code**: `BD.CDN.2YR.DQ.YLD`
- **Description**: Benchmark bond yield: 2-year Government of Canada marketable bonds.
- **Source**: Bank of Canada Selected Bond Yields (`https://www.bankofcanada.ca/rates/interest-rates/canadian-bonds/`).
- **Frequency**: Daily resampled to weekly (Friday close) first-differences ($\Delta y_t$).

## Rationale for Inclusion

1. **Market Inflation Expectations (Fisher Equation)**:  
   Under the Fisher relationship, nominal bond yields reflect expected inflation plus the real interest rate:
   $$\text{Nominal Yield} \approx \text{Real Rate} + \mathbb{E}[\pi]$$
   Shifts in medium-term inflation expectations are reflected in 2-year sovereign yields earlier than official monthly CPI releases.

2. **Maturity Selection**:
   - **3-Month T-Bills**: Strongly pegged to Bank of Canada overnight policy decisions, providing less independent forward-looking information.
   - **10-Year and Long-Term Bonds**: Heavily influenced by global term premia, sovereign risk flows, and global macro conditions rather than domestic near-term CPI dynamics.
   - **2-Year Benchmark**: Balances responsiveness to monetary policy expectations and near-term inflation projections over a 12-to-24 month horizon.

3. **High-Frequency Leading Indicator**:  
   CPI data is released with a lag (typically 2–3 weeks after month-end). Daily and weekly bond market movements provide intra-month information for nowcasting before the official release.