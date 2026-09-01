# Commodity Price Index (BCPI) Predictor Rationale

## Series Metadata
- **Source**: Bank of Canada Weekly Commodity Price Index (BCPI)
- **Sub-components**:
  - `W.BCPI`: Total Commodity Price Index
  - `W.ENER`: Energy Sub-Index (coal, oil, natural gas)
  - `W.BCNE`: Non-Energy Sub-Index (metals and minerals, forestry, agriculture, fish)
  - `W.MTLS`: Metals and Minerals
  - `W.FOPR`: Forestry
  - `W.AGRI`: Agriculture
  - `W.FISH`: Fish

## Selection Rationale

Using only the aggregate index (`W.BCPI`) has limitations: energy commodities constitute a dominant portion of the total index weight in Canada, which can obscure distinct signals from non-energy commodities (agricultural produce, metals, forestry).

Including all individual sub-indices simultaneously increases parameter count significantly: in a MIDAS regression with exponential Almon lag weights, each high-frequency predictor introduces 3 additional parameters ($\beta_m, \theta_{1,m}, \theta_{2,m}$).

To balance parsimony and signal separation, we decompose the commodity index into two aggregate components:
1. **`W.ENER` (Energy)**: Captures fuel, heating, and transportation cost shocks.
2. **`W.BCNE` (Non-Energy)**: Captures input material costs, agricultural price shocks, and broad industrial demand without energy noise.

## Model Formulation

$$y_t = \alpha + \beta_1 \left( \sum_{k=0}^{3} w(k; \theta_{1,1}, \theta_{1,2}) X_{\text{Energy}, t-k} \right) + \beta_2 \left( \sum_{k=0}^{3} w(k; \theta_{2,1}, \theta_{2,2}) X_{\text{Non-Energy}, t-k} \right) + \epsilon_t$$

where $w(k; \theta_1, \theta_2) = \frac{\exp(\theta_1 k + \theta_2 k^2)}{\sum_{j=0}^{K-1} \exp(\theta_1 j + \theta_2 j^2)}$ is the normalized two-parameter exponential Almon lag weight function.