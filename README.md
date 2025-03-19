# Election Forecast for Bundestagswahl 2025 using Bayesian Statistics

## Data Sources
- Historical Data about election results (party shares, turnout rates)
- Polling Data: Voter preferences on the state and national level (include sample size, polling agency, date)
- Demographic Data by State and National: Population Size, Age, income, education,..
- (Party Specific Data)

## Goals
- Predict voter share for the main parties in the "Bundestagswahl 2025" on national and also state level
- The main issue is that I dont have much poll data on the state-level, for some states not even a single one (few data points should suit the bayesian approach well though, as the uncertainty is contained in the posterior distribution)

## Methodology

### Priors
- Historical Election data (and maybe some other variables like GDP?)
- Dirichlet-Priors (or are there better options?) for each state and the whole nation. Could be weighted average of past election with much higher weight on the last election.
- Dirichlet Priors are actually conjugate priors for Multinomial Likelihood 😃

### Likelihood
- Data from Polls
- Distribution: The first distribution that I would think of is the Multinomial


### Useful links

#### Forecasting Model US Election from Economist
https://hdsr.mitpress.mit.edu/pub/nw1dzd02/release/2

https://github.com/TheEconomist/us-potus-model/tree/master

#### Polls in the states
https://www.wahlrecht.de/umfragen/laender.htm

#### Dynamic Bayesian Model for Germany
https://www.marcel-neunhoeffer.com/pdf/papers/pa_forecast-multiparty.pdf

https://github.com/zweitstimme-org/prediction-2025

#### Economist Bundestagswahl Forecast 2021
https://github.com/TheEconomist/2021-germany-election-model-PUBLIC/tree/master

https://archive.ph/3T3aY






