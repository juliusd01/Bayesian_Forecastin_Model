import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets

# set random seed
np.random.seed(42)

# Number of states and parties
NUM_STATES = 16
NUM_PARTIES = 8
STATES = ["BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV", "NI", "NW", "RP", "SL", "SN", "ST", "SH", "TH"]

def get_poll_data(cutoff_date: str, bundestag: bool):
    # Include Polls from 7th of November -> Time when the coalition did break apart
    poll_df = pd.read_csv("/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/Resources_all/Polls/output.csv")

    # Filter for the 7th of November and relevant columns
    poll_df = poll_df[poll_df['survey_date'] >= cutoff_date]
    relevant_cols = ["survey_date", "survey_persons", "parliament_id", "institute_id", "result_cdu-csu", "result_spd", "result_gruene",
                    "result_fdp", "result_linke", "result_afd", "result_bsw", "result_sonstige", "result_cdu", "result_csu"]
    poll_df = poll_df[relevant_cols]

    # Add results of cdu and csu in cdu-csu
    poll_df['result_cdu-csu'] = poll_df['result_cdu-csu'].fillna(0) + poll_df['result_cdu'].fillna(0) + poll_df['result_csu'].fillna(0)
    poll_df = poll_df.drop(columns=['result_cdu', 'result_csu'])

    # Only consider polls for bundestag
    if bundestag:
        poll_df = poll_df[poll_df['parliament_id'] == "bundestag"]
    else:
        poll_df = poll_df[poll_df['parliament_id'] != "bundestag"]
    # Drop prefix result_ from column names
    poll_df.columns = poll_df.columns.str.replace('result_', '')

    return poll_df


def assign_weights_to_polls(national_polls: pd.DataFrame):
    # Define decay rate
    half_life = 30  # Poll looses half of its weight every 30 days
    decay_rate = np.log(2) / half_life

    # Convert survey_date to datetime format
    national_polls['survey_date'] = pd.to_datetime(national_polls['survey_date'])

    # Compute weights based on poll dates
    poll_dates = np.array(national_polls['survey_date'], dtype="datetime64[D]")
    earliest_poll_date = np.max(poll_dates)  
    time_diffs = (earliest_poll_date - poll_dates).astype("timedelta64[D]").astype(int)  # Days since each poll
    weights = np.exp(-decay_rate * time_diffs)  # Exponential decay

    return weights



# Voter share for the whole nation 2021 (8 Parties)
historical_shares_nation = np.array([0.241, 0.257, 0.148, 0.115, 0.049, 0.103, 0.000, 0.087])

baseline_polls = get_poll_data("2024-08-23", True)
baseline_polls = baseline_polls[baseline_polls['survey_date'] < "2024-11-07"]
baseline_polls = baseline_polls.drop(columns=['survey_date', 'survey_persons', 'parliament_id', 'institute_id'])
# avergae over all polls
baseline_polls = baseline_polls.mean() / 100
# round and convert to numpy array
baseline_polls = baseline_polls.round(3)
baseline_polls = baseline_polls.to_numpy()

# Average last election result and baseline polls
voter_shares_nation = (historical_shares_nation + baseline_polls)

# Set alpha parameter for dirichlet distribution of prior
a_nation = voter_shares_nation * 100

nation_polls = get_poll_data("2024-11-07", True)

for var in ["cdu-csu", "spd", "gruene", "fdp", "linke", "afd", "bsw", "sonstige"]:
    nation_polls[var] = round(nation_polls["survey_persons"] * nation_polls[var] / 100)

# Get weights for each poll
weights = assign_weights_to_polls(nation_polls)
# Drop unnecessary columns
nation_polls = nation_polls.drop(columns=["survey_persons", "institute_id", "survey_date", "parliament_id"])

# Data processing and then store it in lists
nation_polls = nation_polls.fillna(0)
nation_polls.reset_index(drop=True, inplace=True)
survey_results_nation = nation_polls.values.tolist()
sample_size_nation = nation_polls.sum(axis=1).to_list()

# Weight the national polls with exponential decay
SCALER = 0.01
survey_results_nation_adj = []
for i in range(len(survey_results_nation)):
    adjusted_shares = [round(weights[i] * voter_share * SCALER) for voter_share in survey_results_nation[i]]
    survey_results_nation_adj.append(adjusted_shares)

sample_size_nation_adj = [sum(adjusted_shares) for adjusted_shares in survey_results_nation_adj]

with pm.Model() as nation_model:
    # Quantify uncertainty in historical results and early polls for the prior by gamma distribution
    alpha_nation = pm.Gamma("alpha_nation", alpha=a_nation, beta=2, shape=8)    
    # Dirichlet-Prior
    nation_prior = pm.Dirichlet("voter_share_nation", a=alpha_nation, shape=(NUM_PARTIES,))

    # National level likelihood based on poll results
    # NOTE: For observed I use a list comprehension to multiply each party's voter share by the corresponding weight
    for i in range(len(survey_results_nation)):
        nation_likelihood = pm.Multinomial(
            f"nation_poll_{i}", 
            n=sample_size_nation_adj[i], 
            p=nation_prior, 
            observed=survey_results_nation_adj[i]
        )

with nation_model:
    trace = pm.sample(
        draws=10000,
        chains=4,
        random_seed=42,
        return_inferencedata=True
        )

summary_nation = az.summary(trace, var_names=["voter_share_nation"], hdi_prob=0.90)
summary_nation.to_csv("/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/results/Final/summary.csv")
mean_forecasted_voter_shares_nation = summary_nation['mean']
party_trends = mean_forecasted_voter_shares_nation - historical_shares_nation
party_trends = pd.Series(party_trends)
# Convert to NumPy array
party_trends = party_trends.to_numpy()

#### STATE MODEL
# Voter share for each German State 2021 (16 States x 8 Parties), Source: Bundeswahlleiterin
# The order of parties: CDU, SPD, GRÜNE, FDP, LINKE, AfD, BSW, Sonstige
historical_shares_states = np.array([
    [0.248, 0.216, 0.172, 0.153, 0.033, 0.096, 0.000, 0.082],
    [0.317, 0.180, 0.141, 0.105, 0.028, 0.090, 0.000, 0.139],
    [0.159, 0.234, 0.224, 0.091, 0.114, 0.084, 0.000, 0.094],
    [0.153, 0.295, 0.090, 0.093, 0.085, 0.181, 0.000, 0.103],
    [0.172, 0.315, 0.209, 0.093, 0.077, 0.069, 0.000, 0.065],
    [0.154, 0.297, 0.249, 0.114, 0.067, 0.050, 0.000, 0.068],
    [0.228, 0.276, 0.158, 0.128, 0.043, 0.088, 0.000, 0.079],
    [0.174, 0.291, 0.078, 0.082, 0.111, 0.180, 0.000, 0.084],
    [0.242, 0.331, 0.161, 0.105, 0.033, 0.074, 0.000, 0.054],
    [0.260, 0.291, 0.161, 0.114, 0.037, 0.073, 0.000, 0.065],
    [0.247, 0.294, 0.126, 0.117, 0.033, 0.092, 0.000, 0.092],
    [0.236, 0.373, 0.000, 0.115, 0.072, 0.100, 0.000, 0.105],
    [0.172, 0.193, 0.086, 0.110, 0.093, 0.246, 0.000, 0.099],
    [0.210, 0.254, 0.065, 0.095, 0.096, 0.196, 0.000, 0.084],
    [0.220, 0.280, 0.183, 0.125, 0.036, 0.068, 0.000, 0.087],
    [0.169, 0.234, 0.066, 0.090, 0.114, 0.240, 0.000, 0.087]
])

state_mapping = {
    "baden-wuerttemberg": "BW",
    "bayern": "BY",
    "berlin": "BE",
    "brandenburg": "BB",
    "bremen": "HB",
    "hamburg": "HH",
    "hessen": "HE",
    "mecklenburg-vorpommern": "MV",
    "niedersachsen": "NI",
    "nordrhein-westfalen-nrw": "NW",
    "rheinland-pfalz": "RP",
    "saarland": "SL",
    "sachsen": "SN",
    "sachsen-anhalt": "ST",
    "schleswig-holstein": "SH",
    "thueringen": "TH"
}

# Include longer timeframe as there might be no/less polls for states
state_polls = get_poll_data("2024-11-07", False)
# create dict with states as keys and the corresponding polls as values
state_polls_dict = {}
state_polls = state_polls.fillna(0)
polls_grouped_by_state = state_polls.groupby('parliament_id')
for state, polls in polls_grouped_by_state:
    state = state_mapping[state]
    for var in ["cdu-csu", "spd", "gruene", "fdp", "linke", "afd", "bsw", "sonstige"]:
        polls[var] = round(polls["survey_persons"] * polls[var] / 100)
    # sum up the polls for each state
    polls = polls.drop(columns=["survey_persons", "institute_id", "survey_date", "parliament_id"])
    polls = polls.fillna(0)
    polls.reset_index(drop=True, inplace=True)
    poll_results_state = polls.values.tolist()
    sample_size_state = polls.sum(axis=1).to_list()
    state_polls_dict[state] = {
        'poll_results': poll_results_state,
        'sample_size': sample_size_state
    }

# Insert values from last available dawum polls
# Bremen: Use results from Bremische Bürgerschaftswahl 2023
# Saarland: https://dawum.de/Saarland/
state_polls_dict["HB"] = {
    'poll_results': [[179, 203, 91, 39, 77, 0, 0, 112]],
    'sample_size': [701]
}
state_polls_dict["SL"] = {
    'poll_results': [[310, 290, 50, 40, 30, 145, 100, 115]],
    'sample_size': [1080]
}

# create numpy array with poll data from states, but the keys should be ordered as in the states array
survey_results_states = np.array([state_polls_dict[state] for state in STATES])

with pm.Model() as state_model:  
    # Hyperprior for trend variability
    sigma_trend = pm.HalfNormal("sigma_trend", 0.05)
    epsilon = pm.Normal("epsilon", mu=0, sigma=1, shape=8)
    
    # National trends: modeled as uncertain around your calculated party_trends
    national_trend = party_trends + sigma_trend * epsilon
    # NOTE: The posterior diverged when running the following code, so i reparameterized with the above line
    # pm.Normal(
    #     "national_trend",
    #     mu=party_trends,  # Your precomputed mean trends
    #     sigma=sigma_trend,  # Estimated variability
    #     shape=8  # One trend per party
    # )  
    # Adjust historical shares with uncertain trends
    trend_adj_hss = historical_shares_states + national_trend
    trend_adj_hss = pm.math.clip(trend_adj_hss, 1e-3, 1)  # Ensure valid probabilities
    
    # Dirichlet prior (pseudo-counts scaled by 100)
    alpha_states = trend_adj_hss * 100

    # Dirichlet-Priors
    state_priors = {}
    for index, state in enumerate(STATES):
        state_prior = pm.Dirichlet(f"voter_share_{state}", a=alpha_states[index])
        state_priors[state] = state_prior
    # State Level likelihood based on poll results -> Iterate through each state and each poll in that state
    for state_index, state in enumerate(STATES):
        state_surveys = survey_results_states[state_index]
        for poll_index in range(len(state_surveys['poll_results'])):
            state_likelihood = pm.Multinomial(
                f"state_likelihood_{state}_{poll_index}",
                n=state_surveys['sample_size'][poll_index], 
                p=state_priors[state], 
                observed=state_surveys['poll_results'][poll_index]
            )

with state_model:
    trace = pm.sample(
        draws=10000,
        chains=4,
        random_seed=42,
        target_accept=0.9,
        return_inferencedata=True
        )
    
summary = az.summary(trace, hdi_prob=0.9)
summary.to_csv("/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/results/Final/summary_states.csv")

party_colors = ['black', 'red', 'green', 'yellow', 'pink', 'blue', 'purple', 'grey']
party_labels = ['CDU', 'SPD', 'Grüne', 'FDP', 'Linke', 'AfD', 'BSW', 'Others']

for state in STATES:
    az.plot_trace(trace, var_names=[f"voter_share_{state}"])
    plt.title(f"State {state}")
    plt.savefig(f"/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/results/Final/state_images/{state}")