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

for C in [1, 10, 100, 1000, 10000]:

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
    a_nation = voter_shares_nation * C

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
    summary_nation.to_csv(f"/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/results/sensitivity/summary_c={C}.csv")

    party_colors = ['black', 'red', 'green', 'yellow', 'pink', 'blue', 'purple', 'grey']
    party_labels = ['CDU', 'SPD', 'Greens', 'FDP', 'Left', 'AfD', 'BSW', 'Others']

    # Plot trace for national posterior
    az.plot_trace(trace, var_names=["voter_share_nation"])
    plt.title("Election Forecast 2025")
    plt.savefig(f"/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/results/sensitivity/posterior_nation_c={C}.png")
