import requests
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
from itertools import permutations

pth = 'data/esg_greenwashing_energy_utilities_industrials_2010_2024.csv'
OLLAMA_URL = "http://localhost:11434"
MODEL_NAMES = ['gemma4:e4b']#['gemma4:31b', 'qwen3:30b', 'gemma4:e4b', 'gemma4:e2b'] # TODO

df_reports = pd.read_csv(pth)
def normalize(df_session):
    cols = ['revenue_usd_bn', 'scope1_emissions_mt_co2e', 
                'scope2_emissions_mt_co2e', 'scope3_emissions_mt_co2e', 
                'esg_score_0_100']
    
    df_norm = df_session[cols].copy()
    for col in cols:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.5
    return  pd.concat([df_norm, df_session.loc[:,~df_session.columns.isin(cols)]],axis=1)


df_reports_norm_all = normalize(df_reports)
df_reports_norm_all.head()
def norm_by_year(df_table):
    df_reports_lst = []
    
    for year in set(df_table.year):
        norm_df = normalize(df_table.query(f"year=={year}"))
        
        df_reports_lst.append(norm_df)
        #.groupby(['year', 'ticker']).apply(normalize)
    return pd.concat(df_reports_lst)

df_reports_norm = norm_by_year(df_reports)
SYSTEM_INSTRUCTION_INDEX = """
You are a quantitative portfolio analyst.
Your task: select exactly 6 indices (0-based) that best match the investor profile.
Base your selection SOLELY on the numerical data provided below.

Return ONLY a JSON array of 6 indices.
No explanation, no markdown, no ```json fences — raw JSON only.
Format: [0, 3, 7, 12, 18, 25]
"""

PROFILES_INDEX = {
    "Financially motivated / conventional investors": """
You are constructing a portfolio for a conventional institutional investor.
The investor's primary objective is long-term financial performance.
Environmental characteristics may be considered only when they do not reduce expected financial performance.
Using the numerical company data below, select exactly six companies that best match this investment philosophy.

Return exactly 6 indices.
""",

    "ESG-Committed / Norm-driven investors": """
You are constructing a portfolio for an institutional investor committed to responsible investing principles.
The investor prioritizes environmental sustainability and high ESG performance over financial considerations. 
Companies with stronger sustainability characteristics should be preferred even if they are not the largest by revenue.
Using the numerical company data below, select exactly six companies that best match this investment philosophy.

Return exactly 6 indices.
""",

    "Long-term / Universal owner investors": """
You are constructing a portfolio for a universal owner pension fund.
The investor seeks long-term resilience by reducing exposure to systemic climate risk while maintaining diversified financial performance.
Using the numerical company data below, select exactly six companies that best match this investment philosophy.

Return exactly 6 indices.
"""
}

def build_prompt_index(profile_name: str, data_str: str) -> str:
    """Build prompt with explicit decision rules."""
    persona = PROFILES_INDEX[profile_name]
    return f"""{SYSTEM_INSTRUCTION_INDEX}

{persona}

COMPANY DATA (each row has an index, 0–29):
{data_str}

Now ANALYZE the data and SELECT exactly 6 indices.
Return ONLY a JSON array: [0, 3, 7, 12, 18, 25]
"""

def model2_request(profile, info, companies, model_name):
    payload = {
            "model": model_name,
            "prompt": build_prompt_index(profile, info),
            "stream": False,
            # "options": {
            #     "temperature": temperature,
            #     "num_predict": max_tokens,
                # "reasoning": False,
                # "think": False,
            # },
        }
    out = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
            ).json()["response"].strip()

    return [companies[item] for item in json.loads(out)] 

test_years = [2020]#, 2021, 2022, 2023, 2024]
train_years = [2010]#list(range(2010, 2020))
all_years = [2010]# list(range(2010, 2025)) # TODO
print([name['name'] for name in requests.get(
            f"{OLLAMA_URL}/api/tags",
        ).json()['models']])
llm_solutions_lst_train = []
for j in range(1): # TODO
    model_names = MODEL_NAMES
    llm_solutions = [('profile', 'llm_name' , 'year', 'asset1', 'asset2', 'asset3', 'asset4', 'asset5', 'asset6')]
    
    for model_name in model_names:
        llm_solution = {}
        print(model_name)
        for profile in list(PROFILES_INDEX.keys()):
            print(profile)
            it = len(all_years)
            i = 0
            years = all_years
    
            while i < len(years):
                try:
                    year = years[i]
                    df_reports_norm_year = df_reports_norm.query(f"year=={year}")
                    company_lst = df_reports_norm_year['company'].tolist()
                    dta = str(df_reports_norm_year[[
                            'revenue_usd_bn','scope1_emissions_mt_co2e','scope2_emissions_mt_co2e', 'scope3_emissions_mt_co2e', 'esg_score_0_100','carbon_intensity_tco2e_per_musd'
                        ]].reset_index(drop=True).reset_index(drop=True))
                    model_selection = model2_request(profile, dta, company_lst, model_name)
                    temp_res = [profile, model_name, year] + model_selection
                    llm_solutions.append(tuple(temp_res))
                    print(f'Round {it}')
                    it -= 1
                    i += 1
                except:
                    print(f'exception {i}')
    
        
        columns=pd.DataFrame(llm_solutions).iloc[0].tolist()
        df_llm_solutions = pd.DataFrame(llm_solutions[1:],columns=llm_solutions[0])
        
    
        def select_most_freq(table):
            asset1_dct,	asset2_dct,	asset3_dct,	asset4_dct,	asset5_dct,	asset6_dct = [defaultdict(int) for _ in range(6)]
            for assets in table[['asset1','asset2','asset3', 'asset4','asset5','asset6']].iterrows():
                asset1,	asset2,	asset3,	asset4,	asset5,	asset6 = assets[1]
                asset1_dct[asset1] += 1
                asset2_dct[asset2] += 1
                asset3_dct[asset3] += 1	
                asset4_dct[asset4] += 1
                asset5_dct[asset5] += 1
                asset6_dct[asset6] += 1
            a1 = sorted(asset1_dct.items(), key=lambda x: x[1], reverse=True)[0][0]
            a2 = sorted(asset2_dct.items(), key=lambda x: x[1], reverse=True)[0][0]
            a3 = sorted(asset3_dct.items(), key=lambda x: x[1], reverse=True)[0][0]
            a4 = sorted(asset4_dct.items(), key=lambda x: x[1], reverse=True)[0][0]
            a5 = sorted(asset5_dct.items(), key=lambda x: x[1], reverse=True)[0][0]
            a6 = sorted(asset6_dct.items(), key=lambda x: x[1], reverse=True)[0][0]
            return pd.DataFrame([a1,a2,a3,a4,a5,a6],columns=['portfolio'])
        df_profile_portfolio = df_llm_solutions.groupby(['profile', 'llm_name']).apply(select_most_freq).reset_index()
        df_profile_portfolio['llm_name'] = model_name
        df_profile_portfolio['round'] = j
        df_profile_portfolio = df_profile_portfolio[[col for col in df_profile_portfolio.columns if 'level' not in col]]
    print('Iteration:', j)
    llm_solutions_lst_train.append(df_profile_portfolio)
    

pd.concat(llm_solutions_lst_train).to_excel('output/profile_portflio_gemma_qwen_prompt2_.xlsx')
llm_runs = pd.read_excel('output/profile_portflio_gemma_qwen_prompt2_.xlsx') # TODO
# llm_run = pd.read_excel("output/profile_portflio_gemma4b_prompt2_test.xlsx")
print(llm_runs.head())

df_cem_runs = pd.read_excel("output/cem_runs_profiles_23072026.xlsx")
df_cem_runs['profile_llm'] = df_cem_runs['profile'] + '_cem'
companies = sorted(list(set(df_cem_runs.company)))
companies_ord = {companies[i]:i for i in range(len(companies))}

def collect_assets(table):
    cols = [c for c in table.columns if 'asset' in c]
    freq = defaultdict(int)
    rows = table.shape[0]
    portfolios = [0 for _ in range(len(companies_ord))]
    table.reset_index(drop=True, inplace=True)
    company = [0 for _ in range(len(companies_ord))]
    for i, row in table[cols].iterrows():
        for asset in row:
            portfolios[companies_ord[asset]] += 1 
            company[companies_ord[asset]] = asset
    portfolios_norm = {com: p/(sum(portfolios) + 1e-5) for com, p in zip(company, portfolios)}
    portfolios_norm_sorted = dict(sorted(portfolios_norm.items(), key=lambda x: -x[1]))

    return pd.DataFrame(portfolios_norm_sorted.items(),columns=['company','score']).head(6)
    
llm_runs['ceed'] = llm_runs['round']
llm_runs_by_models = llm_runs.groupby(["profile","llm_name", "ceed"]).apply(collect_assets).reset_index()
llm_runs_by_models['profile_llm'] = llm_runs_by_models['profile'] + '_' + llm_runs_by_models['llm_name'] 



def weighted_jaccard(w1, w2):
    union = np.maximum(w1, w2).sum()
    return np.minimum(w1, w2).sum() / union if union > 0 else 0.0

def pivot_profile(df, profile_name, seed_col='сeed'):
    sub = df[df['profile_llm'] == profile_name]
    return sub.pivot_table(index=seed_col, columns='company', values='score', fill_value=0)

def compare_profiles_group_test_llm(df, profile_A, profile_B, n_permutations=10000, seed=None, seed_col='ceed'):
    rng = np.random.default_rng(seed)

    A = pivot_profile(df, profile_A, seed_col=seed_col)
    B = pivot_profile(df, profile_B, seed_col=seed_col)
    
    common_cols = A.columns.union(B.columns)
    A = A.reindex(columns=common_cols, fill_value=0).values
    B = B.reindex(columns=common_cols, fill_value=0).values
    
    nA, nB = len(A), len(B)
    pooled = np.vstack([A, B])
    n_total = nA + nB

    def between_group_mean_jaccard(group_idx):
        # group_idx: булев массив длины n_total, True = принадлежит "группе A" в данной перестановке
        grpA = pooled[group_idx]
        grpB = pooled[~group_idx]
        
        vals = [weighted_jaccard(a, b) for a in grpA for b in grpB]
        
        return np.mean(vals)

    labels_real = np.zeros(n_total, dtype=bool)
    
    
    labels_real[:nA] = True
    
    observed = between_group_mean_jaccard(labels_real)
    
    null_vals = np.empty(n_permutations)
    # print(null_vals)
    for i in range(n_permutations):
        perm_labels = np.zeros(n_total, dtype=bool)
        perm_labels[rng.choice(n_total, size=nA, replace=False)] = True
        null_vals[i] = between_group_mean_jaccard(perm_labels)

    # низкий Jaccard = профили сильнее различаются, поэтому значимость — это доля null <= observed
    p_value = max(1, np.sum(null_vals <= observed)) / (n_permutations + 1)

    return {
        "observed_between_jaccard": observed,
        "null_distribution": null_vals,
        "p_value": p_value,
    }


print(set(pd.concat([llm_runs_by_models, df_cem_runs])['profile_llm'].tolist()))
frame_llm = []
for direct in ['fin','esg','long']:
    profiles_2_compare = [prof for prof in set(pd.concat([llm_runs_by_models, df_cem_runs])['profile_llm'].tolist()) if direct in prof.lower()]
    

    frame = []
    pair_collection = set()
    for prof_pairs in permutations(profiles_2_compare,r=2):
        prof_pairs_lst = list(prof_pairs)
    
        if tuple(sorted(prof_pairs_lst)) not in pair_collection:
            result = compare_profiles_group_test_llm(
                    pd.concat([llm_runs_by_models, df_cem_runs]),
                    prof_pairs[0],
                    prof_pairs[1],
                    n_permutations=10000,
                    seed=42
                )
            print(f"{prof_pairs[0]} vs {prof_pairs[1]} \n", 'Jaccard:', result["observed_between_jaccard"], 'P-value:', result["p_value"])
            frame.append(pd.DataFrame([(prof_pairs[0], prof_pairs[1], result["observed_between_jaccard"], result["p_value"])], columns=['profile1','profile2','jaccard','P-value']))
            pair_collection.add(tuple(sorted(prof_pairs_lst)))
            frame_llm.append(pd.DataFrame([(prof_pairs[0], prof_pairs[1], result["observed_between_jaccard"], result["p_value"])], columns=['profile1','profile2','jaccard','P-value']))

    print('==================================')
    
    
        
    