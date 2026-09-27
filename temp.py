import pandas as pd

df = pd.read_pickle("nenci2021_saptdft_pbe0_d4_totals.pkl")
# print(df[["system id"]].to_string())
df['num_id'] = df['system id'].apply(lambda x : int(x.split('_')[0]))
# print(df[["num_id"]].to_string())
df_ion_pi = df[df['num_id'] > 101]
# print(df_ion_pi[["system id"]].to_string())
df_anion_pi = df_ion_pi[(~df_ion_pi['system id'].str.contains('Na')) & (~df_ion_pi['system id'].str.contains('Li'))]
print(df_anion_pi[["system id"]].to_string())
df_anion_pi.to_pickle("anion_pi_nenci.pkl")
