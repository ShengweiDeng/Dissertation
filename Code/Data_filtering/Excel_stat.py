# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Shengwei Deng
@Software: PyCharm
@File:
 input: loc.csv, ts1.csv, ts2.csv, ts3.csv
 output: ts1_out, ts2_out, ts2_out, ts1_stat, ts2_stat, ts3_stat
"""

from tqdm import trange
import pandas as pd

# set env ---
df_loc = pd.read_csv(r"E:\ukdata1\uk_data\uk_data\Code\Code\Data_filtering\input\loc.csv")
df_ts1 = pd.read_csv(r"E:\ukdata1\uk_data\uk_data\Code\Code\Data_filtering\input\ts1.csv", encoding="ISO-8859-1")
df_ts2 = pd.read_csv(r"E:\ukdata1\uk_data\uk_data\Code\Code\Data_filtering\input\ts2.csv", encoding="ISO-8859-1")
df_ts3 = pd.read_csv(r"E:\ukdata1\uk_data\uk_data\Code\Code\Data_filtering\input\ts3.csv", encoding="ISO-8859-1")

# out df ---
# ts1
df_loc0_ind = df_ts1[df_ts1['location'].isin(df_loc['Location'].values)].index
df_loc1_ind = df_ts1[df_ts1['location1'].isin(df_loc['Location'].values)].index
df_loc2_ind = df_ts1[df_ts1['location2'].isin(df_loc['Location'].values)].index

df_ts1_out = df_ts1.loc[df_loc0_ind.union(df_loc1_ind).union(df_loc2_ind), :].loc[:, :'result']

df_ts1_out.loc[df_loc0_ind, 'site'] = df_ts1_out.loc[df_loc0_ind, 'location']
df_ts1_out.loc[df_loc1_ind, 'site'] = df_ts1_out.loc[df_loc1_ind, 'location1']
df_ts1_out.loc[df_loc2_ind, 'site'] = df_ts1_out.loc[df_loc2_ind, 'location2']

# ts2
df_loc0_ind = df_ts2[df_ts2['location'].isin(df_loc['Location'].values)].index
df_loc1_ind = df_ts2[df_ts2['location1'].isin(df_loc['Location'].values)].index
df_loc2_ind = df_ts2[df_ts2['location2'].isin(df_loc['Location'].values)].index

df_ts2_out = df_ts2.loc[df_loc0_ind.union(df_loc1_ind).union(df_loc2_ind), :]

df_ts2_out.loc[df_loc0_ind, 'site'] = df_ts2_out.loc[df_loc0_ind, 'location']
df_ts2_out.loc[df_loc1_ind, 'site'] = df_ts2_out.loc[df_loc1_ind, 'location1']
df_ts2_out.loc[df_loc2_ind, 'site'] = df_ts2_out.loc[df_loc2_ind, 'location2']

# ts3
df_loc0_ind = df_ts3[df_ts3['location'].isin(df_loc['Location'].values)].index
df_loc1_ind = df_ts3[df_ts3['location1'].isin(df_loc['Location'].values)].index
df_loc2_ind = df_ts3[df_ts3['location2'].isin(df_loc['Location'].values)].index

df_ts3_out = df_ts3.loc[df_loc0_ind.union(df_loc1_ind).union(df_loc2_ind), :]

df_ts3_out.loc[df_loc0_ind, 'site'] = df_ts3_out.loc[df_loc0_ind, 'location']
df_ts3_out.loc[df_loc1_ind, 'site'] = df_ts3_out.loc[df_loc1_ind, 'location1']
df_ts3_out.loc[df_loc2_ind, 'site'] = df_ts3_out.loc[df_loc2_ind, 'location2']

# export ---
df_ts1_out.to_csv(r'E:\ukdata1\uk_data\uk_data\Code\Code\Data_filtering\output\ts1_out.csv', index=False)
df_ts2_out.to_csv(r'E:\ukdata1\uk_data\uk_data\Code\Code\Data_filtering\output\ts2_out.csv', index=False)
df_ts3_out.to_csv(r'E:\ukdata1\uk_data\uk_data\Code\Code\Data_filtering\output\ts3_out.csv', index=False)

# stat ---
# df = pd.concat([df_ts1_out.copy(), df_ts2_out.copy(), df_ts3_out.copy()]).copy().reset_index(drop=True)

for dfi, df in enumerate([df_ts1_out, df_ts2_out, df_ts3_out]):

    df_stat = pd.DataFrame()

    for i in trange(df_loc.shape[0], desc=f'Loop locations ({dfi + 1}/3)'):
        ind = df_stat.shape[0]
        location = df_loc.loc[ind, 'Location']
        df_stat.loc[ind, 'Location'] = location
        df_stat.loc[ind, 'Count'] = df[(df['location'] == location) | (df['location'] == location) |
                                       (df['location'] == location)].shape[0]
        # df_stat.loc[ind, 'Time'] = 
        df_stat.loc[ind, 'Mean'] = df['result'][(df['location'] == location) | (df['location'] == location) |
                                                (df['location'] == location)].mean()

    df_stat.dropna(how='any').to_csv(f'E:/ukdata1/uk_data/uk_data/Code/Code/Data_filtering/output/ts{dfi + 1}_stat.csv', index=False)
