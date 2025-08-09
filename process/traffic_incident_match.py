#! -*- coding:utf-8 -*-
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm


def main(args):
    incidents = pd.read_csv(incident_path, sep='\t')
    meta_df = pd.read_csv('new_sensor_meta_feature.csv')

    incidents = incidents[~incidents['Fwy'].isna()]
    incidents['Fwy'] = incidents['Fwy'].astype(int)
    incidents = incidents[~incidents['Abs PM'].isna()]

    output_incidents = pd.DataFrame()

    for fwy in tqdm(incidents['Fwy'].unique()):
        temp = meta_df[meta_df['Fwy']==fwy][['station_id', 'Abs PM']]
        temp_incident = incidents[incidents['Fwy']==fwy][['incident_id', 'Abs PM', 'dt']]
        temp_incident['match_id'] = '1'
        temp['match_id'] = '1'
        t = pd.merge(temp, temp_incident, on='match_id').drop('match_id', axis=1)
        t['dis'] = abs(t['Abs PM_x']-t['Abs PM_y'])
        t = t[t['dis']<=args.dis_threshold].sort_values('dis')
        t = t.groupby('incident_id').head(1)
        output_incidents = pd.concat([output_incidents, t], axis=0, ignore_index=True)

    output_incidents.to_csv('match_incidents.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dis_threshold', type=float, default=0.5)
    parser.add_argument('--incident_path', type=str, default='data/incident_y2023.csv')
    args = parser.parse_args()
    main(args)
