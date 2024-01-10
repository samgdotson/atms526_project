import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
from energy_model import *
from eia_codes import eia_codes
plt.style.use('seaborn')
import scipy.stats as sst


# upload the stream model
power_model = pd.read_csv('linear_models/power_model.csv')
stream_model = pd.read_csv('linear_models/stream_model.csv')
plants = stream_model['Plant ID'].values.astype('int')

N_simulations = 100
N = 12*20 # the number of months * number of years * number of simulations
np.random.seed(235)

ipcc_scenarios = {'RCP2.6':[1.0,0.4,1.6],
                  'RCP4.5':[1.4,0.9,2.0],
                  'RCP6.0':[1.3,0.8,1.8],
                  'RCP8.5':[2.0,1.4,2.6]}


for pid in plants:
# for pid in [6014]:

    rs = stream_model[stream_model['Plant ID']==pid].rsquared.values[0]
    if rs < 0.5:
        continue
    else:
        # upload plant data
        plant_data = pd.read_csv(f'energy_balance/energy_balance_plant_{pid}.csv')

        max_output = plant_data.elc_out_kj.max()
        max_input = plant_data.elc_in_kj.max()
        deltaE = max_input-max_output
        temp_limit = plant_data.temp_limit.values[0]
        R = plant_data.R_combined.values[0]
        max_withdrawal = plant_data.max_withdrawal_rate_kgM.values[0]

        # upload temperature data
        plant_name = eia_codes[pid].replace(' ', '')
        print(plant_name)
        temp_file = f'nrel_psm_data/{plant_name}_Temperature_2007_2020.csv'
        temp_df = pd.read_csv(temp_file)
        temp_df.rename(columns={'time':'date', f'Temp_{plant_name}':'air_tempC'},inplace=True)
        temp_df.index = pd.to_datetime(temp_df.date)
        temp_df.drop(columns='date',inplace=True)
        # resample by month
        temp_monthly = temp_df.resample('M').mean()

        for k in list(ipcc_scenarios.keys()):
            warming_range = ipcc_scenarios[k]
            frames = []
            for delta_shift in warming_range:
                scenario_name = f"{k}_{delta_shift}"
                print(scenario_name+"\n")

                for n in range(N_simulations):
                    # shift the mean
                    df = temp_monthly.copy()
                    curr_mean = df.air_tempC.mean()
                    curr_std = df.air_tempC.std()
                    curr_min = df.air_tempC.min()
                    curr_max = df.air_tempC.max()

                    # sample the new distribution
                    x = np.linspace(0, 45, N)
                    y_i = sst.norm.pdf(x, curr_mean, curr_std)
                    y_shift = sst.norm.pdf(x, curr_mean+delta_shift, curr_std)
                    s_i = np.random.normal(curr_mean+delta_shift, curr_std, N)

                    # create model dataframe
                    model_df = pd.DataFrame({f'modeled_air_temps':s_i})
                    model_df['simID'] = n
                    model_df['mean'] = delta_shift
                    m = stream_model[stream_model['Plant ID'] == pid].m.values[0]
                    b = stream_model[stream_model['Plant ID'] == pid].b.values[0]

                    air_to_stream = lambda T: m*T + b

                    model_df[f'modeled_stream_temps'] = model_df[f'modeled_air_temps'].apply(air_to_stream)

                    flow_rates = np.zeros(len(model_df))
                    for i,T_in in enumerate(model_df.modeled_stream_temps):
                        flow_rates[i] = calculate_max_flow_rate(R,
                                                                max_input,
                                                                max_output,
                                                                T_in,
                                                                temp_limit)
                    model_df['flow_rate'] = flow_rates

                    threshold = 0.01
                    N_iter = 100
                    derate = np.zeros(len(model_df))

                    reduction_list = np.zeros(len(model_df))
                    for j in range(len(model_df)):
                        flow_rate_exceeded = model_df.loc[j, 'flow_rate'] > max_withdrawal
                        reduction_list = np.linspace(0,0.99,N_iter)
                        new_temps = np.zeros(N_iter)
                        for i, r in enumerate(reduction_list):
                            if flow_rate_exceeded:
                                fr = max_withdrawal
                            else:
                                fr = model_df.loc[j, 'flow_rate']

                            t_out = calculate_discharge_temp(Rc=R,
                                                             E_in=max_input*r,
                                                             E_out=max_output*r,
                                                             T_in=model_df.loc[j,'modeled_stream_temps'],
                                                             w_in=fr,
                                                             w_out=fr)
                            new_temps[i] = t_out
                        try:
                            idx = np.argmax(new_temps[new_temps<=temp_limit])
                            derate[j] = reduction_list[idx]
                        except:
                            # print('Shutdown required')
                            pass

                    model_df['derate'] = derate
                    frames.append(model_df)
            plant_impact = pd.concat(frames, axis=0)
            plant_impact.to_csv(f"results/{k}/{plant_name.lower()}_{'_'.join(k.split('.'))}.csv",
                                index=False)
                    # ax = model_df.plot.scatter(x='modeled_stream_temps',y='derate')
                    # ax.set_ylabel('Power Output [Fraction of rated capacity]')
                    # ax.set_title('Power Derate at Brunswick Nuclear Due to Climate Change')
                    # plt.show()
