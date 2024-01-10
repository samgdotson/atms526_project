import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

C_w = 4.181 # kJ/kg*K
h_w = 2256 # kJ/kg = J/g

mmbtu_to_mwh = lambda m: m*1e6/3.412e6
mwh_to_kj = lambda e: e*3.6e6
to_celcius = lambda T: 5/9*(T-32)

def energy_to_kj(dataframe):
    df = dataframe.copy()
    mmbtu_to_mwh = lambda m: m*1e6/3.412e6
    mwh_to_kj = lambda e: e*3.6e6
    df['elc_in_kj'] = df['elc_mmbtu_in'].apply(mmbtu_to_mwh)
    df['elc_in_kj'] = df['elc_in_kj'].apply(mwh_to_kj)
    df['elc_out_kj'] = df['elc_mwh_out'].apply(mwh_to_kj)

    return df


def calculate_discharge_temp_series(dataframe, Rc):

    df = dataframe.copy()

    T_inC = df['avg_intake_temp_C']
    w_inkgMonth = df['withdrawal_rate_kgM'] - df['diversion_rate_kgM']
    w_outkgMonth = df['discharge_rate_kgM']
    w_evap = w_inkgMonth-w_outkgMonth
    E_in_kj = df['elc_in_kj']
    E_out_kj = df['elc_out_kj']

    T_outC = ((1-Rc)*(E_in_kj-E_out_kj) \
              + w_inkgMonth*C_w*(T_inC) - w_evap*h_w)/(w_outkgMonth*C_w)

    return T_outC


def calculate_discharge_temp(Rc,
                             E_in,
                             E_out,
                             T_in,
                             w_in,
                             w_out,
                             w_evap=0.0):

    T_out = ((1-Rc)*(E_in-E_out) + w_in*C_w*(T_in) - w_evap*h_w)/(w_out*C_w)

    return T_out


def calculate_max_flow_rate(Rc,
                            E_in,
                            E_out,
                            T_in,
                            T_out):

    W_in = ((1-Rc)*(E_in-E_out))/(C_w*(T_out-T_in))

    return W_in


def calculate_power(Rc,
                    E_in,
                    W_in,
                    T_in,
                    T_out):

    E_out = E_in - ((C_w*W_in*(T_out-T_in))/(1-Rc))

    return E_out


def calculate_temp_threshold_series(dataframe):

    df = dataframe.copy()

    T_outC = df['temp_limit']
    w_inkgMonth = df['withdrawal_rate_kgM'] - df['diversion_rate_kgM']
    w_outkgMonth = df['discharge_rate_kgM']
    w_evap = w_inkgMonth-w_outkgMonth
    E_in_kj = df['elc_in_kj']
    E_out_kj = df['elc_out_kj']

    T_inC = (-1*(np.ones(len(df))-df['R_combined'])*(E_in_kj-E_out_kj) \
              + w_outkgMonth*C_w*(T_outC) + w_evap*h_w)/(w_inkgMonth*C_w)

    return T_inC


def estimate_R_combined_gridsearch(dataframe, n_iter=100, return_rmse=True):

    df = dataframe.copy()

    T_out_avg = df['avg_discharge_temp_C']

    R_vals = np.linspace(0,0.999,n_iter)

    rmse_list = np.zeros(n_iter)
    for i, R in enumerate(R_vals):
        T_out_calc = calculate_discharge_temp_series(df, R)
        diff = mean_squared_error(T_out_calc,T_out_avg, squared=False)
        rmse_list[i] = diff

    # find which value of R minimizes RMSE.
    r_idx = np.argmin(rmse_list)
    R = R_vals[r_idx]

    if return_rmse:
        return R, rmse_list[r_idx]

    else:
        return R


def linear_fit(df):
    x = df.iloc[:,0]
    y_i = df.iloc[:,1]
    x2 = sm.add_constant(x)
    mod_uni = sm.OLS(y_i,x2).fit()

    return mod_uni
