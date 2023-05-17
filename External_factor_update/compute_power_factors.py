
"""
This code is a Python script that produces power facotrs for the tube leak detection system.
It defines several functions, including mic_tag, filtered_tag, STMA_tag, power_tag, and 
sbl_tag, which return strings that are used as tag names in the df_from_pi_data function. 
It also defines several lists of tag names for different sensor types, such as sbl_tags, 
power_tags, filtered_tags, mic_tags, and STMA_tags.

The script then defines a start time and an end time for a time series query and calls 
osi_pi.df_from_pi_data to fetch the data for all of the tag names specified in the previous
lists. 
The resulting data is then used to compute the power factor change for each unit and sensor. 
This is done using a polynomial regression to fit a line to the data and calculating the slope 
of that line. The results are then stored in a new dataframe called df_pf.

Finally, the script creates scatterplots and timeseries plots of the data for each sensor and
unit combination. The scatterplot shows the relationship between power and microphone measurements,
while the timeseries plot shows the actual time series data for each sensor and unit.
Both plots are saved as PDF files in a specified directory.
"""

import os
import osi_pi
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib, matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
import regex as re
import dplot.combine_pdfs
import warnings
warnings.filterwarnings('ignore')
import os, shutil

def compute_power_factors(previous_power_factor_file_csv='Power_factors 2023-03-07.csv', days=14):
    """
    This function computes the power factors for the tube leak detection system
    and returns a dataframe with the results.

    Args:
    - previous_power_factor_file: string, path to the previous power factors file
    - days: int, number of days to query for the time series data (default=14)

    Returns:
    - df_pf: pandas DataFrame, dataframe with the power factors for each unit and sensor
    - plots the scatterplots and time series plots for each unit and sensor with the power_factors
    """

    # Set up folder paths
    folder_plots_auto = r'Plots\auto'
    output_folder = r'Output data sets'
    previous_power_factor_file = os.path.join(output_folder, previous_power_factor_file_csv)

    # start and end time for the time series query
    last_night = dt.datetime.now().strftime('%Y-%m-%d')
    start_time = (dt.datetime.now() - dt.timedelta(days=days)).strftime('%Y-%m-%d')

    # functions to create tag names
    def filtered_tag(unit=1, sensor=1):
        return f'BW.C.TLD.{unit}.{str(sensor).zfill(2)}.MEASUREMENT.FILTERED'
    def STMA_tag(unit=1, sensor=1):
        return f'BW.C.TLD.{unit}.{str(sensor).zfill(2)}.MEASUREMENT.STMA'
    def power_tag(unit=1):
        return f'BW.{unit}S20G001..XQ01'

    filtered_tags = [filtered_tag(k,i) for k in range(1,5) for i in range(1,23)]
    power_tags = [power_tag(k) for k in range(1,5)]
    STMA_tags = [STMA_tag(k,i) for k in range(1,5) for i in range(1,23)]

    # time series data frame of the tags
    df_ts= osi_pi.df_from_pi_data(
        tag_names=filtered_tags+power_tags+STMA_tags,
        start=start_time,
        end=last_night,
        interval='1m',
        pi_method='interpolated',
        use_cache=True,
        split_query_freq='1w',
        n_threads=4
    )
    df_res = df_ts.copy()
    # resample and keep the minimum value
    df_res[power_tags] = df_res[power_tags].resample('5T',).min().ffill()

    # create a dataframe of the power factors 
    df_pf = pd.DataFrame(columns=['unit', 'sensor', 'previous PF', 'PF change', 'power factor'])
    for unit in range(1,5):
        for i, sensor in enumerate(range(1,23)):
            mic = STMA_tag(unit,sensor)
            power = power_tag(unit=unit)
            df_res_cleaned = df_res[[mic,power]].copy()
            ## BWu4 forms two data clusters (change this later)
            if unit==4:
                df_res_cleaned = df_res_cleaned[df_res_cleaned[mic]<df_res[mic].quantile(0.8)].dropna()
                df_res_cleaned = df_res_cleaned[df_res_cleaned[mic]>df_res[mic].quantile(0.05)].dropna()
            else:
                df_res_cleaned = df_res_cleaned[df_res_cleaned[mic]<df_res[mic].quantile(0.95)].dropna()
                df_res_cleaned = df_res_cleaned[df_res_cleaned[mic]>df_res[mic].quantile(0.05)].dropna()
            # polynimial regression
            fit_np = np.polyfit(x=df_res_cleaned[power],y=df_res_cleaned[mic], deg=1)
            yn = np.poly1d(fit_np)
            power_factor_change = yn[1]

            new_row = {'unit':unit, 'sensor':str(sensor).zfill(2), 'PF change':round(yn[1],5)}
            df_pf = df_pf.append(new_row, ignore_index=True)
            
            # Plots
            # First plot of the scatterplot
            fig, ax= plt.subplots(2,1,figsize=(12,5))
            sns.scatterplot(data=df_res_cleaned, x=power, y=mic, alpha=0.05, ax=ax[0], label=mic, )
            xm = np.linspace(100,700,10)
            ax[0].plot(xm, yn(xm), ls='--', lw=3, color='blue', 
                    label=" ({0:.3f}x+{1:.2f})".format(fit_np[0], fit_np[1]))
            ax[0].legend()
            ax[0].set_xlabel(f'BW-U{unit} sensor{sensor}')

            # Second plot of the timeseries
            ax[1].plot(df_res[mic], label=filtered_tag(unit,sensor))
            ax[1].plot(df_res_cleaned[mic]-power_factor_change*(df_res_cleaned[power_tag(unit)]-400), label='Cleaned')
            ax[1].legend()
            # save the plots
            # plot counter is a 3 digiti number, the first digit is the unit number
            # the last 2 digits are the sensor number (e.g. 01,02, ... 22)
            plot_counter = f'{unit} {str(sensor).zfill(2)}'
            fig.savefig(
                        os.path.join(folder_plots_auto, 
                        '{}.png'.format(plot_counter)),
                        dpi=400,
                        bbox_inches='tight'
                    )
            plt.close()
    # ---------------------------------------------------------------------------------------------------
    # save the power factors to a csv file
    df_pf_old = pd.read_csv(previous_power_factor_file, index_col=0)
    df_pf['previous PF'] = df_pf_old['power factor'] 
    df_pf['power factor'] = df_pf_old['power factor'] + df_pf['PF change']
    df_pf.to_csv(os.path.join(output_folder, f'Power_factors {last_night}.csv'))

    # ---------------------------------------------------------------------------------------------------
    # plot previous and new power factors
    fig, ax = plt.subplots(4,1, figsize=(16,12))
    for unit in range(1,5):
        ax[unit-1].step(data=df_pf[df_pf['unit']==unit], x='sensor', y='previous PF', label='previous PF')
        ax[unit-1].step(data=df_pf[df_pf['unit']==unit], x='sensor', y='power factor', label='new PF')
        ax[unit-1].set_ylabel(f'Unit {unit}')
        ax[unit-1].legend()
    ax[0].set_title(f'Power factors at BW, Date : {last_night}')
    plot_counter = f'0 Power factor summary'
    fig.savefig(os.path.join(folder_plots_auto, 
                    '{}.png'.format(plot_counter)),
                    dpi=400,
                    bbox_inches='tight'
                )
    plt.close()

    # ---------------------------------------------------------------------------------------------------
    #produce the pdf file of the regressions
    fname_pdf_report = 'BW {} TLS_power_reg.pdf'.format(dt.datetime.now().strftime('%Y-%m-%d'))

    dplot.combine_pdfs.convert_images_in_folder_to_pdf(folder_plots_auto, image_extensions='png')
    dplot.combine_pdfs.combine_pdfs_in_folder(
        folder_plots_auto, output_file=os.path.join(r'Plots\PDF', fname_pdf_report))

    folder = folder_plots_auto
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def main():
    compute_power_factors()