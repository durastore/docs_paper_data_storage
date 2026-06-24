#Functions for calculating gaussian distribution
def gaussian(x, amplitude, mean, stddev):
    import numpy as np
    return amplitude * np.exp(-((x - mean) / 4 / stddev) ** 2)

def _1gaussian(x, amp1,cen1,sigma1):
    import numpy as np
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2):
    import numpy as np
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2)))

def base_p_calc(data, channel_info, pos_n, fit_plot, import_lambda_l, export_folder):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    import_lambda = import_lambda_l[0]

    ch_name_l, ch_color_l = channel_info
    channel_n = len(ch_name_l)

    loc_n_l = []
    for i in range(channel_n):
        loc_n_l.append([])
    data_l = [[],[]]
    
    lambda_param_l = []
    for str in data:
        for i in range(channel_n):
            channel = str[i]
            channel_name = ch_name_l[i]
            if import_lambda == True:
                ch_lambda_param = import_lambda_l[2][import_lambda_l[1].index(channel_name)]
                lambda_param_l.append(ch_lambda_param)
            for j in range(pos_n):
                pos = channel[j+1]
                loc_n = len(pos)
                loc_n_l[i].append(loc_n)
                data_l[0].append(loc_n)
                data_l[1].append(channel_name)
    

    p_ch_l=[]
    for i in range(len(loc_n_l)):
        ch = loc_n_l[i]
        channel_name = ch_name_l[i]
        if import_lambda == True:
            ch_lambda_param = lambda_param_l[i]
            a, b, c = em_algorithm_poisson_mixture_non_mixed(observations=ch, lambda_info_l= ch_lambda_param, fit_plot=fit_plot)
        else:
            a, b, c = em_algorithm_poisson_mixture(export_folder=export_folder,observations=ch, fit_plot=fit_plot, channel_name=channel_name)
        p_ch_l.append([a,b,c])

    if fit_plot==True:
        df1 = pd.DataFrame(
            list(zip(data_l[0],data_l[1])),
            columns=['loc_n', 'ch_name'])

        d = {'color': ch_color_l}
        g = sns.FacetGrid(df1, row="ch_name", hue_kws=d, hue='ch_name')
        g.map(sns.histplot, 'loc_n')

        plt.show()
        plt.close()
    return p_ch_l

def em_algorithm_poisson_mixture(channel_name, export_folder, observations, fit_plot, max_iterations=100, convergence_threshold=1e-6):
    import numpy as np
    from scipy.stats import poisson
    """
    Implement EM algorithm for a mixture of two Poisson distributions

    Parameters:
    -----------
    observations : array-like
        List of observed counts (localization numbers)
    max_iterations : int
        Maximum number of iterations for the algorithm
    convergence_threshold : float
        Threshold for convergence check

    Returns:
    --------
    lambda_on : float
        Estimated Poisson parameter for "on" sites
    lambda_off : float
        Estimated Poisson parameter for "off" sites
    p_on : float
        Estimated prior probability of a site being on
    probabilities : array
        Final posterior probabilities of each observation being from an "on" site
    """
    # Convert observations to numpy array
    obs = np.array(observations)
    n_samples = len(obs)

    # Initialize parameters
    # Assume higher counts come from "on" sites
    median_value = np.median(obs)

    # Initial guesses for parameters
    lambda_on = np.mean(obs[obs >= median_value])
    lambda_off = np.mean(obs[obs < median_value])
    lambda_off = max(0.1, lambda_off)  # Ensure lambda_off is not too small
    p_on = 0.5  # Initial prior probability

    # Array to store probabilities that each observation comes from "on" sites
    probabilities = np.zeros(n_samples)

    # Store history for plotting convergence
    history = {
        'lambda_on': [lambda_on],
        'lambda_off': [lambda_off],
        'p_on': [p_on]
    }

    # EM algorithm iterations
    for iteration in range(max_iterations):
        # Store old parameters to check convergence
        old_lambda_on = lambda_on
        old_lambda_off = lambda_off
        old_p_on = p_on

        # E-step: Calculate posterior probabilities
        # P(on|n) for each observation
        p_n_given_on = poisson.pmf(obs, lambda_on)
        p_n_given_off = poisson.pmf(obs, lambda_off)

        # Apply Bayes' theorem
        numerator = p_n_given_on * p_on
        denominator = p_n_given_on * p_on + p_n_given_off * (1 - p_on)
        probabilities = numerator / denominator

        # M-step: Update parameters
        p_on = np.mean(probabilities)

        # Weighted average for lambda parameters
        lambda_on = np.sum(probabilities * obs) / np.sum(probabilities)
        lambda_off = np.sum((1 - probabilities) * obs) / np.sum(1 - probabilities)

        # Ensure lambda values don't get too small
        lambda_off = max(0.1, lambda_off)

        # Store updated parameters
        history['lambda_on'].append(lambda_on)
        history['lambda_off'].append(lambda_off)
        history['p_on'].append(p_on)

        # Check convergence
        param_changes = [
            abs(lambda_on - old_lambda_on) / max(1e-10, old_lambda_on),
            abs(lambda_off - old_lambda_off) / max(1e-10, old_lambda_off),
            abs(p_on - old_p_on) / max(1e-10, old_p_on)
        ]

        if all(change < convergence_threshold for change in param_changes):
            print(f"Converged after {iteration + 1} iterations")
            break

    if iteration == max_iterations - 1:
        print(f"Reached maximum iterations ({max_iterations}) without convergence")

    # Plot convergence history
    if fit_plot == True:
        plot_convergence(history)

    # Plot mixture model fit
    if fit_plot == True:
        plot_mixture_fit_exp(obs, lambda_on, lambda_off, p_on, save_folder=export_folder, channel_name=channel_name)
    return lambda_on, lambda_off, p_on

def lambda_import(lambda_csv_file):
    import csv

    with open(lambda_csv_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        lambda_data_info = [row for row in spamreader]
    info_l = [row for row in lambda_data_info][5:]
    channel_name_l = [x[0] for x in info_l]
    lambda_l = [[x[1], x[2]] for x  in info_l]

    return channel_name_l, lambda_l

def em_algorithm_poisson_mixture_non_mixed(observations, lambda_info_l, fit_plot, max_iterations=100, convergence_threshold=1e-6):
    import numpy as np
    from scipy.stats import poisson
    """
    Implement EM algorithm for a mixture of two Poisson distributions

    Parameters:
    -----------
    observations : array-like
        List of observed counts (localization numbers)
    max_iterations : int
        Maximum number of iterations for the algorithm
    convergence_threshold : float
        Threshold for convergence check

    Returns:
    --------
    lambda_on : float
        Estimated Poisson parameter for "on" sites
    lambda_off : float
        Estimated Poisson parameter for "off" sites
    p_on : float
        Estimated prior probability of a site being on
    probabilities : array
        Final posterior probabilities of each observation being from an "on" site
    """
    # Convert observations to numpy array
    obs = np.array(observations)
    n_samples = len(obs)

    # Initialize parameters
    # Assume higher counts come from "on" sites
    median_value = np.median(obs)

    # Initial guesses for parameters
    lambda_on, lambda_off = lambda_info_l
    lambda_on = float(lambda_on)
    lambda_off = float(lambda_off)
    p_on = 0.5  # Initial prior probability

    # Array to store probabilities that each observation comes from "on" sites
    probabilities = np.zeros(n_samples)

    # Store history for plotting convergence
    history = {
        'lambda_on': [lambda_on],
        'lambda_off': [lambda_off],
        'p_on': [p_on]
    }

    # EM algorithm iterations
    for iteration in range(max_iterations):
        # Store old parameters to check convergence
        old_lambda_on = lambda_on
        old_lambda_off = lambda_off
        old_p_on = p_on

        # E-step: Calculate posterior probabilities
        # P(on|n) for each observation
        p_n_given_on = poisson.pmf(obs, lambda_on)
        p_n_given_off = poisson.pmf(obs, lambda_off)

        # Apply Bayes' theorem
        numerator = p_n_given_on * p_on
        denominator = p_n_given_on * p_on + p_n_given_off * (1 - p_on)
        probabilities = numerator / denominator

        # M-step: Update parameters
        p_on = np.mean(probabilities)

        # Store updated parameters
        history['lambda_on'].append(lambda_on)
        history['lambda_off'].append(lambda_off)
        history['p_on'].append(p_on)

        # Check convergence
        param_changes = [
            abs(lambda_on - old_lambda_on) / max(1e-10, old_lambda_on),
            abs(lambda_off - old_lambda_off) / max(1e-10, old_lambda_off),
            abs(p_on - old_p_on) / max(1e-10, old_p_on)
        ]

        if all(change < convergence_threshold for change in param_changes):
            print(f"Converged after {iteration + 1} iterations")
            break

    if iteration == max_iterations - 1:
        print(f"Reached maximum iterations ({max_iterations}) without convergence")

    # Plot convergence history
    if fit_plot==True:
        plot_convergence(history)

        # Plot mixture model fit
        plot_mixture_fit(obs, lambda_on, lambda_off, p_on)

    return lambda_on, lambda_off, p_on

def plot_convergence(history):
    import matplotlib.pyplot as plt
    """Plot the convergence of parameters during EM iterations"""
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(history['lambda_on'])
    plt.title('Lambda for ON lanterns')
    plt.ylabel('Value')

    plt.subplot(3, 1, 2)
    plt.plot(history['lambda_off'])
    plt.title('Lambda for OFF lanterns')
    plt.ylabel('Value')

    plt.subplot(3, 1, 3)
    plt.plot(history['p_on'])
    plt.title('Prior probability of ON lanterns')
    plt.ylabel('Value')
    plt.xlabel('Iteration')

    plt.tight_layout()
    plt.show()

def plot_mixture_fit(observations, lambda_on, lambda_off, p_on):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import poisson
    import seaborn as sns
    """Plot the histogram of observations with the fitted mixture model"""
    plt.figure(figsize=(10, 6))

    # Histogram of observations

    max_obs = max(observations)
    bins = np.arange(-0.5, max_obs + 1.5, 1)  # Centered bins for integer counts
    #sns.histplot(observations, alpha=0.6,binwidth=bins[1]-bins[0], kde=True, stat="probability", label='Observations')
    #plt.hist(observations,bins=20, density=True, alpha=0.6, label='Observations')
    sns.histplot(observations, stat="probability", bins=20)
    # Generate x values for plotting the Poisson PMFs
    x = np.arange(0, max_obs + 1)

    # Calculate PMF for each component and the mixture
    y_on = poisson.pmf(x, lambda_on)
    y_off = poisson.pmf(x, lambda_off)
    y_mixture = p_on * y_on + (1 - p_on) * y_off

    # Plot the PMFs
    plt.plot(x, y_on, 'r--', label=f'ON Poisson (λ={lambda_on:.2f})')
    plt.plot(x, y_off, 'g--', label=f'OFF Poisson (λ={lambda_off:.2f})')
    plt.plot(x, y_mixture, 'k-', linewidth=2, label='Mixture Model')

    plt.title('Poisson Mixture Model Fit')
    plt.xlabel('Number of Localizations')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_mixture_fit_exp(observations, lambda_on, lambda_off, p_on, save_folder, channel_name):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import poisson
    import seaborn as sns
    import pandas as pd
    import os

    """Plot the histogram of observations with the fitted mixture model and export data."""
    bin_n = 20
    plt.figure(figsize=(10, 6))

    max_obs = max(observations)

    # Generate x values for plotting the Poisson PMFs
    x = np.arange(0, max_obs + 1)

    # Calculate PMF for each component and the mixture
    y_on = poisson.pmf(x, lambda_on)
    y_off = poisson.pmf(x, lambda_off)
    y_mixture = p_on * y_on + (1 - p_on) * y_off

    # Histogram of observations
    sns.histplot(observations, stat="probability", bins=bin_n)

    # Plot the PMFs
    plt.plot(x, y_on, 'r--', label=f'ON Poisson (λ={lambda_on:.2f})')
    plt.plot(x, y_off, 'g--', label=f'OFF Poisson (λ={lambda_off:.2f})')
    plt.plot(x, y_mixture, 'k-', linewidth=2, label='Mixture Model')

    plt.title('Poisson Mixture Model Fit')
    plt.xlabel('Number of Localizations')
    plt.ylabel('Probability')

    # Set the x-limits as requested
    plt.xlim(0, 100)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Handle the CSV data extraction
    counts, bin_edges = np.histogram(observations, bins=bin_n)
    probabilities = counts / len(observations)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

        # Package the curve data into a DataFrame
        # Wrap each list/array in pd.Series() to let Pandas auto-pad the missing rows
        df_plot = pd.DataFrame({
            'Number_of_Localizations_x': pd.Series(x),
            'ON_Poisson_Prob': pd.Series(y_on),
            'OFF_Poisson_Prob': pd.Series(y_off),
            'Mixture_Model_Prob': pd.Series(y_mixture),
            'Histogram_Bin_Centers': pd.Series(bin_centers),
            'Histogram_Probabilities': pd.Series(probabilities)
        })

        # Filter the DataFrame to match the 0 to 100 x-limit you specified
        df_export = df_plot[df_plot['Number_of_Localizations_x'] <= 100]

        save_path = os.path.join(save_folder, 'channel-' + channel_name + '-loc-n-fit.csv')
        df_export.to_csv(save_path, index=False)
        print(f"Plot data exported successfully to: {save_path}")

def predict_pos_status(new_observations, lambda_on, lambda_off, p_on):
    from scipy.stats import poisson
    import numpy as np

    """
    Predict whether position are ON or OFF based on localization counts

    Parameters:
    -----------
    new_observations : array-like
        New localization counts to classify
    lambda_on, lambda_off, p_on : float
        Parameters estimated by the EM algorithm

    Returns:
    --------
    probabilities : array
        Posterior probabilities of each position being ON
    """
    obs = np.array(new_observations)

    # Calculate P(n|on) and P(n|off)
    p_n_given_on = poisson.pmf(obs, lambda_on)
    p_n_given_off = poisson.pmf(obs, lambda_off)

    # Calculate posterior probabilities P(on|n)
    numerator = p_n_given_on * p_on
    denominator = p_n_given_on * p_on + p_n_given_off * (1 - p_on)
    probabilities = numerator / denominator

    return probabilities

def pos_color_p_calc(ch_base_p_l, ch_names_l, pos_loc_n_l):
    # import packages used
    import itertools
    import numpy as np

    ch_pos_p_l = []
    ch_name_l = []
    for i in range(len(pos_loc_n_l)):
        ch_name = ch_names_l[i]
        ch_loc_n = pos_loc_n_l[i]
        lambda_on, lambda_off, p_on = ch_base_p_l[i]
        ch_p = predict_pos_status(new_observations=[ch_loc_n], lambda_on=lambda_on, lambda_off=lambda_off, p_on=p_on)
        ch_pos_p_l.append([ch_p, 1.0-ch_p])
        ch_name_l.append([ch_name, '0'])
    sort_ch_name_l = ['-'.join(x) for x in list(itertools.product(*ch_name_l))]
    sort_ch_pos_p_l = [np.product(x) for x in list(itertools.product(*ch_pos_p_l))]


    return sort_ch_name_l, sort_ch_pos_p_l

def ch_to_code_conv(pos_ch_states_l, pos_ch_states_p_l):

    pos_code_l = []
    for pos_ch_states in pos_ch_states_l:
        pos_code_states = []
        for pos_ch_state in pos_ch_states:
            if 'P2' in pos_ch_state:
                if 'P3' in pos_ch_state:
                    if 'P5' in pos_ch_state:
                        code = 'X'
                    else:
                        code = '3'

                else:
                    if 'P5' in pos_ch_state:
                        code = '4'
                    else:
                        code = '0'

            else:
                if 'P3' in pos_ch_state:
                    if 'P5' in pos_ch_state:
                        code = '5'
                    else:
                        code = '1'

                else:
                    if 'P5' in pos_ch_state:
                        code = '2'
                    else:
                        code = 'X'

            pos_code_states.append(code)

        pos_code_l.append(pos_code_states)

    comp_pos_code_l = []
    comp_pos_p_l = []
    for i in range(len(pos_code_l)):
        comp_pos_code_l.append(['X'])
        comp_pos_p_l.append([0.0])

        pos_codes = pos_code_l[i]
        pos_p = pos_ch_states_p_l[i]
        for j in range(len(pos_codes)):
            code = pos_codes[j]
            p = pos_p[j]
            if code == 'X':
                comp_pos_p_l[-1][0] = comp_pos_p_l[-1][0]+p
            else:
                comp_pos_code_l[-1].append(code)
                comp_pos_p_l[-1].append(p)

    return comp_pos_code_l, comp_pos_p_l

def ch_to_code_conv_non_mixed(pos_ch_states_l, pos_ch_states_p_l):

    pos_code_l = []
    for pos_ch_states in pos_ch_states_l:
        pos_code_states = []
        for pos_ch_state in pos_ch_states:
            if 'P2' in pos_ch_state:
                if 'P3' in pos_ch_state:
                    if 'P5' in pos_ch_state:
                        code = 'X'
                    else:
                        code = 'X'

                else:
                    if 'P5' in pos_ch_state:
                        code = 'X'
                    else:
                        code = '0'

            else:
                if 'P3' in pos_ch_state:
                    if 'P5' in pos_ch_state:
                        code = 'X'
                    else:
                        code = '1'

                else:
                    if 'P5' in pos_ch_state:
                        code = '2'
                    else:
                        code = 'X'

            pos_code_states.append(code)

        pos_code_l.append(pos_code_states)

    comp_pos_code_l = []
    comp_pos_p_l = []
    for i in range(len(pos_code_l)):
        comp_pos_code_l.append(['X'])
        comp_pos_p_l.append([0.0])

        pos_codes = pos_code_l[i]
        pos_p = pos_ch_states_p_l[i]
        for j in range(len(pos_codes)):
            code = pos_codes[j]
            p = pos_p[j]
            if code == 'X':
                comp_pos_p_l[-1][0] = comp_pos_p_l[-1][0]+p
            else:
                comp_pos_code_l[-1].append(code)
                comp_pos_p_l[-1].append(p)

    return comp_pos_code_l, comp_pos_p_l

def ch_to_code_conv_err_corr(pos_ch_states_l, pos_ch_states_p_l):

    pos_code_l = []
    for pos_ch_states in pos_ch_states_l:
        pos_code_states = []
        for pos_ch_state in pos_ch_states:
            if 'P2' in pos_ch_state:
                if 'P3' in pos_ch_state:
                    if 'P5' in pos_ch_state:
                        if 'P0' in pos_ch_state:
                            code = 'X'
                        else:
                            code = 'X'

                    else:
                        if 'P0' in pos_ch_state:
                            code = 'X'
                        else:
                            code = '3'

                else:
                    if 'P5' in pos_ch_state:
                        if 'P0' in pos_ch_state:
                            code = 'X'
                        else:
                            code = '4'
                    else:
                        if 'P0' in pos_ch_state:
                            code = '0'
                        else:
                            code = 'X'

            else:
                if 'P3' in pos_ch_state:
                    if 'P5' in pos_ch_state:
                        if 'P0' in pos_ch_state:
                            code = 'X'
                        else:
                            code = '5'
                    else:
                        if 'P0' in pos_ch_state:
                            code = '1'
                        else:
                            code = 'X'
                else:
                    if 'P5' in pos_ch_state:
                        if 'P0' in pos_ch_state:
                            code = '2'
                        else:
                            code = 'X'
                    else:
                        if 'P0' in pos_ch_state:
                            code = 'X'
                        else:
                            code = 'X'

            pos_code_states.append(code)

        pos_code_l.append(pos_code_states)

    comp_pos_code_l = []
    comp_pos_p_l = []
    for i in range(len(pos_code_l)):
        comp_pos_code_l.append(['X'])
        comp_pos_p_l.append([0.0])

        pos_codes = pos_code_l[i]
        pos_p = pos_ch_states_p_l[i]
        for j in range(len(pos_codes)):
            code = pos_codes[j]
            p = pos_p[j]
            if code == 'X':
                comp_pos_p_l[-1][0] = comp_pos_p_l[-1][0]+p
            else:
                comp_pos_code_l[-1].append(code)
                comp_pos_p_l[-1].append(p)

    return comp_pos_code_l, comp_pos_p_l

def bc_code_p_calc(pos_code_l, pos_code_p_l):
    import itertools
    import numpy as np
    b_comb_name_l = [x for x in list(itertools.product(*pos_code_l))]
    b_comb_p_l = [np.product(x) for x in list(itertools.product(*pos_code_p_l))]
    sort_b_comb_l=sorted(zip(b_comb_name_l, b_comb_p_l), key=lambda x: x[1])[::-1]
    sort_b_code_l = ["".join(x[0]) for x in sort_b_comb_l]
    sort_b_code_p_l = [x[1] for x in sort_b_comb_l]
    return sort_b_code_l, sort_b_code_p_l

def dc_id_det(result_hdf, channel_info, export_crop, export_folder, str_n_thr, bc_cand_n, bc_p_thr, gui_window, err_corr, non_mixed, import_lambda_l, fit_plot):
    #import packages used
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import os, errno
    import cv2
    from PIL import Image
    import itertools
    import csv
    from collections import Counter
    import time
    import datetime

    #Import processing information
    pos_n, ch_name_l, ch_color_l = channel_info
    if err_corr==True:
        error_corr_ch_ind = ch_name_l.index('P0')

    #Remove empty structures
    results_df = pd.read_hdf(result_hdf, key='proc_res')
    filtered_results_df = results_df.loc[results_df['Str_crop'] != 'NaN']

    #Plotting fraction of removed localizations by clustering
    if fit_plot ==True:
        str_loc_removed_l = filtered_results_df['Str_loc_removed'].values.tolist()
        ch_loc_removed_l = [[] for ch in ch_name_l]

        for st in str_loc_removed_l:
            for i in range(len(st)):
                if st[i][0] > 0:
                    ch_loc_removed_l[i].append(st[i][1]/st[i][0])

        fig, ax = plt.subplots(len(ch_name_l), 1)
        for i in range(len(ch_loc_removed_l)):
            channel = ch_loc_removed_l[i]
            channel_name = ch_name_l
            ax[i].hist(channel, bins=10, label=channel_name, color=ch_color_l[i])
            ax[i].set_xlim(0,1.0)

        plt.xlabel('Fraction of events removed')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()
        plt.close()
    
    #Calculate baseline probilities from channels
    str_loc_l = filtered_results_df['Str_loc'].values.tolist()
    ch_base_p_l = base_p_calc(data=str_loc_l, channel_info=[ch_name_l,ch_color_l], pos_n=pos_n, fit_plot=fit_plot, import_lambda_l=import_lambda_l, export_folder=export_folder)

    #Export processing parameters
    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H_%M_%S')

    with open(export_folder + '/ID_det_proc_parameters-' + time_stamp + '.csv', 'w', newline='') as csvfile:

        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        row1 = ['# of predicted barcodes per probe retained',bc_cand_n]
        spamwriter.writerow(row1)
        row2 = ['P cutoff for barcode retention', bc_p_thr]
        spamwriter.writerow(row2)
        row3 = ['Error correction channel used', err_corr]
        spamwriter.writerow(row3)
        row4 = ['Non mixed site sample', non_mixed]
        spamwriter.writerow(row4)
        row5 = ['Imported lambda values used', import_lambda_l[0]]
        spamwriter.writerow(row5)
        if import_lambda_l[0]==True:
            row6 = ['Imported lambda values used', import_lambda_l[-1]]
            spamwriter.writerow(row6)
        row7 = ['Channel name', 'Freq on', 'Freq off', 'P on']
        spamwriter.writerow(row7)
        for i in range(len(ch_name_l)):
            ch_name = ch_name_l[i]
            ch_poisson_param = ch_base_p_l[i]
            poisson_param_list = [ch_name] + ch_poisson_param
            spamwriter.writerow(poisson_param_list)
    #
    str_index_l = filtered_results_df['Str_index'].values.tolist()
    str_crop_img_l = filtered_results_df['Str_crop'].values.tolist()
    if err_corr == True:
        str_err_corr_img_l = [x[error_corr_ch_ind] for x in filtered_results_df['Str_channel_img'].values.tolist()]
    str_rgb_ch_img_l = filtered_results_df['Str_rgb_channel_img'].values.tolist()
    
    

    #list for exporting:
    b_code_l = []
    b_code_p_l = []
    b_crop_img_l = []
    b_error_corr_img_l = []
    b_rgb_img_l = []
    b_index_l = []

    if len(str_n_thr) !=0:
        sample_n = int(str_n_thr)
    else:
        sample_n = len(str_loc_l)

    from tqdm import tqdm

    with tqdm(total=sample_n) as pbar:
        for i in range(sample_n):
            pbar.update(1)
            gui_window['OUTPUT'].update(value=pbar)
            gui_window.refresh()

            str_index = str_index_l[i]
            b_index_l.append(str_index)

            str_crop_img = str_crop_img_l[i]
            b_crop_img_l.append(str_crop_img)

            if err_corr == True:
                str_err_corr_img = str_err_corr_img_l[i]
                b_error_corr_img_l.append(str_err_corr_img)
            
            str_rgb_img = str_rgb_ch_img_l[i]
            b_rgb_img_l.append(str_rgb_img)

            str_loc = str_loc_l[i]
            str_loc_n_l = [[],[],[],[],[]]

            for ch in str_loc:
                for j in range(len(ch)-1):
                    pos = ch[j+1]
                    str_loc_n_l[j].append(len(pos))

            str_ch_p_l = []
            str_ch_name_l = []
            for pos in str_loc_n_l:
                sort_ch_name_l, sort_ch_p_l = pos_color_p_calc(ch_base_p_l=ch_base_p_l, ch_names_l=ch_name_l, pos_loc_n_l=pos)

                str_ch_name_l.append(sort_ch_name_l)
                str_ch_p_l.append(sort_ch_p_l)

            thr_l = bc_cand_n
            if non_mixed == True:
                comp_pos_code_l, comp_pos_p_l = ch_to_code_conv_non_mixed(pos_ch_states_l=str_ch_name_l,
                                                                         pos_ch_states_p_l=str_ch_p_l)
            else:
                if err_corr == True:
                    comp_pos_code_l, comp_pos_p_l = ch_to_code_conv_err_corr(pos_ch_states_l=str_ch_name_l, pos_ch_states_p_l=str_ch_p_l)

                else:
                    comp_pos_code_l, comp_pos_p_l = ch_to_code_conv(pos_ch_states_l=str_ch_name_l, pos_ch_states_p_l=str_ch_p_l)

            sort_b_code_l, sort_b_code_p_l = bc_code_p_calc(pos_code_l=comp_pos_code_l, pos_code_p_l=comp_pos_p_l)

            b_cd_l = sort_b_code_l[0:thr_l]
            b_p_l = sort_b_code_p_l[0:thr_l]
            b_code_l.append(b_cd_l)
            b_code_p_l.append(b_p_l)


    with open(export_folder + '/bc_extr_list-'+time_stamp+'.csv', 'w', newline='') as csvfile:

        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        extr_list = ['BC index']
        for i in range(thr_l):
            extr_list = extr_list + ['BC_code_'+str(i), 'BC_P_' + str(i)]
        spamwriter.writerow(extr_list)

        for i in range(len(b_index_l)):
            str_index = b_index_l[i]
            bc_code = b_code_l[i]
            bc_code_p = b_code_p_l[i]

            extr_result_list = [str_index]
            for j in range(len(bc_code)):
                extr_result_list = extr_result_list + [bc_code[j],bc_code_p[j]]

            spamwriter.writerow(extr_result_list)


    b_code_l_filt_flat= []
    for i in range(len(b_code_l)):
        bc_l = b_code_l[i]
        bc_p_l = b_code_p_l[i]
        for j in range(len(bc_p_l)):
            p = bc_p_l[j]
            bc = bc_l[j]
            r=round(p*100)
            if p >= bc_p_thr:
                for h in range(r):
                    b_code_l_filt_flat.append(bc)

    unique_bc_code_l = list(Counter(b_code_l_filt_flat).keys())
    unique_bc_code_count_l = Counter(b_code_l_filt_flat).values()
    results_l = sorted(zip(unique_bc_code_l, unique_bc_code_count_l), key=lambda x: x[1])[::-1]

    with open(export_folder + '/Joined_bc_list-'+time_stamp+'.csv', 'w', newline='') as csvfile:

        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Full list','','', 'Reduced list'])
        spamwriter.writerow(['BC code','Frequency','','BC code','Frequency'])

        red_result_l = [x for  x in results_l if 'X' not in x[0]]

        red_result_l = [x for x in results_l if 'X' not in x[0]]
        for i in range(len(results_l) - len(red_result_l)):
            red_result_l.append(('', ''))

        for result in zip(results_l, red_result_l):
            spamwriter.writerow([result[0][0], result[0][1], '', result[1][0], result[1][1]])
    
    if export_crop==True:

        new_dir = export_folder + '/crop_img'
        sub_dir_l = []

        for unique_bc_code in unique_bc_code_l:
            sub_dir_l.append(new_dir + '/' +unique_bc_code)

        try:
            os.makedirs(new_dir)
            for sub_dir in sub_dir_l:
                os.makedirs(sub_dir)

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        for i in range(len(b_code_l)):
            barcode_l = b_code_l[i]
            str_index = b_index_l[i]
            crop_img = b_crop_img_l[i]
            if err_corr == True:
                err_corr_img = b_error_corr_img_l[i]
            crop_img[crop_img > 255] = 255
            crop_img = crop_img.astype(np.uint8)
            if err_corr == True:
                err_corr_img = err_corr_img.astype(np.uint8)
            crop_img = crop_img[:, :, ::-1]
            img = Image.fromarray(crop_img)
            if err_corr == True:
                img_err = Image.fromarray(err_corr_img)
            
            rgb_img_l2 =[]
            rgb_img_l = b_rgb_img_l[i]
            for rgb_img in rgb_img_l:
                rgb_img[rgb_img > 255] = 255
                rgb_img = rgb_img.astype(np.uint8)
                rgb_img = rgb_img[:, :, ::-1]
                rgb_img = Image.fromarray(rgb_img)
                rgb_img_l2.append(rgb_img)

            for j in range(len(barcode_l)):

                barcode_code = barcode_l[j]

                if barcode_code in unique_bc_code_l:
                    code_index = unique_bc_code_l.index(barcode_code)
                    barcode_folder = sub_dir_l[code_index]
                    img.save(barcode_folder + '/BC-' + str(str_index)  + '-' + str(j) + '-' + str(barcode_code) + '.png')
                    if err_corr == True:
                        img_err.save(barcode_folder + '/BC-' + str(str_index) + '-' + str(j) + '-' + str(barcode_code) + '-ERR-CH.png')
                    for h in range(len(rgb_img_l2)):
                        name = ch_name_l[h]
                        rgb_img = rgb_img_l2[h]
                        rgb_img.save(barcode_folder + '/BC-' + str(str_index) + '-' + str(j) + '-' + str(barcode_code) +'-CH-'+name+ '.png')

def PAINT_func_res_plot_main_gui():
    # import modules used
    import PySimpleGUI as sg
    import csv

    left_col = [
        [sg.Frame('Data information:',
                  [[sg.Text('Hdf5 file with processing results:'),
                    sg.In(size=(50, 1), enable_events=True, key='results_file'),
                    sg.FileBrowse()],
                   [sg.Text('Csv file with channel info:'),
                    sg.In(size=(50, 1), enable_events=True, key='channel_info_csv'),
                    sg.FileBrowse()]
                   ])],
        [sg.Text('Folder to save data to:'),
         sg.In(size=(50, 1), enable_events=True, key='export_folder'),
         sg.FolderBrowse()],
        [sg.Text('CSV file with rates to import:'),
         sg.In(size=(50, 1), enable_events=True, key='lambda_csv_file'),
         sg.FileBrowse()],
        [sg.Checkbox('Export grouped crop images', size=(36, 1),
                     key='chk_exp_crop')],
        [sg.Text('Processing subset of data (# of probes):', size=(36, 1)),
         sg.InputText(key='str_n_thr', size=(8, 1))],
        [sg.Text('Number of candidates to store per probe (1-x)', size=(36, 1)),
         sg.InputText('1', key='bc_cand_n', size=(8, 1))],
        [sg.Text('P cutoff for candidates (0.0-1.0)', size=(36, 1)),
         sg.InputText('0.0', key='bc_p_thr', size=(8, 1))],
        [sg.Checkbox('Non-mixed channel sites:', size=(36, 1),
                     key='chk_non_mixed')],
        [sg.Checkbox('Error correction channel used:', size=(36, 1),
                     key='chk_err_corr')],
        [sg.Checkbox('Plot data fit:', size=(36, 1),
                     key='chk_plot_fit')],
        [sg.Button('Load data'), sg.Button('Cancel')]
    ]
    right_col = [
        [sg.Frame('Output:', [[sg.Multiline("", size=(50, 30), key='OUTPUT')]])]
    ]

    layout = [[sg.Column(left_col, element_justification='c'),
               sg.Column(right_col, element_justification='c', vertical_alignment="top")]]
    window_result_vis = sg.Window('Results visualization', layout, resizable=True)

    #
    while True:
        event, values = window_result_vis.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_result_vis.close()
            break

        if event == 'Load data':
            window_result_vis['OUTPUT'].update(value='Processing data' + '\n', append=True)
            window_result_vis.refresh()

            try:
                results_hdf = values['results_file']
                if len(results_hdf) == 0:
                    sg.Popup('Results h5 file not provided.')
                    break
                export_crop = values['chk_exp_crop']

                window_result_vis['OUTPUT'].update(value='Hdf5 file loaded:' + '\n', append=True)
                window_result_vis['OUTPUT'].update(value=results_hdf + '\n', append=True)
                window_result_vis.refresh()

                channel_info_csv_file = values['channel_info_csv']
                if len(channel_info_csv_file) == 0:
                    sg.Popup('Channel info csv file not provided.')
                    break

                else:
                    window_result_vis['OUTPUT'].update(value='Csv file loaded:' + '\n', append=True)
                    window_result_vis['OUTPUT'].update(value=channel_info_csv_file + '\n', append=True)
                    window_result_vis.refresh()

                    channel_info = []
                    with open(channel_info_csv_file, newline='') as csvfile:
                        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
                        channel_info = channel_info + [row for row in spamreader]

                    pos_n = int(channel_info[1][1])
                    ch_name_l = channel_info[4][1:]
                    ch_color_l = channel_info[5][1:]

                    channel_info_l = [pos_n, ch_name_l, ch_color_l]

                exp_folder = values['export_folder']
                if len(exp_folder) == 0:
                    sg.Popup('Export folder not specified.')
                    break
                
                lambda_info_csv_file = values['lambda_csv_file']
                if len(lambda_info_csv_file) == 0:
                    import_lambda = False
                    import_lambda_l = [import_lambda]

                else:
                    import_lambda = True
                    channel_name_l, lambda_l = lambda_import(lambda_csv_file=lambda_info_csv_file)
                    import_lambda_l = [import_lambda, channel_name_l, lambda_l, lambda_info_csv_file]
                    window_result_vis['OUTPUT'].update(value='Csv file loaded:' + '\n', append=True)
                    window_result_vis['OUTPUT'].update(value=lambda_info_csv_file + '\n', append=True)
                    window_result_vis.refresh()
                    
                str_n_thr = values['str_n_thr']
                bc_cand_n = int(values['bc_cand_n'])
                bc_p_thr = float(values['bc_p_thr'])
                err_corr = values['chk_err_corr']
                non_mixed = values['chk_non_mixed']
                fit_plot = values['chk_plot_fit']
                dc_id_det(result_hdf=results_hdf, channel_info=channel_info_l, export_crop=export_crop, export_folder=exp_folder, str_n_thr=str_n_thr, bc_cand_n=bc_cand_n, bc_p_thr=bc_p_thr, gui_window=window_result_vis, err_corr = err_corr, non_mixed = non_mixed, import_lambda_l= import_lambda_l, fit_plot=fit_plot)

                window_result_vis['OUTPUT'].update(value='Results exported' + '\n', append=True)
                window_result_vis.refresh()
                
            except ValueError:
                sg.Popup('Could not proceed with processing. Check if all parameters are provided correctly.')