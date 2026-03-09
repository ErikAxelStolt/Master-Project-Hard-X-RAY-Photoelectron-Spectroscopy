import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from mpl_toolkits.mplot3d import Axes3D

import lmfit
from lmfit.models import VoigtModel
from lmfit import Model



#======================================== initialization ========================================

filenames1 = ['A1.txt', 'B2.txt', 'C3.txt', 'A1H.txt', 'B2H.txt', 'C3H.txt']
filenames2 = ['S2_1H0001.txt', 'S2_2H0001.txt', 'S2_3H0001.txt', 'S2_10001.txt', 'S2_20001.txt', 'S2_30001.txt']
filenames3 = ['S3_1H_0001.txt', 'S3_3H_0001.txt', 'S3_2H_0001.txt', 'S3_10001.txt', 'S3_2_0001.txt']#, 'S3_3_0001.txt'] #  last really bad sulfur
filenames4 = ['S4_1_0001.txt']


files = ['A1.txt', 'B2.txt', 'C3.txt', 'S2_10001.txt', 'S2_20001.txt', 'S2_30001.txt', 'S3_10001.txt', 'S3_2_0001.txt', 'S4_1_0001.txt']
files_heating = ['A1H.txt', 'B2H.txt', 'C3H.txt', 'S2_1H0001.txt', 'S2_2H0001.txt', 'S2_3H0001.txt', 'S3_1H_0001.txt', 'S3_3H_0001.txt', 'S3_2H_0001.txt']

# map name to H2S amount
h2s_map = {
    'A1.txt': 1.6,
    'A1H.txt': 1.6,
    'B2.txt': 40,
    'B2H.txt': 40,
    'C3.txt': 8,
    'C3H.txt': 8,

    'S2_10001.txt': 1.1,
    'S2_20001.txt': 15,
    'S2_30001.txt': 25,
    'S2_1H0001.txt': 1.1,
    'S2_2H0001.txt': 15,
    'S2_3H0001.txt': 25,

    'S3_10001.txt': 2.2,
    'S3_2_0001.txt': 3,
    'S3_3_0001.txt': 4,
    'S3_1H_0001.txt': 2.2,
    'S3_2H_0001.txt': 3,
    'S3_3H_0001.txt': 4,

    'S4_1_0001.txt': 0
}

# Known element peak energy values (in eV) from handbook
element_peaks = {
    "S 2s": 228,
    "S 2p1/2": 162, #165 egentligen
    "S 2p3/2": 161, #164 egentligen
    "S 3s": 18,

    "Ag 3d5/2": 368,
    "Ag 3d3/2": 374,
    "Ag 3p3/2": 573.0,
    "Ag 3p1/2": 604.0,
    "Ag 3s": 719,
    "Ag Auger": 392,

    "C 1s": 285,

    "O 1s": 531.0,
    "O 2s": 23,

    "Si 2s": 151.0,
    "Si 2p": 99,

    "Sn 4d": 25,

    "Br 4d": 5
}

antal_matningar = 6



# ========================================= filter functions ========================================
# High-pass filter function
def highpass_filter(data, cutoff=0.1, fs=1.0, order=3):
    """
    highpass filter function for smoothing the data
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = sg.butter(order, normal_cutoff, btype='low', analog=False)
    return sg.filtfilt(b, a, data)



def get_rows(file):
    """"
    Returns the relevant rows for the data spectra for eacg file 
    """
    # if filenames is filenames1 or filenames is filenames4:
    #     rader = np.array([[50, 1911], [2008, 221], [2276, 381], [2704, 251], [3002, 171], [3220, 251]])
    # elif filenames is filenames2:
    #     rader = np.array([[54, 1911], [2016, 221], [2288, 381], [2720, 251], [3022, 171], [3244, 251] ])
    # else:
    #     if file == 'S3_1H_0001.txt' or file == 'S3_3_0001.txt' or file == 'S3_3H_0001.txt':
    #         rader = np.array([[50, 1911], [2008, 221], [2276, 381], [2704, 251], [3002, 171], [3220, 251]])
    #     elif file == 'S3_2H_0001.txt' or file == 'S3_2_0001.txt' or file == 'S3_10001.txt':
    #         rader = np.array([[54, 1911], [2016, 221], [2288, 381], [2720, 251], [3022, 171], [3244, 251]])

    # return rader


    if file in ['A1.txt', 'B2.txt', 'C3.txt', 'A1H.txt', 'B2H.txt', 'C3H.txt', 'S4_1_0001.txt', 'S3_1H_0001.txt', 'S3_3_0001.txt', 'S3_3H_0001.txt']:
        rader = np.array([
            [50, 1911], [2008, 221], [2276, 381],
            [2704, 251], [3002, 171], [3220, 251]
        ])
    
    elif file in ['S2_1H0001.txt', 'S2_2H0001.txt', 'S2_3H0001.txt', 'S2_10001.txt', 'S2_20001.txt', 'S2_30001.txt', 'S3_2H_0001.txt', 'S3_2_0001.txt', 'S3_10001.txt']:
        rader = np.array([
            [54, 1911], [2016, 221], [2288, 381],
            [2720, 251], [3022, 171], [3244, 251]
        ])
    
    else:
        raise ValueError(f"Unknown file format or missing row data: {file}")

    return rader



def get_all_full_spectra(filenames, measurement):
    """
    load in the full spectra for specified measurement
    """

    all_first_measurements = {}

    for file in filenames:
        rader = get_rows(file)
        data = np.loadtxt(file, skiprows=rader[measurement, 0], max_rows=rader[measurement, 1]).T
        #data_filt = highpass_filter(data[1]) # optioonal 
        all_first_measurements[file] = data
        print(f'File {file} done.')

    return all_first_measurements


 #======================================== Fit Functions ========================================
def line(x, slope, intercept):
    """a line"""
    return slope*x + intercept


def fit_two_voigt_peaks(element, x, y, c, slope, peak_positions, delta_E, plot=True):
    """
    fits 2 viogt profiles based on input parameters. Useful for silver which has 2 distinct peaks
    """

    # Define two independent Voigt models
    voigt1 = VoigtModel(prefix='peak1_')
    voigt2 = VoigtModel(prefix='peak2_')
    

    # Combine both models
    model = voigt1 + voigt2 + Model(line)

    # Initial parameter guesses
    pars = model.make_params(amp=5, cen=5, wid={'value': 1, 'min': 0}, slope=slope, intercept=c)


    # Peak 1 (e.g., Sulfur at 164 eV)
    pars['peak1_amplitude'].set(value=np.max(y) * 1, min=np.max(y) * 0.8)
    pars['peak1_center'].set(value=peak_positions[0], min=peak_positions[0]-2, max=peak_positions[0]+2)
    #pars['peak1_sigma'].set(value=0.5, min=0.01, max=1)
    #pars['peak1_gamma'].set(value=1, min=0.1, max=1.0)


    # Peak 2 (e.g., sulfur at 161 eV)
    if element == 'silver':
        pars['peak2_amplitude'].set(expr='peak1_amplitude * 1.5')  # For a 3d5/2 to 3d3/2 ratio of 1.5
    else:
        pars['peak2_amplitude'].set(expr='peak1_amplitude / 2')# constrain intensity
        
    
    pars['peak2_center'].set(expr='peak1_center + %f' % delta_E) # enforce delta_E separation
    pars['peak2_gamma'].set(expr='peak1_gamma') # forces same gamma to be the same
    pars['peak2_sigma'].set(expr='peak1_sigma') # forces same sigma to be the same


    # Fit the model to data
    result = model.fit(y, pars, x=x)

    # Extract the individual peak fits
    peak1_fit = voigt1.eval(result.params, x=x)
    peak2_fit = voigt2.eval(result.params, x=x)

    int1 = result.best_values['peak1_amplitude']
    int2 = result.best_values['peak2_amplitude']
    ratio = int1 / int2 
    int_sum = int1 + int2


    # Plot the results
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(x, y/1e6, 'bo', label='Data')
        plt.plot(x, result.best_fit/1e6, 'r-', label='Total Voigt Fit')
        plt.legend()
        plt.gca().invert_xaxis()
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (Counts $10^6$)")
        plt.title(f"Two-Peak Voigt Fit for {element}")
        plt.show()


    
    return {
        "total_fit": result,
        "peak1_fit": peak1_fit,
        "peak2_fit": peak2_fit,
        "integral_peak1": int1,
        "integral_peak2": int2,
        "ratio": ratio,
        "int_tot": int_sum,
        "parameters": result.best_values
    }



def fit_four_voigt_peaks(element, x, y, c, slope, peak_positions1, peak_positions2, delta_E1, delta_E2, plot=True):
    """
    fits 4 viogt profiles based on input parameters. Useful for sulfur 4 which has 2+2 peaks
    """

    # Define two independent Voigt models
    voigt1 = VoigtModel(prefix='peak1_')
    voigt2 = VoigtModel(prefix='peak2_')
    voigt3 = VoigtModel(prefix='peak3_')
    voigt4 = VoigtModel(prefix='peak4_')

    # Combine both models
    model = voigt1 + voigt2 + voigt3 + voigt4 + Model(line)

    # Initial parameter guesses
    pars = model.make_params(amp=5, cen=5, wid={'value': 1, 'min': 0}, slope=slope, intercept=c)


    # Larger peaks 
    pars['peak1_amplitude'].set(value=np.max(y) * 0.8, min=np.max(y) * 0.2)
    pars['peak1_center'].set(value=peak_positions1[0], min=peak_positions1[0]-0.2, max=peak_positions1[0]+0.2)
    pars['peak1_sigma'].set(value=0.5, min=0.01, max=0.4)
    pars['peak1_gamma'].set(value=0.5, min=0.01, max=1.0)

    pars['peak2_amplitude'].set(expr='peak1_amplitude * 0.5')# constrain intensity 2:1
    pars['peak2_center'].set(expr='peak1_center + %f' % delta_E1) # enforce delta_E separation
    pars['peak2_sigma'].set(expr='peak1_sigma')
    pars['peak2_gamma'].set(expr='peak1_gamma') 

    # Smaller peaks
    pars['peak3_amplitude'].set(value=np.max(y) * 0.4, min=np.max(y) * 0.1)
    pars['peak3_center'].set(value=peak_positions2[0], min=peak_positions2[0]-0.2, max=peak_positions2[0]+0.2)
    pars['peak3_sigma'].set(value=0.5, min=0.1, max=0.4)
    pars['peak3_gamma'].set(value=0.5, min=0.01, max=1.0)

    pars['peak4_amplitude'].set(expr='peak3_amplitude * 0.5')# constrain intensity 2:1
    pars['peak4_center'].set(expr='peak3_center + %f' % delta_E2) # enforce delta_E separation
    pars['peak4_sigma'].set(expr='peak3_sigma')
    pars['peak4_gamma'].set(expr='peak3_gamma')


    # Fit the model to data
    result = model.fit(y, pars, x=x)

    # Extract the individual peak fits
    peak1_fit = voigt1.eval(result.params, x=x)
    peak2_fit = voigt2.eval(result.params, x=x)
    peak3_fit = voigt3.eval(result.params, x=x)
    peak4_fit = voigt4.eval(result.params, x=x)

  
    int1 = result.best_values['peak1_amplitude']
    int2 = result.best_values['peak2_amplitude']
    int_sum1 = int1 + int2

    int3 = result.best_values['peak3_amplitude']
    int4 = result.best_values['peak4_amplitude']
    int_sum2 = int3 + int4

    int_tot = int_sum1 + int_sum2
    ratio = int_sum1 / int_sum2



    # Plot the results
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(x, y/1e6, 'bo', label='Data')
        plt.plot(x, result.best_fit/1e6, 'r-', label='Total Voigt Fit')
        plt.plot(x, peak1_fit/1e6, 'g--', label=f'Voigt Peak 1 ({peak_positions1[0]} eV)')
        plt.plot(x, peak2_fit/1e6, 'g--', label=f'Voigt Peak 2 ({peak_positions1[1]} eV)')
        plt.plot(x, peak3_fit/1e6, 'm--', label=f'Voigt Peak 3 ({peak_positions2[0]} eV)')
        plt.plot(x, peak4_fit/1e6, 'm--', label=f'Voigt Peak 4 ({peak_positions2[1]} eV)')
        plt.legend()
        plt.gca().invert_xaxis()
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (Counts $10^6$)")
        plt.title(f"Two-Peak Voigt Fit for {element}")
        plt.show()


    return {
        "total_fit": result,
        "peak1_fit": peak1_fit,
        "peak2_fit": peak2_fit,
        "peak3_fit": peak3_fit,
        "peak4_fit": peak4_fit,
        "integral_peak1": int1,
        "integral_peak2": int2,
        "integral_peak3": int3,
        "integral_peak4": int4,
        "ratio": ratio,
        "int_tot": int_tot,
        "int_sum1": int_sum1,
        "int_sum2": int_sum2,
        "parameters": result.best_values
    }



#= ======================================== Plot Functions For The Report ========================================


def superpose(spectra, element, labels):
    """
    Superimposes all spectra in a single plot for phase comparison with NO intensity y values.
    """

    # Find global min and max to ensure uniform scaling
    global_min = min(spectra[sample][1].min() for sample in filenames)
    global_max = max(spectra[sample][1].max() for sample in filenames)
    global_range = global_max - global_min  # Total intensity range

    if element == 'silver':
        buffert = -2e6
    elif element == 'svavel':
        buffert = -2e5 
    elif element == 'full':
        buffert = 0.25e5
    else:
        raise ValueError("Choose valid known element for superpose plot.")

    offset = global_max + buffert

    plt.figure(figsize=(8, 6))  

    for index, sample in enumerate(filenames):
        x = spectra[sample][0]  # Energy values
        y = spectra[sample][1]  # Intensity values

        baseline = index * offset  # Define new baseline for each plot
        
        adjusted_y = y + baseline  # Offset spectrum

        plt.plot(x, adjusted_y, '-', label=f'{labels[index]}')  # Plot spectrum

    plt.gca().invert_xaxis()  # Reverse x-axis
    plt.xlabel("Energy (eV)", fontsize=12)
    plt.ylabel("Intensity (Counts)", fontsize=12)  # Only label, no tick values
    plt.title(f"Intensity vs Energy for {element}", fontsize=14)

    plt.yticks([])  # Remove all y-tick values
    plt.legend()
    plt.show()


def plot_ratios_combined(filenames_normal, filenames_heating, element1, element2, plot):
    """
    Plots the ratio for both non-heated and heated samples in the same plot.
    """

    sorted_normal = sort_files_by_h2s(filenames_normal) # sorts the filenames based on H2S amount
    sorted_heating = sort_files_by_h2s(filenames_heating) 

    labels_normal = sorted([float(h2s) for h2s in file_to_h2s(sorted_normal)]) # Get the labels for the normal samples
    labels_heating = sorted([float(h2s) for h2s in file_to_h2s(sorted_heating)])

    ratios_normal = get_ratios(sorted_normal, element1, element2, plot) # Get the ratios for the normal samples
    ratios_heating = get_ratios(sorted_heating, element1, element2, plot)

    ratio_label = get_ratios_label(element1, element2)

    plt.figure(figsize=(8, 6))
    plt.plot(labels_normal, ratios_normal, '-o', label='Room Temp')
    plt.plot(labels_heating, ratios_heating, '-o', label='Heated')

    plt.xlabel("H$_2$S", fontsize=12)
    plt.ylabel(f"Ratio ({ratio_label})", fontsize=12)
    plt.title(f"{ratio_label} vs H$_2$S", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================== Get And Plot Ratios ===============================



def plot_ratios(ratios, ratio_label, heating):
    """
    given ratio list and elements it plots the relation
    """

    plt.plot(labels, ratios, '-o') #ratios agrees with order of labels in get_all_ratios

    plt.legend()
    plt.xlabel("H$_2$S", fontsize=12)
    plt.ylabel(f"Ratio ({ratio_label})", fontsize=12)

    if heating:
        plt.title("Ratio Vs H$_2$S With Heating", fontsize=14)
    else:
        plt.title("Ratio Vs H$_2$S Without Heating", fontsize=14)

    plt.show()



def get_ratios(filenames, element1, element2, plot):

    """
    Given the filenames and elements it returns the ratios for the elements
    """

    ratios = []

    if element1 == "silver": 
        spectra1 = get_all_full_spectra(filenames, 1) 
    elif element1 == 'svavel':
        spectra1 = get_all_full_spectra(filenames, 2)
    else:
        raise ValueError("unknown ratio for element1")
    
    if element2 == 'silver':
        raise ValueError("unknown ratio")
    elif element2 == 'svavel_bound' or element2 == 'svavel_free':
        spectra2 = get_all_full_spectra(filenames, 2) 
    else:
        raise ValueError("unknown ratio for element2")
    

    for sample in filenames:
        fit_results1 = voigt(spectra1[sample], element1, plot)
        fit_results2 = voigt(spectra2[sample], element2, plot)

        if element1 == "silver" and element2 == "svavel_bound":
            ratios.append(fit_results1['int_tot'] / fit_results2['int_sum1'])
        elif element1 == "silver" and element2 == "silver":
            raise ValueError("unknown ratio")
        elif element1 == "svavel" and element2 == "svavel_bound":
            ratios.append(fit_results1['int_tot'] / fit_results2['int_sum1'])
        elif element1 == "svavel" and element2 == "svavel_free":
            ratios.append(fit_results1['int_tot'] / fit_results2['int_sum2'])
        else:
            raise ValueError("unknown ratio")

    return ratios

# ========================================= Voigt calling function ========================================

def voigt(spectra, element, plot):
    """
    Curve fitting function for Voigt profile. It takes the spectra and element as input and returns the fit results.
    """

    positions1, positions2, x_range, y_range, c, slope, delta_E1, delta_E2 = set_parameters(element) 
    
    x_data = spectra[0]  # Energy values (eV)
    y_data = spectra[1]  # Intensity values


    # Apply filtering to keep only values within range
    mask = (x_data >= x_range) & (x_data <= y_range)
    filtered_x = x_data[mask]  # energy values
    filtered_y = y_data[mask]  # intensities


    if element == 'silver':# or element == 'silver':
        fit_results = fit_two_voigt_peaks(element, filtered_x, filtered_y, c, slope, positions1, delta_E1, plot)
    elif element == 'svavel_4' or element == 'svavel' or element == 'svavel_bound' or element == 'svavel_free':
        fit_results = fit_four_voigt_peaks(element, filtered_x, filtered_y, c, slope, positions1, positions2, delta_E1, delta_E2, plot)
    else:
        raise ValueError(f"Unknown element: {element}")

    return fit_results


# ================================== set parameters ========================================
def set_parameters(element):
    """
    sets the parameters to voigt fit accordingly
    """

    if element == "silver":
        positions1 = [374, 367]

        positions2 = None
        
        x_range = 365
        y_range = 377
        c = 0.5e6
        slope = 0
        delta_E1 = -6

        delta_E2 = None
    
    # elif element == "svavel":
    #     positions1 = [162.2, 161.1]

    #     positions2 = None

    #     x_range = 158
    #     y_range = 165
    #     c = 0.1e6
    #     slope = -0.1e6
    #     delta_E1 = 1.18

    #     delta_E2 = None
    

    elif element == "svavel" or element == "svavel_free" or element == "svavel_bound":
        positions1 = [161.2, 162.3]
        positions2 = [160.7, 162.5]
        x_range = 158
        y_range = 165
        c = 2.36e5
        slope = 0
        delta_E1 = 1.1
        delta_E2 = 1.7
    
    else:
        raise ValueError(f"Unknown element: {element}")

    
    return positions1, positions2, x_range, y_range, c, slope, delta_E1, delta_E2



def get_reference_peak(element):
    """
    returns the reference peak for the element
    """
    if element == 'silver':
        return 375
    elif element == 'svavel':
        return 162
    elif element == 'svavel_4':
        return 161
    else:
        raise ValueError(f"Unknown element: {element}")


def get_heating(filenames):
    """
    returns True if filenames contains heating, False otherwise
    """
    for file in filenames:
        if 'H' in file:
            return True
    return False

def get_ratios_label(element1, element2):
    """
    returns the label for the ratio plot
    """
    if element1 == 'silver' and element2 == 'svavel_bound':
        return "Ag/Ag$_2$S"
    elif element1 == 'silver' and element2 == 'silver':
        return "unknown"
    elif element1 == 'svavel' and element2 == 'svavel_bound':
        return "S/Ag$_2$S"
    elif element1 == 'svavel' and element2 == 'svavel_free':
        return "S/S-S"
    else:
        raise ValueError(f"Unknown elements: {element1}, {element2}")


# ================================== Data Processing ========================================

def add_phase(spectra, element, reference_peak):
    """
    shifts the spectra to align with the reference peak
    """

    shifted_spectra = {}

    for sample, (x, y) in spectra.items():
        try: # try to get parameters for the element
            positions1, _, x_range, y_range, _, _, _, _ = set_parameters(element)
        except:
            print(f"Skipping {sample}: no parameters for element {element}")
            continue

        mask = (x >= x_range) & (x <= y_range) # Filter x and y based on the range
        x_filtered = x[mask]
        y_filtered = y[mask]

        if len(y_filtered) <= 12:
            print(f"{sample}: Not enough data points for filtering. Skipping.")
            continue

        y_filt = highpass_filter(y_filtered)
        peaks, _ = sg.find_peaks(y_filt, prominence=1e4)

        if len(peaks) == 0:
            print(f"{sample}: No peaks found. Skipping.")
            continue

        peak_index = peaks[np.argmax(y_filtered[peaks])] # Find the index of the peak with the highest intensity
        peak_energy = x_filtered[peak_index]

        shift = reference_peak - peak_energy
        shifted_x = x + shift

        shifted_spectra[sample] = (shifted_x, y)
        print(f"{sample}: Peak at {peak_energy:.2f} eV shifted by {shift:.2f} eV")

    return shifted_spectra


def file_to_h2s(filenames):
    """
    Returns list of new labels based on amount of H2S for each file.
    """
    labels = []

    for filename in filenames:
        h2s_amount = h2s_map.get(filename) # Get the H2S amount from the map
        if h2s_amount is not None:
            labels.append(f"{h2s_amount}") # Append the H2S amount to the label
        else:
            labels.append("Unknown H₂S")

    return labels


def sort_files_by_h2s(filenames):
    """
    returns sorted list of filenames based on H₂S amount.
    """

    known_files = [f for f in filenames if f in h2s_map] # Filter out unknown files
    sorted_files = sorted(known_files, key=lambda f: h2s_map[f]) # Sort based on H2S amount

    return sorted_files

#=================================== Main ========================================

# ------------ input parameters ----------------------------------------------------------------

measurement = 2                           # 0:full spectra, 1 silver (inzoomad), 2:svavel (inzomad)
element = 'svavel'                      # have to match with the measurement e.g. silver of measurement 1, svavel/svavel_4 for 2

reference = get_reference_peak(element) # 375 for silver, 162 for svavel, 161 for svavel_4
filenames = files_heating                       #filenames1, filenames2, filenames3, filenames4 depending on series 
heating = get_heating(filenames)        # True if filenames contains heating, False otherwise

sorted_filenames = sort_files_by_h2s(filenames)
labels = file_to_h2s(sorted_filenames)

plot = False

# ----------------------------------------------------------------------------------------------


spectra = get_all_full_spectra(sorted_filenames, measurement)
spectra = add_phase(spectra, element, reference)


# superpose(spectra, element, labels)


# ================================= Ratios ========================================
# Ag/Ag2S, Ag/Ag-Ag, S/Ag2S och S/S-S.

# silver, svavel_bound, svavel_free, svavel=total svavel
element1 = 'silver'
element2 = 'svavel_bound' 
ratio_label = get_ratios_label(element1, element2)


# ratios = get_ratios(filenames, element1, element2, plot) 
# plot_ratios(ratios, ratio_label, heating) 

# ============================= Combined Ratios ===================================
plot_ratios_combined(files, files_heating, element1, element2, plot)












