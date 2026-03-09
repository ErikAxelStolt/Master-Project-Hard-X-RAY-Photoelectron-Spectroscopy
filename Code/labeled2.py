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



# Known element peak energy values (in eV) from handbook
element_peaks = {
    "S 2s": 228,
    "S 2p1/2": 162, #165 egentligen
    "S 2p3/2": 161, #164 egentligenlll
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
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = sg.butter(order, normal_cutoff, btype='low', analog=False)
    return sg.filtfilt(b, a, data)



def get_rows(file):
    """"
    Returns the number of rows 
    """
    if filenames is filenames1 or filenames is filenames4:
        rader = np.array([[50, 1911], [2008, 221], [2276, 381], [2704, 251], [3002, 171], [3220, 251]])
    elif filenames is filenames2:
        rader = np.array([[54, 1911], [2016, 221], [2288, 381], [2720, 251], [3022, 171], [3244, 251] ])
    else:
        if file == 'S3_1H_0001.txt' or file == 'S3_3_0001.txt' or file == 'S3_3H_0001.txt':
            rader = np.array([[50, 1911], [2008, 221], [2276, 381], [2704, 251], [3002, 171], [3220, 251]])
        elif file == 'S3_2H_0001.txt' or file == 'S3_2_0001.txt' or file == 'S3_10001.txt':
            rader = np.array([[54, 1911], [2016, 221], [2288, 381], [2720, 251], [3022, 171], [3244, 251]])

    return rader
        
        

def get_all_full_spectra(measurement):
    """
    load in the full spectra for specified measurement
    """

    all_first_measurements = {}

    for file in filenames:
        rader = get_rows(file)
        data = np.loadtxt(file, skiprows=rader[measurement, 0], max_rows=rader[measurement, 1]).T
        #data_filt = highpass_filter(data[1])
        all_first_measurements[file] = data
        print(f'File {file} done.')

    return all_first_measurements


 # ======================================== plot functions for inspection ========================================

def plot_all():
    """
    Plots all the measurements in the files
    """

    for filename in filenames:

        rader = get_rows(filename)

        for i in range(antal_matningar):
            data = np.loadtxt(filename, skiprows=rader[i, 0], max_rows=rader[i, 1]).T
            data_filt = highpass_filter(data[1])
            yrange = np.max(data[1]) - np.min(data[1])
            
            peaks = sg.find_peaks(data_filt, height=yrange * 0.01, distance=10, prominence=5e4)
            x_indices = peaks[0]
            
            plt.figure()
            plt.plot(data[0], data[1] * 1e-6, linewidth=1)
            plt.scatter(data[0, x_indices], data[1, x_indices] * 1e-6, color='red', s=5)
            
            offset = np.max(data[1]) * 1e-6 * 1e-2
            for index in x_indices:
                energy = data[0, index]
                label = f'{energy:.0f} eV'
                
                # Identify potential element matches
                for element, peak_energy in element_peaks.items():
                    if abs(energy - peak_energy) < 2.0:
                        label += f' ({element})'
                        break
                
                plt.text(data[0, index], data[1, index] * 1e-6 + offset, label,
                        horizontalalignment='right', size='small')
            
            plt.ylim(np.min(data[1]) * 1e-6 * 0.9, np.max(data[1]) * 1.1e-6)
            plt.title(f'{filename[:-4]} Measurement {i+1}')
            plt.gca().invert_xaxis()
            plt.xlabel('Energy (eV)')
            plt.ylabel('Electron Count ($10^6$)')
            plt.show()
            
        print(f'File {filename} done.')



def plot3d(all_data):
    """
    plts a 3D surface plot of the measurements
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i, (filename, data) in enumerate(all_data.items()):
        ax.plot(data[0], np.full_like(data[0], i), data[1] * 1e-6, label=labels[i])
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Measurement')
    ax.set_zlabel('Electron Count ($10^6$)')
    ax.set_title('3D Surface of Measurements')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)  
    ax.invert_xaxis()
    plt.show()

    

def plot_heatmap(all_data):
    """
    Plots a heatmap of the measurements
    """

    energy_values = None  # To store common x-axis values
    measurement_indices = list(range(len(all_data)))
    intensity_matrix = []
    
    for i, (filename, data) in enumerate(all_data.items()):
        if energy_values is None:
            energy_values = data[0]  # Use the first file's energy values as reference
        intensity_matrix.append(data[1] * 1e-6)  # Normalize intensity
    
    intensity_matrix = np.array(intensity_matrix)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(intensity_matrix, aspect='auto', cmap='viridis', origin='lower',
               extent=[energy_values[0], energy_values[-1], 0, len(all_data)], interpolation='nearest')
    plt.colorbar(label='Electron Count ($10^6$)')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Measurement')
    plt.title('Heatmap of Measurements')
    plt.gca().invert_xaxis()  # Match previous plot style
    plt.yticks(ticks=np.arange(len(all_data)), labels=labels)
    plt.show()







 #======================================== fit functions ========================================
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

    pars['peak2_gamma'].set(expr='peak1_gamma')  # forces same gamma
    pars['peak2_sigma'].set(expr='peak1_sigma')  # forces same gamma



    # Fit the model to data
    result = model.fit(y, pars, x=x)

    # Extract the individual peak fits
    peak1_fit = voigt1.eval(result.params, x=x)
    peak2_fit = voigt2.eval(result.params, x=x)


    int1 = result.best_values['peak1_amplitude']
    int2 = result.best_values['peak2_amplitude']
    ratio = int1 / int2
    int_sum = (int1 + int2)/1e6



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
        "int_sum": int_sum,
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


    # Peak 1 (e.g., Sulfur at 164 eV)
    pars['peak1_amplitude'].set(value=np.max(y) * 0.8, min=np.max(y) * 0.2)
    pars['peak1_center'].set(value=peak_positions1[0], min=peak_positions1[0]-0.2, max=peak_positions1[0]+0.2)
    pars['peak1_sigma'].set(value=0.5, min=0.3, max=0.4)
    #pars['peak1_gamma'].set(value=1, min=0.1, max=1.0)

    # Peak 2 (e.g., sulfur at 161 eV)
    #pars['peak2_amplitude'].set(value=np.max(y) * 0.6, min=np.max(y) * 0.2)
    pars['peak2_amplitude'].set(expr='peak1_amplitude * 0.5')# constrain intensity 2:1
    pars['peak2_center'].set(expr='peak1_center + %f' % delta_E1) # enforce delta_E separation
    #pars['peak2_sigma'].set(value=0.5, min=0.01, max=1)
    pars['peak2_sigma'].set(expr='peak1_sigma')

    # pars['peak2_gamma'].set(value=1, min=0.1, max=1.0)


    pars['peak3_amplitude'].set(value=np.max(y) * 0.4, min=np.max(y) * 0.1)
    pars['peak3_center'].set(value=peak_positions2[0], min=peak_positions2[0]-0.2, max=peak_positions2[0]+0.2)
    pars['peak3_sigma'].set(value=0.5, min=0.1, max=0.4)

    pars['peak4_amplitude'].set(expr='peak3_amplitude * 0.5')# constrain intensity 2:1
    pars['peak4_center'].set(expr='peak3_center + %f' % delta_E2) # enforce delta_E separation
    # pars['peak4_sigma'].set(value=0.5, min=0.1, max=0.4)
    pars['peak4_sigma'].set(expr='peak3_sigma')


    # Fit the model to data
    result = model.fit(y, pars, x=x)

    # Extract the individual peak fits
    peak1_fit = voigt1.eval(result.params, x=x)
    peak2_fit = voigt2.eval(result.params, x=x)
    peak3_fit = voigt3.eval(result.params, x=x)
    peak4_fit = voigt4.eval(result.params, x=x)


  
    int1 = result.best_values['peak1_amplitude']
    int2 = result.best_values['peak2_amplitude']
    int_sum1 = (int1 + int2)/1e6

    int3 = result.best_values['peak3_amplitude']
    int4 = result.best_values['peak4_amplitude']
    int_sum2 = (int3 + int4)/1e6

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
        "int_sum1": int_sum1,
        "int_sum2": int_sum2,
        "parameters": result.best_values
    }



#= ======================================== plot functions for the report ========================================


def superpose_plot(spectra, element):
    """
    Superimposes all spectra in a single plot for phase comparison.
    """


    # Find global min and max to ensure uniform scaling
    global_min = min(spectra[sample][1].min() for sample in filenames)
    global_max = max(spectra[sample][1].max() for sample in filenames)
    global_range = global_max - global_min  # Total intensity range

    if element == 'silver':
        buffert = 0.25e6
        scaling = 1e6
        scaling_label = "$10^6$"
    elif element == 'svavel':
        buffert = 0.25e5 
        scaling = 1e4
        scaling_label = "$10^4$"
    elif element == 'full':
        buffert = 0.25e5
        scaling = 1e5
        scaling_label = "$10^5$"
    else:
        raise ValueError("Choose valid nknown element for superpose plot.")

    
    offset = global_max + buffert

    plt.figure(figsize=(8, 6))  

    y_ticks = []  
    y_tick_labels = []  


    for index, sample in enumerate(filenames):
        x = spectra[sample][0]  # Energy values
        y = spectra[sample][1]  # Intensity values

        baseline = index * offset  # Define new baseline for each plot
        
        # Normalize intensity values to ensure consistent scaling across all plots
        adjusted_y = y + baseline #(y - global_min) / global_range * offset + baseline

        plt.plot(x, adjusted_y, '-', label=f'{labels[index]}')  # Plot spectrum
        
        plt.axhline(baseline, color='gray', linestyle='dashed', linewidth=0.5)  # Baseline reference line

        # Add "0" tick at baseline
        y_ticks.append(baseline)
        y_tick_labels.append("0")

        # Add a few extra y-ticks for real intensity values (cleaned up)
        y_step = offset / 4  # Define a step for y-ticks to avoid clutter
        for i in range(1, 4):  # Add 3 more tick levels per plot
            tick_value = baseline + i * y_step # tick values that we will display
            y_ticks.append(tick_value)
            
            real_value = i * y_step / 4 # real values that are on the y axis
            y_tick_labels.append(f"{real_value / scaling:.2f}")  


    plt.gca().invert_xaxis()  # Reverse x-axis
    plt.xlabel("Energy (Ev)", fontsize=12)
    plt.ylabel(f"Intensity (Counts x {scaling_label})", fontsize=12)
    plt.title(f"Intensity vs Energy for {element}", fontsize=14)
    
    plt.yticks(y_ticks, y_tick_labels)  # Set custom y-axis ticks
    plt.legend()
    plt.show()




def superpose(spectra, element):
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





# ========================================= get  and plot ratios ========================================

def plot_svavel_4_ints(int_sums1, int_sums2, element):
    """
    given two lists of intensities it plits them
    """

    plt.plot(labels, int_sums1, '-o', label="Stora toppen i svavel int_tot")
    plt.plot(labels, int_sums2, '-o', label='Lilla toppen i svavel int_tot')

    plt.legend()
    plt.xlabel("Measurement", fontsize=12)
    plt.ylabel("I_tot (Counts $10^6$)", fontsize=12)
    plt.title(f"Intensitet vs Measurement for {element}", fontsize=14)

    plt.show()



def plot_ratios(ratios, element):
    """
    given ratio list and elements it plots the relation
    """

    plt.plot(labels, ratios, '-o') #ratios agrees with order of labels in get_all_ratios

    plt.legend()
    plt.xlabel("Measurement", fontsize=12)
    plt.ylabel("Ratio", fontsize=12)
    plt.title("Ratio of integrals", fontsize=14)

    plt.show()


def get_all_ratios(spectra, element, plot=True):
    """
    gets all the ratios given a spectra for an element
    """
    ratios = []

    for sample in filenames:
        fit_results = voigt(spectra[sample], element, plot)
        ratios.append(fit_results['ratio'])

    return ratios




def get_ag_S_ratios(element1, element2, plot=True):
    """
    gets all the ratios given a spectra for an element
    """
    ratios = []

    spectra_ag = get_all_full_spectra(1)
    spectra_S = get_all_full_spectra(2)

    for sample in filenames:
        fit_results_ag = voigt(spectra_ag[sample], element1, plot)
        fit_results_S = voigt(spectra_S[sample], element2, plot)

        if element2 == "svavel":
            ratios.append(fit_results_ag['int_sum'] / fit_results_S['int_sum'])
        elif element2 == "svavel_4":
            ratios.append(fit_results_ag['int_sum'] / fit_results_S['int_sum2'])
        else:
            None

    return ratios


# ========================================= Voigt calling function ========================================

def voigt(spectra, element, plot):
    """
    kurvanpassar två eller fyra voigt toppar givet inputsen
    """

    positions1, positions2, x_range, y_range, c, slope, delta_E1, delta_E2 = set_parameters(element) 
    
    x_data = spectra[0]  # Energy values (eV)
    y_data = spectra[1]  # Intensity values


    # Apply filtering to keep only values within range
    mask = (x_data >= x_range) & (x_data <= y_range)
    filtered_x = x_data[mask]  # energy values
    filtered_y = y_data[mask]  # intensities


    if element == 'svavel' or element == 'silver':
        fit_results = fit_two_voigt_peaks(element, filtered_x, filtered_y, c, slope, positions1, delta_E1, plot)
    elif element == 'svavel_4':
        fit_results = fit_four_voigt_peaks(element, filtered_x, filtered_y, c, slope, positions1, positions2, delta_E1, delta_E2, plot)
    else:
        None

    return fit_results


def get_svavel_4_ints(spectra, element, plot):
    """
    gets all the int_sums given a spectra for an element
    """
    int_sums1 = []
    int_sums2 = []

    for sample in filenames:
        fit_results = voigt(spectra[sample], element, plot)
        int_sums1.append(fit_results['int_sum1'])
        int_sums2.append(fit_results['int_sum2'])

    return int_sums1, int_sums2




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
    
    elif element == "svavel":
        positions1 = [162.2, 161.1]

        positions2 = None

        x_range = 158
        y_range = 165
        c = 0.1e6
        slope = -0.1e6
        delta_E1 = 1.18

        delta_E2 = None
    

    elif element == "svavel_4":
        positions1 = [161.2, 162.3]
        positions2 = [160.7, 162.5]
        x_range = 158
        y_range = 165
        c = 2.36e5
        slope = 0
        delta_E1 = 1.1
        delta_E2 = 1.7
    
    else:
        return None

    
    return positions1, positions2, x_range, y_range, c, slope, delta_E1, delta_E2



#=================================== Main ========================================

measurement = 2  # 0:full spectra, 1 silver (inzoomad), 2:svavel (inzomad)
element = 'svavel' #can also use get_element(measurement) 'svavel' or 'silver' or 'svavel_4'
#sample = "A1.txt"
filenames = filenames3 #filenames1, filenames2, filenames3, filenames4 depending on series 
labels = [i.replace(".txt", "") for i in filenames]



spectra = get_all_full_spectra(measurement)


# plot3d(spectra)
superpose(spectra, element)


# int_sums1, int_sums2 = get_svavel_4_ints(spectra, element, False)
# plot_svavel_4_ints(int_sums1, int_sums2, element)


ratios = get_all_ratios(spectra, element, True) # only makes sense for sulfur
plot_ratios(ratios, element) 


#ratios = get_ag_S_ratios('silver', 'svavel_4', False)
#plot_ratios(ratios, element) 


#fit_results = voigt(spectra[sample], element, False) #fits a single measurement sample viogt profile
ratios = get_all_ratios(spectra, element, True)
plot_ratios(ratios, element)


# ratios = get_all_ratios(spectra, element, True) # ger bara 2 efter låst förhållande 
#plot_ratios(ratios, element)


# plot_all()
# plot_heatmap(spectra)


## 4 för tvåan med nones???

# ----------------------------------------information
"""

* Alla H har samma mängd fast med heating
* A,B,C refererar till slot
----S1---------
1: 1,6 H2S
2: 40 H2S
3: 8,0 H2S
----S2----------
1: 
2: 
3: 
----S3---------
1: 
2: 
3: 

"""
