from testbenches.sram_6t_core_testbench import Sram6TCoreTestbench
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# def set_default():
plt.rcParams.update({
    'font.family': 'serif',           # 设置字体族
    'font.sans-serif': 'Century',
    'font.size': 22,                  # 基础字体大小
    'axes.labelsize': 22,             # 轴标签字体大小
    'axes.titlesize': 22,             # 标题字体大小
    'xtick.labelsize': 22,            # x轴刻度标签大小
    'ytick.labelsize': 22,            # y轴刻度标签大小
    'legend.fontsize': 22,            # 图例字体大小
    'figure.figsize': [8, 8],         # 图形大小
    'figure.dpi': 300,                # 分辨率
})

# Change default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#d62728', '#ff7f0e', '#2ca02c',
                                                    '#1f77b4', '#9467bd', '#8c564b',
                                                    '#e377c2', '#000000', '#17becf',
                                                    '#808080'])

# Or use built-in style sheets
plt.style.use('ggplot')  # Options: 'seaborn', 'fivethirtyeight', 'dark_background', etc.

def plot_delay(row, r_delay_mean, r_delay_std, w_delay_mean, w_delay_std,
               labelr, labelw, figname, ylim_b=0.05):
    # set_default()

    # Convert to nanoseconds for better readability
    r_delay_mean_ns = [x * 1e9 for x in r_delay_mean]
    r_delay_std_ns = [x * 3e9 for x in r_delay_std]
    w_delay_mean_ns = [x * 1e9 for x in w_delay_mean]
    w_delay_std_ns = [x * 3e9 for x in w_delay_std]

    # Set up the figure with a specified size
    plt.figure()

    # Create the plot with error bars for read delay
    plt.errorbar(
        row, 
        r_delay_mean_ns, 
        yerr=r_delay_std_ns, 
        fmt='o-', 
        linewidth=2, 
        capsize=6, 
        capthick=2, 
        markersize=8,
        color='#1f77b4',
        ecolor='#ff7f0e',
        label=labelr,
    )

    # Add write delay data to the same plot
    plt.errorbar(
        row, 
        w_delay_mean_ns, 
        yerr=w_delay_std_ns, 
        fmt='s-', 
        linewidth=2, 
        capsize=6, 
        capthick=2, 
        markersize=8,
        color='#2ca02c',
        ecolor='#d62728',
        label=labelw,
    )

    # Set labels and title
    plt.xlabel('Row Size', fontsize=24)
    plt.ylabel('Delay (ns)', fontsize=24)
    # plt.title('SRAM Read and Write Delay vs Row Size')

    # Add legend
    plt.legend(frameon=False)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=1.0)

    # Customize the tick parameters - removing font size settings
    plt.xticks()
    plt.yticks()

    # Add a light gray background to highlight the plot area
    # plt.gca().set_facecolor('#f8f8f8')

    # Improve layout
    plt.tight_layout()
    # Option 1: Adjust subplot parameters directly # Increase left margin
    plt.subplots_adjust(left=0.16)

    # Show log scale on y-axis to better display both datasets
    plt.yscale('log')
    plt.ylim(bottom=ylim_b)  # Start y-axis slightly above 0

    # Add a subtle box around the plot
    # plt.box(True)

    # Display the plot
    plt.show()

    plt.savefig(f'plots/{figname}.pdf')

    
def plot_power(row, r_pavg_mean, r_pavg_std, w_pavg_mean, w_pavg_std, 
               labelr, labelw, figname):
    # Convert to microwatts (μW) for better readability
    r_pavg_mean_uw = [x * 1e6 for x in r_pavg_mean]
    r_pavg_std_uw = [x * 3e6 for x in r_pavg_std]
    w_pavg_mean_uw = [x * 1e6 for x in w_pavg_mean]
    w_pavg_std_uw = [x * 3e6 for x in w_pavg_std]

    # Set up the figure
    plt.figure()

    # Create the plot with error bars for read power
    plt.errorbar(
        row, 
        r_pavg_mean_uw, 
        yerr=r_pavg_std_uw, 
        fmt='o-', 
        linewidth=2, 
        capsize=6, 
        capthick=2, 
        markersize=8,
        color='#1f77b4',
        ecolor='#ff7f0e',
        label=labelr
    )

    # Add write power data to the same plot
    plt.errorbar(
        row, 
        w_pavg_mean_uw, 
        yerr=w_pavg_std_uw, 
        fmt='s-', 
        linewidth=2, 
        capsize=6, 
        capthick=2, 
        markersize=8,
        color='#2ca02c',
        ecolor='#d62728',
        label=labelw
    )

    # Set labels and title
    plt.xlabel('Row Size', fontsize=24)
    plt.ylabel('Average Power (μW)', fontsize=24)
    # plt.title('SRAM Read and Write Power vs Row Size')

    # Add grid
    plt.grid(True, linestyle='--', alpha=1.0)

    # Add legend
    plt.legend(frameon=False)

    # Add a light gray background to highlight the plot area
    # plt.gca().set_facecolor('#f8f8f8')

    # Set y-axis to log scale since there's a large range of values
    plt.yscale('log')

    # Improve layout
    plt.tight_layout()

    # Display the plot
    plt.savefig(f'plots/{figname}.pdf')


def plot_rc_delay(row, rc_delay_mean, rc_delay_std, orc_delay_mean, orc_delay_std, 
                #   c_delay_mean, c_delay_std, 
                  labelrc, labelorc, #labelc, 
                  figname):
    # set_default()

    # Convert to nanoseconds for better readability
    rc_delay_mean_ns = [x * 1e9 for x in rc_delay_mean]
    rc_delay_std_ns = [x * 3e9 for x in rc_delay_std]
    orc_delay_mean_ns = [x * 1e9 for x in orc_delay_mean]
    orc_delay_std_ns = [x * 3e9 for x in orc_delay_std]

    # Set up the figure with a specified size
    plt.figure()

    # Create the plot with error bars for read delay
    plt.errorbar(
        row, 
        rc_delay_mean_ns, 
        yerr=rc_delay_std_ns, 
        fmt='o-', 
        linewidth=2, 
        capsize=6, 
        capthick=2, 
        markersize=8,
        color='#1f77b4',
        ecolor='#ff7f0e',
        label=labelrc,
    )

    # Add write delay data to the same plot
    plt.errorbar(
        row, 
        orc_delay_mean_ns, 
        yerr=orc_delay_std_ns, 
        fmt='s-', 
        linewidth=2, 
        capsize=6, 
        capthick=2, 
        markersize=8,
        color='#2ca02c',
        ecolor='#d62728',
        label=labelorc,
    )

    # Set labels and title
    plt.xlabel('Row Size', fontsize=24)
    plt.ylabel('Delay (ns)', fontsize=24)
    # plt.title('SRAM Read and Write Delay vs Row Size')

    # Add legend
    plt.legend(frameon=False)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=1.0)

    # Customize the tick parameters - removing font size settings
    plt.xticks()
    plt.yticks()

    # Add a light gray background to highlight the plot area
    # plt.gca().set_facecolor('#f8f8f8')

    # Improve layout
    plt.tight_layout(pad=1.5)

    # Show log scale on y-axis to better display both datasets
    plt.yscale('log')
    plt.ylim(bottom=6e-3)  # Start y-axis slightly above 0

    # Add a subtle box around the plot
    # plt.box(True)

    # Display the plot
    plt.show()

    plt.savefig(f'plots/{figname}.png')


def plot_leak_delay(row, r_delay_mean, r_delay_std, w_delay_mean, w_delay_std,
               labelr, labelw, figname, ylim_b=0.05, ylim_t=7):
    # set_default()

    # Convert to nanoseconds for better readability
    r_delay_mean_ns = [x * 1e9 for x in r_delay_mean]
    r_delay_std_ns = [x * 1e9 for x in r_delay_std]
    w_delay_mean_ns = [x * 1e9 for x in w_delay_mean]
    w_delay_std_ns = [x * 1e9 for x in w_delay_std]

    # Set up the figure with a specified size
    plt.figure()

    # Create the plot with error bars for read delay
    plt.errorbar(
        row, 
        r_delay_mean_ns, 
        yerr=r_delay_std_ns, 
        fmt='o-', 
        linewidth=2, 
        capsize=6, 
        capthick=2, 
        markersize=8,
        color='#1f77b4',
        ecolor='#ff7f0e',
        label=labelr,
        alpha=0.7,
    )

    # Add write delay data to the same plot
    plt.errorbar(
        row, 
        w_delay_mean_ns, 
        yerr=w_delay_std_ns, 
        fmt='s-', 
        linewidth=2, 
        capsize=6, 
        capthick=2, 
        markersize=8,
        color='#2ca02c',
        ecolor='#d62728',
        label=labelw,
        alpha=0.7,
    )

    # Set labels and title
    plt.xlabel('VDD (V)', fontsize=24)
    plt.ylabel('Delay (ns)', fontsize=24)
    # plt.title('SRAM Read and Write Delay vs Row Size')

    # Add legend
    plt.legend(frameon=False)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=1.0)

    # Customize the tick parameters - removing font size settings
    plt.xticks()
    plt.yticks()

    # Add a light gray background to highlight the plot area
    # plt.gca().set_facecolor('#f8f8f8')

    # Improve layout
    plt.tight_layout()
    # Option 1: Adjust subplot parameters directly # Increase left margin
    # plt.subplots_adjust(left=0.16)

    # Show log scale on y-axis to better display both datasets
    # plt.yscale('log')
    plt.ylim(bottom=ylim_b, top=ylim_t)  # Start y-axis slightly above 0

    # Add a subtle box around the plot
    # plt.box(True)

    # Display the plot
    plt.show()

    plt.savefig(f'plots/{figname}.pdf')

if __name__ == '__main__':
    r_delay_mean = [1.386000e-10, 2.402947e-10, 4.396346e-10, 8.434029e-10, 1.646921e-09, 3.273307e-09,] 
    r_delay_std = [6.397186e-12, 1.252712e-11, 2.172452e-11, 4.576142e-11, 9.092243e-11, 1.754441e-10] 
    w_delay_mean = [9.151580e-11, 9.148292e-11, 9.118616e-11, 9.234190e-11, 9.315216e-11, 9.514691e-11]
    w_delay_std = [4.195672e-12, 4.285453e-12, 4.118874e-12, 3.783552e-12, 3.559204e-12, 2.834615e-12]
    row = ['8', '16', '32', '64', '128', '256']
    r_pavg_mean =[1.413105e-05, 2.777633e-05, 5.508502e-05, 1.094728e-04, 2.182667e-04, 4.345131e-04]
    r_pavg_std =[7.431630e-08, 1.217739e-07, 2.815874e-07, 5.487543e-07, 1.179591e-06, 2.234966e-06]
    w_pavg_mean =[2.719423e-06, 2.936455e-06, 3.302843e-06, 3.919095e-06, 5.343889e-06, 8.408444e-06]
    w_pavg_std =[1.724884e-07, 2.234122e-07, 3.574260e-07, 5.893348e-07, 1.243927e-06, 2.029968e-06]
    # plot_delay(row, r_delay_mean, r_delay_std, w_delay_mean, w_delay_std, labelr='Read', labelw='Write', figname='access_delay_vs_row')
    # plot_power(row, r_pavg_mean, r_pavg_std, w_pavg_mean, w_pavg_std, labelr='Read', labelw='Write', figname='access_power_vs_row')
    
    # assert 0
    r_delay_mean_worc = [1.396194e-11, 2.492658e-11, 4.127327e-11, 6.695017e-11, 1.048438e-10, 1.747913e-10]
    r_delay_std_worc = [4.469588e-12, 4.722821e-12, 4.492268e-12, 4.484446e-12, 5.408408e-12, 9.406319e-12]
    r_pavg_mean_worc =[1.244997e-06, 2.038084e-06, 3.661807e-06, 6.868877e-06, 1.345126e-05, 2.598651e-05]
    r_pavg_std_worc =[6.223379e-08, 1.402892e-07, 2.671098e-07, 4.934379e-07, 1.067156e-06, 2.123307e-06]
    
    # plot_delay(row, r_delay_mean, r_delay_std, r_delay_mean_worc, r_delay_std_worc, 'w/ RC', 'w/o RC', 'rc_read_vs_row', ylim_b=6e-3)
    # plot_power(row, r_pavg_mean, r_pavg_std, r_pavg_mean_worc, r_pavg_std_worc, 'w/ RC', 'w/o RC', 'rc_power_vs_row')
    
    # assert 0
    r_delay_mean_wc = [1.369211e-10, 2.385366e-10, 4.365987e-10, 8.341607e-10, 1.649365e-09, 3.271069e-09]
    r_delay_std_wc = [6.854351e-12, 1.351489e-11, 2.028318e-11, 4.481658e-11, 7.597533e-11, 1.550094e-10]
    row = ['4', '8', '16', '32', '64', '256']
    r_pavg_mean_wc =[1.414657e-05, 2.780670e-05, 5.508002e-05, 1.095496e-04, 2.182179e-04, 4.347182e-04]
    r_pavg_std_wc =[8.149906e-08, 1.557332e-07, 2.596893e-07, 5.576341e-07, 2.210737e-06, 9.947269e-07]

    # 64x4 array vs. vdd
    r0_delay_mean = [6.702204e-09, 3.851744e-09, 1.914726e-09, 1.364176e-09, 1.090848e-09, 9.309606e-10, 8.434029e-10]
    r0_delay_std = [2.207034e-09, 8.956956e-10, 2.975255e-10, 1.310010e-10, 8.723298e-11, 5.036981e-11, 4.576142e-11]
    r0_pavg_mean = [5.618400e-06, 1.094728e-04, 3.926230e-05, 5.342455e-05, 6.986006e-05, 8.853872e-05, 1.094728e-04]
    r0_pavg_std = [8.078408e-08, 5.487543e-07, 1.276220e-07, 1.837578e-07, 2.858239e-07, 3.484549e-07, 5.487543e-07]
    vdd = ['0.45', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    r1_delay_mean = [6.377010e-09, 3.943235e-09, 1.942409e-09, 1.360166e-09, 1.086421e-09, 9.387288e-10, 8.440601e-10]
    r1_delay_std = [1.960650e-09, 9.234266e-10, 2.580048e-10, 1.144108e-10, 6.582480e-11, 5.484934e-11, 3.546146e-11]
    r1_pavg_mean = [5.627842e-06, 1.371640e-05, 3.926542e-05, 5.345351e-05, 6.988124e-05, 8.857163e-05, 1.095417e-04]
    r1_pavg_std = [7.391658e-08, 7.987265e-08, 1.178377e-07, 1.716268e-07, 2.508248e-07, 3.773809e-07, 5.367693e-07]
    plot_leak_delay(
        vdd, r0_delay_mean, r0_delay_std, r1_delay_mean, r1_delay_std, 
        '0 Idle Cells', '1 Idle Cells', 'leakage_vs_vdd', 
        ylim_b=0.8, ylim_t=7)
