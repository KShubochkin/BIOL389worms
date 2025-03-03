import ActiveZoneAnalysis as a
import pandas as pd

control_files = ["C:/Users/corna/Downloads/389 worms - control1rawxlsx.csv",
                 "C:/Users/corna/Downloads/389 worms - control2rawxlsx.csv",
                 "C:/Users/corna/Downloads/389 worms - control3rawxlsx.csv",
                 "C:/Users/corna/Downloads/389 worms - control4rawxlsx.csv",
                 "C:/Users/corna/Downloads/389 worms - control5rawxlsx.csv"]
for file in control_files:
    df, e_locations, e_times, r_locations, total_pumps, sample_rate, e_e_ipi_threshold, r_e_ipi_threshold = a.read_data(file)
    a.find_active_pump_rate(e_locations, r_locations,r_e_ipi_threshold)
    a.plot_active_inactive_timeline()
    a.plot_raster(e_times)
