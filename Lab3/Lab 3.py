# %% [markdown]
# # **Lab 3**

# %% [markdown]
# ## **Task 1**

# %%
'''
For each case tabulate Re_c and Cd_o using numerical integration
'''

# Imports
import numpy as np
import pandas as pd

# Create dataframes
file_path = 'Lab_3_Data_Template_with_Data.xlsx'

df_50 = pd.read_excel(file_path, sheet_name="50 fps", usecols=[0,1,2,4], skiprows = 1)
df_100 = pd.read_excel(file_path, sheet_name="100 fps", usecols=[0,1,2,4,6], skiprows = 1)
df_150 = pd.read_excel(file_path, sheet_name="150 fps", usecols=[0,1,2,4], skiprows = 1)

for df in [df_50, df_100, df_150]:
    df["Actual Measurement (in)"] = [12.00, 11.75, 11.50, 11.25, 11.00, 10.75, 10.50, 10.25, 10.00, 9.75, 9.50, 9.38, 9.25, 9.13, 9.00, 8.88, 8.75, 8.63, 8.50, 8.25, 8.00, 7.75, 7.50, 7.25, 7.00]
    df["Relative Measurement (in)"] = df["Actual Measurement (in)"] - 9

# Constants
c = 6 * .0254 # Chord length in meters converted from inches
nu = 15.67e-6 # Kinematic viscosity in m2/s at 80F
g = 9.81 # Gravitational constant in m/s2

rho = [1.182, 1.182, 1.177] # Density in kg/m3 based on average temperature for each velocity

# Calculates Re
def calculate_re(u_atm):
    u_atm = u_atm * .3048 # Convert ft/s to m/s
    return u_atm * c / nu # Re

# Calculates downstream velocity (u2) based on recorded q
def calculate_u2(q, rho):
    q = q * 249.0889 # Convert inH20 to Pa
    return (q * 2 * rho) ** .5 # U2 in m/s

# Add Reynold's number to each dataframe
df_50["Re"] = calculate_re(50)
df_100["Re"] = calculate_re(100)
df_150["Re"] = calculate_re(150)

# Calculate Cd_o for each speed/aoa combo
i = 0
for df in [df_50, df_100, df_150]:
    u_atm = 50 * (1+i) * .3048
    ro = rho[i]
    q_atm = ro * u**2 / 2 

    df["U2 - 0"] = calculate_u2(df["Dynamic Pressure (in H2O)"], ro) # Downstream velocity for each location
    df["U2 - 8"] = calculate_u2(df["Dynamic Pressure (in H2O).1"], ro)
    try: df["U2 - 12"] = calculate_u2(df["Dynamic Pressure (in H2O).2"], ro)
    except: pass

    df["Cdo - 0 - Components"] = 1/q_atm * (ro * (u_atm**2 - df["U2 - 0"]**2) / g) # Each location's component of Cd_o
    df["Cdo - 8 - Components"] = 1/q_atm * (ro * (u_atm**2 - df["U2 - 8"]**2) / g)
    try: df["Cdo - 12 - Components"] = 1/q_atm * (ro * (u_atm**2 - df["U2 - 12"]**2) / g)
    except: pass

    df["Cdo - 0"] = df["Cdo - 0 - Components"].sum() # Total Cd_o for the AoA
    df["Cdo - 8"] = df["Cdo - 8 - Components"].sum()
    try: df["Cdo - 12"] = df["Cdo - 12 - Components"].sum()
    except: pass

    i += 1 

# %%
df_100

# %% [markdown]
# $$ C_{d_o} = \frac{1}{\frac{1}{2} \rho {U_{atm}^2}} \Sigma \left[ \frac{\rho (U_{atm}^2 - U_{downstream}^2)}{g} \right]$$

# %%
# Refining data
index_values = [
    "50 FPS - 0 AoA",
    "50 FPS - 8 AoA",
    "100 FPS - 0 AoA",
    "100 FPS - 8 AoA",
    "100 FPS - 12 AoA",
    "150 FPS - 0 AoA",
    "150 FPS - 8 AoA"
]
refined_table = pd.DataFrame(index = index_values)
refined_table["Re"] = [df_50["Re"][0], df_50["Re"][0], df_100["Re"][0], df_100["Re"][0], df_100["Re"][0], df_150["Re"][0], df_150["Re"][0]]
refined_table["Cdo"] = [df_50["Cdo - 0"][0], df_50["Cdo - 8"][0], df_100["Cdo - 0"][0], df_100["Cdo - 8"][0], df_100["Cdo - 12"][0], df_150["Cdo - 0"][0], df_150["Cdo - 8"][0]]

# %%
refined_table

# %% [markdown]
# ## **Task 2**

# %%
''' 
Calculate viscous drag by subtracting the form drag from the total drag. Calculate the
percentage of total drag for each component at each condition 
'''

form_drag = [-0.00457, 0.01646, 0.08246] # Form drag data provided by Dr. Cuppoletti at 100 ft/s and 0, 8, 12 AoA
total_drag = [refined_table["Cdo"][3], refined_table["Cdo"][4], refined_table["Cdo"][5]]

columns = pd.MultiIndex.from_tuples([("Value1", "Metric1"), ("Value1", "Metric2"), ("Value2", "Metric1"), ("Value2", "Metric2")], names=["Metric", "Type"])
viscous_drag = pd.DataFrame(index = index_values[2:5]) 
viscous_drag["Cdo"] = total_drag
viscous_drag["Form Drag"] = form_drag
viscous_drag["Viscous Drag"] = viscous_drag["Cdo"] - viscous_drag["Form Drag"]

# %%
viscous_drag

# %%
columns = pd.MultiIndex.from_tuples([("50", "0 AoA"), ("50", "8 AoA"), ("100", "0 AoA"), ("100", "8 AoA"), ("100", "12 AoA"), ("150", "0 AoA"), ("150", "8 AoA")], names=["", "Position"])
drag_percentages = pd.DataFrame(index = [12.00, 11.75, 11.50, 11.25, 11.00, 10.75, 10.50, 10.25, 10.00, 9.75, 9.50, 9.38, 9.25, 9.13, 9.00, 8.88, 8.75, 8.63, 8.50, 8.25, 8.00, 7.75, 7.50, 7.25, 7.00], columns = columns)

# %%
drag_percentages.iloc[:, 0] = (df_50["Cdo - 0 - Components"].to_list()      / df_50["Cdo - 0 - Components"].sum() * 100)
drag_percentages.iloc[:, 1] = (df_50["Cdo - 8 - Components"].to_list()      / df_50["Cdo - 8 - Components"].sum() * 100)
drag_percentages.iloc[:, 2] = (df_100["Cdo - 0 - Components"].to_list()     / df_100["Cdo - 0 - Components"].sum() * 100)
drag_percentages.iloc[:, 3] = (df_100["Cdo - 8 - Components"].to_list()     / df_100["Cdo - 8 - Components"].sum() * 100)
drag_percentages.iloc[:, 4] = (df_100["Cdo - 12 - Components"].to_list()    / df_100["Cdo - 12 - Components"].sum() * 100)
drag_percentages.iloc[:, 5] = (df_150["Cdo - 0 - Components"].to_list()     / df_150["Cdo - 0 - Components"].sum() * 100)
drag_percentages.iloc[:, 6] = (df_150["Cdo - 8 - Components"].to_list()     / df_150["Cdo - 8 - Components"].sum() * 100)

# Format all values in the drag_percentages DataFrame as percentages with two decimal places
drag_percentages = drag_percentages.applymap('{:.2f}%'.format)

