import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the data
df = pd.read_csv("medical_examination.csv")

# Step 2: Add an overweight column (BMI > 25)
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2  # BMI calculation
df['overweight'] = (df['bmi'] > 25).astype(int)

# Step 3: Normalize cholesterol and gluc columns
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Step 4: Draw the Categorical Plot
def draw_cat_plot():
    # Step 4a: Create DataFrame for categorical plot
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
    # Step 4b: Group by cardio and calculate counts
    df_cat = df_cat.groupby(["cardio", "variable", "value"]).size().reset_index(name="total")
    # Step 4c: Rename columns to ensure proper usage in catplot
    df_cat = df_cat.rename(columns={"value": "health_status", "total": "count"})
    
    # Step 4d: Create the categorical plot
    fig = sns.catplot(x="variable", hue="health_status", col="cardio", data=df_cat, kind="count")
    return fig

# Step 5: Draw the Heatmap
def draw_heat_map():
    # Step 5a: Clean data based on the conditions provided
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) & 
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    # Step 5b: Calculate correlation matrix
    corr = df_heat.corr()
    
    # Step 5c: Generate a mask for the upper triangle
    mask = np.triu(corr)
    
    # Step 5d: Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Step 5e: Plot the correlation matrix
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={"shrink": .8})
    plt.show()

# Calling the functions to generate the plots
cat_plot_fig = draw_cat_plot()  # This will generate the categorical plot
draw_heat_map()  # This will generate the heatmap

