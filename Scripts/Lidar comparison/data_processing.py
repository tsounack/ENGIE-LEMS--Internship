#df_anemo contains the data recorded by the meteo station (anemo for anemometer)
#imported from the .txt using the pandas read_csv function.
#by parsing dates and adding them to the index, we create a timeseries dataframe (very useful)

#df lidar contains the wind speeds recorded by the lidar 
df_lidar = pd.read_csv(f"{maindir}/1 Datasets/1 Wind LiDAR/{site}_Wind_Data.csv", sep=",", parse_dates=True, index_col=0)
df_lidar = df_lidar.loc[:,df_lidar.columns.str.contains("Average Horizontal Wind Speed")]
df_lidar.dropna(axis=0, inplace=True) # we drop any rows which we do not have valid data at every height (how much data do we lose by doing this?)

df_lidar_backup = pd.read_csv(f"{maindir}/1 Datasets/1 Wind LiDAR/{site}_Wind_Data_Backup.csv", sep=";", parse_dates=True, index_col=0)
df_lidar_backup.dropna(axis=0, inplace=True) # we drop any rows which we do not have valid data at every height (how much data do we lose by doing this?)

regex = ""
for i in range(len(heights)):
    z = heights[i]
    regex += str(z)
    if i!=len(heights)-1:
        regex += "|"

df_lidar_filtered = df_lidar.filter(regex=regex)
df_lidar_backup_filtered = df_lidar_backup.filter(regex=regex)

df_lidar_filtered.columns = [f"WS_{z}m" for z in heights]    
df_lidar_backup_filtered.columns = [f"WS_{z}m_backup" for z in heights]


# we can merge our df_anemo and df_lidar datasets using the datetime indexes. I've called the resulting dataframe "df" (0 points for creativity)
df = df_lidar_filtered.merge(df_lidar_backup_filtered, right_index=True, left_index=True)
df.dropna(inplace=True)

