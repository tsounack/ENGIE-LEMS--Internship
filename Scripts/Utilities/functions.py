def datasets(ls_site):
    
    print(f"[1/{len(ls_site)+1}] Importing power dataset...")
    globals()["df_power"] = pd.read_csv(f"{maindir}/1 Datasets/LEMS_SWT167-8,4MW_0,1.txt", sep="\t", index_col=0)

    globals()["heights"] = [37, 56.5, 78, 96.5, 118, 135, 155, 175, 198]
        
    for i in range(len(ls_site)):
        site = ls_site[i]
        print(f"[{i+2}/{len(ls_site)+1}] Site {site}:")
        
        print("\t Importing ERA5's dataset...")
        globals()[f"df_era_{site}"] = pd.read_csv(f"{maindir}/1 Datasets/2 ERA5/{site}_ERA.csv", sep=",", index_col=0, parse_dates=True)
        globals()[f"df_era_{site}"].index = pd.to_datetime(globals()[f"df_era_{site}"].index)
        globals()[f"df_era_{site}"].drop(["longitude", "latitude"], axis=1, inplace=True)
        globals()[f"df_era_{site}"].dropna(inplace=True)
        
        if site == "A":
            
            print("\t Importing lidar and meteo station's datasets...")
            sub_heights_A = [40,57,77,97,117,137,157,177,197]
            
            regex = ""
            for i in range(len(sub_heights_A)):
                z = sub_heights_A[i]
                regex += str(z)
                if i!=len(sub_heights_A)-1:
                    regex += "|"
            
            df_meteo = pd.read_csv(f"{maindir}/1 Datasets/0 Meteo Station/Site A/A_MeteoStation_Data.csv", sep=";", index_col=0, parse_dates=True)
            df_meteo = df_meteo.loc[:,["wind_speed [m/s]", "air_temperature [ｰC]", "wind_direction [ｰ]"]]
            
            df_meteo.index = pd.to_datetime(df_meteo.index)
            df_meteo = df_meteo.groupby(pd.Grouper(freq="1h")).mean()
            df_meteo.dropna(inplace=True)
                
            df_temp = df_meteo.merge(globals()[f"df_era_{site}"], right_index=True, left_index=True)
            df_temp.dropna(inplace=True)
            
            df_subtract = df_temp["air_temperature [ｰC]"].subtract(df_temp["sst"]) + 273.15
            globals()[f"df_anemo_{site}"] = pd.DataFrame([df_temp["wind_speed [m/s]"], df_temp["wind_direction [ｰ]"], df_subtract]).transpose()
            globals()[f"df_anemo_{site}"].columns = ["WS", "Wdir", "Delta T"] 
            
            df_temp_lidar = pd.read_csv(f"{maindir}/1 Datasets/1 Wind LiDAR/A_Wind_Data.txt", sep="\t", parse_dates=True, index_col=0)
            df_temp_lidar = df_temp_lidar.loc[:,df_temp_lidar.columns.str.contains("wind_speed ")]
            df_temp_lidar = df_temp_lidar.groupby(pd.Grouper(freq="1h")).mean()
            df_temp_lidar.dropna(inplace=True)
            globals()[f"df_lidar_{site}"] = df_temp_lidar.filter(regex=regex)
            globals()[f"df_lidar_{site}"].columns = [f"Average Horizontal Wind Speed {z}m" for z in heights]
            
        
        if site == "B":
            
            print("\t Importing lidar and meteo station's datasets...")
            sub_heights_B = [40,57,77,97,117,137,157,177,197]
            
            regex = ""
            for i in range(len(sub_heights_B)):
                z = sub_heights_B[i]
                regex += str(z)
                if i!=len(sub_heights_B)-1:
                    regex += "|"
            
            df_meteo = pd.read_csv(f"{maindir}/1 Datasets/0 Meteo Station/Site B/B_MeteoStation_Data.csv", sep=";", index_col=0, parse_dates=True)
            df_meteo = df_meteo.loc[:,["wind_speed [m/s]", "air_temperature [?C]", "wind_direction [?]"]]
            
            df_meteo.index = pd.to_datetime(df_meteo.index)
            df_meteo = df_meteo.groupby(pd.Grouper(freq="1h")).mean()
            df_meteo.dropna(inplace=True)
                
            df_temp = df_meteo.merge(globals()[f"df_era_{site}"], right_index=True, left_index=True)
            df_temp.dropna(inplace=True)
            
            df_subtract = df_temp["air_temperature [?C]"].subtract(df_temp["sst"]) + 273.15
            globals()[f"df_anemo_{site}"] = pd.DataFrame([df_temp["wind_speed [m/s]"], df_temp["wind_direction [?]"], df_subtract]).transpose()
            globals()[f"df_anemo_{site}"].columns = ["WS", "Wdir", "Delta T"] 
            
            df_temp_lidar = pd.read_csv(f"{maindir}/1 Datasets/1 Wind LiDAR/B_Wind_Data.csv", sep=";", parse_dates=True, index_col=0)
            df_temp_lidar = df_temp_lidar.loc[:,df_temp_lidar.columns.str.contains("m/s")]
            df_temp_lidar = df_temp_lidar.groupby(pd.Grouper(freq="1h")).mean()
            df_temp_lidar.dropna(inplace=True)
            globals()[f"df_lidar_{site}"] = df_temp_lidar.filter(regex=regex)
            globals()[f"df_lidar_{site}"].columns = [f"Average Horizontal Wind Speed {z}m" for z in heights]
            
            
        if site == "C":
            
            print("\t Importing lidar and meteo station's datasets...")
            sub_heights_C = [38,58,78,98,118,138,158,178,198]
            
            regex = ""
            for i in range(len(sub_heights_C)):
                z = sub_heights_C[i]
                regex += str(z)
                if i!=len(sub_heights_C)-1:
                    regex += "|"
            
            df_meteo = pd.read_csv(f"{maindir}/1 Datasets/0 Meteo Station/Site C/C_MeteoStation_Data.csv", sep=";", index_col=0, parse_dates=True)
            df_meteo = df_meteo.loc[:,["meteo_Sm_avg [m/s]", "meteo_Ta_avg [?C]", "meteo_Dir_bear [?]"]]
            
            df_meteo.index = pd.to_datetime(df_meteo.index)
            df_meteo = df_meteo.groupby(pd.Grouper(freq="1h")).mean()
            df_meteo.dropna(inplace=True)
            
            df_temp = df_meteo.merge(globals()[f"df_era_{site}"], right_index=True, left_index=True)
            df_temp.dropna(inplace=True)
            
            df_subtract = df_temp["meteo_Ta_avg [?C]"].subtract(df_temp["sst"]) + 273.15
            globals()[f"df_anemo_{site}"] = pd.DataFrame([df_temp["meteo_Sm_avg [m/s]"], df_temp["meteo_Dir_bear [?]"], df_subtract]).transpose()
            globals()[f"df_anemo_{site}"].columns = ["WS", "Wdir", "Delta T"]       
            
            df_temp_lidar = pd.read_csv(f"{maindir}/1 Datasets/1 Wind LiDAR/C_Wind_Data.csv", sep=";", parse_dates=True, index_col=0)
            df_temp_lidar = df_temp_lidar.loc[:,df_temp_lidar.columns.str.contains("WS")]
            df_temp_lidar = df_temp_lidar.groupby(pd.Grouper(freq="1h")).mean()
            df_temp_lidar.dropna(inplace=True)
            globals()[f"df_lidar_{site}"] = df_temp_lidar.filter(regex=regex)
            globals()[f"df_lidar_{site}"].columns = [f"Average Horizontal Wind Speed {z}m" for z in heights]


        if site == "NOY" or site == "TRE":            
            
            if site == "NOY":
                print("\t Importing lidar and meteo station's datasets...")
            if site == "TRE":
                print("\t Importing lidar and meteo station's datasets...")
                
            sub_heights_NOY = [35,55,80,95,120,130,150,170,200]
            
            regex = ""
            for i in range(len(sub_heights_NOY)):
                z = sub_heights_NOY[i]
                regex += str(z)
                if i!=len(sub_heights_NOY)-1:
                    regex += "|"
            
            df_meteo = pd.read_csv(f"{maindir}/1 Datasets/0 Meteo Station/Site {site}/{site}_MeteoStation_WindTemp.txt", sep="\t", index_col=0, parse_dates=True)
            df_temp = df_meteo["Air Temp"].subtract(df_meteo["Sea Temp"])
            # df_temp = globals()[f"df_era_{site}"]["t2m"].subtract(globals()[f"df_era_{site}"]["sst"])
            df_meteo["Delta T"] = df_temp
            df_meteo.drop(["Air Temp", "Sea Temp"], axis=1, inplace=True)
            df_meteo = df_meteo.groupby(pd.Grouper(freq="1h")).mean()
            df_meteo.dropna(inplace=True)
                        
            globals()[f"df_anemo_{site}"] = df_meteo
            globals()[f"df_anemo_{site}"].columns = ["WS", "Wdir", "Delta T"]
            
            df_temp_lidar = pd.read_csv(f"{maindir}/1 Datasets/1 Wind LiDAR/{site}_Wind_Data.csv", sep=",", parse_dates=True, index_col=0)
            df_temp_lidar = df_temp_lidar.loc[:,df_temp_lidar.columns.str.contains("Average Horizontal Wind Speed")]
            df_temp_lidar = df_temp_lidar.groupby(pd.Grouper(freq="1h")).mean()
            df_temp_lidar.dropna(inplace=True)
            globals()[f"df_lidar_{site}"] = df_temp_lidar.filter(regex=regex)
            globals()[f"df_lidar_{site}"].columns = [f"Average Horizontal Wind Speed {z}m" for z in heights]
            
        print("\t Exporting the dataset...")
                
        df_temp_X = globals()[f"df_anemo_{site}"].merge(globals()[f"df_era_{site}"], right_index=True, left_index=True)
        df_temp_X["month"] = df_temp_X.index.month
        df_temp_X = df_temp_X[df_temp_X["WS"] != 0]
        df_temp_X.dropna(inplace=True)
        
        df_temp_y = globals()[f"df_lidar_{site}"].groupby(pd.Grouper(freq="1h")).mean()        
        df_temp_y.dropna(inplace=True)
        
        df_temp_Xy = df_temp_X.merge(df_temp_y, right_index=True, left_index=True)
        df_temp_Xy.dropna(inplace=True)
        
        globals()[f"df_Xy_{site}"] = df_temp_Xy
        globals()[f"df_X_{site}"] = df_temp_Xy.loc[:, ~df_temp_Xy.columns.str.contains("Average Horizontal Wind Speed")]
        globals()[f"df_y_{site}"] = df_temp_Xy.loc[:, df_temp_Xy.columns.str.contains("Average Horizontal Wind Speed")]
        
        

def model0(site, ls_results, Save_files, directional_alpha, new_site):
    
    title = f"Model 0 - site {site}"
    
    # Setting the heights list and color map 
    heights = [37, 56.5, 78, 96.5, 118, 135, 155, 175, 198]

    if site == "NOY":
        cmap="Greens"
    if site == "TRE":
        cmap = "Blues"
    if site == "A":
        cmap = "Oranges"
    if site == "B":
        cmap = "Reds"
    if site == "C":
        cmap = "Purples"


    #df alpha contains the shear exponent for each sector at our site
    df_alpha = pd.read_csv(f"{maindir}/1 Datasets/3 Alpha Values/{site}_Alpha_0.txt", sep="\t", index_col=0)

    N_Sectors = 12 # We could use more sectors (if we had more shear coefficients)
    edges = np.linspace(360/N_Sectors/2,360-360/N_Sectors/2,N_Sectors)
    z0 = 4

    if directional_alpha == "constant":
        cst_alpha = df_alpha.loc["All",:].values[0]
        df_alpha[:] = cst_alpha
    
    if new_site == True:
        df_alpha[:] = 0.08
        
        
    # Copying dataset
    df_Xy = globals()[f"df_Xy_{site}"].copy()


    # Binning the data
    df_Xy["sector"] = np.digitize(df_Xy["Wdir"], edges) + 1
    df_Xy["sector"].replace({13:1}, inplace=True)
    df = df_Xy


    # We apply the "powerlaw" function we defined above, and make use of the generic lambda function
    for i in range(len(heights)):
        z = heights[i]
        df[f"WS_{z}"] = df.apply(lambda x: powerlaw(x["WS"], z0, z, df_alpha["Alpha"][int(x["sector"])]), axis=1)
    
    globals()[f"df_{site}"] = df
    
    # Plotting graphs
    if "global" in ls_results:
        plot_global(heights=heights, df=df, cmap=cmap, Save_files=Save_files, title=title)
    if "monthly" in ls_results:
        plot_monthly(heights=heights, df=df, cmap=cmap, Save_files=Save_files, title=title)
    if "3d" in ls_results:
        plot_3Dchart(heights=heights, df=df, cmap=cmap, Save_files=Save_files, title=title)
        
        
        

def model1(ls_train, ls_test, ls_results, Save_files, min_length, polynomial_alpha, ratio=None):
    
    t_train = ""
    t_test = ""
    t_ratio = ""
    
    for i in range(len(ls_train)):
        site = ls_train[i]
        if i==0:
            t_train += f"{site}"
        if i!=0:
            t_train += f", {site}"

    for i in range(len(ls_test)):
        site = ls_test[i]
        if i==0:
            t_test += f"{site}"
        if i!=0:
            t_test += f", {site}"
    
    if ratio:
        t_ratio=f" ; ratio: {ratio*100}%"

    title = "Model 1 - estimating polynomial with " + t_train + " - applying it to " + t_test + t_ratio
    
    # Setting the heights list and color map
    heights = [37, 56.5, 78, 96.5, 118, 135, 155, 175, 198]
    
    for site in ls_train:
        globals()[f"df_Xy_1_{site}"] = globals()[f"df_Xy_{site}"].copy()
    for site in ls_test:
        globals()[f"df_Xy_1_{site}"] = globals()[f"df_Xy_{site}"].copy()

    if len(ls_train)!=1 and min_length==True:
        
        min_len = min(len(globals()[f"df_X_{site}"]) for site in ls_train)
        
        for site in ls_train:
            globals()[f"df_Xy_1_{site}"] = globals()[f"df_Xy_{site}"].sample(n=min_len, replace=True)
    
    
    if ratio and ls_train == ls_test:

        for site in ls_train:
            globals()[f"df_Xy_copy"] = globals()[f"df_Xy_{site}"].copy()
            globals()[f"df_Xy_1_{site}"] = globals()[f"df_Xy_copy"][(globals()[f"df_Xy_copy"].index < np.percentile(globals()[f"df_Xy_copy"].index, ratio*100))]
            globals()[f"df_Xy2_1_{site}"] = globals()[f"df_Xy_copy"][(globals()[f"df_Xy_copy"].index > np.percentile(globals()[f"df_Xy_copy"].index, ratio*100))]
    
    
    if len(ls_test)==1:
        site = ls_test[0]
        if site == "NOY":
            cmap="Greens"
        if site == "TRE":
            cmap = "Blues"
        if site == "A":
            cmap = "Oranges"
        if site == "B":
            cmap = "Reds"
        if site == "C":
            cmap = "Purples"
    
    if len(ls_test)!=1:
        cmap = "Greys"
    
    
    # Creating the train dataframe with ls_train
    df = pd.concat([globals()[f'df_Xy_1_{site}'] for site in ls_train])
    
    
    # Filtering 
    df_filtered = df.copy()

    if filter != "":
        df_filtered.index = pd.to_datetime(df.index)
        df_filtered = df_filtered.groupby(pd.Grouper(freq=filter)).mean()
        df_filtered.dropna(inplace=True)
    
    
    # Binning in increments of Delta T and dropping values where bins have less than 25 elements
    edges = np.linspace(-12.5, 4.5, 18)
    df_filtered["sector"] = np.digitize(df_filtered["Delta T"], edges) - 13

    for i in df_filtered["sector"].unique():
        if df_filtered["sector"].value_counts()[i] < 25:
            df_filtered = df_filtered[df_filtered.sector != i]
            
    
    # Estimating alpha with a linear regression based on all heights --- this is where we take the global polynomial
    df_avg = df_filtered.assign(Alpha = lambda x: linear_reg_alpha([x[f"Average Horizontal Wind Speed {z}m"] for z in heights], x["WS"], heights, 4))
    df_graph = df_avg[["sector", "Alpha"]].copy()
    df_graph = df_graph.groupby(["sector"], as_index=False).mean()

    coefficients = np.polyfit(df_graph["sector"], df_graph["Alpha"], poly_deg)
    poly_definitive = np.poly1d(coefficients)
    
    
    # Estimating alpha with a linear regression based on all heights - monthly --- this is where we take the monthly polynomials
    df_coeffs = pd.DataFrame(columns=["month", "coeffs"])
    
    for month in range(1,13):
                
        df_temp = df_filtered.loc[df_filtered.index.month == month,:]
        df_avg = df_temp.assign(Alpha = lambda x: linear_reg_alpha([x[f"Average Horizontal Wind Speed {z}m"] for z in heights], x["WS"], heights, 4))
        df_graph = df_avg[["sector", "Alpha"]].copy()
        df_graph = df_graph.groupby(["sector"], as_index=False).mean()
        
        if len(df_graph.index) != 0:
            coefficients = np.polyfit(df_graph["sector"], df_graph["Alpha"], poly_deg)
            df_coeffs_temp = pd.DataFrame({"month" : [f"{calendar.month_name[month]}"], "coeffs" : [coefficients]})
            df_coeffs = pd.concat([df_coeffs, df_coeffs_temp], ignore_index=True)
            
        if len(df_graph.index) == 0:
            df_coeffs_temp = pd.DataFrame({"month" : [f"{calendar.month_name[month]}"], "coeffs" : [[0 for i in range(poly_deg+1)]]})
            df_coeffs = pd.concat([df_coeffs, df_coeffs_temp], ignore_index=True)
    
    # Setting the testing dataframe based on ls_test
    if ratio and ls_train == ls_test:
        df = pd.concat([globals()[f'df_Xy2_1_{site}'] for site in ls_train])
    
    if not ratio or ls_train != ls_test:
        df = pd.concat([globals()[f'df_Xy_1_{site}'] for site in ls_test])
    
    
    # Applying the polynomial interpolation found to calculate alpha at each time stamp, computing the windspeed at each height based on this alpha
    if polynomial_alpha == "constant":
        df["Alpha"] = poly_definitive(df["Delta T"])
        title += "\n \u03B1 = " + prettyprintPolynomial(poly_definitive)
        
    if polynomial_alpha == "monthly":
        df["Alpha"] = np.nan
        for month in range(1,13):
            poly_definitive = np.poly1d(df_coeffs["coeffs"][month-1])
            df.loc[df.index.month==month,"Alpha"] = poly_definitive(df.loc[df.index.month==month, "Delta T"])

    for i in range(len(heights)):
        z = heights[i]
        # we apply the "powerlaw" function we defined above, and make use of the generic lambda function
        df[f"WS_{z}"] = df.apply(lambda x: powerlaw(x["WS"], 4, z, x["Alpha"]), axis=1) 

    globals()[f"df_{site}"] = df


    # Plotting graphs
    if "global" in ls_results:
        plot_global(heights=heights, df=df, cmap=cmap, Save_files=Save_files, title=title)
    if "monthly" in ls_results:
        plot_monthly(heights=heights, df=df, cmap=cmap, Save_files=Save_files, title=title)
    if "3d" in ls_results:
        plot_3Dchart(heights=heights, df=df, cmap=cmap, Save_files=Save_files, title=title)


def plot_global(heights, df, cmap, Save_files, file_loc="", file_name="", title=""):    
    print("\t Plotting global graph...")
    fig = plt.figure(figsize=(18,14))
    fig.tight_layout(pad=3)

    spec = GridSpec(int(np.ceil(len(heights)/3))+2,3, figure=fig)

    column_headers = []

    row_headers = ["R²", "Avg. error (m/s)", "Avg. absolute error (m/s)", "Standard deviation (m/s)", "Avg. relative error (%)", "Est. P avg. / Real P avg."]

    cell_text = np.zeros((len(row_headers), len(heights)))

    r = 0
    c = 0

    for i in range(len(heights)):
        
        z = heights[i]
        
        if len(heights)%3 == 1:
            
            if c > 2 and i != len(heights)-1:
                c = 0
                r += 1
            
            if i == 9:
                c = 1
                r += 1
        
        if len(heights)%3 != 1:
        
            if c > 2:
                c = 0
                r += 1
        
        column_headers += [str(z)+"m"]
        
        df_temp = df.loc[:,df.columns.str.contains(f"{z}")]
        
        
        # R² value
        lr = st.linregress(df_temp[f"Average Horizontal Wind Speed {z}m"],df_temp[f"WS_{z}"]) #scipy stats built-in linear regression function
        
        cell_text[0][i] = '{: 0.2f}'.format(lr.rvalue**2)
        
        
        # Mean error
        df_subtract = df_temp[f"WS_{z}"].subtract(df_temp[f"Average Horizontal Wind Speed {z}m"], axis = 0)
        subtract_avg = df_subtract.mean()
        
        cell_text[1][i] = '{: 0.2f}'.format(subtract_avg)
        
        
        # Mean absolute error
        df_subtract_abs = df_subtract.abs()
        subtract_abs_avg = df_subtract_abs.mean()
        
        cell_text[2][i] = '{: 0.2f}'.format(subtract_abs_avg)
        
        
        # Standard deviation
        cell_text[3][i] = '{: 0.2f}'.format(np.std(df_temp[f"WS_{z}"]))
        
        
        # Average relative error
        df_relative_error = 100 * df_subtract_abs.divide(df_temp[f"Average Horizontal Wind Speed {z}m"], axis = 0)
        relative_error_avg = df_relative_error.mean()
        cell_text[4][i] = '{: 0.2f}'.format(relative_error_avg)
        
        
            
        df_WS_pred_round = df_temp[f"WS_{z}"].round(1)
        df_WS_real_round = df_temp[f"Average Horizontal Wind Speed {z}m"].round(1)
        
        df_power_est = df_WS_pred_round.map(df_power["P"])
        df_power_real = df_WS_real_round.map(df_power["P"])              
        
        # Estimated power avg / Real power avg
        cell_text[5][i] = '{: 0.2f}'.format(df_power_est.mean()/df_power_real.mean())
            
            
            
        ## Histogram
        
        ax = fig.add_subplot(spec[r, c])
        
        ax.hist2d(df_temp[f"Average Horizontal Wind Speed {z}m"], df_temp[f"WS_{z}"], 100, cmin=0.0001, cmap=cmap) 
        
        if c == 0:
            ax.set_ylabel("WS Predicted [m/s]")

        if r == int(np.ceil(len(heights)/3))-1:
            ax.set_xlabel("WS Flidar [m/s]")
            
        
        ax.set_ylim([0, 25])
            
        xpoints = ypoints = ax.get_xlim()
        ax.axline((xpoints[0], ypoints[0]), (xpoints[1], ypoints[1]), linestyle='--', color='k', lw=1)
        
        # on the title, I write the height of the plot and the R2 value
        ax.set_title(f"WS Correlation at {z}m; $R^2$ = {lr.rvalue**2: 0.2f}")

        c += 1


    if cmap == 'Blues':
        rcolors = plt.cm.Blues(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.Blues(np.full(len(column_headers), 0.1))
    if cmap == 'Greens':
        rcolors = plt.cm.Greens(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.Greens(np.full(len(column_headers), 0.1))
    if cmap == 'Purples':
        rcolors = plt.cm.Purples(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.Purples(np.full(len(column_headers), 0.1))
    if cmap == 'Oranges':
        rcolors = plt.cm.Oranges(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.Oranges(np.full(len(column_headers), 0.1))
    if cmap == 'Reds':
        rcolors = plt.cm.Reds(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.Reds(np.full(len(column_headers), 0.1))
    if cmap == 'Greys':
        rcolors = plt.cm.Greys(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.Greys(np.full(len(column_headers), 0.1))
    
    ax = fig.add_subplot(spec[int(np.ceil(len(heights)/3)):int(np.ceil(len(heights)/3))+2, :])

    ytable = ax.table(cellText=cell_text,
                rowLabels=row_headers,
                rowColours=rcolors,
                rowLoc='center',
                colColours=ccolors,
                colLabels=column_headers,
                loc='center',
                bbox=[0.1, 0.05, 0.9, 0.9])

    ax.axis('tight')
    ax.axis('off')

    fig.suptitle(title, fontweight="bold")
    

    cellDict = ytable.get_celld()
    for i in range(0, len(cell_text[0])):
        cellDict[(0,i)].set_height(.15)
        for j in range(1, len(cell_text)+1):
            cellDict[(j,i)].set_height(.1)

    for i in range(1, len(cell_text)+1):
        cellDict[(i,-1)].set_height(.1)

    fig.subplots_adjust(left=0.05, bottom=0.04, right=0.95, top=0.92, hspace=0.5, wspace=0.2)

    if Save_files == "Yes":
            fig.savefig(f"{file_loc}/{file_name}.jpg", dpi=480)
            plt.close()   
    
    
    
    
    
    
def plot_monthly(heights, df, cmap, Save_files, file_loc="", file_name="", title=""):
    print("\t Plotting monthly graphs...")
    for i in range(len(heights)):

        fig = plt.figure(figsize=(18,14))
        fig.tight_layout(pad=3)

        spec = GridSpec(6,3, figure=fig)

        column_headers = []

        row_headers = ["Number of samples", "R²", "Avg. error (m/s)", "Avg. absolute error (m/s)", "Standard deviation (m/s)", "Avg. relative error (%)", "Est. P avg. / Real P avg."]

        z = heights[i]
        
        cell_text = np.zeros((len(row_headers), 12))

        r = 0
        c = 0


        for j in range(0,12):
            
            month = j+1
            
            if c > 2:
                c = 0
                r += 1
                            
            column_headers += [str(calendar.month_name[month])]
            
            df_temp = df.loc[df.index.month == month,df.columns.str.contains(f"{z}")]
            
            if len(df_temp.index) !=0:
            
                # Number of samples
                cell_text[0][j] = str(round(len(df_temp.index)))
                
                
                # R² value
                lr = st.linregress(df_temp[f"Average Horizontal Wind Speed {z}m"],df_temp[f"WS_{z}"]) #scipy stats built-in linear regression function
                
                cell_text[1][j] = '{: 0.2f}'.format(lr.rvalue**2)
                
                
                # Mean error
                df_subtract = df_temp[f"WS_{z}"].subtract(df_temp[f"Average Horizontal Wind Speed {z}m"], axis = 0)
                subtract_avg = df_subtract.mean()
                
                cell_text[2][j] = '{: 0.2f}'.format(subtract_avg)
                
                
                # Mean absolute error
                df_subtract_abs = df_subtract.abs()
                subtract_abs_avg = df_subtract_abs.mean()
                
                cell_text[3][j] = '{: 0.2f}'.format(subtract_abs_avg)
                
                
                # Standard deviation
                cell_text[4][j] = '{: 0.2f}'.format(np.std(df_subtract))
                
                
                # Average relative error
                df_relative_error = 100 * df_subtract_abs.divide(df_temp[f"Average Horizontal Wind Speed {z}m"], axis = 0)
                relative_error_avg = df_relative_error.mean()
                cell_text[5][j] = '{: 0.2f}'.format(relative_error_avg)
                
                
                
                    
                df_WS_pred_round = df_temp[f"WS_{z}"].round(1)
                df_WS_real_round = df_temp[f"Average Horizontal Wind Speed {z}m"].round(1)

                df_power_est = df_WS_pred_round.map(df_power["P"])
                df_power_real = df_WS_real_round.map(df_power["P"])                
                
                # Estimated power avg / Real power avg
                cell_text[6][j] = '{: 0.2f}'.format(df_power_est.mean()/df_power_real.mean())
                    
                    
                    
                ## Histogram
                
                ax = fig.add_subplot(spec[r, c])
                
                ax.hist2d(df_temp[f"Average Horizontal Wind Speed {z}m"], df_temp[f"WS_{z}"], 100, cmin=0.0001, cmap=cmap) 
                
                if c == 0:
                    ax.set_ylabel("WS Predicted [m/s]")

                if r == 3:
                    ax.set_xlabel("WS Flidar [m/s]")
                    
                ax.set_xlim([0, 25])
                ax.set_ylim([0, 25])
                    
                xpoints = ypoints = ax.get_xlim()
                ax.axline((xpoints[0], ypoints[0]), (xpoints[1], ypoints[1]), linestyle='--', color='k', lw=1)
                
                # on the title, I write the height of the plot and the R2 value
                ax.set_title(f"WS Correlation in {calendar.month_name[month]}; $R^2$ = {lr.rvalue**2: 0.2f}; N = {int(len(df_temp.index))}")

                c += 1
                
            if len(df_temp.index) ==0:
                cell_text[0][j] = 0
                cell_text[1][j] = 0
                cell_text[2][j] = 0
                cell_text[3][j] = 0
                cell_text[4][j] = 0
                cell_text[5][j] = 0
                cell_text[6][j] = 0
                
                ax = fig.add_subplot(spec[r, c])
                
                ax.hist2d(df_temp[f"Average Horizontal Wind Speed {z}m"], df_temp[f"WS_{z}"], 100, cmin=0.0001, cmap=cmap) 
                
                if c == 0:
                    ax.set_ylabel("WS Predicted [m/s]")

                if r == 3:
                    ax.set_xlabel("WS Flidar [m/s]")
                    
                ax.set_xlim([0, 25])
                ax.set_ylim([0, 25])
                    
                xpoints = ypoints = ax.get_xlim()
                ax.axline((xpoints[0], ypoints[0]), (xpoints[1], ypoints[1]), linestyle='--', color='k', lw=1)
                
                # on the title, I write the height of the plot and the R2 value
                ax.set_title(f"WS Correlation in {calendar.month_name[month]}; $R^2$ = n/a; N = {int(len(df_temp.index))}")

                c += 1
                


        if cmap == 'Blues':
            rcolors = plt.cm.Blues(np.full(len(row_headers), 0.1))
            ccolors = plt.cm.Blues(np.full(len(column_headers), 0.1))
        if cmap == 'Greens':
            rcolors = plt.cm.Greens(np.full(len(row_headers), 0.1))
            ccolors = plt.cm.Greens(np.full(len(column_headers), 0.1))
        if cmap == 'Purples':
            rcolors = plt.cm.Purples(np.full(len(row_headers), 0.1))
            ccolors = plt.cm.Purples(np.full(len(column_headers), 0.1))
        if cmap == 'Oranges':
            rcolors = plt.cm.Oranges(np.full(len(row_headers), 0.1))
            ccolors = plt.cm.Oranges(np.full(len(column_headers), 0.1))
        if cmap == 'Reds':
            rcolors = plt.cm.Reds(np.full(len(row_headers), 0.1))
            ccolors = plt.cm.Reds(np.full(len(column_headers), 0.1))
        if cmap == 'Greys':
            rcolors = plt.cm.Greys(np.full(len(row_headers), 0.1))
            ccolors = plt.cm.Greys(np.full(len(column_headers), 0.1))

        ax = fig.add_subplot(spec[4:6, :])

        ytable = ax.table(cellText=cell_text,
                    rowLabels=row_headers,
                    rowColours=rcolors,
                    rowLoc='center',
                    colColours=ccolors,
                    colLabels=column_headers,
                    loc='center',
                    bbox=[0.1, 0.05, 0.9, 0.9])

        ax.axis('tight')
        ax.axis('off')
        
        fig.suptitle(title + f" - {z}m", fontweight="bold")
        
        cellDict = ytable.get_celld()
        for i in range(0, len(cell_text[0])):
            cellDict[(0,i)].set_height(.15)
            for j in range(1, len(cell_text)+1):
                cellDict[(j,i)].set_height(.1)

        for i in range(1, len(cell_text)+1):
            cellDict[(i,-1)].set_height(.1)

        fig.subplots_adjust(left=0.05, bottom=0.02, right=0.95, top=0.92, hspace=0.5, wspace=0.2)

        if Save_files == "Yes":
            fig.savefig(f"{file_loc}/{file_name}.jpg", dpi=480)
            plt.close()
            
            
   
   
            
def plot_3Dchart(heights, df, cmap, Save_files, file_loc="", file_name="", title=""):
    print("\t Plotting 3D bar chart...")
    
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(111, projection='3d')

    plt.rcParams["figure.autolayout"] = True

    xpos = []
    ypos = []
    zpos = []

    dx = []
    dy = []
    dz = []

    xticks = [calendar.month_name[month] for month in range(1,13)]
    yticks = [str(heights[i])+'m' for i in range(len(heights))]

    for month in range(1,13):
        
        for i in range(len(heights)):
            
            z = heights[i]
            
            df_temp = df.loc[df.index.month == month,df.columns.str.contains(f"{z}")]
            
            if len(df_temp.index) != 0:
                lr = st.linregress(df_temp[f"Average Horizontal Wind Speed {z}m"],df_temp[f"WS_{z}"]) #scipy stats built-in linear regression function
                df_subtract = df_temp[f"WS_{z}"].subtract(df_temp[f"Average Horizontal Wind Speed {z}m"], axis = 0)
                df_subtract_abs = df_subtract.abs()
                df_relative_error = 100 * df_subtract_abs.divide(df_temp[f"Average Horizontal Wind Speed {z}m"], axis = 0)
                relative_error_avg = df_relative_error.mean()
                
                dz.append(float('{: 0.2f}'.format(relative_error_avg)))
            
            if len(df_temp.index) == 0:
                dz.append(0)
                
            xpos.append(month-1)
            ypos.append(z)
            
            
    zpos = np.zeros(len(xpos))
    dx = np.ones(len(xpos))
    dy = [heights[-1]/len(heights) for i in range(len(xpos))]

    if cmap == 'Blues':
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = 'blue')
    if cmap == 'Greens': 
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = 'green')
    if cmap == 'Purples':
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = 'purple')
    if cmap == 'Oranges':
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = 'orange')
    if cmap == 'Reds':
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = 'red')
    if cmap == 'Greys':
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = 'grey')

    ax1.set_xticks(np.arange(len(xticks)))
    ax1.set_xticklabels(xticks)

    ax1.set_yticklabels(yticks)

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Height')
    ax1.set_zlabel('Average relative error (%)')
    
    fig.suptitle(title, fontweight="bold")

    if Save_files == "Yes":
        fig.savefig(f"{file_loc}/{file_name}", dpi=480)
        plt.close()
        
    
def linear_reg_alpha(U, ur, Z, zr):
    Y = [np.log(u/ur) for u in U]
    X = [np.log(z/zr) for z in Z]
    coefficients = np.polyfit(X, Y, 1)
    return coefficients[0]


def alpha(u, ur, Z, zr):
    a = np.log(u/ur)
    b = np.log(z/zr)
    return a/b


# Power law function to use later in vector operation
def powerlaw(v0,h0,h1,alpha):
    v1 = v0*(h1/h0)**alpha
    return v1


def prettyprintPolynomial(p):
    """ Small function to print nicely the polynomial p as we write it in maths, in ASCII text."""
    coefs = p.coef[::-1]  # List of coefficient, sorted by increasing degrees
    res = ""  # The resulting string
    for i, a in enumerate(coefs):
        if int(a) == a:  # Remove the trailing .0
            a = int(a)
        if i == 0:  # First coefficient, no need for X
            if a > 0:
                res += "{a:.5f} + ".format(a=a)
            elif a < 0:  # Negative a is printed like (a)
                res += "({a:.5f}) + ".format(a=a)
            # a = 0 is not displayed 
        elif i == 1:  # Second coefficient, only X and not X**i
            if a == 1:  # a = 1 does not need to be displayed
                res += "\u0394T + "
            elif a > 0:
                res += "{a:.5f} * \u0394T + ".format(a=a)
            elif a < 0:
                res += "({a:.5f}) * \u0394T + ".format(a=a)
        else:
            if a == 1:
                res += "\u0394T**{i} + ".format(i=i)
            elif a > 0:
                res += "{a:.5f} * \u0394T**{i} + ".format(a=a, i=i)
            elif a < 0:
                res += "({a:.5f}) * \u0394T**{i} + ".format(a=a, i=i)
    return res[:-3] if res else ""


def test_model(model_name, heights, ls_results, Save_files, df_X_test, df_y_test):
    
    columns = [f"WS_{z}" for z in heights]
    sub_df = pd.DataFrame(data = model_name.predict(df_X_test), columns = columns)
    sub_df["Date"] = df_X_test.index
    sub_df["Date"] = pd.to_datetime(sub_df["Date"])
    sub_df.set_index("Date", inplace=True)
    df = pd.concat([df_X_test, df_y_test, sub_df], axis=1)

    globals()[f"df_test"] = df

    site="A"
    cmap="Oranges"
    
    # Plotting graphs
    if "global" in ls_results:
        plot_global(heights=heights, df=df, cmap=cmap, Save_files=Save_files)
    if "monthly" in ls_results:
        plot_monthly(heights=heights, df=df, cmap=cmap, Save_files=Save_files)
    if "3d" in ls_results:
        plot_3Dchart(heights=heights, df=df, cmap=cmap, Save_files=Save_files)


def test_model_hh(model_name, objective_height, df_X_test, df_y_test):
    
    columns = [f"WS_{objective_height}"]
    sub_df = pd.DataFrame(data = model_name.predict(df_X_test), columns = columns)
    sub_df["Date"] = df_X_test.index
    sub_df["Date"] = pd.to_datetime(sub_df["Date"])
    sub_df.set_index("Date", inplace=True)
    df = pd.concat([df_X_test, df_y_test, sub_df], axis=1)
    
    df["Error"] = 100*np.abs(df[f"WS_{objective_height}"] - df[f"Average Horizontal Wind Speed {objective_height}m"])/df[f"Average Horizontal Wind Speed {objective_height}m"]
    df["bias"] = 100*(df[f"WS_{objective_height}"] - df[f"Average Horizontal Wind Speed {objective_height}m"])/df[f"Average Horizontal Wind Speed {objective_height}m"]

    return df["Error"].mean()


def split(df_Xy_, ratio):
    df_Xy_train = df_Xy_[(df_Xy_.index < np.percentile(df_Xy_.index, ratio*100))]
    df_Xy_test = df_Xy_[(df_Xy_.index > np.percentile(df_Xy_.index, ratio*100))]

    df_X_temp = df_Xy_train.loc[:, ~df_Xy_train.columns.str.contains("Average Horizontal Wind Speed")]
    df_y_temp = df_Xy_train.loc[:, df_Xy_train.columns.str.contains("Average Horizontal Wind Speed")]

    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X_temp, df_y_temp)

    df_X_test2 = df_Xy_test.loc[:, ~df_Xy_test.columns.str.contains("Average Horizontal Wind Speed")]
    df_y_test2 = df_Xy_test.loc[:, df_Xy_test.columns.str.contains("Average Horizontal Wind Speed")]

    return df_X_train, df_X_test, df_X_test2, df_y_train, df_y_test, df_y_test2