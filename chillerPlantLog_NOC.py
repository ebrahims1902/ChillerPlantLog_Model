import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os
import yaml
import pickle



# folder to load config file
CONFIG_PATH = "./"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config



config = load_config("Config.yaml")

df = pd.read_excel(config["data_name"])

df.rename(columns = {'Hartman Fan Speed SP (%)': 'FSP (%)'}, inplace = True)

df.rename(columns = {'CHWST (°C)': 'CHWST (Celsius)', 'CHWRT (°C)': 'CHWRT (Celsius)', 'Outside Air Temp. (°C)':'Outside Air Temp. (Celsius)','CHWST SP (°C)':'CHWST SP (Celsius)', 'ECWT SP (°C)': 'ECWT SP (Celsius)', 'ECWT (°C)': 'ECWT (Celsius)', 'LCWT (°C)': 'LCWT (Celsius)', 'Differential Temp. (°C)': 'Differential Temp. (Celsius)'},  inplace = True )

df1 = df[(df['Plant KW/Ton']>config["plantKW_Ton_Column_Range"])&(df['Capacity (Tons)']>=config["capacity_Tons_column_Range"])]

capacityTons = df1['Capacity (Tons)']

df1['Capacity (%)'] = (capacityTons - capacityTons.min()) / (capacityTons.max() - capacityTons.min()) * 100

remove_dup_df = df1.drop_duplicates(subset=config["duplicate_row_from_required_column_NOC"])

outerlier_removed = remove_dup_df[(remove_dup_df['Capacity (%)'] > config["capacity_Percentage_MoreThan_NOC"])
                 &(remove_dup_df['CHWST (Celsius)']>config["chwst_cel_MoreThan_NOC"])
                 &(remove_dup_df['CHWST (Celsius)']<config["chwst_cel_LessThan_NOC"])
                 &(remove_dup_df['CHWRT (Celsius)']>config["chwrt_cel_MoreThan_NOC"])
                 &(remove_dup_df['CHWRT (Celsius)']<config["chwrt_cel_LessThan_NOc"])
                 &(remove_dup_df['Relative Humidity (%)']>config["relative_Humidity_Per_MoreThan_NOC"])
                 &(remove_dup_df['Relative Humidity (%)']<config["relative_Humidity_Per_LessThan_NOC"])
                 &(remove_dup_df['Outside Air Temp. (Celsius)']>config["outsideAirTemp_MoreThan_NOC"])
                 &(remove_dup_df['Outside Air Temp. (Celsius)']<config["outsideAirTemp_LessThan_NOC"])
                 &(remove_dup_df['CHWST SP (Celsius)']>config["chwst_SP_cel_MoreThan_NOC"])
                 &(remove_dup_df['CHWST SP (Celsius)']<config["chwst_SP_cel_LessThan_NOC"])
                 &(remove_dup_df['ECWT (Celsius)']>config["ecwt_cel_MoreThan_NOC"])
                 &(remove_dup_df['ECWT (Celsius)']<config["ecwt_cel_LessThan_NOC"])
                 &(remove_dup_df['LCWT (Celsius)']>config["lcwt_cel_MoreThan_NOC"])
                 &(remove_dup_df['LCWT (Celsius)']<config["lcwt_cel_LessThan_NOC"])
                 &(remove_dup_df['Differential Temp. (Celsius)']>config["diff_Temp_cel_MoreThan_NOC"])
                 &(remove_dup_df['Differential Temp. (Celsius)']<config["diff_Temp_cel_LessThan_NOC"])]

columns = config["required_columns_NOC"]

data = outerlier_removed[columns]

x = data.drop(config["target_name_NOC"], axis=1).values
y = data[config["target_name_NOC"]].values


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=config["test_size"], random_state = config["random_state_train"])

# NN Model
Nn = MLPClassifier(hidden_layer_sizes=(15, 15, 15), activation=config["activation"], solver=config["solver"], max_iter=config["max_iter"])
Nn.fit(X_train, y_train)

print(Nn.score(X_train, y_train))

# save our RF in the model directory
pickle.dump(Nn, open(config["model_name_NOC"], 'wb'))
print(">>>> ALL DONE")
