import pandas

data = pandas.read_csv('Time_struct_subject0_trial3_keshav1.csv')
data = data.drop(['Present Timestamp','Previous Label(one frame)'],axis=1)
cols = data.columns.tolist()
cols = [col for col in data if col != 'New label (Present Label)'] + ['New label (Present Label)'] 
data = data[cols]
data['New label (Present Label)'] = data['New label (Present Label)'].replace(['free_gestures','stop_gesture','init_gesture'],0)
data['New label (Present Label)'] = data['New label (Present Label)'].replace(['gesture'],1)
data.to_csv('Time_struct_subject0_trial3_keshav2.csv' ,  index = False, header=False)