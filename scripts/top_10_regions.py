import pandas as pd

### Display the top 10 regions with the highest and lowest coefficients

CONTROL_ONLY = False
MALE = False
FEMALE = False
SAVE_RESULTS = True

if MALE:
    sex = '_male'
elif FEMALE:
    sex = '_female'
else:
    sex = ''

if CONTROL_ONLY:
    file_name = 'control'
else:
    file_name = 'control_moderate'

SC_regions_info = pd.read_csv(f'data/regions_info/aal_roi_vals/SC_regions_info_roi_vals_{file_name}{sex}.csv')
FC_regions_info = pd.read_csv(f'data/regions_info/aal_roi_vals/FC_regions_info_roi_vals_{file_name}{sex}.csv')

if SAVE_RESULTS:
    # Open a text file to write the output
    with open(f'data/regions_info/top_10_regions_{file_name}{sex}.txt', 'w') as f:
        f.write("SC\n")
        f.write(SC_regions_info.sort_values(by='SC_pos_coefs', ascending=False).head(10)[['ROI', 'Yeo_Network', 'SC_pos_coefs']].to_string(index=False) + '\n')
        f.write(SC_regions_info.sort_values(by='SC_neg_coefs', ascending=True).head(10)[['ROI', 'Yeo_Network', 'SC_neg_coefs']].to_string(index=False) + '\n')

        f.write("\nFC\n")
        f.write(FC_regions_info.sort_values(by='FC_pos_coefs', ascending=False).head(10)[['ROI', 'Yeo_Network', 'FC_pos_coefs']].to_string(index=False) + '\n')
        f.write(FC_regions_info.sort_values(by='FC_neg_coefs', ascending=True).head(10)[['ROI', 'Yeo_Network', 'FC_neg_coefs']].to_string(index=False) + '\n')

        f.write("\nFCgsr\n")
        f.write(FC_regions_info.sort_values(by='FCgsr_pos_coefs', ascending=False).head(10)[['ROI', 'Yeo_Network', 'FCgsr_pos_coefs']].to_string(index=False) + '\n')
        f.write(FC_regions_info.sort_values(by='FCgsr_neg_coefs', ascending=True).head(10)[['ROI', 'Yeo_Network', 'FCgsr_neg_coefs']].to_string(index=False) + '\n')

    print(f"Values written to data/regions_info/top_10_regions_{file_name}{sex}.txt")
else:
    print("SC")
    print(SC_regions_info.sort_values(by='SC_pos_coefs', ascending=False).head(10)[['ROI', 'Yeo_Network', 'SC_pos_coefs']].to_string(index=False))
    print(SC_regions_info.sort_values(by='SC_neg_coefs', ascending=True).head(10)[['ROI', 'Yeo_Network', 'SC_neg_coefs']].to_string(index=False))

    print("FC")
    print(FC_regions_info.sort_values(by='FC_pos_coefs', ascending=False).head(10)[['ROI', 'Yeo_Network', 'FC_pos_coefs']].to_string(index=False))
    print(FC_regions_info.sort_values(by='FC_neg_coefs', ascending=True).head(10)[['ROI', 'Yeo_Network', 'FC_neg_coefs']].to_string(index=False))

    print("FCgsr")
    print(FC_regions_info.sort_values(by='FCgsr_pos_coefs', ascending=False).head(10)[['ROI', 'Yeo_Network', 'FCgsr_pos_coefs']].to_string(index=False))
    print(FC_regions_info.sort_values(by='FCgsr_neg_coefs', ascending=True).head(10)[['ROI', 'Yeo_Network', 'FCgsr_neg_coefs']].to_string(index=False))