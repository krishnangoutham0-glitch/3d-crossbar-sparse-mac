    import pandas as pd
    import numpy as np

    # === CONFIG ===
    conductance_csv = "/Users/gouthamkrishnan/PycharmProjects/multplication/Q_new_models/layer2_weights_neg.csv"  # Your conductance matrix file
    input_csv = '/Users/gouthamkrishnan/PycharmProjects/multplication/Q_new_models/side_out/mac_output_l1_final.csv'              # Your MNIST-style input file
    output_spice = '/Users/gouthamkrishnan/PycharmProjects/multplication/Q_new_models/side_out/l2_neg.spice'
    output_netlist = '/Users/gouthamkrishnan/PycharmProjects/multplication/Q_new_models/side_out/l2_neg.netlist'

    df = pd.read_csv("/Users/gouthamkrishnan/PycharmProjects/multplication/Q_new_models/model_unique_res - layer2_unique_res_neg.csv")
    dict_res  = dict(zip(df['resistance'], zip(df['w'], df['l'])))



    # === LOAD FILES ===
    conductance_df = pd.read_csv(conductance_csv, header=None)
    conductance_df = conductance_df.iloc[1:, ].reset_index(drop=True).astype(float)

    input_df = pd.read_csv(input_csv, header=None)
    conductance = conductance_df.to_numpy()
    norm_inputs = input_df[5].values
    print(norm_inputs.shape)


    # Normalize input to [0, 0.1]


    num_rows, num_cols = conductance.shape

    # === BUILD NETLIST ===
    netlist = []

    # Header
    netlist.append("** Auto-generated Crossbar Netlist with Bitlines and Gate Voltage\n")
    netlist.append("**.subckt crossbar\n")

    resistor_count = 1

    # for row in range(num_rows):
    #     for col in range(num_cols):
    #         G = conductance[row, col]
    #         if G == 0:
    #             continue  # skip open circuits
    #
    #         R_value = 1.0 / G
    #         # Find closest match in dict_res
    #         res_keys = np.array(list(dict_res.keys()))
    #         closest_res = res_keys[np.argmin(np.abs(res_keys - R_value))]
    #         W, L = dict_res[closest_res]
    #
    #         # Nets
    #         net_top = f"WL{row + 1}"
    #         net_bottom = f"BL{col + 1}"

    for row in range(num_rows):
        for col in range(num_cols):
            G = conductance[row, col]
            if G == 0:
                continue

            R_value = 1.0 / G
            res_keys = np.array(list(dict_res.keys()))
            closest_res = res_keys[np.argmin(np.abs(res_keys - R_value))]
            W, L = dict_res[closest_res]

            # Node names
            wl = f"WL{row + 1}"             # NMOS drain
            bl = f"BL{col + 1}"             # Bottom of resistor
            nmos_src = f"N{resistor_count}" # NMOS source = resistor top





            # FET line
            fet_line = (
                f"XM{resistor_count} {wl} VGATE {nmos_src} GND "
                f"sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1 "
                f"ad='int((1 + 1)/2) * 1 / 1 * 0.29' as='int((1 + 2 ) /2) * 1 / 1 * 0.29' "
                f"pd='2*int((1 + 1)/2) * (1 / 1 + 0.29)' ps='2*int((1 + 2)/2) * (1 / 1 + 0.29)' "
                f"nrd='0.29 / 1' nrs='0.29 / 1' sa=0 sb=0 sd=0 mult=1 m=1")

            # Resistor line
            res_line = (f"XR{resistor_count} {bl} {nmos_src}  GND "
                        f"sky130_fd_pr__res_xhigh_po L={L} W=0.15 mult=1 m=1")


            netlist.append(fet_line)
            netlist.append(res_line)

            resistor_count += 1

    # === VOLTAGE SOURCES ===
    netlist.append("\n* Wordline Voltage sources")
    for row in range(num_rows):
        voltage_value = norm_inputs[row]
        netlist.append(f"V_IN{row + 1} WL{row + 1} GND {voltage_value:.4f}")

    netlist.append("\n* Bitline Voltage sources (0V)")
    for col in range(num_cols):
        netlist.append(f"V_BL{col + 1} BL{col + 1} GND 0")

    # Gate Voltage Source (1.8V)
    netlist.append("\n* Common Gate Voltage Source")
    netlist.append("V_GATE VGATE GND 1")

    # === CONTROL SECTION ===
    netlist.append("\n**** begin user architecture code")
    netlist.append(".lib /usr/local/share/pdk/sky130B/libs.tech/combined/sky130.lib.spice tt")
    netlist.append("\n.control \n set compat = ng \n op \n .endc")


     # Example: plotting V1 current

    netlist.append("\n**** end user architecture code")
    netlist.append("\n.GLOBAL GND")
    netlist.append(".end\n")

    # === SAVE FILES ===
    with open(output_spice, 'w') as f:
        f.write('\n'.join(netlist))

    with open(output_netlist, 'w') as f:
        f.write('\n'.join(netlist))

    print(f"Netlist generated: {output_spice}, {output_netlist}")