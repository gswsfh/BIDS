import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data = [
    ("Destination Port", 0.1046),
    ("Flow Duration", 0.00015),
    ("Total Fwd Packets", 0.000104),
    ("Total Backward Packets", 0),
    ("Total Length of Fwd Packets", 0.00094),
    ("Total Length of Bwd Packets", 0.00038),
    ("Fwd Packet Length Max", 0.00468),
    ("Fwd Packet Length Min", 0.000369),
    ("Fwd Packet Length Mean", 0.00033),
    ("Fwd Packet Length Std", 0.00966),
    ("Bwd Packet Length Max", 0.0001176),
    ("Bwd Packet Length Min", 0.001276),
    ("Bwd Packet Length Mean", 0.00096),
    ("Bwd Packet Length Std", 0.3754),
    ("Flow Bytes/s", 0.0004479),
    ("Flow Packets/s", 0.00011876),
    ("Flow IAT Mean", 0.000394),
    ("Flow IAT Std", 0.000793),
    ("Flow IAT Max", 0.0006652),
    ("Flow IAT Min", 0.0034749),
    ("Fwd IAT Total", 0.000405),
    ("Fwd IAT Mean", 0.0033225),
    ("Fwd IAT Std", 0.00020),
    ("Fwd IAT Max", 0.0035),
    ("Fwd IAT Min", 0.0054),
    ("Bwd IAT Total", 2.52e-05),
    ("Bwd IAT Mean", 0.00204),
    ("Bwd IAT Std", 0.00193),
    ("Bwd IAT Max", 0.000113),
    ("Bwd IAT Min", 0.000138),
    ("Fwd PSH Flags", 3.1899e-06),
    ("Bwd PSH Flags", 0),
    ("Fwd URG Flags", 0),
    ("Bwd URG Flags", 0),
    ("Fwd Header Length", 0.0011),
    ("Bwd Header Length", 0.13266),
    ("Fwd Packets/s", 0.0002328),
    ("Bwd Packets/s", 0.00143956),
    ("Min Packet Length", 0.00032),
    ("Max Packet Length", 0.093),
    ("Packet Length Mean", 0.001195),
    ("Packet Length Std", 2.555e-06),
    ("FIN Flag Count", 9.7212e-05),
    ("SYN Flag Count", 0),
    ("RST Flag Count", 0),
    ("PSH Flag Count", 3.695e-05),
    ("ACK Flag Count", 0.000219),
    ("URG Flag Count", 0.00039),
    ("CWE Flag Count", 0),
    ("ECE Flag Count", 0),
    ("Down/Up Ratio", 0.000277),
    ("Average Packet Size", 0.1897766),
    ("Avg Fwd Segment Size", 8.10824e-05),
    ("Avg Bwd Segment Size", 0.00149),
    ("Fwd Header Length", 0.00048),
    ("Fwd Avg Bytes/Bulk", 0),
    ("Fwd Avg Packets/Bulk", 0),
    ("Fwd Avg Bulk Rate", 0),
    ("Bwd Avg Bytes/Bulk", 0),
    ("Bwd Avg Packets/Bulk", 0),
    ("Bwd Avg Bulk Rate", 0),
    ("Subflow Fwd Packets", 0.0001358),
    ("Subflow Fwd Bytes", 0.00011299),
    ("Subflow Bwd Packets", 3.133e-06),
    ("Subflow Bwd Bytes", 0.00037),
    ("Init_Win_bytes_forward", 0.0266),
    ("Init_Win_bytes_backward", 0.00685),
    ("act_data_pkt_fwd", 0),
    ("min_seg_size_forward", 0.01866),
    ("Active Mean", 6.44e-06),
    ("Active Std", 0.002079),
    ("Active Max", 7.9855e-07),
    ("Active Min", 0.00011),
    ("Idle Mean", 9.3282e-06),
    ("Idle Std", 1.4129e-07),
    ("Idle Max", 6.3487e-06),
    ("Idle Min", 1.5381e-05)
]

if __name__ == '__main__':

    df = pd.DataFrame(data, columns=["Feature", "Importance"])

    df_sorted = df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    mean_importance = df_sorted["Importance"].mean()

    N_SHOW = 20
    df_top = df_sorted.head(N_SHOW)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(df_top["Feature"], df_top["Importance"], color='skyblue', edgecolor='black')

    plt.axvline(mean_importance, color='red', linestyle='--', linewidth=2, label=f'Average importance: {mean_importance:.5f}')

    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.title(f"Top {N_SHOW} of feature importance (red line indicates average importance)", fontsize=16)
    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        plt.text(width + max(df_top["Importance"]) * 0.005, bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', va='center', fontsize=10)

    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Average importance: {mean_importance:.6f}")