io_features = [
    "VolumeCred",
    "VolumeCred_CA",
    "TransactionsCred",
    "TransactionsCred_CA",
    "VolumeDeb",
    "VolumeDeb_CA",
    "VolumeDebCash_Card",
    "VolumeDebCashless_Card",
    "VolumeDeb_PaymentOrder",
    "TransactionsDeb",
    "TransactionsDeb_CA",
    "TransactionsDebCash_Card",
    "TransactionsDebCashless_Card",
    "TransactionsDeb_PaymentOrder",
]
prod_bal_features = [
    "Count_CA",
    "Count_SA",
    "Count_MF",
    "Count_OVD",
    "Count_CC",
    "Count_CL",
    "ActBal_CA",
    "ActBal_SA",
    "ActBal_MF",
    "ActBal_OVD",
    "ActBal_CC",
    "ActBal_CL",
]
dem_features = ["Sex", "Age", "Tenure"]
sale_target = ["Sale_MF", "Sale_CC", "Sale_CL"]
revenue_target = ["Revenue_MF", "Revenue_CC", "Revenue_CL"]

categorical_features = [
    "Sex",
    "Count_CA",
    "Count_SA",
    "Count_MF",
    "Count_OVD",
    "Count_CC",
    "Count_CL",
]

continuous_features = [
    "Age",
    "Tenure",
    "ActBal_CA",
    "ActBal_SA",
    "ActBal_MF",
    "ActBal_OVD",
    "ActBal_CC",
    "ActBal_CL",
    "VolumeCred",
    "VolumeCred_CA",
    "TransactionsCred",
    "TransactionsCred_CA",
    "VolumeDeb",
    "VolumeDeb_CA",
    "VolumeDebCash_Card",
    "VolumeDebCashless_Card",
    "VolumeDeb_PaymentOrder",
    "TransactionsDeb",
    "TransactionsDeb_CA",
    "TransactionsDebCash_Card",
    "TransactionsDebCashless_Card",
    "TransactionsDeb_PaymentOrder",
]

asset_prods_suffix = ["CA", "SA", "MF"]
liability_prods_suffix = ["OVD", "CC", "CL"]

new_categorical_features = ["Count_Total_Asset", "Count_Total_Liability"]
new_continuous_features = [
    "ActBal_Total_Asset",
    "ActBal_Total_Liability",
    "ActBal_Total",
    "Debt_To_Balance",
    "Inflow_Outflow_Ratio",
]

sale_cc_features = [
    "ActBal_Total_Asset",
    "ActBal_Total_Liability",
    "ActBal_Total",
    "Age",
    "Count_Total_Asset",
    "Count_Total_Liability",
    "Debt_To_Balance",
    "Inflow_Outflow_Ratio",
    "TransactionsDebCashless_Card",
    "VolumeDeb_PaymentOrder",
]

revenue_cc_features = [
    "ActBal_Total_Asset",
    "ActBal_Total_Liability",
    "ActBal_Total",
    "Age",
    "Count_Total_Asset",
    "Count_Total_Liability",
    "Debt_To_Balance",
    "Inflow_Outflow_Ratio",
    "Tenure",
    "TransactionsDeb_CA",
    "TransactionsDeb",
    "TransactionsDebCashless_Card",
    "VolumeDebCashless_Card",
]

sale_cl_features = [
    "ActBal_OVD",
    "ActBal_Total_Asset",
    "ActBal_Total_Liability",
    "ActBal_Total",
    "Age",
    "Count_Total_Asset",
    "Count_Total_Liability",
    "Debt_To_Balance",
    "Inflow_Outflow_Ratio",
    "Tenure",
    "TransactionsCred",
]

revenue_cl_features = [
    "ActBal_CA",
    "ActBal_CC",
    "ActBal_Total_Asset",
    "ActBal_Total_Liability",
    "ActBal_Total",
    "Age",
    "Count_Total_Asset",
    "Count_Total_Liability",
    "Debt_To_Balance",
    "Inflow_Outflow_Ratio",
    "TransactionsCred",
    "TransactionsDeb_PaymentOrder",
    "TransactionsDeb",
]

sale_mf_features = [
    "ActBal_Total_Asset",
    "ActBal_Total_Liability",
    "ActBal_Total",
    "Age",
    "Count_MF",
    "Count_Total_Asset",
    "Count_Total_Liability",
    "Debt_To_Balance",
    "Inflow_Outflow_Ratio",
    "TransactionsCred_CA",
    "TransactionsCred",
]

revenue_mf_features = [
    "ActBal_OVD",
    "ActBal_Total_Asset",
    "ActBal_Total_Liability",
    "ActBal_Total",
    "Age",
    "Count_Total_Asset",
    "Count_Total_Liability",
    "Debt_To_Balance",
    "Inflow_Outflow_Ratio",
    "TransactionsDeb_PaymentOrder",
    "TransactionsDebCash_Card",
    "VolumeCred_CA",
    "VolumeDeb_PaymentOrder",
]

suffixes = ["CC", "CL", "MF"]
