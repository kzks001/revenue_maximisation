import pandas as pd


def load_excel_sheets(file_path: str) -> dict[str, pd.DataFrame]:
    """
    Loads all sheets from an Excel file into a dictionary.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        dict[str, pd.DataFrame]: Dictionary with sheet names as keys and DataFrames as values.
    """
    return pd.read_excel(file_path, sheet_name=None, engine="openpyxl")


def extract_data_dictionary(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Extracts the data dictionary (assumed to be the first sheet).

    Args:
        sheets (dict[str, pd.DataFrame]): Dictionary of sheet names and DataFrames.

    Returns:
        pd.DataFrame: Data dictionary from the first sheet.
    """
    sheet_names = list(sheets.keys())
    return sheets[sheet_names[0]]


def merge_data_sheets_by_client(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merges all sheets except the first one into a single DataFrame, using the "Client" column.

    Args:
        sheets (dict[str, pd.DataFrame]): Dictionary of sheet names and DataFrames.

    Returns:
        pd.DataFrame: Merged DataFrame of all data sheets on the "Client" column.
    """
    sheet_names = list(sheets.keys())
    data_frames = [sheets[sheet] for sheet in sheet_names[1:]]

    # Ensure "Client" column exists in all dataframes
    for df in data_frames:
        if "Client" not in df.columns:
            raise ValueError("Column 'Client' is missing in one of the sheets.")

    # Iteratively merge the data frames on the "Client" column
    merged_data = data_frames[0]
    for df in data_frames[1:]:
        merged_data = pd.merge(merged_data, df, on="Client", how="outer")

    # Sort the resulting DataFrame by "Client"
    merged_data = merged_data.sort_values(by="Client").reset_index(drop=True)
    return merged_data


def ingest_and_merge_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingests an Excel file and merges all data sheets into a single DataFrame,
    excluding the first sheet, which is treated as the data dictionary.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - The data dictionary (first sheet) as a DataFrame.
            - A merged DataFrame of all other sheets, sorted by "Client".
    """
    sheets = load_excel_sheets(file_path)
    data_dictionary = extract_data_dictionary(sheets)
    merged_data = merge_data_sheets_by_client(sheets)
    return data_dictionary, merged_data
