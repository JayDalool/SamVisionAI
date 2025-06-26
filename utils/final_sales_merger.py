import pandas as pd
from typing import List

class FinalSalesMerger:
    """Utility class to merge cleaned sales records from multiple sources."""

    EXPECTED_COLUMNS: List[str] = [
        "listing_date",
        "season",
        "mls_number",
        "neighborhood",
        "address",
        "list_price",
        "sold_price",
        "sell_list_ratio",
        "dom",
        "bedrooms",
        "bathrooms",
        "garage_type",
        "house_type",
        "style",
        "type",
        "sqft",
        "lot_size",
    ]

    @classmethod
    def _validate_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize a DataFrame to the expected schema."""
        missing = [col for col in cls.EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        # Drop any extra columns
        return df[cls.EXPECTED_COLUMNS]

    @classmethod
    def merge_and_save(cls, text_df: pd.DataFrame, pdf_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
        """Merge text and PDF sale DataFrames into a single CSV.

        Parameters
        ----------
        text_df: pd.DataFrame
            Cleaned sales parsed from text files.
        pdf_df: pd.DataFrame
            Cleaned sales parsed from PDF files.
        output_path: str
            Destination path for the merged CSV.
        """
        if text_df is None and pdf_df is None:
            raise ValueError("At least one DataFrame must be provided")

        frames = []
        if text_df is not None and not text_df.empty:
            frames.append(cls._validate_df(text_df))
        if pdf_df is not None and not pdf_df.empty:
            frames.append(cls._validate_df(pdf_df))

        if not frames:
            raise ValueError("No data to merge")

        merged = pd.concat(frames, ignore_index=True)
        # Drop duplicate addresses keeping the last occurrence
        merged = merged.drop_duplicates(subset=["address"], keep="last")
        # Ensure column order
        merged = merged[cls.EXPECTED_COLUMNS]

        merged.to_csv(output_path, index=False)
        return merged
