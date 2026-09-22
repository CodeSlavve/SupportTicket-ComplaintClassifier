import os
import zipfile

import pandas as pd
import requests

# ============================================================
# CONFIGURATION
# ============================================================

# Paste your CFPB archive ZIP URLs here.
ARCHIVE_URLS = [
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_21_August_2026.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_20_July_2026.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_19_June_2026.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_18_May_2026.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_17_April_2026.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_16_March_2026.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_15_January_2026_through_February_2026.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_14_November_2025_through_December_2025.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_13_September_2025_through_October_2025.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_12_July_2025_through_August_2025.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_11_May_2025_through_June_2025.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_10_March_2025_through_April_2025.zip",
    "https://files.consumerfinance.gov/f/documents/CCDB_Export_9_January_2025_through_February_2025.zip",
]

OUTPUT_PATH = "data/cfpb_sample_50k.csv"

DOWNLOAD_DIR = "data/cfpb_archives"

CHUNK_SIZE = 100_000

TARGET_ROWS = 50_000

RANDOM_STATE = 42


# ============================================================
# DOWNLOAD
# ============================================================


def download_file(url, output_path):

    if os.path.exists(output_path):
        print(f"Already downloaded: {output_path}")
        return

    print("\nDownloading:")
    print(url)

    response = requests.get(url, stream=True, timeout=120)

    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    downloaded = 0

    with open(output_path, "wb") as f:

        for chunk in response.iter_content(chunk_size=1024 * 1024):

            if chunk:

                f.write(chunk)

                downloaded += len(chunk)

                if total_size:
                    percent = downloaded / total_size * 100

                    print(f"\rProgress: {percent:.1f}%", end="")

    print("\nDownload complete.")


# ============================================================
# FIND CSV
# ============================================================


def find_csv_in_zip(zip_path):

    with zipfile.ZipFile(zip_path) as z:

        csv_files = [name for name in z.namelist() if name.lower().endswith(".csv")]

        if not csv_files:
            raise RuntimeError(f"No CSV found in {zip_path}")

        if len(csv_files) > 1:

            print("Multiple CSV files found:")

            for csv in csv_files:
                print(" ", csv)

        return csv_files[0]


# ============================================================
# FIND PRODUCT CATEGORIES
# ============================================================


def find_top_categories(zip_path, csv_name):

    print("\nFinding top Product categories...")

    product_counts = {}

    with zipfile.ZipFile(zip_path) as z, z.open(csv_name) as f:

        reader = pd.read_csv(
            f, usecols=["Product"], chunksize=CHUNK_SIZE, low_memory=False
        )

        for chunk in reader:

            counts = chunk["Product"].value_counts()

            for product, count in counts.items():

                product_counts[product] = product_counts.get(product, 0) + count

    counts = pd.Series(product_counts).sort_values(ascending=False)

    return counts.head(8).index.tolist()


# ============================================================
# PROCESS ARCHIVE
# ============================================================


def process_archive(zip_path, csv_name, quotas, collected):

    print(f"\nProcessing: {csv_name}")

    columns = [
        "Date received",
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Consumer complaint narrative",
        "Company public response",
        "Company",
        "State",
        "ZIP code",
        "Tags",
        "Submitted via",
        "Date sent to company",
        "Company response to consumer",
        "Timely response?",
        "Complaint ID",
    ]

    with zipfile.ZipFile(zip_path) as z, z.open(csv_name) as f:

        reader = pd.read_csv(
            f, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False
        )

        for chunk_number, chunk in enumerate(reader, start=1):

            # ------------------------------------------------
            # Stop reading if all categories are complete
            # ------------------------------------------------

            active_categories = [
                category for category, quota in quotas.items() if quota > 0
            ]

            if not active_categories:
                print("\nAll category quotas filled.")
                break

            # ------------------------------------------------
            # Remove empty narratives
            # ------------------------------------------------

            narrative = (
                chunk["Consumer complaint narrative"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            chunk = chunk[narrative.ne("")]

            # ------------------------------------------------
            # Keep only categories still needing rows
            # ------------------------------------------------

            chunk = chunk[chunk["Product"].isin(active_categories)]

            if chunk.empty:
                continue

            # ------------------------------------------------
            # Fill each category
            # ------------------------------------------------

            for category in active_categories:

                remaining = quotas[category]

                if remaining <= 0:
                    continue

                category_rows = chunk[chunk["Product"] == category]

                if category_rows.empty:
                    continue

                # Never collect more than quota
                take = min(remaining, len(category_rows))

                sampled = category_rows.sample(n=take, random_state=RANDOM_STATE)

                collected[category].append(sampled)

                quotas[category] -= take

                print(
                    f"\r{category}: "
                    f"{TARGET_ROWS // 8 - quotas[category]:,} / "
                    f"{TARGET_ROWS // 8:,}",
                    end="",
                )

        print()

    return quotas


# ============================================================
# MAIN
# ============================================================


def main():

    if not ARCHIVE_URLS:

        raise RuntimeError(
            "ARCHIVE_URLS is empty. " "Add your CFPB archive ZIP links first."
        )

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    os.makedirs("data", exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1
    # Get categories from FIRST archive
    # ---------------------------------------------------------

    first_url = ARCHIVE_URLS[0]

    first_filename = os.path.basename(first_url.split("?")[0])

    first_zip = os.path.join(DOWNLOAD_DIR, first_filename)

    download_file(first_url, first_zip)

    first_csv = find_csv_in_zip(first_zip)

    categories = find_top_categories(first_zip, first_csv)

    print("\nTarget categories:")

    for category in categories:
        print(f"  - {category}")

    # ---------------------------------------------------------
    # STEP 2
    # Create equal quotas
    # ---------------------------------------------------------

    quota_per_category = TARGET_ROWS // len(categories)

    quotas = {category: quota_per_category for category in categories}

    collected = {category: [] for category in categories}

    print(f"\nTarget per category: " f"{quota_per_category:,}")

    # ---------------------------------------------------------
    # STEP 3
    # Process archives sequentially
    # ---------------------------------------------------------

    for archive_index, url in enumerate(ARCHIVE_URLS, start=1):

        print(f"\n{'=' * 60}")

        print(f"ARCHIVE {archive_index} " f"/ {len(ARCHIVE_URLS)}")

        print(f"{'=' * 60}")

        # ---------------------------------------------
        # Download
        # ---------------------------------------------

        filename = os.path.basename(url.split("?")[0])

        zip_path = os.path.join(DOWNLOAD_DIR, filename)

        download_file(url, zip_path)

        # ---------------------------------------------
        # Find CSV
        # ---------------------------------------------

        csv_name = find_csv_in_zip(zip_path)

        print(f"CSV: {csv_name}")

        # ---------------------------------------------
        # Process
        # ---------------------------------------------

        quotas = process_archive(zip_path, csv_name, quotas, collected)

        # ---------------------------------------------
        # Progress report
        # ---------------------------------------------

        print("\nCurrent progress:")

        total_collected = 0

        for category in categories:

            filled = quota_per_category - quotas[category]

            total_collected += filled

            print(f"{category}: " f"{filled:,} / " f"{quota_per_category:,}")

        print(f"\nTotal: " f"{total_collected:,} / " f"{TARGET_ROWS:,}")

        # ---------------------------------------------
        # Stop if complete
        # ---------------------------------------------

        if total_collected >= TARGET_ROWS:

            print("\nAll 50,000 rows collected.")

            break

    # ---------------------------------------------------------
    # STEP 4
    # Build final dataset
    # ---------------------------------------------------------

    print("\nCombining collected data...")

    final_parts = []

    for category in categories:

        if not collected[category]:
            continue

        category_df = pd.concat(collected[category], ignore_index=True)

        final_parts.append(category_df)

    if not final_parts:

        raise RuntimeError("No usable complaint narratives found.")

    final_df = pd.concat(final_parts, ignore_index=True)

    # ---------------------------------------------------------
    # Final shuffle
    # ---------------------------------------------------------

    final_df = final_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 60}")

    print("FINAL DATASET")

    print(f"{'=' * 60}")

    print(f"Shape: {final_df.shape}")

    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFinal category distribution:")

    print(final_df["Product"].value_counts())


if __name__ == "__main__":
    main()
