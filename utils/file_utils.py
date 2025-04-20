import pandas as pd
import glob
import os

from dawn_vok.utils.dir_utils import DirUtils


class FileUtils:
    @classmethod
    def combine_csv_files(cls, input_path, output_path, file_pattern, output_filename):
        # --- Configuration ---

        # 1. Specify the path to the directory containing your CSV files
        #    Replace 'path/to/your/csv/files' with the actual path
        csv_directory = DirUtils.get_raw_data_dir(path=input_path)
        
        # 2. Specify a pattern to match your CSV files
        #    '*.csv' will match all files ending with .csv in the directory
        file_pattern = '*.csv'

        # 3. Specify the name for the output combined CSV file

        # 4. Specify the full path for the output file
        output_path = os.path.join(csv_directory, output_filename)

        # --- Logic ---

        # Construct the full search pattern
        search_pattern = os.path.join(csv_directory, file_pattern)

        # Find all files matching the pattern
        all_files = glob.glob(search_pattern)

        # Ensure the output file itself is not included in the list of files to combine
        # (in case you run the script multiple times in the same directory)
        try:
            all_files.remove(output_path)
        except ValueError:
            pass # Output file doesn't exist yet or wasn't matched, which is fine

        if not all_files:
            print(f"No CSV files found matching '{search_pattern}'. Please check the directory and pattern.")
        else:
            print(f"Found {len(all_files)} files to combine:")
            for f in all_files:
                print(f" - {os.path.basename(f)}")

            # List to hold DataFrames
            li = []

            # Loop through all found CSV files
            for filename in all_files:
                try:
                    # Read each CSV file into a DataFrame
                    # Add options like `header=None` if your files don't have headers,
                    # or `encoding='your_encoding'` if they use a specific encoding.
                    df = pd.read_csv(filename, index_col=None, header=0)
                    li.append(df)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    print("Skipping this file.")


            # Concatenate all DataFrames in the list into one
            if li:
                combined_df = pd.concat(li, axis=0, ignore_index=True)

                # Write the combined DataFrame to a new CSV file
                try:
                    # index=False prevents pandas from writing the DataFrame index as a column
                    combined_df.to_csv(output_path, index=False, encoding='utf-8')
                    print(f"\nSuccessfully combined files into '{output_path}'")
                except Exception as e:
                    print(f"\nError writing combined file '{output_path}': {e}")
            else:
                print("\nNo valid dataframes were created. Combined file not generated.")

if __name__ == "__main__":
    FileUtils.combine_csv_files(input_path='provider/raw/ims', output_path='provider/raw/ims', file_pattern='*.csv', output_filename='ims_afeq_78.csv')