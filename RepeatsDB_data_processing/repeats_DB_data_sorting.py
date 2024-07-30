import pandas as pd
import numpy as np
import sys
from Bio.PDB import PDBList

def main(input_file):
    # Read in data
    LRR_data = pd.read_table(input_file)

    # Remove the first two columns: review and code number
    LRR_data.drop(LRR_data.columns[[0, 1]], axis=1, inplace=True)

    # Group data by PDB accession code
    PDB_groups = LRR_data.groupby('RepeatsDB ID')

    # Get the PDB/chain ID for each repeat structure
    PDB_dict = list(PDB_groups.groups.keys())

    with open("cmd_file.txt", "w") as f:
        for pdb_entry in PDB_dict:
            PDB_ID = pdb_entry[:4]
            chain_ID = pdb_entry[-1]

            subgroup = PDB_groups.get_group(pdb_entry)
            insertion = subgroup[subgroup['Type'].str.contains("insertion")].reset_index()
            subgroup_repeats = subgroup[~subgroup['Type'].str.contains("insertion")].reset_index()

            if len(subgroup_repeats) <= 4:
                continue

            insertion_full = np.array([], dtype=int)
            if not insertion.empty:
                for _, row in insertion.iterrows():
                    insertion_range = np.arange(row["start"], row["end"] + 1)
                    insertion_full = np.concatenate([insertion_full, insertion_range])

            column_start = subgroup_repeats["start"]
            column_end = subgroup_repeats["end"]
            repeat_length = column_end - column_start
            standard_deviation = np.std(repeat_length)

            if standard_deviation > 3:
                continue

            start_full = column_start.tolist()
            average_repeat_length = int(np.mean(repeat_length))
            start_res = column_start.min()
            end_res = column_end.max()
            
            pdbl=PDBList()
            pdbl.retrieve_pdb_file(str(PDB_ID), file_format="pdb", overwrite=True, pdir=".")

            f.write(f"python rep_pam.py {PDB_ID} pdb{PDB_ID} {chain_ID} {start_res} {end_res} "
                    f"{average_repeat_length} --Ca {' '.join(map(str, start_full))} "
                    f"--insert {' '.join(map(str, insertion_full))}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    main(input_file)
