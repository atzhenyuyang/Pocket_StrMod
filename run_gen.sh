python generate_molecule.py -pkt ./pockets/Mpro_second_round/C2_pocket10.pdb -lig ./pockets/Mpro_second_round/C2_ligand_kekulized.sdf --idx "[9, 12，17]" --ckpt ./trained_model/parameter.pt -n 500 --folder_name Mpro_new_generate --path_name gen_results_new -d cuda:0 --lig_max 33