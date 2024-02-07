from pocket_strmod import Protein, Ligand, SplitPocket 

a = Protein("/export/home/yangzhenyu/pocket_strmod_/pockets/Mpro_initial/6lze_protein.pdb")
l = Ligand("/export/home/yangzhenyu/pocket_strmod_/pockets/Mpro_initial/structure_select_initial.sdf")
result_pdb,result_ligand = SplitPocket._split_pocket_with_surface_atoms(a,l,15)
with open("pockets/Mpro_initial/Mpro_select_initial_pocket15.pdb", "w") as f:
    f.write(result_pdb)

