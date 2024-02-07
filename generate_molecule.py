import argparse
import time
from pocket_strmod import pocket_strmod_WithEdgeNew, Generate
from pocket_strmod.utils import *
from pocket_strmod.utils import mask_node, Protein, ComplexData
from distutils.util import strtobool



def parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='cuda:x or cpu')
    parser.add_argument('--ckpt', type=str, default='None', help='the model parameter file path')
    parser.add_argument('-pkt', '--pocket', type=str, default='None', help='the pdb file of pocket in receptor')
    parser.add_argument('-lig', '--ligand', type=str, default='None', help='the molecular scaffold file of pocket in receptor')
    parser.add_argument('--idx', type=str, default= None, help='the sites where atoms and covalent bonds are generated')
    parser.add_argument('-n', '--num_gen', type=int, default=100, help='the number of molecules to be generated')
    parser.add_argument('--name', type=str, default='receptor', help='The folder where generated molecules files are saved')
    parser.add_argument('--max_atom_num', type=int, default=40, help='the maximum atom number limitation of molecules which are generated')
    parser.add_argument('--with_print', type=lambda x:bool(strtobool(x)), default=False, help='whether print SMILES in generative process')
    parser.add_argument('--root_path', type=str, default='gen_results', help='the file path for saving generation results')
    parser.add_argument('--readme', '-rm', type=str, default='None', help='description of this genrative task')
    parser.add_argument('--substruct', type=str, default= None, help='The substructure to be excluded from generation')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    args = parameter()
    #args.idx = eval(args.idx)
    if args.substruct == None:
        substruct = []
    else:
        substruct = open(args.substruct).read().split("\n")

    args.pocket = 'pockets/Mpro_initial/Mpro_initial_pocket15.pdb'
    args.ligand = 'pockets/Mpro_initial/Mpro_initial_molecule_scaffold.sdf'
    args.ckpt = './trained_model/parameter.pt' 
    args.root_path = 'gen_results/'
    args.name = 'Mpro_fragment_new'
    args.num_gen = 10
    args.idx = [12, 14, 15, 16]
    args.with_print = True
    args.device = 'cuda:0'
    args.with_print = True



    if args.name == 'receptor':
        args.name = args.pocket.split('/')[-1].split('-')[0]
    ## Load Target
    assert args.pocket != 'None', 'Please specify pocket !'
    assert args.ckpt != 'None', 'Please specify model !'
    pdb_file = args.pocket
    
    if args.ligand == 'None':
        pro_dict = Protein(pdb_file).get_atom_dict(removeHs=True, get_surf=True)
        lig_dict = Ligand.empty_dict()
        complex_data = ComplexData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pro_dict),
                        ligand_dict=torchify_dict(lig_dict),
                    )
    else:
        pro_dict = Protein(pdb_file).get_atom_dict(removeHs=True, get_surf=True)
        lig_dict = Ligand(args.ligand).to_dict()
        complex_data = ComplexData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pro_dict),
                        ligand_dict=torchify_dict(lig_dict),
                    )

    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
    pivotal_masker = pivotalMaker(r=6.0, num_work=16, atomic_numbers=[6,7,8,9,15,16,17,35,53])
    atom_composer = AtomComposer(knn=16, num_workers=16, for_gen=True, use_protein_bond=True)

    complex_data = RefineData()(complex_data)
    complex_data = LigandCountNeighbors()(complex_data)
    complex_data = protein_featurizer(complex_data)
    complex_data = ligand_featurizer(complex_data)
    node4mask = torch.arange(complex_data.ligand_pos.size(0))
    if args.ligand == 'None':
        complex_data = mask_node(complex_data, torch.empty([0], dtype=torch.long), node4mask, num_atom_type=9, y_pos_std=0.)
    else:
        complex_data = mask_node(complex_data, node4mask, torch.empty([0], dtype=torch.long), num_atom_type=9, y_pos_std=0.)
    complex_data = atom_composer.run(complex_data)

    ## Load model
    print('Loading model ...')
    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt['config']
    model = pocket_strmod_WithEdgeNew(config).to(device)
    model.load_state_dict(ckpt['model'])

    print('Generating molecules ...')
    temperature = [1.0, 1.0]
    generate = Generate(model, atom_composer.run, temperature=temperature, atom_type_map=[6,7,8,9,15,16,17,35,53],
                        num_bond_type=4, max_atom_num=args.max_atom_num, focus_threshold= 0.5, 
                        max_double_in_6ring=0, min_dist_inter_mol=3.0,
                        bond_length_range=(1.0,2.0), choose_max= True, device=device, gen_idx = args.idx,
                        sub_structure = substruct)
    start = time.time()
    
    generate.generate(complex_data, num_gen=args.num_gen, rec_name=args.name, with_print=args.with_print,
                      root_path=args.root_path)
    os.system('cp {} {}'.format(args.ckpt, generate.out_dir))
    gen_config = '\n'.join(['{}: {}'.format(k,v) for k,v in args.__dict__.items()])
    with open(generate.out_dir + '/readme.txt', 'w') as fw:
        fw.write(gen_config)
    end = time.time()
    print('Time: {}'.format(timewait(end-start)))
