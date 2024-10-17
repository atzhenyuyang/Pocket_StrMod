import torch
import time
from rdkit import Chem
from .molfilter import Filter
from torch_geometric.nn import knn, radius
from .utils import get_tri_edges, add_ligand_atom_to_data, data2mol, modify, check_valency, check_alert_structures, verify_dir_exists, substructure
from .autoregressive_process import embed_compose
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def selection(pivotal_pred, idx, start_idx):
    for i in range(0,start_idx):
        if i in idx:
            continue
        else:
            pivotal_pred[i] = -100
    return pivotal_pred
    
max_valence_dict = {
                6:torch.LongTensor([4]), 7:torch.LongTensor([3]), 8:torch.LongTensor([2]),
                9:torch.LongTensor([1]), 15:torch.LongTensor([5]), 16:torch.LongTensor([6]),
                17:torch.LongTensor([1]), 35:torch.LongTensor([1]), 53:torch.LongTensor([1]),
                1:torch.LongTensor([1])
                }

class Generate(object):
    def __init__(self, model, transform, temperature=[1.0, 1.0], possible_atom_type=[6,7,8,9,15,16,17,35,53],
                 num_bond_type=4, lig_max=35, pivotal_threshold=0.5, max_double_in_6ring=0, min_dist_inter_mol=3.0, bond_length_range=(1.0, 2.0),
                 choose_max=True, device='cuda:0', max_resample_node=100, max_resample_edge=100, remove_tri_ring=True, gen_idx = None, sub_structure = [], filter = None):
       
        self.model = model
        self.transform = transform
        self.temperature = temperature
        self.possible_atom_type = possible_atom_type
        self.num_bond_type = num_bond_type
        self.lig_max = lig_max
        self.pivotal_threshold = pivotal_threshold
        self.max_double_in_6ring = max_double_in_6ring
        self.min_dist_inter_mol = min_dist_inter_mol
        self.bond_length_range = bond_length_range
        self.choose_max = choose_max
        self.hidden_channels = model.config.hidden_channels
        self.knn = model.config.encoder.knn
        self.device = device
        self.max_resample_node = max_resample_node
        self.max_resample_edge = max_resample_edge
        self.remove_tri_ring = remove_tri_ring
        self.bond_type_map =  {
            1: Chem.rdchem.BondType.SINGLE, 
            2: Chem.rdchem.BondType.DOUBLE, 
            3: Chem.rdchem.BondType.TRIPLE
            }
        self.gen_idx = gen_idx
        self.sub_structure = sub_structure

    #@staticmethod
    def __choose_pivotal(self, pivotal_net, h_ctx, ctx_idx, pivotal_threshold, choose_max, surf_mask=None):
        pivotal_pred = pivotal_net(h_ctx, ctx_idx)
        if self.gen_idx != None:
            pivotal_pred = selection(pivotal_pred, self.gen_idx, self.start_idx)
        pivotal_prob = torch.sigmoid(pivotal_pred).view(-1)
        if choose_max:
            max_idx = pivotal_pred.argmax()
            pivotal_idx_candidate = ctx_idx[max_idx].view(-1)
            pivotal_prob = pivotal_prob[max_idx].view(-1)
        else:
            if isinstance(surf_mask, torch.Tensor) and surf_mask.sum() > 0:
                surf_idx = torch.nonzero(surf_mask).view(-1)
                pivotal_prob_surf = pivotal_prob[surf_mask]
                surf_pivotal_mask = pivotal_prob_surf > pivotal_threshold
                pivotal_idx_candidate = surf_idx[surf_pivotal_mask]
                pivotal_prob = pivotal_prob_surf[surf_pivotal_mask]
                if surf_pivotal_mask.sum() == 0:
                    return False, pivotal_prob
            else:
                pivotal_mask = (pivotal_prob >= pivotal_threshold).view(-1)
                pivotal_idx_candidate = ctx_idx[pivotal_mask]
                pivotal_prob = pivotal_prob[pivotal_mask]
                if pivotal_mask.sum() == 0:
                    return False, pivotal_prob
        return pivotal_idx_candidate, pivotal_prob

    def choose_pivotal(self, h_cpx, cpx_index, idx_ligand_ctx_in_cpx, data, atom_idx):
        if atom_idx == 0:
            if 'protein_surface_mask' in data:
                surf_mask = data.protein_surface_mask
            else:
                surf_mask = None
            pivotal_idx_, pivotal_prob = self.__choose_pivotal(
                self.model.pivotal_net, h_cpx, cpx_index, 
                self.pivotal_threshold, self.choose_max, 
                surf_mask=surf_mask)
        else:
            pivotal_idx_, pivotal_prob = self.__choose_pivotal(
                self.model.pivotal_net, h_cpx, idx_ligand_ctx_in_cpx, 
                self.pivotal_threshold, 1
                )
        if pivotal_idx_ is False:
            return False
        
        pivotal_valence_check = False
        if data.ligand_context_element.size(0) > 3 and pivotal_idx_ is not False and atom_idx != 0:
            max_valence = data.max_atom_valence[pivotal_idx_]
            valence_in_ligand_context_pivotal = data.ligand_context_valence[pivotal_idx_]
            valence_mask = max_valence > valence_in_ligand_context_pivotal
            pivotal_idx_ = pivotal_idx_[valence_mask]
            pivotal_prob = pivotal_prob[valence_mask]
            pivotal_valence_check = valence_mask.sum() == 0
            if pivotal_valence_check:
                return False
            else:
                self.counter += 1 
        if pivotal_valence_check:
            return False
        return pivotal_idx_, pivotal_prob

    def atom_generate(self, h_cpx, pivotal_idx, pivotal_prob, atom_idx):
        latent_atom = self.prior_node.sample([pivotal_idx.size(0)])
        latent_atom = self.model.atom_flow.reverse(latent_atom, h_cpx, pivotal_idx)
        if self.choose_max:
            if atom_idx == 0:
                new_atom_type_prob, new_atom_type_pool = torch.max(latent_atom, -1)
                new_atom_idx_with_max_prob = torch.flip(new_atom_type_prob.argsort(), dims=[0])#[:10]
                new_atom_type_pool = new_atom_type_pool[new_atom_idx_with_max_prob]
                new_atom_type_prob = torch.log(new_atom_type_prob[new_atom_idx_with_max_prob])

                pivotal_idx_ = pivotal_idx[new_atom_idx_with_max_prob]
                pivotal_choose_idx = torch.multinomial(pivotal_prob, 1)
                pivotal_idx_ = pivotal_idx[pivotal_choose_idx]
                new_atom_type = new_atom_type_pool[pivotal_choose_idx]
            else:
                new_atom_type_prob, new_atom_type = torch.max(latent_atom, -1)
                pivotal_idx_ = pivotal_idx
        else:
            new_atom_type_prob, new_atom_type_pool = torch.max(latent_atom, -1)
            new_atom_idx_with_max_prob = torch.flip(new_atom_type_prob.argsort(), dims=[0])#[:10]
            new_atom_type_pool = new_atom_type_pool[new_atom_idx_with_max_prob]
            new_atom_type_prob = torch.log(new_atom_type_prob[new_atom_idx_with_max_prob])

            pivotal_idx_ = pivotal_idx[new_atom_idx_with_max_prob]
            pivotal_choose_idx = torch.multinomial(pivotal_prob, 1)
            pivotal_idx_ = pivotal_idx[pivotal_choose_idx]
            new_atom_type = new_atom_type_pool[pivotal_choose_idx]
        return new_atom_type, pivotal_idx_
    
    def atom_generate_(self, h_cpx, pivotal_idx, pivotal_prob, atom_idx):
        new_atom_type, pivotal_idx_ = self.atom_generate(h_cpx, pivotal_idx, pivotal_prob, atom_idx)
        if atom_idx == 0:
            while new_atom_type.item() in {2,4,8}:
                new_atom_type, pivotal_idx_ = self.atom_generate(h_cpx, pivotal_idx, pivotal_prob, atom_idx)
                self.resample_node += 1
                if self.resample_node > self.max_resample_node:
                    new_atom_type = torch.tensor(1).to(self.device)
                    self.resample_node =- 20
                    break
        else:
            while new_atom_type.item() in {4,8}:
                new_atom_type, pivotal_idx_ = self.atom_generate(h_cpx, pivotal_idx, pivotal_prob, atom_idx)
                self.resample_node += 1
                if self.resample_node > self.max_resample_node:
                    break
        return new_atom_type, pivotal_idx_
    
    def pos_generate(self, h_cpx, atom_type_emb, pivotal_idx, cpx_pos, atom_idx):
        new_relative_pos, new_abs_pos, sigma, pi = self.model.pos_predictor(
                        h_cpx, 
                        pivotal_idx,
                        cpx_pos, 
                        atom_type_emb=atom_type_emb
                        )
        new_relative_pos = new_relative_pos.view(-1,3)
        new_abs_pos = new_abs_pos.view(-1,3)
        pi = pi.view(-1)
        dist = torch.norm(new_relative_pos, p=2, dim=-1)

        if atom_idx != 0:
            dist_mask = (dist>self.bond_length_range[0]) == (dist<self.bond_length_range[1])
            new_pos_to_add_ = new_abs_pos[dist_mask]
            check_pos = new_pos_to_add_.size(0) != 0
            if check_pos:
                pi_ = pi[dist_mask]
                pos_choose_idx = torch.multinomial(pi_, 1)
                new_pos_to_add = new_pos_to_add_[pos_choose_idx]
                return new_pos_to_add
            else:
                return False
        else:
            dist_mask = dist > self.min_dist_inter_mol
            new_pos_to_add_ = new_abs_pos[dist_mask]
            check_pos = new_pos_to_add_.size(0) != 0
            if check_pos:
                pi_ = pi[dist_mask]
                pos_choose_idx = torch.multinomial(pi_, 1)
                new_pos_to_add = new_pos_to_add_[pos_choose_idx]
                return new_pos_to_add
            else:
                new_pos_to_add = new_abs_pos[torch.argmax(dist).view(-1)]
                return new_pos_to_add

    def bond_generate(self, h_cpx, data, new_pos_to_add, atom_type_emb, atom_idx, rw_mol):
        if atom_idx == 0:
            is_check = False
            new_edge_idx = torch.empty([2, 0], dtype=torch.long)
            new_bond_type_to_add = torch.empty([0], dtype=torch.long)
        else:
            edge_index_query = radius(data.ligand_context_pos, new_pos_to_add, r=4.0, num_workers=16)
            pos_query_knn_edge_idx = knn(
                x=data.cpx_pos, y=new_pos_to_add, k=self.model.config.encoder.knn, num_workers=16
                )
            index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat = get_tri_edges(
                edge_index_query, 
                new_pos_to_add, 
                data.context_idx, 
                data.ligand_context_bond_index, 
                data.ligand_context_bond_type
                )
            resample_edge = 0
            no_bond = True
            while no_bond:
                if resample_edge >= self.max_resample_edge:
                    self.resample_edge_faild = True
                    rw_mol.RemoveAtom(atom_idx)
                    return False
                latent_edge = self.prior_edge.sample([edge_index_query.size(1)])
                edge_latent = self.model.edge_flow.reverse(
                    edge_latent=latent_edge,
                    pos_query=new_pos_to_add, 
                    edge_index_query=edge_index_query, 
                    cpx_pos=data.cpx_pos, 
                    node_attr_compose=h_cpx, 
                    edge_index_q_cps_knn=pos_query_knn_edge_idx, 
                    index_real_cps_edge_for_atten=index_real_cps_edge_for_atten, 
                    tri_edge_index=tri_edge_index, 
                    tri_edge_feat=tri_edge_feat,
                    atom_type_emb=atom_type_emb
                    )
                edge_pred_type = edge_latent.argmax(-1)
                edge_pred_mask = edge_pred_type > 0
                if edge_pred_mask.sum() > 0:
                    new_bond_type_to_add = edge_pred_type[edge_pred_mask]
                    new_edge_idx = edge_index_query[:,edge_pred_mask]

                    new_edge_vec = new_pos_to_add[new_edge_idx[0]]-data.cpx_pos[new_edge_idx[1]]
                    new_edge_dist = torch.norm(new_edge_vec, p=2, dim=-1)
                    if (new_edge_dist > self.bond_length_range[1]).sum() > 0:
                        if resample_edge >= self.max_resample_edge:
                            self.resample_edge_faild = True
                            rw_mol.RemoveAtom(atom_idx)
                            return False
                        else:
                            resample_edge += 1
                            continue
                        
                    for ix in range(new_edge_idx.size(1)):
                        i, j = new_edge_idx[:, ix].tolist()
                        bond_type = new_bond_type_to_add[ix].item()
                        rw_mol.AddBond(atom_idx, j, self.bond_type_map[bond_type])
                    valency_valid = check_valency(rw_mol)
                    has_alert = check_alert_structures(rw_mol, ['[C,N,O,S]=[C]=[C,N,O,S]'])
                    if valency_valid and has_alert == False:
                        no_bond = False
                        break
                    else:
                        for ix in range(new_edge_idx.size(1)):
                            i, j = new_edge_idx[:, ix].tolist()
                            bond_type = new_bond_type_to_add[ix].item()
                            rw_mol.RemoveBond(atom_idx, j)
                        resample_edge += 1
                        if resample_edge >= self.max_resample_edge:
                            self.resample_edge_faild = True
                            rw_mol.RemoveAtom(atom_idx)
                            return False
                else:
                    resample_edge += 1
        return rw_mol, new_edge_idx, new_bond_type_to_add


    def run(self, data):
        if data.idx_ligand_ctx_in_cpx.size(0) == 0:
            data.max_atom_valence = torch.empty(0, dtype=torch.long)
        else:
            data.max_atom_valence = torch.Tensor([max_valence_dict[i.item()] for i in data.ligand_context_element]).long()
        data = data.to(self.device)
        with torch.no_grad():
            self.prior_node = torch.distributions.normal.Normal(
                    torch.zeros([len(self.possible_atom_type)]).cuda(data.cpx_pos.device), 
                    self.temperature[0] * torch.ones([len(self.possible_atom_type)]).cuda(data.cpx_pos.device)
                    )
            self.prior_edge = torch.distributions.normal.Normal(
                    torch.zeros([self.num_bond_type]).cuda(data.cpx_pos.device), 
                    self.temperature[1] * torch.ones([self.num_bond_type]).cuda(data.cpx_pos.device)
                    )
            begin_atom_idx = 0 + data.idx_ligand_ctx_in_cpx.size(0)
            rw_mol = Chem.RWMol()
            if data.idx_ligand_ctx_in_cpx.size(0) != 0:
                for ele in data.ligand_context_element:
                    rw_mol.AddAtom(Chem.Atom(ele.item()))
                l_ = []
                for b_ix, b in enumerate(data.ligand_context_bond_index.T):
                    b1 = (b[0].item(), b[1].item())
                    b2 = (b[1].item(), b[0].item())
                    if b1 in l_ or b2 in l_:
                        continue
                    else:
                        bond_type = data.ligand_context_bond_type[b_ix].item()
                        rw_mol.AddBond(b1[0], b1[1], self.bond_type_map[bond_type])
                        l_.append(b1)
                        l_.append(b2)

            self.counter = 0
            for atom_idx in range(begin_atom_idx, self.lig_max):
                data = data.to(self.device)
                h_cpx = embed_compose(
                    data.cpx_feature.float(), data.cpx_pos, data.idx_ligand_ctx_in_cpx, 
                    data.idx_protein_in_cpx, self.model.ligand_atom_emb, 
                    self.model.protein_atom_emb, self.model.emb_dim
                    )
                # encoding context
                h_cpx = self.model.encoder(
                    node_attr = h_cpx,
                    pos = data.cpx_pos,
                    edge_index = data.cpx_edge_index,
                    edge_feature = data.cpx_edge_feature
                )

                self.resample_edge_faild = False
                self.check_node = True
                self.resample_node = 0
                while self.check_node:
                    if self.resample_node > self.max_resample_node:
                        break
                    
                    # choose pivotal atom
                    pivotal_out = self.choose_pivotal(
                            h_cpx, data.idx_protein_in_cpx, data.idx_ligand_ctx_in_cpx, 
                            data, atom_idx
                            )
                    if pivotal_out is False:
                        break
                    else:
                        pivotal_idx, pivotal_prob = pivotal_out
                    
                    # generate atom type
                    new_atom_type, pivotal_idx = self.atom_generate_(
                        h_cpx, pivotal_idx, pivotal_prob, atom_idx
                        )
                    # predict position of new atom
                    atom_type_emb = self.model.atom_type_embedding(new_atom_type).view(-1, self.hidden_channels)
                    new_pos_to_add = self.pos_generate(
                        h_cpx, atom_type_emb, pivotal_idx, data.cpx_pos, atom_idx
                        )
                    if new_pos_to_add is False:
                        self.resample_node += 1
                        continue
                    else:
                        rw_mol.AddAtom(Chem.Atom(self.possible_atom_type[new_atom_type]))

                    # generate covalent bonds
                    bond_out = self.bond_generate(
                        h_cpx, data, new_pos_to_add, atom_type_emb, atom_idx, rw_mol
                        )
                    if bond_out is not False:
                        rw_mol, new_edge_idx, new_bond_type_to_add = bond_out
                        has_alert = check_alert_structures(rw_mol, ['[O]-[O,Br,Cl,I,F]','[N,P]-[Br,Cl,I,F,P]','[S,P]-[Br,Cl,I,F]','[P]-[O]-[P]', '[C]-[S]=[C]',
                                                                    '[Br,Cl,I,F]-[Br,Cl,I,F]','[C,N,O,S]=[C]=[C,N,O,S]','[C,N,S,I]=[P]','[I]=[N,C,O]', '[C]-[I]-[C]']+self.sub_structure)
                        if has_alert:
                            for ix in range(new_edge_idx.size(1)):
                                i, j = new_edge_idx[:, ix].tolist()
                                rw_mol.RemoveBond(atom_idx, j)
                            rw_mol.RemoveAtom(atom_idx)
                            self.resample_node += 1
                            continue
                        else:
                            break
                    if self.resample_edge_faild:
                        break
                
                if self.resample_edge_faild or self.resample_node > self.max_resample_node or pivotal_out is False:
                    break
                else:
                    data = data.to('cpu')
                    data = add_ligand_atom_to_data(
                        data, 
                        new_pos_to_add.to('cpu'), 
                        new_atom_type.to('cpu'), 
                        new_edge_idx.to('cpu'), 
                        new_bond_type_to_add.to('cpu'), 
                        type_map=self.possible_atom_type,
                        remove_tri_ring=self.remove_tri_ring
                        )
                    data = self.transform(data)
        try:
            mol = data2mol(data)
            modified_mol = modify(mol, max_double_in_6ring=self.max_double_in_6ring)
            return modified_mol, mol
        except:
            mol_ = rw_mol.GetMol()
            print('Invalid mol: ', Chem.MolToSmiles(mol_))
            mol = data2mol(data)
            return None

    def unique_struct(self, mol_list):
            mol_dict = {}
            for mol in mol_list:
                s = Chem.MolToSmiles(mol)
                if s not in mol_dict.keys():
                    mol_dict[s] = Chem.MolToMolBlock(mol)
                else:
                    continue
            return mol_dict


    def generate(self, data, generate_number=100, folder_name='recptor', print_SMILES=True, path_name='gen_results'):
            date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            out_dir = path_name+'/'+folder_name+'/'+date+'/'
            self.out_dir = out_dir
            verify_dir_exists(out_dir)
            valid_mol = []
            smiles_list = []
            self.start_idx = data.ligand_context_pos.size(0)
            valid_conuter = 0

# Parameters
            i = 0
            number_of_output = 0
            mol_dict = {}
            #f_Modify = Filter(sa_threshhold=5.0, qed_threshhold=0.6, mw_threshhold=(350,500))     
            f_Modify = Filter(sa_threshhold=10.0, qed_threshhold=0.0, mw_threshhold=(350,900))       
            while number_of_output < generate_number:   
                i = i + 1 
                data_clone = data.clone().detach()
                out = self.run(data_clone)
                if out:
                    mol, mol_NoModify = out     
                s = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
                filter_1 = f_Modify._filter(Chem.MolFromSmiles(s))  
                if filter_1:
                    if s not in mol_dict:
                        mol_dict[s] = None
                    else:
                        continue
                else:
                    continue
                number_of_output = number_of_output + 1

                del data_clone
                if mol is not None:
                    mol.SetProp('_Name', 'No_%s-%s'%(valid_conuter, out_dir))
                    mol_NoModify.SetProp('_Name', 'No_%s-%s' %(valid_conuter, out_dir))
                    smi = Chem.MolToSmiles(mol)
                    smi_NoModify = Chem.MolToSmiles(mol_NoModify)
                    if print_SMILES:
                        print(smi)
                    with open(out_dir+'generated.sdf', 'a') as sdf_writer:
                        mol_block = Chem.MolToMolBlock(mol)
                        sdf_writer.write(mol_block + '\n$$$$\n')
                    with open(out_dir+'generated_NoModify.sdf', 'a') as sdf_writer1:
                        mol_block_NoModify = Chem.MolToMolBlock(mol_NoModify)
                        sdf_writer1.write(mol_block_NoModify + '\n$$$$\n')
                    with open(out_dir+'generated.smi','a') as smi_writer:
                        smi_writer.write(smi+'\n')
                    with open(out_dir+'generated_NoModify.smi','a') as smi_writer1:
                        smi_writer1.write(smi_NoModify+'\n')
                    smiles_list.append(smi)
                    valid_mol.append(mol)
                    valid_conuter += 1       
            print(i)
            print(len(smiles_list))
            print(len(set(smiles_list)))
            print('Validity: {:.4f}'.format(len(smiles_list)/generate_number))
            print('Unique: {:.4f}'.format(len(set(smiles_list))/len(smiles_list)))
            out_statistic = {
                'Validity':len(smiles_list)/generate_number,
                'Unique':len(set(smiles_list))/len(smiles_list)
                }
            ring_size_statis = substructure([valid_mol])
            out_statistic['ring_size'] = ring_size_statis
            with open(out_dir+'metrics.dir','w') as fw:
                fw.write(str(out_statistic))