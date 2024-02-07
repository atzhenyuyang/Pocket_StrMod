from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Crippen, Lipinski
import sascorer

def calculate_SA_and_qed_based_on_smiles(s):
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(s))) 
    print(QED.qed(mol))
    print(sascorer.calculateScore(mol))

#calculate_SA_and_qed_based_on_smiles("O=C(N1CC(C2=NC=CC=C2)=O)N(C3=CC=C(C(F)(F)F)C(C#N)=C3)N=C(CN)C1=O")


def SA_Glide_sort(file_path,bound):  # Input the sdf file with molecules are sorted using docking score.
    SA_score_l = []
    docking_score = []
    molec = Chem.SDMolSupplier(file_path)
    molec_docking = Chem.SDMolSupplier(file_path)
    for c in molec:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(c))
        try:
            SA_score_l.append((sascorer.calculateScore(mol), Chem.MolToSmiles(mol),c.GetProp("_Name")))
        except:
            continue

    SA_score_l_result = sorted(SA_score_l)
    result = [i[2] for i in SA_score_l_result][:bound]
    
    string_result =["{},{},{}".format(*i) for i in result]
    f = open("SA_ds_result.txt","w")
    f.write("\n".join(string_result))
    f.close()

#SA_Glide_sort(file_path,bound)


def qed_calculation(file_path):
    l = 0
    molec = Chem.SDMolSupplier(file_path)
    sum_qed = 0
    for c in molec:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(c))
        try:
            print(QED.qed(mol))
        except:
            continue
#qed_calculation("XXXXXXX.sdf")

def qed_sort(file_path):
    qed_score_l = []
    molec = Chem.SDMolSupplier(file_path)
    for c in molec:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(c))
        try:
            qed_score_l.append((QED.qed(mol), Chem.MolToSmiles(mol)))
        except:
            continue
    QED_l_result = sorted(qed_score_l)
    string_QED_score_l_result = [str(i) for i in QED_l_result]
    f = open("XXXXXX.txt","w")
    f.write("\n".join(string_QED_score_l_result))
    f.close()

#qed_sort("XXXXXX.sdf")

def SA_calculation(file_path):
    l = 0
    molec = Chem.SDMolSupplier(file_path)
    sum_sascore = 0
    for c in molec:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(c))
        try:
            print(sascorer.calculateScore(mol))
        except:
            continue

#SA_calculation("XXXXXX.sdf")


def SA_score_sort(file_path):
    SA_score_l = []
    molec = Chem.SDMolSupplier(file_path)
    for c in molec:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(c))
        try:
            SA_score_l.append((sascorer.calculateScore(mol), Chem.MolToSmiles(mol),c.GetProp("_Name")))
        except:
            continue
    SA_score_l_result = sorted(SA_score_l)
    string_SA_score_l_result = [str(i) for i in SA_score_l_result]
    f = open("SA_sort.txt","w")
    f.write("\n".join(string_SA_score_l_result))
    f.close()

#SA_score_sort("XXXXXX.sdf")


def SA_qed_sort(file_path):
    SA_qed_score_l = []
    molec = Chem.SDMolSupplier(file_path)
    for c in molec:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(c))
        try:
            SA_qed_score_l.append((sascorer.calculateScore(mol), QED.qed(mol), Chem.MolToSmiles(mol)))
        except:
            continue
    SA_qed_score_l_result = sorted(SA_qed_score_l)[:25]
    result = sorted(SA_qed_score_l_result, key=lambda x:x[1],reverse = True)
    qed_sa_result = ["{},{},{}".format(*i) for i in result]
    f = open("SA_qed_sort_result.txt","w")
    f.write("\n".join(qed_sa_result))
    f.close()

#SA_qed_sort("XXXXXX.sdf")

def molecule_weight_calculation(file_path):
    l = 0
    molec = Chem.SDMolSupplier(file_path)
    molecular_weight = 0
    for c in molec:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(c))
        try:
            print(Descriptors.MolWt(mol))
        except:
            continue
    
#molecule_weight_calculation("XXXXXX.sdf")

def linpinski_percentage(file_path):
    v = 0
    lp = 0
    molec = Chem.SDMolSupplier(file_path)
    for c in molec:
        try:
            v = v + 1
            mw = Descriptors.MolWt(c)
            logp = Crippen.MolLogP(c)
            hbd = Lipinski.NumHDonors(c)
            hba = Lipinski.NumHAcceptors(c)
            if sum([mw<=700, logp<=5, hba<=10, hbd<=5]) == 4:
                lp = lp + 1
        except:
            continue
    linp =  lp/v  
    print("The percentage of molecules which match linpinski rules is: %f" % linp)


#linpinski_percentage("XXXXXX.sdf")