import math
import tensorflow as tf
from model.mc_dropout import mc_dropout
from model.utils import convert_to_graph, maxSum, np_sigmoid
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdmolfiles import MolFragmentToSmiles, MolFromSmiles
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
import io
import base64
from rdkit.Chem import AllChem

class Predictor:
    def __init__(self, ckpt_path, FLAGS):
        tf.compat.v1.reset_default_graph()

        self.model = mc_dropout(FLAGS)
        self.FLAGS = FLAGS
        self.sess = tf.compat.v1.Session()

        vars_in_graph = tf.compat.v1.trainable_variables()
        saver = tf.compat.v1.train.Saver(var_list=vars_in_graph)
        saver.restore(self.sess, ckpt_path)

    def predict(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
        
        AllChem.Compute2DCoords(mol)

        A, X = convert_to_graph([smiles], self.FLAGS.max_atoms)

        if A.shape[0] == 0 or X.shape[0] == 0:
            return {"error": "Graph conversion failed"}
        
        y_mean, _, _ = self.model.test(self.sess, A, X, np.zeros((1,)))
        y_sigmoid = 1.0 / (1.0 + np.exp(-y_mean))
        prediction = float(np.around(y_sigmoid[0][0]))
        probability = float(y_sigmoid[0][0])

        weights_arr = np_sigmoid(self.model.get_feature(self.sess, A, X, np.array([prediction])))
        adj_len = len(GetAdjacencyMatrix(mol))
        
        if(math.ceil(0.4 * adj_len) >= 6):
            start_atoms = maxSum(weights_arr.flatten(), adj_len, math.ceil(0.4 * adj_len))
        else:
            start_atoms = maxSum(weights_arr.flatten(), adj_len, 6)

        if (math.ceil(0.4 * adj_len) >= 6):
            fragment_size = math.ceil(0.4 * adj_len)
        else:
            fragment_size = 6
            start_atoms = maxSum(weights_arr.flatten(), adj_len, 6)
            fragment_size = 6

        explanation = self.generateSVG(mol, start_atoms, weights_arr.flatten(), adj_len, fragment_size)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "explanation": {
                "image_base64": explanation.get("image_base64"),
                "atoms": explanation.get("atoms")
            }
        }
    
    def generateSVG(self, iMol, start_atoms, tracker, adj_len, full_size):
        tmp = MolFragmentToSmiles(iMol, atomsToUse = start_atoms)
        j = 0

        while(MolFromSmiles(tmp) is None and len(tmp) != 0):
            j += 1
            if(full_size >= 6):
                full_size = full_size - j
                start_atoms = maxSum(tracker, adj_len, full_size)
            else:
                break
            
            if(len(start_atoms) > 0):
                tmp = MolFragmentToSmiles(iMol, atomsToUse = start_atoms)
            else:
                break
            
        try:
            drawer = Draw.rdMolDraw2D.MolDraw2DSVG(350, 350)
            drawer.DrawMolecule(iMol, highlightAtoms = start_atoms)
            drawer.FinishDrawing()
            svg_content = drawer.GetDrawingText()
            svg_bytes = svg_content.encode('utf-8')
            b64_bytes = base64.b64encode(svg_bytes)
            b64_string = b64_bytes.decode('utf-8')

            return {"image_base64": b64_string, "atoms": start_atoms}
        except Exception as e:
            print(f"Erro ao gerar imagem SVG: {e}")

            return { "image_base64": None, 'atoms': None}
