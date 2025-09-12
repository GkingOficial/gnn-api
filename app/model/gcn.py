import numpy as np
import os
import csv
import time
import sys
import random
import itertools
from utils import shuffle_two_list, load_input_HIV, convert_to_graph, split_train_eval_test, load_folds
from rdkit import Chem

from mc_dropout import mc_dropout
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score,recall_score,matthews_corrcoef
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
import rdkit
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG

from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
from IPython.display import SVG
from rdkit import rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
np.set_printoptions(precision=3)
import os
print(os.getcwd())
def get_match_bond_indices(query, mol, match_atom_indices):
    bond_indices = []
    for query_bond in query.GetBonds():
        atom_index1 = match_atom_indices[query_bond.GetBeginAtomIdx()]
        atom_index2 = match_atom_indices[query_bond.GetEndAtomIdx()]
        bond_indices.append(mol.GetBondBetweenAtoms(
             atom_index1, atom_index2).GetIdx())
    return bond_indices

def processUnit(iMol,start,i,batch_size,count,tracker,adj_len,full_size, fold_index, probability=None, prediction=None, ground_truth=None):

    # print(prediction, ground_truth)
    # input()

    size = (120, 120)
    tmp = rdkit.Chem.rdmolfiles.MolFragmentToSmiles(iMol, atomsToUse=start)
    tmp1=tmp
    j=0
    full_size=full_size

    with open(f"mutag200_preds_{fold_index}.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"{i * batch_size + count}.png", probability, prediction, ground_truth])

    start=start
    #print(start)
    while(rdkit.Chem.rdmolfiles.MolFromSmiles(tmp)==None and len(tmp)!=0):
        j=j+1
        if(full_size>=6):
            full_size=full_size-j
            start = maxSum(tracker,adj_len,full_size)

            #print(start)
        else:
            fig = Draw.MolToFile(iMol, f"./mutag200/Fold {fold_index}/" + str(i * batch_size + count) + '.png', size=size,
                                 highlightAtoms=start)
            print("bad")
            return max(tmp1.split('.'), key=len)
        if(len(start)>0):
            tmp = rdkit.Chem.rdmolfiles.MolFragmentToSmiles(iMol, atomsToUse=start)
            #print(tmp)
        else:
            fig = Draw.MolToFile(iMol, f"./mutag200/Fold {fold_index}/" + str(i * batch_size + count) + '.png', size=size,
                                 highlightAtoms=start)
            return max(tmp1.split('.'), key=len)
    fig = Draw.MolToFile(iMol, f"./mutag200/Fold {fold_index}/" + str(i * batch_size + count) + '.png', size=size,
                         highlightAtoms=start)

    #print("ok")

    return max(tmp.split('.'), key=len)

def np_sigmoid(x):
    return 1. / (1. + np.exp(-x))
def maxSum(arr, n, k):
    #print("n",n)
    #print("k",k)
    # n must be greater
    if (n < k):
        k=n
    #print(arr)

    # Compute sum of first
    # window of size k
    res = 0
    start=0
    end=k
    for i in range(k):
        #0.5 also used
        if (arr[i] > 0):
            res += arr[i]
        else:
            if(end<n):
                start=start+1
                end=end+1
            else:
                if(k>1):
                    k=k-1
                else:
                    start=0
                    end=0
                    return start, end

        # Compute sums of remaining windows by
    # removing first element of previous
    # window and adding last element of
    # current window.
    curr_sum = res
    for i in range(k, n):
        if(arr[i]>0):
            curr_sum += arr[i] - arr[i - k]
        if(curr_sum<res):
            res = curr_sum
            start=i-k+1
            end=i+1






    return list(range(start,end))
def calc_stats(Y_batch_total,Y_pred_total):
    True_positive = 0
    False_postive = 0
    True_negative = 0
    False_negative = 0
    Exp = Y_batch_total

    unique, counts = np.unique(Exp, return_counts=True)
    # print("Distribuição das classes no conjunto de treinamento:", dict(zip(unique, counts)))
    # input('PRED')

    # print("Previsões antes do arredondamento:", np.unique(Y_pred_total, return_counts=True))
    # input('ANTES')

    Pred = np.around(Y_pred_total)
    for i in range(len(Exp)):
        if (Exp[i] == Pred[i] and Exp[i] == 1):
            True_positive += 1
        if (Exp[i] != Pred[i] and Exp[i] == 0):
            False_postive += 1
        if (Exp[i] == Pred[i] and Exp[i] == 0):
            True_negative += 1
        if (Exp[i] != Pred[i] and Exp[i] == 1):
            False_negative += 1

    count_TP = True_positive

    count_FP = False_postive

    count_FN = False_negative

    count_TN = True_negative

    # print('count_TP: ' + str(count_TP))
    # print('count_FP: ' + str(count_FP))
    # print('count_FN: ' + str(count_FN))
    # print('count_TN: ' + str(count_TN))
    # input("Calc")

    import math

    if ((count_TN + count_FP) * (count_TN + count_FN) * (count_TP + count_FP) * (count_TP + count_FN)):
        MCC = (count_TP * count_TN - count_FP * count_FN) / math.sqrt(abs((count_TN
                                                                        + count_FP)
                                                                        * (count_TN + count_FN)
                                                                        * (count_TP + count_FP)
                                                                        * (count_TP + count_FN)))
    else:
        MCC = 0
        
    if (count_TN + count_FP):
        Specificity = count_TN / (count_TN + count_FP)
    else:
        Specificity = 0
    
    if (count_TP + count_FN):
        Recall = (count_TP / (count_TP + count_FN))
    else:
        Recall = 0


    return Recall,MCC,Specificity


def training(model, FLAGS, model_name, smi_train, prop_train, smi_test, prop_test, fold_index):
# def training(model, FLAGS, model_name, smi_total, prop_total):
    np.set_printoptions(threshold=sys.maxsize)
    print("Start Training XD")
    stuff = open("./TeststructNIHSUH.txt", 'w')
    stuff1 = open("./JPredNIHSUH.txt", 'w')
    stuff2 = open("./JALENIHSUH.txt", 'w')
    stuff3 = open("./JEpiNIHSUH.txt", 'w')
    
    num_epochs = FLAGS.epoch_size
    batch_size = FLAGS.batch_size
    init_lr = FLAGS.init_lr
    total_st = time.time()
    # smi_train, smi_eval, smi_test = split_train_eval_test(smi_total, 0.9, 0.2, 0.1)
    # prop_train, prop_eval, prop_test = split_train_eval_test(prop_total, 0.9, 0.2, 0.1)

    # prop_eval = np.asarray(prop_eval)
    prop_test = np.asarray(prop_test)
    num_train = len(smi_train)
    # num_eval = len(smi_eval)
    num_test = len(smi_test)
    smi_train = smi_train[:num_train]
    prop_train = prop_train[:num_train]
    num_batches_train = (num_train // batch_size) + 1
    # num_batches_eval = (num_eval // batch_size) + 1
    num_batches_test = (num_test // batch_size) + 1
    num_sampling = 20
    total_iter = 0

    # best_auroc = 0
    # patience = 20
    # epochs_without_improvement = 0

    # print("Number of-  training data:", num_train, "\t evaluation data:", num_eval, "\t test data:", num_test)
    print("Number of-  training data:", num_train, "\t test data:", num_test)
    for epoch in range(num_epochs):
        st = time.time()
        lr = init_lr * 0.5 ** (epoch // 10)
        model.assign_lr(lr)
        #smi_train, prop_train = shuffle_two_list(smi_train, prop_train)
        prop_train = np.asarray(prop_train)

        # TRAIN
        num = 0
        train_loss = 0.0
        Y_pred_total = np.array([])
        Y_batch_total = np.array([])

        for i in range(num_batches_train):
            num += 1
            st_i = time.time()
            total_iter += 1
            #tmp=smi_train[i * batch_size:(i + 1) * batch_size]

            A_batch, X_batch = convert_to_graph(smi_train[i * batch_size:(i + 1) * batch_size], FLAGS.max_atoms)
            Y_batch = prop_train[i * batch_size:(i + 1) * batch_size]

            # print(f"Shape de a_batch (entrada): {len(A_batch)}")
            # print(f"Shape de x_batch (rótulos): {len(X_batch)}")
            # print(f"Shape de y_batch (rótulos): {len(Y_batch)}")
            # input('IIN')

            #mtr = np.abs(model.get_feature(A_batch, X_batch, Y_batch))
            # print(np.shape(mtr))
            #print(len(tmp))
            #count = -1
            # for i in tmp:
            #     count += 1
            #     iMol = Chem.MolFromSmiles(i.strip())
            #
            #     start= (np.argpartition((mtr[count]),-10))
            #     start=np.array((start[start<len(Chem.rdmolops.GetAdjacencyMatrix(iMol))])).tolist()[0:9]
            #     #stuff.write(str(smi_test[count][start:end + 1]) + "\n")
            #     #print(len(Chem.rdmolops.GetAdjacencyMatrix(iMol)))
            #     print(start)
            #     print(rdkit.Chem.rdmolfiles.MolFragmentToSmiles(iMol,start))

            Y_mean, _, loss = model.train(A_batch, X_batch, Y_batch)

            train_loss += loss
            Y_pred = np_sigmoid(Y_mean.flatten())

            Y_pred_total = np.concatenate((Y_pred_total, Y_pred), axis=0)

            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)

            et_i = time.time()

        # print(Y_pred_total.shape, Y_batch_total.shape)
        # input('antes')

        train_loss /= num
        train_accuracy = accuracy_score(Y_batch_total, np.around(Y_pred_total).astype(int))

        train_recall,train_mcc,train_specificity=calc_stats(Y_batch_total, np.around(Y_pred_total).astype(int))

        try:
            train_auroc = roc_auc_score(Y_batch_total, Y_pred_total)
        except:
            train_auroc = 0.0

        # train_auroc = 0.0
        # try:
        #     train_auroc = roc_auc_score(Y_batch_total, Y_pred_total)

        #     if train_auroc > best_auroc:
        #         best_auroc = train_auroc
        #         epochs_without_improvement = 0
        #     else:
        #         epochs_without_improvement += 1
        #         if epochs_without_improvement >= patience:
        #             print(f"⏹️ Early stopping at epoch {epoch} — no improvement in AUROC for {patience} epochs.")
        #             break
        # except:
        #     train_auroc = 0.0

            # Eval
        # Y_pred_total = np.array([])
        # Y_batch_total = np.array([])

        # num = 0
        # eval_loss = 0.0
        # for i in range(num_batches_eval):
        #     evalbatch=smi_eval[i * batch_size:(i + 1) * batch_size]
        #     A_batch, X_batch = convert_to_graph(evalbatch, FLAGS.max_atoms)
        #     Y_batch = prop_eval[i * batch_size:(i + 1) * batch_size]
        #     # mtr_eval = np.abs(model.get_feature(A_batch, X_batch, Y_batch))
        #     # print(np.shape(mtr_eval))
        #     # count=-1
        #     # print(len(evalbatch))
        #     # for i in evalbatch:
        #     #     count += 1
        #     #     iMol = Chem.MolFromSmiles(i.strip())
        #     #
        #     #     #start= (np.argpartition((mtr_eval[count]),-10))
        #     #     start=mtr_eval[count]
        #     #     start=start[start>0.1]
        #     #     start=np.array((start[start<len(Chem.rdmolops.GetAdjacencyMatrix(iMol))])).tolist()[0:9]
        #     #     #stuff.write(str(smi_test[count][start:end + 1]) + "\n")
        #     #     #print(len(Chem.rdmolops.GetAdjacencyMatrix(iMol)))
        #     #     print(start)
        #     #     print(rdkit.Chem.rdmolfiles.MolFragmentToSmiles(iMol,start))
        #     # MC-sampling
        #     P_mean = []
        #     for n in range(1):
        #         num += 1
        #         Y_mean, _, loss = model.test(A_batch, X_batch, Y_batch)
        #         eval_loss += loss
        #         P_mean.append(Y_mean.flatten())

        #     P_mean = np_sigmoid(np.asarray(P_mean))
        #     mean = np.mean(P_mean, axis=0)

        #     Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
        #     Y_pred_total = np.concatenate((Y_pred_total, mean), axis=0)

        # eval_loss /= num
        # eval_accuracy = accuracy_score(Y_batch_total, np.around(Y_pred_total).astype(int))
        # eval_recall,eval_mcc,eval_specificity=calc_stats(Y_batch_total, np.around(Y_pred_total).astype(int))
        # eval_auroc = 0.0
        # try:
        #     eval_auroc = roc_auc_score(Y_batch_total, Y_pred_total)
        # except:
        #     eval_auroc = 0.0

            # Save network!
        ckpt_path = 'tmp/' + model_name + '.ckpt'
        model.save(ckpt_path, epoch)
        et = time.time()

        # Print Results
        # print("Time for", epoch, "-th epoch: ", et - st)
        # print("Loss        Train:", round(train_loss, 3), "\t Evaluation:", round(eval_loss, 3))
        # print("Accuracy    Train:", round(train_accuracy, 3), "\t Evaluation:", round(eval_accuracy, 3))
        # print("AUROC       Train:", round(train_auroc, 3), "\t Evaluation:", round(eval_auroc, 3))
        # print("train_mcc:",train_mcc,"train_recall",train_recall,"train_spec",train_specificity)
        # print("eval_mcc:", eval_mcc, "eval_recall", eval_recall, "eval_spec", eval_specificity)

        print("Time for", epoch, "-th epoch: ", et - st)
        print("Loss        Train:", round(train_loss, 3))
        print("Accuracy    Train:", round(train_accuracy, 3))
        print("AUROC       Train:", round(train_auroc, 3))
        print("train_mcc:",train_mcc,"train_recall",train_recall,"train_spec",train_specificity)

    total_et = time.time()

    total_train_time = total_et - total_st

    print("Finish training! Total required time for training : ", (total_et - total_st))

    # Test
    test_st = time.time()
    Y_pred_total = np.array([])
    Y_batch_total = np.array([])
    ale_unc_total = np.array([])
    epi_unc_total = np.array([])
    tot_unc_total = np.array([])
    num = 0
    test_loss = 0.0
    count_total = 0
    for i in range(num_batches_test):
        num += 1
        testBatch=smi_test[i * batch_size:(i + 1) * batch_size]
        # print(f"Batch {i} - tamanho: {len(testBatch)}")
        # input("BATCH")
        A_batch, X_batch = convert_to_graph(testBatch, FLAGS.max_atoms)
        Y_batch = prop_test[i * batch_size:(i + 1) * batch_size]

        if A_batch.shape[0] == 0 or X_batch.shape[0] == 0:
            print(f"⚠️  Batch {i} ignorado — sem amostras válidas.")
            continue

        mtr_test = np_sigmoid(model.get_feature(A_batch, X_batch, Y_batch))
        #print(np.shape(mtr_test))
        count = -1
        #print(len(testBatch))

        P_mean = []
        for n in range(5):
            Y_mean, _, loss = model.test(A_batch, X_batch, Y_batch)
            P_mean.append(Y_mean.flatten())
            # mtr = np.abs(model.get_feature(A_batch, X_batch, Y_batch))
            # print(np.shape(mtr))
            # for j in range(len(Y_batch)):
            #     count += 1
            #

            #     start, end = maxSum(mtr[j], 503, 15)
            #     stuff.write(str(smi_test[count][start:end + 1]) + "\n")

        P_mean = np_sigmoid(np.asarray(P_mean))

        mean = np.mean(P_mean, axis=0)
        ale_unc = np.mean(P_mean * (1.0 - P_mean), axis=0)
        epi_unc = np.mean(P_mean ** 2, axis=0) - np.mean(P_mean, axis=0) ** 2
        tot_unc = ale_unc + epi_unc

        for j in testBatch:
            count += 1
            count_total+=1
            iMol = Chem.MolFromSmiles(j.strip())
            adj_len=len(Chem.rdmolops.GetAdjacencyMatrix(iMol))
            #start= (np.argpartition((mtr_test[count]),-10))
            import math
            if(math.ceil(0.4 * adj_len)>=6):
                start=maxSum(mtr_test[count],adj_len,math.ceil(0.4*adj_len))
            else:
                start=maxSum(mtr_test[count],adj_len,6)
            #start = mtr_test[count]
            #print("adj_len",adj_len)
            #start = (np.squeeze(np.argwhere(start > 1)))
            #print("this is start",start)
            #print("adj len",adj_len)
            #print(j)
            #print(start)
            #start = np.array((start[start < adj_len])).tolist()[0:9]
            # stuff.write(str(smi_test[count][start:end + 1]) + "\n")
            # print(len(Chem.rdmolops.GetAdjacencyMatrix(iMol)))
            #print(start)
            #print(rdkit.Chem.rdmolfiles.MolFragmentToSmiles(iMol, start))

            #bondNum=Chem.rdchem.Mol.GetNumBonds(iMol)
            #tmp=rdkit.Chem.rdmolfiles.MolFragmentToSmarts(iMol, atomsToUse=start,bondsToUse=list(range(1,bondNum)),isomericSmarts=False)
            #print(tmp)
            #tmp4=(rdkit.Chem.rdmolfiles.MolFragmentToSmarts(iMol, atomsToUse=start))

            #print(tmp4)
            import math
            if (math.ceil(0.4 * adj_len) >= 6):
                tmpS=math.ceil(0.4 * adj_len)
            else:
                tmpS = 6
                start=maxSum(mtr_test[count],adj_len,6)
                tmpS=6
            
            probability = mean[count]
            prediction = np.around(mean[count])
            ground_truth = Y_batch[count]
            stuff.write(processUnit(iMol,start,i,batch_size,count,mtr_test[count],adj_len,tmpS, fold_index, probability, prediction, ground_truth) + "\n")
            #stuff.write(tmp4+ "\n")
            #uncomment this for the drawing.
            # fig = Draw.MolToFile(iMol, "./amesfirstmodImg3/"+str(i*batch_size+count)+'.png', size=size, highlightAtoms=start)
        # MC-sampling

        # P_mean = []
        # for n in range(5):
        #     Y_mean, _, loss = model.test(A_batch, X_batch, Y_batch)
        #     P_mean.append(Y_mean.flatten())
        #     # mtr = np.abs(model.get_feature(A_batch, X_batch, Y_batch))
        #     # print(np.shape(mtr))
        #     # for j in range(len(Y_batch)):
        #     #     count += 1
        #     #

        #     #     start, end = maxSum(mtr[j], 503, 15)
        #     #     stuff.write(str(smi_test[count][start:end + 1]) + "\n")

        # P_mean = np_sigmoid(np.asarray(P_mean))

        # mean = np.mean(P_mean, axis=0)
        # ale_unc = np.mean(P_mean * (1.0 - P_mean), axis=0)
        # epi_unc = np.mean(P_mean ** 2, axis=0) - np.mean(P_mean, axis=0) ** 2
        # tot_unc = ale_unc + epi_unc

        Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
        Y_pred_total = np.concatenate(( np.around(Y_pred_total), mean), axis=0)

        with open(f"preds/mutag200_predictions_fold{fold_index}.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for pred, label in zip(mean, Y_batch):
                writer.writerow([pred, label])

        ale_unc_total = np.concatenate((ale_unc_total, ale_unc), axis=0)
        epi_unc_total = np.concatenate((epi_unc_total, epi_unc), axis=0)
        tot_unc_total = np.concatenate((tot_unc_total, tot_unc), axis=0)
    stuff1.write(str((Y_pred_total)) + "\n")
    stuff2.write(str(ale_unc_total) + "\n")
    stuff3.write(str(epi_unc_total) + "\n")
    print(Y_pred_total)
    print("ale:",ale_unc_total)
    print("epi",epi_unc_total)

    True_positive = 0
    False_postive = 0
    True_negative = 0
    False_negative = 0
    Exp = Y_batch_total
    Pred = np.around(Y_pred_total)
    print("pred",Pred)
    #print("Exp",Exp)
    for i in range(len(Exp)):
        if (Exp[i] == Pred[i] and Exp[i] == 1):
            True_positive += 1
        if (Exp[i] != Pred[i] and Exp[i] == 0):
            False_postive += 1
        if (Exp[i] == Pred[i] and Exp[i] == 0):
            True_negative += 1
        if (Exp[i] != Pred[i] and Exp[i] == 1):
            False_negative += 1
    count_TP = True_positive
    print("True Positive:", count_TP)
    count_FP = False_postive
    print("False Positive", count_FP)
    count_FN = False_negative
    print("False Negative:", count_FN)
    count_TN = True_negative
    print("True negative:", count_TN)
    Accuracy = (count_TP + count_TN) / (count_TP + count_FP + count_FN + count_TN)
    print("Accuracy:", Accuracy)
    import math
    print("testAuroc",roc_auc_score(Y_batch_total, Y_pred_total))

    if ((count_TN + count_FP) * (count_TN + count_FN) * (count_TP + count_FP) * (count_TP + count_FN)):
        MCC = (count_TP * count_TN - count_FP * count_FN) / math.sqrt(abs((count_TN
                                                                        + count_FP)
                                                                        * (count_TN + count_FN)
                                                                        * (count_TP + count_FP)
                                                                        * (count_TP + count_FN)))
    else:
        MCC = 0
        
    print("MCC", MCC)

    if (count_TN + count_FP):
        Specificity = count_TN / (count_TN + count_FP)
    else:
        Specificity = 0
    
    print("Specificity:", Specificity)
    
    if (count_TP + count_FP):
        Precision = (count_TP / (count_TP + count_FP))
    else:
        Precision = 0
    print("Precision:", Precision)

    # sensitivity
    if (count_TP + count_FN):
        Recall = (count_TP / (count_TP + count_FN))
    else:
        Recall = 0

    print("Recall:", Recall)

    # F1
    Fmeasure = (2 * count_TP) / (2 * count_TP + count_FP + count_FN)
    print("Fmeasure", Fmeasure)

    test_et = time.time()

    total_test_time = test_et - test_st

    print("Finish Testing, Total time for test:", (test_et - test_st))

    test_auroc = roc_auc_score(Y_batch_total, Y_pred_total)
    test_stats = {
        "tp": count_TP,
        "fp": count_FP,
        "fn": count_FN,
        "tn": count_TN,
        "accuracy": Accuracy,
        "auroc": test_auroc,
        "mcc": MCC,
        "specificity": Specificity,
        "precision": Precision,
        "recall": Recall,
        "f1": Fmeasure
    }
    train_stats = {
        "loss": round(train_loss, 3),
        "accuracy": round(train_accuracy, 3),
        "auroc": round(train_auroc, 3),
        "mcc": train_mcc,
        "recall": train_recall,
        "specificity": train_specificity,
        "last_epoch_time": round(et - st, 3)
    }
    log_fold_result(
        fold_index=fold_index,
        smi_train=smi_train,
        prop_train=prop_train,
        smi_test=smi_test,
        prop_test=prop_test,
        train_stats=train_stats,
        test_stats=test_stats,
        total_train_time=total_train_time,
        total_test_time=total_test_time
    )

    return

# TOXICIDADE
dim1 = 62
dim2 = 768
max_atoms = 170
num_layer = 4
batch_size = 8
epoch_size = 200
learning_rate = 0.0003
regularization_scale = 4e-4

# MUTAGENICIDADE
# dim1 = 62
# dim2 = 256
# max_atoms = 170
# num_layer = 4
# batch_size = 512
# epoch_size = 200
# learning_rate = 0.001
# regularization_scale = 1e-4

beta1 = 0.9
beta2 = 0.98
smi_total, prop_total = load_input_HIV()
num_total = len(smi_total) #118
num_test = int(num_total * 0.2) #23.6
num_train = num_total - num_test #94.4
num_eval = int(num_train * 0.1) #9.44
num_train -= num_eval #84.96

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set FLAGS for environment setting
flags = tf.app.flags
FLAGS = flags.FLAGS
# Hyperparameters for a transfer-trained model
flags.DEFINE_string('task_type', 'classification', '')
flags.DEFINE_integer('hidden_dim', dim1, '')
flags.DEFINE_integer('latent_dim', dim2, '')
flags.DEFINE_integer('max_atoms', max_atoms, '')
flags.DEFINE_integer('num_layers', num_layer, '# of hidden layers')
flags.DEFINE_integer('num_attn', 4, '# of heads for multi-head attention')
flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
flags.DEFINE_integer('epoch_size', epoch_size, 'Epoch size')
flags.DEFINE_integer('num_train', num_train, 'Number of training data')
flags.DEFINE_float('regularization_scale', regularization_scale, '')
flags.DEFINE_float('beta1', beta1, '')
flags.DEFINE_float('beta2', beta2, '')
flags.DEFINE_string('optimizer', 'Adam', 'Options : Adam, SGD, RMSProp')
flags.DEFINE_float('init_lr', learning_rate, 'Batch size')

model_name = 'MC-Dropout_HIV'
print("Do Single-Task Learning")
print("Hidden dimension of graph convolution layers:", dim1)
print("Hidden dimension of readout & MLP layers:", dim2)
print("Maximum number of allowed atoms:", max_atoms)
print("Batch sise:", batch_size, "Epoch size:", epoch_size)
print("Initial learning rate:", learning_rate, "\t Beta1:", beta1, "\t Beta2:", beta2,
      "for the Adam optimizer used in this training")

# model = mc_dropout(FLAGS)
# training(model, FLAGS, model_name, smi_total, prop_total)

def log_fold_result(fold_index, smi_train, prop_train, smi_test, prop_test, train_stats, test_stats, total_train_time, total_test_time, log_path="log_mutagenicidade.txt"):
    with open('./LOG_mutag.txt', "a", encoding="utf-8") as f:
        f.write(f"\n===== Fold {fold_index + 1}/5 =====\n")
        f.write(f"Treino: {dict(zip(*np.unique(prop_train, return_counts=True)))}\n")
        f.write(f"Teste: {dict(zip(*np.unique(prop_test, return_counts=True)))}\n\n")

        f.write(f"Time for {FLAGS.epoch_size - 1} -th epoch:  {train_stats['last_epoch_time']}\n")
        f.write(f"Loss        Train: {train_stats['loss']}\n")
        f.write(f"Accuracy    Train: {train_stats['accuracy']}\n")
        f.write(f"AUROC       Train: {train_stats['auroc']}\n")
        f.write(f"train_mcc: {train_stats['mcc']} train_recall {train_stats['recall']} train_spec {train_stats['specificity']}\n")
        f.write(f"Finish training! Total required time for training :  {total_train_time}\n\n")

        f.write(f"True Positive: {test_stats['tp']}\n")
        f.write(f"False Positive {test_stats['fp']}\n")
        f.write(f"False Negative: {test_stats['fn']}\n")
        f.write(f"True negative: {test_stats['tn']}\n")
        f.write(f"Accuracy: {test_stats['accuracy']}\n")
        f.write(f"testAuroc {test_stats['auroc']}\n")
        f.write(f"MCC {test_stats['mcc']}\n")
        f.write(f"Specificity: {test_stats['specificity']}\n")
        f.write(f"Precision: {test_stats['precision']}\n")
        f.write(f"Recall: {test_stats['recall']}\n")
        f.write(f"Fmeasure {test_stats['f1']}\n")
        f.write(f"Finish Testing, Total time for test: {total_test_time}\n")


model_name = "Mutag200GCN"
folds = load_folds()
for i in range(5):
    print(f"\n===== Fold {i+1}/5 =====")
    
    smi_train = []
    prop_train = []
    
    for j, (smi, prop) in enumerate(folds):
        if j == i:
            smi_test = smi
            prop_test = prop
        else:
            smi_train.extend(smi)
            prop_train.extend(prop)
    
    smi_train = np.array(smi_train)
    prop_train = np.array(prop_train)
    smi_test = np.array(smi_test)
    prop_test = np.array(prop_test)

    unique, counts = np.unique(prop_train, return_counts=True)
    print("Treino:", dict(zip(unique, counts)))

    unique, counts = np.unique(prop_test, return_counts=True)
    print("Teste:", dict(zip(unique, counts)))

    tf.compat.v1.reset_default_graph()
    model = mc_dropout(FLAGS)

    training(model, FLAGS, f"{model_name}_fold{i}", smi_train, prop_train, smi_test, prop_test, i)

# ckpt_path = "model/weights/Tox250GCN_fold0.ckpt-299"
# saver = tf.train.Saver()
# saver.restore(sess, ckpt_path)