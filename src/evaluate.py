import json
import time
import argparse
import torch
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from data import load_data
from torchvision import transforms
import torchvision.models as models
from precision_recall import prd_score
from unittest.mock import patch
from util import kid_score, roc_pc, typicality_test, compute_nlls, preprocess_samples, sample_model, extract_features

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def evaluate_made(args):
    seed = args.seed
    result_dir = args.result_dir 
    data_dir = args.data_dir
    num_samples = args.num_samples
    subset_size = args.subset_size
    K = args.K
    alpha = args.alpha
    model_type = args.model_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    results['directory'] = result_dir
    results['config'] = {'seed': seed, 'num_samples': num_samples,
                         'subset_size': subset_size, 'K': K, 'alpha': alpha}

    torch.manual_seed(seed)
    model = torch.load(result_dir + '/model_best.model', map_location = device).to(device)
    model.eval()

    # Sample Quality (KID)
    start_time = time.time()
    _, _, test_data = load_data('mnist', data_dir = data_dir, binarize = True, eval = True, val = False)
    test_loader = DataLoader(test_data, batch_size = num_samples, shuffle = True)
    true = next(iter(test_loader))[0].to(device)
    samples = sample_model(model = model, n = num_samples, model_type = model_type).to(device)
    samples = preprocess_samples(samples, preprocess).to(device)
    n_subsets = num_samples // subset_size

    inception_model = models.inception_v3(pretrained=False)
    state_dict = torch.load('./inception_v3_google-1a9a5a14.pth')
    inception_model.load_state_dict(state_dict)
    inception_model.fc = torch.nn.Identity()
    inception_model.to(device)
    inception_model.eval()
    real_features = extract_features(true, inception_model)
    generated_features = extract_features(samples, inception_model)
    kids, kid_stats = kid_score(torch.tensor(real_features, dtype=torch.float32).to(device),  
                                torch.tensor(generated_features, dtype=torch.float32).to(device), 
                                n_subsets = n_subsets, subset_size = subset_size)
    results['KID'] = {'kids': kids.tolist(), 
                      'std': kid_stats}
    end_time = time.time()
    print('Sampling Time:' + f'{end_time-start_time}')
    
    # Sample Quality (PRD)
    precisions, recalls = prd_score.compute_prd_from_embedding(real_features, generated_features)

    results['PRD'] = {'precisions': precisions.tolist(), 
                      'recalls': recalls.tolist()}
    end_time = time.time()
    print('PRD Time:' + f'{end_time-start_time}')
    
    # OOD (Visualization)
    train_data, _, test_data = load_data('mnist', data_dir = data_dir, binarize = True, val = False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle = True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle = True)
    _, _, aug_test_data = load_data('mnist', data_dir, binarize = True, augment = True, val = False)
    aug_test_loader = DataLoader(aug_test_data, batch_size=1, shuffle=False)
    _, _, test_data_emnist = load_data('emnist', data_dir = data_dir, binarize = True, val = False)
    test_loader_emnist = DataLoader(test_data_emnist, batch_size=1, shuffle = False)
    aug_nll_mnist = compute_nlls(model, aug_test_loader, model_type = model_type)
    train_nll_mnist = compute_nlls(model, train_loader, model_type = model_type)
    nll_mnist= compute_nlls(model, test_loader, model_type = model_type)
    nll_emnist = compute_nlls(model, test_loader_emnist, model_type = model_type)

    results['OOD_NLLs_test'] = {'mnist': nll_mnist.tolist(), 
                                'emnist': nll_emnist.tolist()}
    results['train_nll_mnist'] = train_nll_mnist.tolist()
    results['aug_nll_mnist'] = aug_nll_mnist.tolist()
    end_time = time.time()
    print('Viz Time:' + f'{end_time-start_time}')

    # OOD (ROC/AUC)
    fpr, tpr, thresholds, roc_auc, precision, recall, pr_thresholds, pr_auc, nll_mnist, nll_emnist \
                        = roc_pc(test_loader = test_loader, test_loader_ood = test_loader_emnist, 
                        model = model, model_type = model_type, nll_mnist = nll_mnist, nll_ood = nll_emnist)
    
    results['roc_pc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist(), 
                        'roc_auc': roc_auc, 'precision': precision.tolist(), 'recall': recall.tolist(),
                        'pr_thesholds': pr_thresholds.tolist(),'pr_auc': pr_auc}
    end_time = time.time()
    print('ROC Time:' + f'{end_time-start_time}')
    # OOD (Typicality)
    train_data, val_data, test_data = load_data('mnist', data_dir = data_dir, binarize = True, val = True)
    _, _, test_data_emnist = load_data('emnist', data_dir = data_dir, binarize = True, val = False)
    results['typicality'] = {}
    Ms = [2,4,8,16,24,32,40,48,56,64]
    for M in Ms:
        ood_mnist, ood_emnist = typicality_test(model = model, train_data = train_data, val_data = val_data, 
                                                test_data = test_data, test_data_ood = test_data_emnist, 
                                                K=K, alpha=alpha, model_type=model_type, M=M)
        results['typicality'][f'M={M}'] = {'mnist': ood_mnist, 'emnist': ood_emnist}
    end_time = time.time()
    print('Typicality Time:' + f'{end_time-start_time}')
    with open(result_dir + '/evaluate_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return results


def evaluate_pc(args):
    seed = args.seed
    result_dir = args.result_dir 
    data_dir = args.data_dir
    num_samples = args.num_samples
    subset_size = args.subset_size
    K = args.K
    alpha = args.alpha
    model_type = args.model_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    results['directory'] = result_dir
    results['config'] = {'seed': seed, 'num_samples': num_samples,
                         'subset_size': subset_size, 'K': K, 'alpha': alpha}

    torch.manual_seed(seed)
    circuit = torch.load(result_dir + '/circuit.pt', map_location = device).to(device).eval()
    pf_circuit = torch.load(result_dir + '/pf_circuit.pt', map_location = device).to(device).eval()
    model = (circuit, pf_circuit)

    # Sample Quality (KID)
    start_time = time.time()
    _, _, test_data = load_data('mnist', data_dir = data_dir, binarize = False, eval = True, val = False)
    test_loader = DataLoader(test_data, batch_size = num_samples, shuffle = True)
    true = next(iter(test_loader))[0].to(device)
    samples = sample_model(model = model, n = num_samples//10, model_type = 'PC').to(device)
    for i in range(9):
        samples = torch.cat((samples, sample_model(model = model, n = num_samples//10, model_type = 'PC')), dim = 0)
    samples = samples.to(torch.float32).to(device)
    samples = preprocess_samples(samples, preprocess).to(device)
    n_subsets = num_samples // subset_size

    inception_model = models.inception_v3(pretrained=False)
    state_dict = torch.load('./inception_v3_google-1a9a5a14.pth')
    inception_model.load_state_dict(state_dict)
    inception_model.fc = torch.nn.Identity()
    inception_model.to(device)
    inception_model.eval()
    real_features = extract_features(true, inception_model)
    generated_features = extract_features(samples, inception_model)
    kids, kid_stats = kid_score(torch.tensor(real_features, dtype=torch.float32).to(device),  
                                torch.tensor(generated_features, dtype=torch.float32).to(device), 
                                n_subsets = n_subsets, subset_size = subset_size)
    results['KID'] = {'kids': kids.tolist(), 
                      'std': kid_stats}
    end_time = time.time()
    print('Sampling Time:' + f'{end_time-start_time}')
    
    # Sample Quality (PRD)
    start_time = time.time()
    precisions, recalls = prd_score.compute_prd_from_embedding(real_features, generated_features)
    end_time = time.time()
    print('PRD:' + f'{end_time-start_time}')
    results['PRD'] = {'precisions': precisions.tolist(), 
                      'recalls': recalls.tolist()}
    
    # OOD (Visualization)
    start_time = time.time()
    train_data, _, test_data = load_data('mnist', data_dir = data_dir, binarize = False, val = False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle = True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle = True)
    _, _, aug_test_data = load_data('mnist', data_dir, binarize = False, augment = True, val = False)
    aug_test_loader = DataLoader(aug_test_data, batch_size=1, shuffle=False)
    _, _, test_data_fashion = load_data('fashion', data_dir = data_dir, binarize = False, val = False)
    test_loader_fashion = DataLoader(test_data_fashion, batch_size=1, shuffle = False)
    train_nll_mnist = compute_nlls(model, train_loader, model_type = model_type)
    nll_mnist = compute_nlls(model, test_loader, model_type = model_type)
    aug_nll_mnist = compute_nlls(model, aug_test_loader, model_type = model_type)
    nll_fashion = compute_nlls(model, test_loader_fashion, model_type = model_type)
    end_time = time.time()
    print('Viz:' + f'{end_time-start_time}')

    results['OOD_NLLs_test'] = {'mnist': nll_mnist.tolist(), 
                                'fashion': nll_fashion.tolist()}
    results['train_nll_mnist'] = train_nll_mnist.tolist()
    results['aug_nll_mnist'] = aug_nll_mnist.tolist()

    # OOD (ROC/AUC)
    fpr, tpr, thresholds, roc_auc, precision, recall, pr_thresholds, pr_auc, nll_mnist, nll_fashion \
                        = roc_pc(test_loader = test_loader, test_loader_ood = test_loader_fashion, 
                        model = model, model_type = model_type, nll_mnist = nll_mnist, nll_ood = nll_fashion)
    
    results['roc_pc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist(), 
                        'roc_auc': roc_auc, 'precision': precision.tolist(), 'recall': recall.tolist(),
                        'pr_thesholds': pr_thresholds.tolist(),'pr_auc': pr_auc}
    
    # OOD (Typicality)
    start_time = time.time()
    train_data, val_data, test_data = load_data('mnist', data_dir = data_dir, binarize = False, val = True)
    _, _, test_data_fashion = load_data('fashion', data_dir = data_dir, binarize = False, val = False)
    results['typicality'] = {}
    Ms = [2,4,8,16,24,32,40,48,56,64]
    for M in Ms:
        ood_mnist, ood_fashion = typicality_test(model = model, train_data = train_data, val_data = val_data, 
                                                test_data = test_data, test_data_ood = test_data_fashion, 
                                                K=K, alpha=alpha, model_type=model_type, M=M)
        results['typicality'][f'M={M}'] = {'mnist': ood_mnist, 'fashion': ood_fashion}
    end_time = time.time()
    print('Typicality:' + f'{end_time-start_time}')
    with open(result_dir + '/evaluate_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Generative Model.")
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory for model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data Directory')
    parser.add_argument('--num_samples', type=int, required=True, help='Number of samples for KID and PRD')
    parser.add_argument('--subset_size', type=int, required=True, help='Subset size for KID calculaiton')
    parser.add_argument('--K', type=int, required=True, help='Number of bootstrap samples for typicality test')
    parser.add_argument('--alpha', type=float, required=True, help='Confidence level for typicality test')
    parser.add_argument('--model_type', type=str, default='MADE', choices=['MADE', 'PC'], help='Type of model to evaluate')
    args = parser.parse_args()
    if args.model_type == 'MADE':
        evaluate_made(args)
    elif args.model_type == 'PC':
        evaluate_pc(args)
    else:
        print(f"Model type {args.model_type} is not supported for evaluation in this script.")

if __name__ == '__main__':
    main()
    