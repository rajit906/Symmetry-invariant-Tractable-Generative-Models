import json
import argparse
import torch
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from data import load_data
from torchvision import transforms
import torchvision.models as models
from precision_recall import prd_score
from util import compute_KID, roc_pc, typicality_test, compute_nlls, preprocess_samples, sample_model, extract_features

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
    model= torch.load(result_dir + '/model_best.model', map_location = device).to(device)
    model.eval()

    # Sample Quality (KID)
    _, _, test_data = load_data('mnist', data_dir = data_dir, binarize = True, eval = True, val = False)
    test_loader = DataLoader(test_data, batch_size = num_samples, shuffle = True)
    true = next(iter(test_loader))[0].to(device)
    samples = sample_model(model = model, n = num_samples, model_type = model_type)
    samples = preprocess_samples(samples, preprocess)
    kid_mean, kid_std = compute_KID(true, samples, subset_size = subset_size, device = device)
    results['KID'] = {'mean': kid_mean, 
                      'std': kid_std}
    
    # Sample Quality (PRD)
    inception_model = models.inception_v3(pretrained=True) 
    inception_model.eval()  # Set the model to evaluation mode
    inception_model.fc = torch.nn.Identity()
    real_features = extract_features(true, inception_model)
    generated_features = extract_features(samples, inception_model)

    precisions, recalls = prd_score.compute_prd_from_embedding(real_features, generated_features)

    results['PRD'] = {'precisions': precisions.tolist(), 
                      'recalls': recalls.tolist()}
    
    # OOD (Visualization)
    _, _, test_data = load_data('mnist', data_dir = data_dir, binarize = True, val = False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle = True)
    _, _, test_data_omniglot = load_data('omniglot', data_dir = data_dir, binarize = True, val = False)
    test_loader_omniglot = DataLoader(test_data_omniglot, batch_size=1, shuffle = False)
    nll_mnist= compute_nlls(model, test_loader, model_type = model_type)
    nll_omniglot = compute_nlls(model, test_loader_omniglot, model_type = model_type)

    results['OOD_NLLs_test'] = {'mnist': nll_mnist.tolist(), 
                                'omniglot': nll_omniglot.tolist()}

    # OOD (ROC/AUC)

    fpr, tpr, thresholds, roc_auc, precision, recall, pr_thresholds, pr_auc, nll_mnist, nll_omniglot \
                        = roc_pc(test_loader = test_loader, test_loader_ood = test_loader_omniglot, 
                        model = model, model_type = model_type, nll_mnist = nll_mnist, nll_ood = nll_omniglot)
    
    results['roc_pc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist(), 
                        'roc_auc': roc_auc, 'precision': precision.tolist(), 'recall': recall.tolist(),
                        'pr_thesholds': pr_thresholds.tolist(),'pr_auc': pr_auc}
    
    # OOD (Typicality)
    train_data, val_data, test_data = load_data('mnist', data_dir = data_dir, binarize = True, val = True)
    _, _, test_data_omniglot = load_data('omniglot', data_dir = data_dir, binarize = True, val = False)
    results['typicality'] = {}
    Ms = [2,4,8,16,24,32,40,48,56,64]
    for M in Ms:
        ood_mnist, ood_omniglot = typicality_test(model = model, train_data = train_data, val_data = val_data, 
                                                test_data = test_data, test_data_ood = test_data_omniglot, 
                                                K=K, alpha=alpha, model_type=model_type, M=M)
        results['typicality'][f'M={M}'] = {'mnist': ood_mnist, 'omniglot': ood_omniglot}
    
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
    _, _, test_data = load_data('mnist', data_dir = data_dir, binarize = False, eval = True, val = False)
    test_loader = DataLoader(test_data, batch_size = num_samples, shuffle = True)
    true = next(iter(test_loader))[0].to(device)
    samples = sample_model(model = model, n = num_samples, model_type = model_type).to(float)
    samples = preprocess_samples(samples, preprocess)
    kid_mean, kid_std = compute_KID(true, samples, subset_size = subset_size, device = device)
    results['KID'] = {'mean': kid_mean, 
                      'std': kid_std}
    
    # Sample Quality (PRD)
    inception_model = models.inception_v3(pretrained=True) 
    inception_model.eval()  # Set the model to evaluation mode
    inception_model.fc = torch.nn.Identity()
    real_features = extract_features(true, inception_model)
    generated_features = extract_features(samples, inception_model)

    precisions, recalls = prd_score.compute_prd_from_embedding(real_features, generated_features)

    results['PRD'] = {'precisions': precisions.tolist(), 
                      'recalls': recalls.tolist()}
    
    # OOD (Visualization)
    _, _, test_data = load_data('mnist', data_dir = data_dir, binarize = False, val = False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle = True)
    _, _, test_data_fashion = load_data('fashion', data_dir = data_dir, binarize = False, val = False)
    test_loader_fashion = DataLoader(test_data_fashion, batch_size=1, shuffle = False)
    nll_mnist= compute_nlls(model, test_loader, model_type = model_type)
    nll_fashion = compute_nlls(model, test_loader_fashion, model_type = model_type)

    results['OOD_NLLs_test'] = {'mnist': nll_mnist.tolist(), 
                                'fashion': nll_fashion.tolist()}

    # OOD (ROC/AUC)

    fpr, tpr, thresholds, roc_auc, precision, recall, pr_thresholds, pr_auc, nll_mnist, nll_fashion \
                        = roc_pc(test_loader = test_loader, test_loader_ood = test_loader_fashion, 
                        model = model, model_type = model_type, nll_mnist = nll_mnist, nll_ood = nll_fashion)
    
    results['roc_pc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist(), 
                        'roc_auc': roc_auc, 'precision': precision.tolist(), 'recall': recall.tolist(),
                        'pr_thesholds': pr_thresholds.tolist(),'pr_auc': pr_auc}
    
    # OOD (Typicality)
    train_data, val_data, test_data = load_data('mnist', data_dir = data_dir, binarize = False, val = True)
    _, _, test_data_fashion = load_data('fashion', data_dir = data_dir, binarize = False, val = False)
    results['typicality'] = {}
    Ms = [2,4,8,16,24,32,40,48,56,64]
    for M in Ms:
        ood_mnist, ood_fashion = typicality_test(model = model, train_data = train_data, val_data = val_data, 
                                                test_data = test_data, test_data_ood = test_data_fashion, 
                                                K=K, alpha=alpha, model_type=model_type, M=M)
        results['typicality'][f'M={M}'] = {'mnist': ood_mnist, 'fashion': ood_fashion}
    
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
    