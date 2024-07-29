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
    result_dir = args.model_dir 
    num_samples = args.num_samples
    subset_size = args.subset_size
    K = args.K
    alpha = args.alpha
    model_type = args.model_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    results['directory'] = 'result_dir'
    results['config'] = {'seed': seed, 'num_samples': num_samples, 
                         'subset_size': subset_size, 'K': K, 'alpha': alpha}

    torch.manual_seed(seed)
    model= torch.load(result_dir + '/model_best.model', map_location=device).to(device)
    model.eval()

    # Sample Quality (KID)
    _, _, test_data = load_data('mnist', data_dir = './data', binarize=True, eval = True, val = False)
    test_loader = DataLoader(test_data, batch_size=num_samples, shuffle = True)
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
    _, _, test_data = load_data('mnist', data_dir = './data', binarize=True, val = False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle = True)
    nll_mnist= compute_nlls(model, test_loader, model_type = model_type)
    nll_emnist = compute_nlls(model, test_loader_emnist, model_type = model_type)

    results['OOD_NLLs_test'] = {'mnist': nll_mnist.tolist(), 
                                'emnist': nll_emnist.tolist()}

    # OOD (ROC/AUC)
    _, _, test_data = load_data('mnist', './data', binarize=True, val = False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    _, _, test_data_emnist = load_data('emnist', './data', binarize=True, val = True)
    test_loader_emnist = DataLoader(test_data_emnist, batch_size=1, shuffle=False)

    fpr, tpr, thresholds, roc_auc, precision, recall, pr_thresholds, pr_auc, nll_mnist, nll_emnist \
                                        = roc_pc(test_loader, test_loader_emnist, model, 
                                                 nll_mnist, nll_emnist, model_type=model_type)
    
    results['roc_pc'] = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds, 
                        'roc_auc':roc_auc, 'precision':precision, 'recall':recall,
                        'pr_thesholds':pr_thresholds,'pr_auc':pr_auc}
    
    # OOD (Typicality)
    train_data, val_data, test_data = load_data('mnist', './data', binarize=True, val = True)
    _, _, test_data_emnist = load_data('emnist', './data', binarize=True, val = True)
    ood_mnist, ood_emnist= typicality_test(model, train_data, val_data, 
                                                            test_data, test_data_emnist, K=K, 
                                                            alpha=alpha, model_type=model_type)

    results['typicality'] = {'mnist': ood_mnist, 
                             'emnist': ood_emnist}
    
    with open(result_dir + 'evaluate_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return results


def evaluate_PC():
    pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Generative Model.")
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory for model')
    parser.add_argument('--num_samples', type=int, required=True, help='Number of samples for KID and PRD')
    parser.add_argument('--subset_size', type=int, required=True, help='Subset size for KID calculaiton')
    parser.add_argument('--K', type=int, required=True, help='Number of bootstrap samples for typicality test')
    parser.add_argument('--alpha', type=float, required=True, help='Confidence level for typicality test')
    parser.add_argument('--model_type', type=str, default='MADE', choices=['MADE', 'PC'], help='Type of model to evaluate')

    args = parser.parse_args()
    if args.model_type == 'MADE':
        evaluate_made(args)
    else:
        print(f"Model type {args.model_type} is not supported for evaluation in this script.")

if __name__ == '__main__':
    main()
    