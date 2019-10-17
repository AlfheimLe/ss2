from get_args import process_args
from model import DSGM
import os
import pprint
import torch
if __name__ == '__main__':
    args = process_args()

    args.device = torch.device(
        'cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda != None else 'cpu')
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    model = DSGM(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.restore_file:
        # load model and optimizer states
        state = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(state)
        # set up paths
        args.output_dir = os.path.dirname(args.restore_file)
    args.results_file = os.path.join(args.output_dir, args.results_file)

    print('Loaded settings and model:')
    print(pprint.pformat(args.__dict__))
    print(model)
    print(pprint.pformat(args.__dict__), file=open(args.results_file, 'a'))
    print(model, file=open(args.results_file, 'a'))

    if args.train:
        train_and_evaluate(model, labeled_dataloader, unlabeled_dataloader, test_dataloader, loss_components_fn,
                           optimizer, args)

    if args.evaluate:
        evaluate(model, test_dataloader, None, args)

    if args.generate:
        generate(model, test_dataloader.dataset, args)

    if args.vis_styles:
        vis_styles(model, args)