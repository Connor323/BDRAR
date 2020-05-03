import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import sbu_testing_root
from misc import check_mkdir, crf_refine
from model import BDRAR

def compute_stats(pred, gt): 
    """Calculate statistics"""
    TP = ((gt != 0) * (pred != 0)).sum()
    FP = ((gt == 0) * (pred != 0)).sum()
    FN = ((gt != 0) * (pred == 0)).sum()

    return [TP, FP, FN]

torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'BDRAR-xView2-new'
args = {
    'snapshot': '3001',
    'scale': 416
}

img_transform = transforms.Compose([
    transforms.Resize(args['scale']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'sbu': sbu_testing_root}

to_pil = transforms.ToPILImage()


def main():
    net = BDRAR().cuda()

    if len(args['snapshot']) > 0:
        print 'load snapshot \'%s\' for testing' % args['snapshot']
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    test_tp, test_fp, test_fn = 0., 0., 0.
    net.eval()
    with torch.no_grad():
        for name, root in to_test.iteritems():
            img_list = sorted([img_name for img_name in os.listdir(os.path.join(root, 'images')) if
                        img_name.endswith('.jpg')])
            gt_list = sorted([img_name for img_name in os.listdir(os.path.join(root, 'masks')) if
                        img_name.endswith('.png')])
            assert len(img_list) == len(gt_list), "%d != %d".format(len(img_list), len(gt_list))

            for idx, (img_name, gt_name) in enumerate(zip(img_list, gt_list)):
                print 'predicting for %s: %d / %d' % (name, idx + 1, len(img_list))
                check_mkdir(
                    os.path.join(ckpt_path, exp_name, '(%s) %s_prediction_%s' % (exp_name, name, args['snapshot'])))
                img = Image.open(os.path.join(root, 'images', img_name))
                gt = Image.open(os.path.join(root, 'masks', gt_name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                res = net(img_var)
                prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
                prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                pred_binary = (np.array(prediction) > 0).astype(int)
                gt = (np.array(gt) > 0).astype(int)
                tp, fp, fn = compute_stats(pred_binary, gt)
                test_tp += tp
                test_fp += fp
                test_fn += fn

    test_prec = test_tp / (test_tp + test_fp)
    test_rec  = test_tp / (test_tp + test_fn)
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec)
    print('Testing percision: %.2f, recall: %.2f, f1: %.2f' % (test_prec, test_rec, test_f1))


if __name__ == '__main__':
    main()
