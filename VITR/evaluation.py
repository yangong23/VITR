"""Evaluation"""
from __future__ import print_function
import logging
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from transformers import BertTokenizer
from lib import data
from lib.model import VSEModel
import pickle

logger = logging.getLogger(__name__)
import os


def main():
    evalrank("./runs/F30K/model/model_best.pth",
             data_path="/home/datasets",
             split="test")

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)

def encode_data(model, data_loader, log_step=10, logging=logger.info):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    for i, data_i in enumerate(data_loader):

        images, image_lengths, captions, lengths, images2, ids, img_ids, lsa, capembs, cbembs = data_i

        model.logger = val_logger

        # compute the embeddings
        # if not backbone:
        img_emb, img_embg, cap_emb, bemb, cap_len, cap_len2 = model.forward_emb(images, captions, lengths, images2, capembs, cbembs, image_lengths=image_lengths)

        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                img_embgs = np.zeros((len(data_loader.dataset), img_embg.size(1), img_embg.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                img_embgs = np.zeros((len(data_loader.dataset), img_embg.size(1)))

            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.shape[1], cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
            cap_lens2 = [0] * len(data_loader.dataset)
            bembs = np.zeros((len(data_loader.dataset), bemb.size(1), bemb.size(2)))
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        img_embgs[ids] = img_embg.data.cpu().numpy().copy()

        cap_embs[ids, :, :] = cap_emb[:, :, :].data.cpu().numpy().copy()
        bembs[ids, :, :] = bemb[:, :, :].data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]
            cap_lens2[nid] = cap_len2[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time,
                e_log=str(model.logger)))
        del images, captions

    return img_embs, img_embgs, cap_embs, bembs, cap_lens, cap_lens2


def evalrank(model_path, data_path=None, split='test'):

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.workers = 5
    opt.batch_size = 128

    logger.info(opt)
    if not hasattr(opt, 'caption_loss'):
        opt.caption_loss = False

    # load vocabulary used by the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    if data_path is not None:
        opt.data_path = os.path.join(data_path, opt.dataset)
        data_path = opt.data_path

    # construct model
    model = VSEModel(opt)

    model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    test_loader = data.get_test_loader(split, tokenizer, opt.batch_size, opt.workers, opt)
    with torch.no_grad():
        img_embs, img_embgs, cap_embs, bembs, cap_lens, lengths2 = encode_data(
            model, test_loader, opt.log_step, logging.info)

        ID = np.loadtxt(data_path + '/precomp/' + split + '_ids.txt', dtype=int, delimiter=',')

        lenImg = max(ID)

        img_embs2 = []
        img_embgs2 = []
        for i in range(0, lenImg + 1):
            img_embs2.append(img_embs[i])
            img_embgs2.append(img_embgs[i])
        img_embs = np.array(img_embs2)
        img_embgs = np.array(img_embgs2)
        print('img: ', img_embs.shape)

    start = time.time()

    print('text: ', cap_embs.shape)
    sims = shard_attn_scores(100000, model, img_embs, img_embgs, cap_embs, bembs, cap_lens, lengths2, opt,
                             shard_size=1000)
    end = time.time()
    logger.info("calculate similarity time: {}".format(end - start))

    print('sims: ', sims.shape)

    # caption retrieval
    npts1 = img_embs.shape[0]
    npts2 = cap_embs.shape[0]

    (r1, r5, r10, medr, meanr) = i2t(npts1, sims, Path=data_path, split=split)
    sumi = r1 + r5 + r10
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, sumi))

    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, sumi))

    (r1i, r5i, r10i, medri, meanr) = t2i(npts2, sims, Path=data_path, split=split)
    sumt = r1i + r5i + r10i
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, sumt))

    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, sumt))

    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('Current rsum is {}'.format(currscore))

def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities

def shard_attn_scores(epoch, model, img_embs, img_embgs, cap_embs, bembs, cap_lens, cap_lens2, opt, shard_size=100):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            # sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                img = torch.from_numpy(img_embgs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                cag = torch.from_numpy(bembs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                l2 = cap_lens2[ca_start:ca_end]

                sim = model.forward_sim(epoch, im, img, ca, cag, l, l2)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()

    # sys.stdout.write('\n')
    return sims

def i2t(npts, sims, Path = '', split='dev', return_ranks=False):

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    sims = sims
    releIDs = load_obj(Path + '/precomp/' + split +'_releIDi2t.pkl')

    for index in tqdm(range(npts)):
        releID = releIDs[str(index)]
        inds = np.argsort(sims[index])[::-1]

        rind = []
        for j in range(len(releID)):
            rind.append(np.where(inds == releID[j])[0][0])

        rkind = min(rind)
        ranks[index] = rkind
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i(npts, sims, Path = '', split='dev', return_ranks=False):

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    sims = sims.T
    releIDs = load_obj(Path + '/precomp/' + split + '_releIDt2i.pkl')

    for index in tqdm(range(npts)):
        releID = releIDs[str(index)]
        inds = np.argsort(sims[index])[::-1]

        rind = []
        for j in range(len(releID)):
            rind.append(np.where(inds == releID[j])[0][0])

        rkind = min(rind)
        ranks[index] = rkind
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

if __name__ == '__main__':
    main()
