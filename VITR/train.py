import time
import numpy as np
import torch
from transformers import BertTokenizer
import os.path as osp
from lib import data
from lib.model import VSEModel
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores
import logging
import tensorboard_logger as tb_logger
import random
import arguments
from SVD_descriptions import SVDdescriptions
import os

def main():

    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    opt.data_path = os.path.join(opt.data_path, opt.dataset)

    # SVD descriptions
    SVDpath = osp.join(opt.data_path, 'precomp/')
    if not os.path.exists(SVDpath+'train_svd.txt'):
        print('SVD Descriptions')
        SVDdescriptions(SVDpath, SVDpath)

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    train_loader, val_loader = data.get_loaders(
        opt.data_path, tokenizer, opt.batch_size, opt.workers, opt)

    model = VSEModel(opt)
    lr_schedules = [opt.lr_update, 2*opt.lr_update, 3*opt.lr_update, 4*opt.lr_update, 5*opt.lr_update,]

    # optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            if not model.is_data_parallel:
                model.make_data_parallel()
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another training
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))

            # if opt.reset_start_epoch:
            start_epoch = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))

    if not model.is_data_parallel:
        model.make_data_parallel()

    # Train the Model
    best_rsum = 0
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch, lr_schedules)

        if epoch>=opt.max_epochs:
            opt.max_violation = True
            model.set_max_violation(opt.max_violation)

        # train for one epoch
        best_rsum = train(opt, tokenizer, train_loader, model, epoch, val_loader, best_rsum)

        # evaluate on validation set
        rsum = validate(epoch, opt, tokenizer, val_loader, model, Path = opt.data_path)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint.pth'.format(epoch), prefix=opt.model_name + '/')


def train(opt, tokenizer, train_loader, model, epoch, val_loader, best_rsum):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    logger.info('image encoder trainable parameters: {}'.format(count_params(model.img_enc)))
    logger.info('txt encoder trainable parameters: {}'.format(count_params(model.txt_enc)))

    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1

    end = time.time()

    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        images, img_lengths, captions, lengths, images2, ids, img_ids, lsa, capembs, cbembs = train_data
        model.train_emb(epoch, images, captions, lengths, images2, ids, img_ids, lsa, capembs, cbembs, image_lengths=img_lengths)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader.dataset) // train_loader.batch_size + 1, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # evaluate on validation set
        if model.Eiters % opt.val_step == 0 and epoch>=5:
            rsum = validate(epoch, opt, tokenizer, val_loader, model, Path = opt.data_path)

            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            if not os.path.exists(opt.model_name):
                os.mkdir(opt.model_name)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint.pth'.format(epoch), prefix=opt.model_name + '/')
    return best_rsum

def validate(epoch, opt, tokenizer, val_loader, model, Path = ''):
    print('Dev: ')
    currscore_dev = val_dev(epoch, opt, val_loader, model, Path = Path)

    print('Test: ')
    currscore_test = val_test(epoch, opt, tokenizer, model, Path=Path)

    currscore = currscore_dev

    return currscore

def val_dev(epoch, opt, val_loader, model, Path = ''):
    logger = logging.getLogger(__name__)

    model.val_start()

    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        img_embs, img_embgs, cap_embs, bembs, cap_lens, lengths2 = encode_data(
            model, val_loader, opt.log_step, logging.info)

        ID = np.loadtxt(Path + '/precomp/dev_ids.txt', dtype=int, delimiter=',')

        lenImg = max(ID)

        img_embs2 = []
        for i in range(0, lenImg + 1):
            img_embs2.append(img_embs[i])
        img_embs = np.array(img_embs2)
        print('img: ', img_embs.shape)

    start = time.time()
    print('text: ', cap_embs.shape)
    sims = shard_attn_scores(epoch, model, img_embs, img_embgs, cap_embs, bembs, cap_lens, lengths2, opt, shard_size=1000)
    end = time.time()
    logger.info("calculate similarity time: {}".format(end - start))

    print('sims: ', sims.shape)

    # caption retrieval
    npts1 = img_embs.shape[0]
    npts2 = cap_embs.shape[0]

    (r1, r5, r10, medr, meanr) = i2t(npts1, sims, Path=Path, split='dev')
    sumi = r1 + r5 + r10
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, sumi))

    (r1i, r5i, r10i, medri, meanr) = t2i(npts2, sims, Path = Path, split='dev')
    sumt = r1i + r5i + r10i
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, sumt))

    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('Current rsum is {}'.format(currscore))

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def val_test(epoch, opt, tokenizer, model, Path=''):
    logger = logging.getLogger(__name__)

    model.val_start()
    test_loader = data.get_test_loader('test', tokenizer, opt.batch_size, opt.workers, opt)
    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        img_embs, img_embgs, cap_embs, bembs, cap_lens, lengths2 = encode_data(
            model, test_loader, opt.log_step, logging.info)

        ID = np.loadtxt(Path + '/precomp/test_ids.txt', dtype=int, delimiter=',')

        lenImg = max(ID)

        img_embs2 = []
        for i in range(0, lenImg + 1):
            img_embs2.append(img_embs[i])
        img_embs = np.array(img_embs2)
        print('img: ', img_embs.shape)

    start = time.time()
    print('text: ', cap_embs.shape)
    sims = shard_attn_scores(epoch, model, img_embs, img_embgs, cap_embs, bembs, cap_lens, lengths2, opt, shard_size=1000)
    end = time.time()
    logger.info("calculate similarity time: {}".format(end - start))

    print('sims: ', sims.shape)

    # caption retrieval
    npts1 = img_embs.shape[0]
    npts2 = cap_embs.shape[0]

    (r1, r5, r10, medr, meanr) = i2t(npts1, sims, Path=Path, split='test')
    sumi = r1 + r5 + r10
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, sumi))

    (r1i, r5i, r10i, medri, meanr) = t2i(npts2, sims, Path=Path, split='test')
    sumt = r1i + r5i + r10i
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, sumt))

    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('Current rsum is {}'.format(currscore))

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore

def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def adjust_learning_rate(opt, optimizer, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    main()
