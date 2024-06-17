import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from lib.loss import ContrastiveLossLSEH
from lib.reasoning_aggregation_modules import EncoderReasoningAggregation
import logging

logger = logging.getLogger(__name__)

def get_text_encoder(bert_size, embed_size):
    return EncoderText(bert_size, embed_size)


def get_image_encoder(img_dim, embed_size, bert_size):
    return EncoderImage(img_dim, embed_size, bert_size)


class EncoderImage(nn.Module):
    def __init__(self, img_dim, embed_size, bert_size):
        super(EncoderImage, self).__init__()
        self.bert_size = bert_size
        self.embed_size = embed_size
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths, images2):
        bert_size = self.bert_size
        img_embg = images2
        img_emb = images[:, :, :]
        img_emb = self.fc(img_emb)

        return img_emb, img_embg

class EncoderText(nn.Module):
    def __init__(self, bert_size, embed_size):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(bert_size, embed_size)
        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, x, lengths, capembs, bemb):

        bert_emb = capembs[:, :, :]
        cap_len = lengths.cpu()
        cap_emb = self.fc(bert_emb)

        return cap_emb, bemb

class VSEModel(object):

    def __init__(self, opt):

        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(opt.img_dim, opt.embed_size, opt.bert_size)
        self.txt_enc = get_text_encoder(opt.bert_size, opt.embed_size)
        self.sim_enc = EncoderReasoningAggregation(opt.embed_size, opt.sim_dim, opt.bert_size, opt.thre_cat)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True

        self.criterion = ContrastiveLossLSEH(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())

        self.params = params
        self.opt = opt

        decay_factor = 1e-4
        if self.opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW([
                {'params': self.txt_enc.parameters(), 'lr': opt.learning_rate},
                {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                {'params': self.sim_enc.parameters(), 'lr': opt.learning_rate},
            ],
                lr=opt.learning_rate, weight_decay=decay_factor)
        elif self.opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
        else:
            raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)
        self.sim_enc.load_state_dict(state_dict[2], strict=False)

    def train_start(self):

        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):

        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, lengths, images2, capembs, cbembs, image_lengths=None):

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            image_lengths = image_lengths.cuda()
        img_emb, img_embg = self.img_enc(images, image_lengths, images2)

        lengths = torch.Tensor(lengths).cuda()
        cap_emb, bemb = self.txt_enc(captions, lengths, capembs, cbembs)
        return img_emb, img_embg, cap_emb, bemb, lengths, image_lengths

    def forward_sim(self, epoch, img_embs, img_embgs, cap_embs, bembs, cap_lens, lengths2):

        sims = self.sim_enc(epoch, img_embs, img_embgs, cap_embs, bembs, cap_lens, lengths2)
        return sims

    def forward_loss(self, sims, img_emb, cap_emb, ids, img_ids, lsa, bemb):

        loss, lossS, lossIm = self.criterion(sims, img_emb, cap_emb, ids, img_ids, lsa, bemb)
        self.logger.update('Le', loss.data.item(), img_emb.size(0))
        self.logger.update('Le_S', lossS.item(), img_emb.size(0))
        self.logger.update('Le_Im', lossIm.item(), img_emb.size(0))
        return loss

    def train_emb(self, epoch, images, captions, lengths, lengths2, ids, img_ids, lsa, capembs, cbembs, image_lengths=None, warmup_alpha=None):

        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_emb, img_embg, cap_emb, bemb, cap_lens, lengths2 = self.forward_emb(images, captions, lengths, lengths2, capembs, cbembs, image_lengths=image_lengths)
        sims = self.forward_sim(epoch, img_emb, img_embg, cap_emb, bemb, cap_lens, lengths2)

        self.optimizer.zero_grad(set_to_none = True)
        loss = self.forward_loss(sims, img_emb, cap_emb, ids, img_ids, lsa, bemb)

        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()




