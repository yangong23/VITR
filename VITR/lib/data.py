import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class PrecompRegionDataset(data.Dataset):

    def __init__(self, data_path, data_split, tokenizer, opt, train):
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train
        self.data_path = data_path

        self.initial_prob = 0
        self.max_prob = 8
        self.prob_increment = 1
        self.current_prob = self.initial_prob

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')

        IdPath = osp.join(loc_cap, '%s_ids.txt' % data_split)
        self.ID = np.loadtxt(IdPath, dtype=int, delimiter=',')

        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        self.captions2 = self.captions
        self.images = np.load(os.path.join(loc_image, '%s_ims_RN.npy' % data_split))
        if self.images.shape[1]==50:
            self.images = self.images[:,1:,:]
        self.images2 = np.load(os.path.join(loc_image, '%s_ims_ViT.npy' % data_split))
        self.images2 = self.images2[:,:1,:opt.bert_size]
        self.capembs = np.load(os.path.join(loc_cap, '%s_text_ViT.npy' % data_split))
        self.cbembs = np.load(os.path.join(loc_cap, '%s_text_ViT2.npy' % data_split))
        self.cbembs = self.cbembs[:,:1,:]
        self.length = len(self.captions)

        print(data_split, 'images: ', len(self.images))
        print(data_split, 'captions: ', len(self.captions))

        if data_split == 'train':
            self.dv = np.loadtxt(os.path.join(loc_cap, '%s_svd.txt' % data_split), dtype=float, delimiter=',')
        else:
            self.dv = np.random.rand(len(self.captions), 400) * 2 - 1

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = self.ID[index]

        caption = self.captions[index]

        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        target = process_caption(self.tokenizer, caption_tokens, self.train)
        capembs = self.capembs[index]
        capembs = torch.Tensor(capembs)
        cbembs = self.cbembs[index]
        cbembs = torch.Tensor(cbembs)
        dv = self.dv[index]
        dv = torch.Tensor(dv)
        image = self.images[img_index]

        prob = random.randint(1, 100)
        if prob < self.current_prob and self.train:
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            filtered_indices = np.where(rand_list < 0.20)[0]
            if len(filtered_indices) > 0:
                image[filtered_indices] = 0.0

        image2 = self.images2[img_index]
        image = torch.Tensor(image)
        image2 = torch.Tensor(image2)

        return image, target, image2, index, img_index, dv, capembs, cbembs

    def __len__(self):
        return self.length

def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
                    # print(output_tokens)
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
                # print(deleted_idx)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ["<|startoftext|>"] + output_tokens + ["<|endoftext|>"]

    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, images2, ids, img_ids, lsa, capembs, cbembs = zip(*data)
    img_lengths = [len(image) for image in images]
    all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
    img_lengths2 = [len(image2) for image2 in images2]
    all_images2 = torch.zeros(len(images2), max(img_lengths2), images2[0].size(-1))

    for i, image in enumerate(images):
        end = img_lengths[i]
        all_images[i, :end] = image[:end]
    img_lengths = torch.Tensor(img_lengths)

    for i, image2 in enumerate(images2):
        end2 = img_lengths2[i]
        all_images2[i, :end2] = image2[:end2]
    img_lengths2 = torch.Tensor(img_lengths2)

    numSE = 2
    lengths = []
    for cap in captions:
        le = len(cap) + numSE
        if le > 77:
            le = 77
        lengths.append(le)

    lengths2 = lengths
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i] - numSE
        targets[i, :end] = cap[:end]
    lsa = torch.stack(lsa, 0)
    capembs = torch.stack(capembs, 0)
    cbembs = torch.stack(cbembs, 0)

    return all_images, img_lengths, targets, lengths, all_images2, ids, img_ids, lsa, capembs, cbembs

def get_loader(data_path, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    if train:
        drop_last = True
    else:
        drop_last = False
    dset = PrecompRegionDataset(data_path, data_split, tokenizer, opt, train)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
    return data_loader


def get_loaders(data_path, tokenizer, batch_size, workers, opt):
    train_loader = get_loader(data_path, 'train', tokenizer, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, 'dev', tokenizer, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, tokenizer, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, split_name, tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader


