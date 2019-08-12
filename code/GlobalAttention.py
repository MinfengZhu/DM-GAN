"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        #self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context_key, content_value):#
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context_key.size(0), context_key.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        #sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        #sourceT = self.conv_context(sourceT).squeeze(3)
        sourceT = context_key

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)

        text_weighted = None
        # text_attn = torch.transpose(attn, 1, 2).contiguous() # batch x sourceL x queryL
        # text_attn = text_attn.view(batch_size*sourceL, queryL)
        # if self.mask is not None:
        #     mask = self.mask.repeat(queryL, 1)
        #     mask = mask.view(batch_size, queryL, sourceL)
        #     mask = torch.transpose(mask, 1, 2).contiguous()
        #     mask = mask.view(batch_size*sourceL, queryL)
        #     text_attn.data.masked_fill_(mask.data, -float('inf'))
        # text_attn = self.sm(text_attn)
        # text_attn = text_attn.view(batch_size,sourceL, queryL)
        # text_attn = torch.transpose(text_attn, 1, 2).contiguous() # batch x queryL x sourceL
        # # (batch x idf x queryL) * (batch x queryL x sourceL) -> batch x idf x sourceL
        # text_weighted = torch.bmm(target, text_attn)

        # --> batch*queryL x sourceL
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(content_value, attn)  #
        #weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn


class GlobalAttention_text(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttention_text, self).__init__()
        self.conv_context = nn.Conv1d(cdf, idf, kernel_size=1, stride=1, padding=0)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = self.conv_context(context)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)

        # --> batch*queryL x sourceL
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        #attn_o = self.sm(attn)  # Eq. (2)
        #attn_o = attn_o.view(batch_size, queryL, sourceL)

        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.nn.Softmax(dim=1)(attn)

        #import ipdb;
        #ipdb.set_trace()  # BREAKPOINT

        # (batch x idf x queryL) * (batch x queryL x sourceL) -> batch x idf x sourceL
        text_weighted = torch.bmm(target, attn)

        return text_weighted
