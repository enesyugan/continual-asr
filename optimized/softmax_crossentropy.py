import torch


class SoftmaxCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, padding_idx=-100, half_to_float=False):
        losses, max_log_sum_exp = xentropy_cuda.forward(
            logits, labels, smoothing, half_to_float)
        losses.masked_fill_(labels == padding_idx, 0)

        ctx.save_for_backward(logits, max_log_sum_exp, labels,
                              torch.FloatTensor([smoothing]),
                              torch.LongTensor([padding_idx]))

        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logits, max_log_sum_exp, labels, smoothing, padding_idx = ctx.saved_tensors

        if not grad_loss.is_contiguous():
            grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels == padding_idx.item(), 0)
        grad_logits = xentropy_cuda.backward(
            grad_loss.contiguous(), logits, max_log_sum_exp,
            labels, smoothing.item())

        return grad_logits, None, None, None, None


try:
    import xentropy_cuda

    softmax_xentropy = SoftmaxCrossEntropyLoss.apply
    fast_xentropy = True
except (ModuleNotFoundError, AttributeError):
    softmax_xentropy = None
    fast_xentropy = False

if fast_xentropy:
    print("[INFO] Fast entropy is active.")