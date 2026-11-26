import math

POLICY = {'minimum_completed_epoch': 20, 'patience_epochs': 10, 'minimum_delta': 1e-12}


def recovery_decision(history, policy=None):
    policy = POLICY if policy is None else policy
    rows = [r for r in history if r['epoch'] > 0]
    if not rows:
        return {'stop': False, 'reason': 'no_trained_epochs', 'epoch0_excluded': True}
    epochs = [r['epoch'] for r in rows]
    if epochs != list(range(1, epochs[-1]+1)):
        raise ValueError('Expected complete, ordered trained epochs starting at1')
    peaks, last = {}, {}
    for key in ('acc2', 'f1_weighted'):
        peak, changed = -float('inf'), None
        for row in rows:
            metric = row['valid'].get('M', row['valid'])
            value = metric[key]
            if not math.isfinite(value):
                raise ValueError('Nonfinite VALID metric')
            if value > peak+policy['minimum_delta']:
                peak, changed = value, row['epoch']
        peaks[key], last[key] = peak, changed
    epoch = epochs[-1]
    stalled = all(epoch-last[k] >= policy['patience_epochs'] for k in last)
    stop = epoch >= policy['minimum_completed_epoch'] and stalled
    return {'stop': stop, 'completed_epoch': epoch, 'epoch0_excluded': True,
            'best_trained_metrics': peaks, 'last_improvement_epochs': last,
            'epochs_without_improvement': {k: epoch-last[k] for k in last},
            'reason': 'trained_validation_stalled' if stop else 'continue_observation'}
