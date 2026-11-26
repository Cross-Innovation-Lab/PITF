def mono_stop_decision(stage, epoch, best_epoch, *, patience, min_epochs,
                       save_after_epoch, checkpoint_exists):
    if stage not in ('T', 'V', 'A') or patience <= 0:
        return None
    minimum = max(min_epochs, save_after_epoch + 1)
    if best_epoch is None or not checkpoint_exists or epoch < minimum:
        return None
    stale = epoch - best_epoch
    if stale < patience:
        return None
    return {'reason': 'VALID MAE did not improve', 'epoch': epoch,
            'best_epoch': best_epoch, 'epochs_without_improvement': stale,
            'patience': patience, 'min_epochs': minimum,
            'test_used': False}
