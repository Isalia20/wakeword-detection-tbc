import torch

class BatchCollator:
    def __init__(self):
        super().__init__()
    
    def __call__(self, batch):
        batch = list(zip(*[i for i in batch if i is not None]))
        waveforms = batch[0]
        targets = batch[1]
        return waveforms, targets
    
def make_dataloader(dataset, phase, batch_size, num_workers):
    collator = BatchCollator()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        shuffle=phase=="train",
    )
    return data_loader